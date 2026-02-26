"""
Training eval hook: run evals periodically during training.

Integrates evals into the training loop and returns a flat
dict of metrics suitable for wandb.log().

Usage from train.py:

    from evals.training_eval_hook import run_training_evals

    # Inside training loop, every N steps:
    metrics = run_training_evals(
        model, tokenizer, model_name="Qwen/Qwen3-8B",
        step=global_step, device="cuda",
        eval_dir="data/evals", max_items_per_eval=20,
    )
    wandb.log(metrics, step=global_step)
"""

from __future__ import annotations

import gc
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any

import torch

# Import from the existing eval infrastructure
from evals.common import (
    EvalItem,
    CompletedEvalItem,
    load_eval_items,
    load_eval_items_hf,
    list_hf_evals,
    parse_oracle_binary,
    determine_ground_truth,
)
from evals.score_oracle import (
    EVAL_PARSING,
    _score_sentence_insertion,
    _score_reconstruction_metrics,
)
from evals.run_evals import (
    ORACLE_PROMPTS_TEMPLATES,
    _oracle_prompt,
    _extract_answer,
    _token_match_rate,
    _token_unigram_kl,
    _rot13,
    set_oracle_mode,
    _ORACLE_MODE,
)
from evals.activation_cache import (
    ActivationBundle,
    extract_activations as _extract_activations,
    cache_path as _cache_path,
    maybe_load_cached_bundle as _maybe_load_cached,
    save_bundle_with_metadata as _save_bundle_with_metadata,
)
from core.ao import (
    run_oracle_on_activations as _run_oracle_raw,
    load_extra_adapter,
    layer_percent_to_layer,
    get_batched_steering_hook,
    get_hf_submodule,
    add_hook,
    _active_adapter_name,
    TRAINED_PLACEHOLDER,
    SPECIAL_TOKEN,
)
from cot_utils import get_injection_layers


# Evals to run during training, in order of cost (cheapest first).
# This is the fallback if config doesn't specify eval.unfaith_evals.
TRAINING_EVALS = [
    "hinted_mcq_truthfulqa",
    "sycophancy_v2_riya",
    "sentence_insertion",
    "reasoning_termination_riya",
    "atypical_answer_riya",
    "atypical_answer_mcq",
    "cybercrime_ood",
    "hint_admission",
    "rot13_reconstruction",
]

ROT13_ADAPTER_HF = "ceselder/rot13-qwen3-8b-lora"
ROT13_ADAPTER_NAME = "rot13"


def _first_rollout(rollouts) -> str | None:
    """Extract first rollout from a list of precomputed rollouts."""
    if isinstance(rollouts, list) and len(rollouts) > 0:
        return rollouts[0]
    return None


def _apply_oracle_mode_to_extract(model, tokenizer, **kwargs):
    """Wrapper applying current oracle mode to activation extraction.

    Delegates to extract_activations() with stride and layers from _ORACLE_MODE.
    """
    kwargs.setdefault("stride", _ORACLE_MODE["stride"])
    kwargs.setdefault("layers", _ORACLE_MODE.get("layers"))
    kwargs.pop("max_boundaries", None)
    return _extract_activations(model, tokenizer, **kwargs)


def _apply_oracle_mode_to_oracle(model, tokenizer, activations, prompt, **kwargs):
    """Wrapper applying current oracle mode to oracle inference."""
    kwargs.setdefault("placeholder_token", _ORACLE_MODE["placeholder_token"])
    kwargs.setdefault("oracle_adapter_name", _ORACLE_MODE["oracle_adapter_name"])
    # Pass layers list as act_layer so the prefix matches training format
    layers = _ORACLE_MODE.get("layers")
    if layers and len(layers) > 1:
        kwargs["act_layer"] = layers
    return _run_oracle_raw(model, tokenizer, activations, prompt, **kwargs)


@torch.no_grad()
def _batched_oracle_generate(
    model,
    tokenizer,
    items: list[tuple[torch.Tensor, str]],
    model_name: str,
    device: str = "cuda",
    injection_layer: int = 1,
    max_new_tokens: int = 100,
    eval_batch_size: int = 8,
) -> list[str]:
    """Batched oracle generation with per-item activation steering.

    Uses get_batched_steering_hook (from Adam's AO) for ragged-batch support:
    each item can have different K (number of activation positions).

    Args:
        items: List of (activations [K_i, D], oracle_prompt_text) tuples.
        model_name: HF model name (for adapter resolution).
        device: Target device.
        injection_layer: Layer to inject at (default 1).
        max_new_tokens: Max tokens to generate per item.
        eval_batch_size: Mini-batch size for generation.

    Returns:
        List of oracle response strings, one per input item.
    """
    if not items:
        return []

    dtype = torch.bfloat16
    ph_token = _ORACLE_MODE.get("placeholder_token") or SPECIAL_TOKEN
    oracle_adapter_name = _ORACLE_MODE.get("oracle_adapter_name")
    layers = _ORACLE_MODE.get("layers")

    # Build prefix layer string once
    if layers and len(layers) > 1:
        layers_str = ", ".join(str(l) for l in layers)
    else:
        act_layer = layer_percent_to_layer(model_name, 50)
        layers_str = str(act_layer)

    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Expected single token for '{ph_token}', got {len(ph_id)}"
    ph_id = ph_id[0]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # --- Phase 1: Tokenize all items and find placeholder positions ---
    all_input_ids: list[list[int]] = []
    all_ph_positions: list[list[int]] = []

    for activations, oracle_prompt in items:
        num_positions = activations.shape[0]
        prefix = f"Layer: {layers_str}\n" + ph_token * num_positions + " \n"
        full_prompt = prefix + oracle_prompt

        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)

        positions = []
        for i, tid in enumerate(input_ids):
            if tid == ph_id and len(positions) < num_positions:
                positions.append(i)
        assert len(positions) == num_positions, (
            f"Found {len(positions)} placeholder positions, expected {num_positions}"
        )

        all_input_ids.append(input_ids)
        all_ph_positions.append(positions)

    # --- Phase 2: Set adapter once for all batches ---
    previous_adapter = _active_adapter_name(model)
    if oracle_adapter_name is not None:
        model.set_adapter(oracle_adapter_name)

    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    was_training = model.training
    model.eval()

    # --- Phase 3: Generate in mini-batches ---
    all_responses: list[str] = [""] * len(items)

    try:
        for batch_start in range(0, len(items), eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, len(items))

            try:
                batch_ids = all_input_ids[batch_start:batch_end]
                batch_pre_pad_pos = all_ph_positions[batch_start:batch_end]
                batch_acts = [items[i][0] for i in range(batch_start, batch_end)]

                # Left-pad to max length in this mini-batch
                max_len = max(len(ids) for ids in batch_ids)
                padded_ids = []
                attention_masks = []
                batch_padded_positions = []

                for j, ids in enumerate(batch_ids):
                    pad_len = max_len - len(ids)
                    padded_ids.append([pad_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))
                    # Shift placeholder positions by padding offset
                    batch_padded_positions.append([p + pad_len for p in batch_pre_pad_pos[j]])

                input_tensor = torch.tensor(padded_ids, device=device)
                attn_mask = torch.tensor(attention_masks, device=device)

                hook_fn = get_batched_steering_hook(
                    vectors=batch_acts,
                    positions=batch_padded_positions,
                    device=device,
                    dtype=dtype,
                )

                with add_hook(injection_submodule, hook_fn):
                    outputs = model.generate(
                        input_ids=input_tensor,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )

                # Decode generated tokens (everything after the padded prompt)
                for j in range(len(batch_ids)):
                    generated = outputs[j][max_len:]
                    all_responses[batch_start + j] = tokenizer.decode(
                        generated, skip_special_tokens=True,
                    )
            except Exception as e:
                print(f"    [batched_oracle] Mini-batch {batch_start}-{batch_end} failed: {e}")
                # Leave responses as "" for this mini-batch
    finally:
        if was_training:
            model.train()
        if (previous_adapter
                and previous_adapter in getattr(model, "peft_config", {})
                and previous_adapter != oracle_adapter_name):
            model.set_adapter(previous_adapter)

    return all_responses


def _subsample(items: list[EvalItem], max_items: int, seed: int) -> list[EvalItem]:
    """Deterministically subsample items for fast training evals.

    Oversamples by 1.5x to compensate for items that become "indeterminate"
    during scoring (ground truth filtering).  The scoring functions cap the
    final scoreable count at ``max_items`` so we never report *more* than
    the caller intended.
    """
    oversample = int(max_items * 1.5)
    if len(items) <= oversample:
        return items
    rng = random.Random(seed)
    return rng.sample(items, oversample)


def _try_load_cached(cache_dir: Path | None, eval_name: str, example_id: str, device: str):
    """Try to load a cached activation bundle. Returns bundle or None."""
    return _maybe_load_cached(
        cache_dir,
        eval_name=eval_name,
        example_id=example_id,
        map_location=device,
        stride=_ORACLE_MODE.get("stride"),
        layers=_ORACLE_MODE.get("layers"),
    )


def _auto_cache_bundle(
    cache_dir: Path | None,
    *,
    eval_name: str,
    example_id: str,
    prompt: str,
    cot_text: str,
    activations: torch.Tensor | None,
    boundary_positions: list[int],
    clean_response: str = "",
    test_response: str = "",
    clean_answer: str | None = None,
    test_answer: str | None = None,
) -> None:
    """Save an activation bundle to cache for reuse in future evals."""
    if cache_dir is None or activations is None:
        return
    bundle = ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=boundary_positions,
        sentences=[],
        clean_response=clean_response,
        test_response=test_response,
        clean_answer=clean_answer,
        test_answer=test_answer,
    )
    _save_bundle_with_metadata(
        bundle, cache_dir,
        stride=_ORACLE_MODE.get("stride"),
        layers=_ORACLE_MODE.get("layers"),
    )


def _collect_and_batch_oracle(
    extracted: list[dict],
    model,
    tokenizer,
    model_name: str,
    device: str,
    eval_name: str,
    eval_batch_size: int = 8,
    max_new_tokens: int = 100,
):
    """Run batched oracle generation on extracted items and store responses.

    Each dict in `extracted` should have:
    - "activations": torch.Tensor | None
    - "oracle_prompt": str (empty string if no activations)

    Sets "oracle_response" on each dict that had activations+prompt.
    """
    oracle_items = []
    oracle_indices = []
    for i, ex in enumerate(extracted):
        if ex.get("activations") is not None and ex.get("oracle_prompt"):
            oracle_items.append((ex["activations"], ex["oracle_prompt"]))
            oracle_indices.append(i)

    if not oracle_items:
        return

    try:
        responses = _batched_oracle_generate(
            model, tokenizer, oracle_items, model_name=model_name,
            device=device, eval_batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        for j, idx in enumerate(oracle_indices):
            extracted[idx]["oracle_response"] = responses[j]
    except Exception as e:
        print(f"    [{eval_name}] Batched oracle generation failed: {e}")
        traceback.print_exc()


def _run_standard_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    eval_name: str,
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """Run a standard binary eval (hinted_mcq, sycophancy_v2_riya, etc.).

    Flow (batched):
    1. Extract activations + metadata for all items
    2. Batch all oracle queries via _batched_oracle_generate
    3. Assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract activations and collect metadata ---
    extracted = []

    for item in items:
        try:
            # Try cached bundle first (has activations + precomputed responses)
            cached = _try_load_cached(cache_dir, eval_name, item.example_id, device)
            if cached is not None:
                clean_response = cached.clean_response or ""
                test_response = cached.test_response or ""
                clean_answer = cached.clean_answer or _extract_answer(clean_response, eval_name)
                test_answer = cached.test_answer or _extract_answer(test_response, eval_name)
                activations = cached.activations
            else:
                # Use precomputed text responses from metadata if available
                # Support multiple field naming conventions across evals
                precomp_clean = (
                    item.metadata.get("qwen3_8b_clean_response")
                    or _first_rollout(item.metadata.get("clean_rollouts"))
                )
                precomp_test = (
                    item.metadata.get("qwen3_8b_test_response")
                    or item.metadata.get("representative_response")  # sycophancy_v2_riya
                    or item.metadata.get("cot_text")  # atypical_answer evals (precomputed CoTs)
                    or _first_rollout(item.metadata.get("hinted_rollouts"))
                )
                # For evals with no nudge (clean == test prompt), cot_text IS the clean response
                if not precomp_clean and precomp_test and item.clean_prompt == item.test_prompt:
                    precomp_clean = precomp_test

                if precomp_clean and precomp_test:
                    clean_response = precomp_clean
                    test_response = precomp_test
                    clean_answer = (
                        item.metadata.get("qwen3_8b_clean_answer")
                        or _extract_answer(clean_response, eval_name)
                    )
                    test_answer = (
                        item.metadata.get("qwen3_8b_test_answer")
                        or _extract_answer(test_response, eval_name)
                    )
                elif precomp_test:
                    # Missing clean response — all eval datasets should have this precomputed
                    print(f"    *** PANIC [{eval_name}] {item.example_id}: no precomputed clean_response! "
                          f"Has: test_response but missing clean. Dataset needs reprocessing. ***")
                    test_response = precomp_test
                    test_answer = (
                        item.metadata.get("qwen3_8b_test_answer")
                        or _extract_answer(test_response, eval_name)
                    )
                    clean_response = ""
                    clean_answer = None
                else:
                    # Missing BOTH responses — dataset is broken for this item
                    print(f"    *** PANIC [{eval_name}] {item.example_id}: no precomputed responses at all! "
                          f"Keys: {list(item.metadata.keys())[:10]}. Skipping. ***")
                    continue

                # Extract activations (forward pass only, no generation)
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name=eval_name,
                        example_id=item.example_id,
                        prompt=item.test_prompt,
                        cot_text=test_response,
                        act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                        device=device,
                        max_boundaries=10,
                        generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [{eval_name}] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None

                # Auto-cache for next eval run
                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        clean_response=clean_response, test_response=test_response,
                        clean_answer=clean_answer, test_answer=test_answer,
                    )

            # Build oracle prompt
            oracle_prompt = ""
            if activations is not None:
                n_positions = activations.shape[0]
                template = ORACLE_PROMPTS_TEMPLATES.get(eval_name, "What is this model doing?")
                oracle_prompt = _oracle_prompt(n_positions, template)

            ground_truth = determine_ground_truth(item, clean_answer, test_answer)
            extracted.append({
                "item": item,
                "clean_response": clean_response,
                "test_response": test_response,
                "clean_answer": clean_answer,
                "test_answer": test_answer,
                "activations": activations,
                "ground_truth": ground_truth,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name=eval_name, eval_batch_size=eval_batch_size,
    )

    # --- Pass 3: Build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        completed.append(CompletedEvalItem(
            eval_name=eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            clean_response=ex["clean_response"],
            test_response=ex["test_response"],
            clean_answer=ex["clean_answer"],
            test_answer=ex["test_answer"],
            ground_truth_label=ex["ground_truth"],
            oracle_response=ex["oracle_response"],
            activations_path=None,
            metadata={**item.metadata},
        ))

    return completed


def _run_decorative_cot_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """Decorative CoT eval. Uses cached activations + static labels when available.

    Flow (batched):
    1. Extract activations + metadata for all items
    2. Batch all oracle queries
    3. Assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract ---
    extracted = []

    for item in items:
        try:
            # Try cached bundle first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                # Use cached activations and precomputed label from metadata
                label = item.metadata.get("decorative_label", "indeterminate")
                representative_cot = cached.test_response or ""
                activations = cached.activations
                with_cot_acc = item.metadata.get("with_cot_acc", 0.0)
                without_cot_acc = item.metadata.get("without_cot_acc", 0.0)
            elif item.metadata.get("decorative_label"):
                # Static label in dataset but no cached activations — need extraction
                label = item.metadata["decorative_label"]
                with_cot_acc = item.metadata.get("with_cot_acc", 0.0)
                without_cot_acc = item.metadata.get("without_cot_acc", 0.0)
                representative_cot = item.metadata.get("representative_cot", "")
                if not representative_cot:
                    print(f"    *** PANIC [decorative_cot] {item.example_id}: has decorative_label but "
                          f"no representative_cot in metadata. Dataset needs reprocessing. ***")
                activations = None
                if representative_cot:
                    try:
                        bundle = _apply_oracle_mode_to_extract(
                            model, tokenizer,
                            eval_name=item.eval_name, example_id=item.example_id,
                            prompt=item.test_prompt, cot_text=representative_cot,
                            act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                            device=device,
                            max_boundaries=10, generation_adapter_name=None,
                        )
                        if bundle and bundle.activations is not None:
                            activations = bundle.activations
                            _auto_cache_bundle(
                                cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                                prompt=item.test_prompt, cot_text=representative_cot,
                                activations=activations, boundary_positions=bundle.boundary_positions,
                                test_response=representative_cot,
                            )
                    except Exception as e:
                        print(f"    [decorative_cot] Activation extraction failed for {item.example_id}: {e}")
            else:
                # No decorative_label AND no cache — dataset is missing precomputed labels
                print(f"    *** PANIC [decorative_cot] {item.example_id}: no decorative_label in metadata! "
                      f"Keys: {list(item.metadata.keys())[:10]}. Skipping. ***")
                continue

            # Build oracle prompt
            oracle_prompt = ""
            if activations is not None:
                template = ORACLE_PROMPTS_TEMPLATES["decorative_cot"]
                oracle_prompt = _oracle_prompt(activations.shape[0], template)

            extracted.append({
                "item": item,
                "representative_cot": representative_cot,
                "label": label,
                "with_cot_acc": with_cot_acc,
                "without_cot_acc": without_cot_acc,
                "activations": activations,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: decorative_cot item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name="decorative_cot", eval_batch_size=eval_batch_size,
    )

    # --- Pass 3: Build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        completed.append(CompletedEvalItem(
            eval_name=item.eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=None,
            clean_response="",
            test_response=ex["representative_cot"],
            clean_answer=None,
            test_answer=None,
            ground_truth_label=ex["label"],
            oracle_response=ex["oracle_response"],
            activations_path=None,
            metadata={
                **item.metadata,
                "with_cot_acc": ex["with_cot_acc"],
                "without_cot_acc": ex["without_cot_acc"],
            },
        ))

    return completed


def _run_sentence_insertion_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """Run sentence insertion eval from pre-spliced CoTs in metadata.

    Flow (batched):
    1. Extract activations for all items
    2. Batch all oracle queries
    3. Assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract ---
    extracted = []

    for item in items:
        try:
            test_response = item.metadata.get("spliced_cot_text", "")

            # Try cached activations first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)
            if cached is not None:
                activations = cached.activations
            else:
                # Extract activations (forward pass only)
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                        device=device,
                        max_boundaries=30, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [sentence_insertion] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=test_response,
                    )

            # Build oracle prompt
            oracle_prompt = ""
            if activations is not None:
                template = ORACLE_PROMPTS_TEMPLATES["sentence_insertion"]
                oracle_prompt = _oracle_prompt(activations.shape[0], template)

            ground_truth = determine_ground_truth(item, None, None)
            extracted.append({
                "item": item,
                "test_response": test_response,
                "ground_truth": ground_truth,
                "activations": activations,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: sentence_insertion item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name="sentence_insertion", eval_batch_size=eval_batch_size,
    )

    # --- Pass 3: Build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        completed.append(CompletedEvalItem(
            eval_name=item.eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            clean_response="",
            test_response=ex["test_response"],
            clean_answer=None,
            test_answer=None,
            ground_truth_label=ex["ground_truth"],
            oracle_response=ex["oracle_response"],
            activations_path=None,
            metadata={**item.metadata},
        ))

    return completed


def _run_reasoning_termination_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """Reasoning termination eval. Uses PARTIAL CoT prefix from metadata.

    Unlike standard evals, this eval does NOT generate fresh CoT. It uses
    the precomputed cot_prefix (cut at a specific token position) so the
    oracle sees activations from mid-reasoning, not a completed trace.

    Flow (batched):
    1. Extract activations for all items
    2. Batch all oracle queries
    3. Assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract ---
    extracted = []

    for item in items:
        try:
            # Try cached bundle first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                test_response = cached.test_response or ""
                activations = cached.activations
            else:
                # Use cot_prefix from metadata (the PARTIAL CoT, not a full generation)
                cot_prefix = item.metadata.get("cot_prefix", "")
                test_response = (
                    item.metadata.get("qwen3_8b_test_response")
                    or cot_prefix
                )

                if not test_response.strip():
                    # No prefix available — skip (labels are pending_precompute)
                    continue

                # Extract activations from the partial CoT
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                        device=device,
                        max_boundaries=10, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [reasoning_term] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=test_response,
                    )

            # Build oracle prompt
            oracle_prompt = ""
            if activations is not None:
                template = ORACLE_PROMPTS_TEMPLATES.get(item.eval_name, "What is this model doing?")
                oracle_prompt = _oracle_prompt(activations.shape[0], template)

            extracted.append({
                "item": item,
                "test_response": test_response,
                "ground_truth": item.correct_answer,  # "will_terminate" or "will_continue"
                "activations": activations,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: reasoning_term item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name="reasoning_termination", eval_batch_size=eval_batch_size,
    )

    # --- Pass 3: Build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        completed.append(CompletedEvalItem(
            eval_name=item.eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            clean_response="",
            test_response=ex["test_response"],
            clean_answer=None,
            test_answer=None,
            ground_truth_label=ex["ground_truth"],
            oracle_response=ex["oracle_response"],
            activations_path=None,
            metadata={**item.metadata},
        ))

    return completed


def _run_rot13_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """Run ROT13 model-organism reconstruction eval.

    With cached activations: just runs oracle inference (fast).
    Without cache: generates rot13 + normal CoTs and extracts activations (slow).

    Flow (batched):
    1. Extract activations for all items
    2. Batch all oracle queries (max_new_tokens=1024 for reconstruction)
    3. Score and assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract ---
    extracted = []
    rot13_loaded = False

    for item in items:
        try:
            # Try cached bundle first (has rot13 activations + precomputed CoTs)
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                rot13_cot = cached.test_response or ""
                normal_cot = cached.clean_response or ""
                activations = cached.activations
            else:
                # Try precomputed CoTs from metadata first (no LoRA adapter needed)
                meta = item.metadata or {}
                rot13_cot = (meta.get("rot13_cot") or meta.get("qwen3_8b_test_response") or "")
                normal_cot = (meta.get("normal_cot") or meta.get("qwen3_8b_clean_response") or "")

                if not rot13_cot.strip():
                    print(f"    *** PANIC [rot13] {item.example_id}: no precomputed rot13_cot! "
                          f"Keys: {list(meta.keys())[:10]}. Skipping. ***")
                    continue
                if not normal_cot.strip():
                    print(f"    *** PANIC [rot13] {item.example_id}: no precomputed normal_cot! "
                          f"Keys: {list(meta.keys())[:10]}. Skipping. ***")

                # Extract activations from the rot13 CoT using the ROT13 adapter
                # (activations must come from the ROT13 model, not base)
                activations = None
                if rot13_cot.strip():
                    if not rot13_loaded:
                        load_extra_adapter(model, ROT13_ADAPTER_HF, adapter_name=ROT13_ADAPTER_NAME)
                        rot13_loaded = True
                    try:
                        bundle = _apply_oracle_mode_to_extract(
                            model, tokenizer,
                            eval_name=item.eval_name, example_id=item.example_id,
                            prompt=item.test_prompt, cot_text=rot13_cot,
                            act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                            device=device,
                            max_boundaries=20, generation_adapter_name=ROT13_ADAPTER_NAME,
                        )
                        if bundle and bundle.activations is not None:
                            activations = bundle.activations
                            _auto_cache_bundle(
                                cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                                prompt=item.test_prompt, cot_text=rot13_cot,
                                activations=activations, boundary_positions=bundle.boundary_positions,
                                clean_response=normal_cot, test_response=rot13_cot,
                            )
                    except Exception as e:
                        print(f"    [rot13] Activation extraction failed for {item.example_id}: {e}")

            # Build oracle prompt
            oracle_prompt = ""
            if activations is not None:
                template = ORACLE_PROMPTS_TEMPLATES["rot13_reconstruction"]
                oracle_prompt = _oracle_prompt(activations.shape[0], template)

            extracted.append({
                "item": item,
                "rot13_cot": rot13_cot,
                "normal_cot": normal_cot,
                "activations": activations,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: rot13 item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation (higher max_new_tokens for reconstruction) ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name="rot13_reconstruction", eval_batch_size=eval_batch_size,
        max_new_tokens=1024,
    )

    # --- Pass 3: Score and build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        oracle_response = ex["oracle_response"]
        normal_cot = ex["normal_cot"]
        n_positions = ex["activations"].shape[0] if ex["activations"] is not None else 0

        # Score
        target_cot = normal_cot
        predicted_for_match = oracle_response
        if target_cot and oracle_response:
            direct_match = _token_match_rate(tokenizer, target_cot, oracle_response)[2]
            decoded_match = _token_match_rate(tokenizer, target_cot, _rot13(oracle_response))[2]
            if decoded_match > direct_match:
                predicted_for_match = _rot13(oracle_response)

        kl = _token_unigram_kl(tokenizer, target_cot, predicted_for_match)
        if not math.isfinite(kl):
            kl = None
        matched, total_ref, match_rate = _token_match_rate(tokenizer, target_cot, predicted_for_match)

        completed.append(CompletedEvalItem(
            eval_name=item.eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            clean_response=normal_cot,
            test_response=ex["rot13_cot"],
            clean_answer=None,
            test_answer=None,
            ground_truth_label="pending_reconstruction",
            oracle_response=oracle_response,
            activations_path=None,
            metadata={
                **item.metadata,
                "positions_used": n_positions,
                "reference_token_count": total_ref,
                "matched_tokens": matched,
                "token_match_rate": match_rate,
                "kl_divergence": kl,
            },
        ))

    return completed


def _word_token_f1(prediction: str, reference: str) -> float:
    """Word-level F1 between prediction and reference (used for task evals)."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _token_f1(tokenizer, reference: str, predicted: str) -> float:
    """Token-level F1 between reference and predicted texts."""
    from collections import Counter
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    pred_ids = tokenizer.encode(predicted, add_special_tokens=False)
    if not ref_ids or not pred_ids:
        return 0.0
    common = Counter(ref_ids) & Counter(pred_ids)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_ids)
    recall = num_common / len(ref_ids)
    return 2 * precision * recall / (precision + recall)


def _run_compqa_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int | list[int],
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
    eval_batch_size: int = 8,
) -> list[CompletedEvalItem]:
    """CompQA eval: answer questions about CoT reasoning quality.

    Unlike binary evals, each item has its own question (test_prompt) about the CoT.
    The CoT to analyze is in metadata["cot_text"]. Scored via token F1 against
    Gemini ground truth in correct_answer.

    Flow (batched):
    1. Extract activations for all items
    2. Batch all oracle queries (max_new_tokens=256)
    3. Score and assemble CompletedEvalItem list
    """
    # --- Pass 1: Extract ---
    extracted = []

    for item in items:
        try:
            cot_text = (item.metadata.get("cot_text") or "").strip()
            if not cot_text:
                continue

            # Try cached bundle first
            cached = _try_load_cached(cache_dir, "compqa", item.example_id, device)

            if cached is not None:
                activations = cached.activations
            else:
                # Extract activations from clean_prompt + cot_text
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name="compqa", example_id=item.example_id,
                        prompt=item.clean_prompt, cot_text=cot_text,
                        act_layer=act_layer if isinstance(act_layer, int) else act_layer[0],
                        device=device,
                        max_boundaries=15, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [compqa] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name="compqa", example_id=item.example_id,
                        prompt=item.clean_prompt, cot_text=cot_text,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=cot_text,
                    )

            # Build oracle prompt — per-item question, not a template
            oracle_prompt = ""
            if activations is not None:
                oracle_prompt = _oracle_prompt(activations.shape[0], item.test_prompt)

            extracted.append({
                "item": item,
                "cot_text": cot_text,
                "activations": activations,
                "oracle_prompt": oracle_prompt,
                "oracle_response": "",
            })
        except Exception as e:
            print(f"  [training_eval] Warning: compqa item {item.example_id} failed: {e}")
            continue

    # --- Pass 2: Batched oracle generation ---
    _collect_and_batch_oracle(
        extracted, model, tokenizer, model_name, device,
        eval_name="compqa", eval_batch_size=eval_batch_size,
        max_new_tokens=256,
    )

    # --- Pass 3: Score and build CompletedEvalItem list ---
    completed = []
    for ex in extracted:
        item = ex["item"]
        oracle_response = ex["oracle_response"]

        # Compute per-item token F1
        f1 = 0.0
        if oracle_response and item.correct_answer:
            f1 = _token_f1(tokenizer, item.correct_answer, oracle_response)

        completed.append(CompletedEvalItem(
            eval_name=item.eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            clean_response="",
            test_response=ex["cot_text"],
            clean_answer=None,
            test_answer=None,
            ground_truth_label="pending_token_f1",
            oracle_response=oracle_response,
            activations_path=None,
            metadata={
                **item.metadata,
                "token_f1": f1,
            },
        ))

    return completed


def _score_binary_eval(
    eval_name: str,
    items: list[CompletedEvalItem],
    max_score: int | None = None,
) -> dict[str, float]:
    """Score a binary eval using oracle keyword parsing. Returns accuracy.

    Args:
        max_score: If set, cap the scoreable items to this count so that
            oversampling in _subsample() doesn't inflate n beyond the
            intended max_items_per_eval.
    """
    parsing_config = EVAL_PARSING.get(eval_name)
    if not parsing_config:
        return {}

    skip_labels = {
        "indeterminate", "pending_pair_resolution", "pending_multi_run",
        "pending_manual", "pending_reconstruction",
        "pending_kl_scoring",
    }

    scoreable = [
        item for item in items
        if item.ground_truth_label not in skip_labels
        and item.oracle_response
    ]

    # Cap to max_score so oversampling doesn't inflate reported n
    if max_score and len(scoreable) > max_score:
        scoreable = scoreable[:max_score]

    if not scoreable:
        return {}

    correct = 0
    total = 0
    unparsed = 0
    for item in scoreable:
        pred = parse_oracle_binary(
            item.oracle_response,
            parsing_config["positive_keywords"],
            parsing_config["negative_keywords"],
        )
        if pred is None:
            unparsed += 1
            continue

        pred_label = (
            parsing_config["positive_label"]
            if pred == "positive"
            else parsing_config["negative_label"]
        )
        total += 1
        if pred_label == item.ground_truth_label:
            correct += 1

    if total == 0:
        return {f"eval_n/{eval_name}_parse_fail": unparsed / len(scoreable) if scoreable else 0}

    return {
        f"eval/{eval_name}_acc": correct / total,
        f"eval_n/{eval_name}": total,
        f"eval_n/{eval_name}_parse_fail": unparsed / len(scoreable),
    }


def _save_table_to_disk(log_dir: Path, name: str, step: int, columns: list[str], rows: list[list]):
    """Save a wandb-style table to disk as nicely formatted JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    records = [dict(zip(columns, row)) for row in rows]
    path = log_dir / f"{name}_step{step}.json"
    with open(path, "w") as f:
        json.dump({"step": step, "name": name, "n": len(records), "rows": records}, f, indent=2, default=str)


def precache_eval_activations(
    model,
    tokenizer,
    model_name: str,
    device: str = "cuda",
    eval_dir: str = "data/evals",
    activation_cache_dir: str | None = None,
    eval_names: list[str] | None = None,
    oracle_adapter_name: str = "default",
    stride: int | str = None,
):
    """Pre-extract and cache activation bundles for all eval items.

    Run this once before training so that evals during training
    are pure cache lookups (no live generation/extraction = no NCCL timeouts).

    Uses multi-layer extraction matching training format (layers 9, 18, 27).
    """
    from tqdm.auto import tqdm

    if not activation_cache_dir:
        print("  [precache] No activation_cache_dir set, skipping")
        return
    if stride is None:
        raise ValueError("stride must be explicitly set for precache_eval_activations")

    cache_dir = Path(activation_cache_dir)

    # Configure oracle mode — layers match training (from CONFIGURED_LAYERS)
    act_layers = get_injection_layers(model_name)
    set_oracle_mode(trained=True, oracle_adapter_name=oracle_adapter_name, stride=stride, layers=act_layers)
    print(f"  [precache] Extraction: layers={act_layers}, stride={stride}")

    model.eval()
    eval_list = eval_names or TRAINING_EVALS

    total_cached, total_new, total_stale = 0, 0, 0
    for eval_name in eval_list:
        items = load_eval_items_hf(eval_name, eval_dir=eval_dir)

        # Find uncached items (stale caches auto-invalidated by stride/layer mismatch)
        uncached = []
        for item in items:
            existed = _cache_path(cache_dir, eval_name, item.example_id).exists()
            bundle = _maybe_load_cached(
                cache_dir,
                eval_name=eval_name,
                example_id=item.example_id,
                map_location="cpu",
                stride=stride,
                layers=act_layers,
            )
            if bundle is not None:
                total_cached += 1
            else:
                if existed:
                    total_stale += 1
                uncached.append(item)

        if not uncached:
            print(f"  [precache] {eval_name}: all {len(items)} items cached")
            continue

        stale_msg = f" ({total_stale} stale)" if total_stale else ""
        print(f"  [precache] {eval_name}: {len(uncached)}/{len(items)} items need extraction{stale_msg}")

        for item in tqdm(uncached, desc=f"  {eval_name}", leave=False):
            # Get CoT text from precomputed metadata
            cot_text = (
                item.metadata.get("qwen3_8b_test_response")
                or item.metadata.get("representative_response")
                or item.metadata.get("cot_text")
                or _first_rollout(item.metadata.get("hinted_rollouts"))
            )
            if not cot_text:
                # Fall back to clean response
                cot_text = (
                    item.metadata.get("qwen3_8b_clean_response")
                    or _first_rollout(item.metadata.get("clean_rollouts"))
                )
            if not cot_text:
                continue

            bundle = _apply_oracle_mode_to_extract(
                model, tokenizer,
                eval_name=eval_name,
                example_id=item.example_id,
                prompt=item.test_prompt,
                cot_text=cot_text,
                act_layer=act_layers[0],
                device=device,
                max_boundaries=10,
                generation_adapter_name=None,
            )
            if bundle:
                # Store text responses too
                clean_text = (
                    item.metadata.get("qwen3_8b_clean_response")
                    or _first_rollout(item.metadata.get("clean_rollouts"))
                    or ""
                )
                _auto_cache_bundle(
                    cache_dir, eval_name=eval_name, example_id=item.example_id,
                    prompt=item.test_prompt, cot_text=cot_text,
                    activations=bundle.activations,
                    boundary_positions=bundle.boundary_positions,
                    clean_response=clean_text, test_response=cot_text,
                    clean_answer=item.metadata.get("qwen3_8b_clean_answer"),
                    test_answer=item.metadata.get("qwen3_8b_test_answer"),
                )
                total_new += 1

        torch.cuda.empty_cache()

    print(f"  [precache] Done: {total_new} new bundles cached, {total_cached} already existed, {total_stale} stale replaced")


def run_training_evals(
    model,
    tokenizer,
    model_name: str,
    step: int,
    device: str = "cuda",
    eval_dir: str = "data/evals",
    max_items_per_eval: int = 20,
    skip_rot13: bool = False,
    oracle_adapter_name: str = "default",
    activation_cache_dir: str | None = None,
    eval_names: list[str] | None = None,
    log_dir: Path | str | None = None,
    eval_batch_size: int = 8,
    stride: int | str = None,
    task_eval_datasets: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Run evals and return results dict for wandb logging.

    Args:
        model: PeftModel with trained adapter.
        tokenizer: Tokenizer matching the model.
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen3-8B").
        step: Current training step (for deterministic subsampling).
        device: Device string.
        eval_dir: Path to directory with eval JSON files.
        max_items_per_eval: Maximum items per eval (for speed).
        skip_rot13: If True, skip ROT13 eval (it is expensive).
        oracle_adapter_name: Name of the trained oracle adapter in the PeftModel.
        activation_cache_dir: Path to precomputed activation bundles.
        eval_names: List of eval names to run. If None, uses TRAINING_EVALS default.
        log_dir: Path to directory for disk logging of eval tables.
        eval_batch_size: Mini-batch size for batched oracle generation.
        stride: Position extraction stride (int for fixed, "punctuation" for punctuation).
        task_eval_datasets: dict of task_name -> list[TrainingDataPoint] with pre-materialized
            steering_vectors. If provided, runs task-level generation evals (token F1) before
            detection evals.

    Returns:
        Flat dict of metrics suitable for wandb.log().
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if stride is None:
        raise ValueError("stride must be explicitly set for run_training_evals")

    # Configure oracle mode — layers match training (from CONFIGURED_LAYERS)
    act_layers = get_injection_layers(model_name)
    set_oracle_mode(trained=True, oracle_adapter_name=oracle_adapter_name, stride=stride, layers=act_layers)
    print(f"  [training_eval] Extraction: layers={act_layers}, stride={stride}, batch_size={eval_batch_size}")

    # Save training state and switch to eval
    was_training = model.training
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    all_metrics: dict[str, Any] = {}
    _log_dir = Path(log_dir) if log_dir else None
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    if cache_dir:
        print(f"  [training_eval] Activation cache dir: {cache_dir} (auto-populates on first run)")

    # ── Task-level evals (generation + token F1) ──
    if task_eval_datasets:
        import wandb
        print(f"  [training_eval] Running {len(task_eval_datasets)} task evals...")
        task_scores = {}
        for ds_name, dp_list in task_eval_datasets.items():
            # Build extracted dicts for _collect_and_batch_oracle
            extracted = []
            for dp in dp_list:
                if dp.steering_vectors is None:
                    continue
                oracle_prompt = dp.meta_info.get("prompt", "")
                extracted.append({
                    "activations": dp.steering_vectors,
                    "oracle_prompt": oracle_prompt,
                    "oracle_response": "",
                    "target": dp.target_output,
                    "datapoint_type": dp.datapoint_type,
                })

            if not extracted:
                continue

            _collect_and_batch_oracle(
                extracted, model, tokenizer, model_name, device,
                eval_name=ds_name, eval_batch_size=eval_batch_size,
            )

            # Score with word-level token F1
            scores = []
            columns = ["id", "type", "oracle_prompt", "prediction", "target", "token_f1", "pred_tokens", "target_tokens"]
            table = wandb.Table(columns=columns)
            rows = []
            for i, ex in enumerate(extracted):
                pred = ex["oracle_response"].strip()
                target = ex["target"].strip()
                score = _word_token_f1(pred, target)
                scores.append(score)
                row = [i, ex["datapoint_type"], ex["oracle_prompt"][:300], pred[:500], target[:500], round(score, 3), len(pred.split()), len(target.split())]
                table.add_data(*row)
                rows.append(row)

            avg_score = sum(scores) / len(scores)
            all_metrics[f"eval/{ds_name}"] = avg_score
            all_metrics[f"eval_n/{ds_name}"] = len(scores)
            print(f"    {ds_name}: token_f1={avg_score:.3f} (n={len(scores)})")

            if extracted:
                print(f"      pred='{extracted[0]['oracle_response'].strip()[:120]}'")
                print(f"      targ='{extracted[0]['target'].strip()[:120]}'")

            all_metrics[f"eval_table/{ds_name}"] = table
            if _log_dir and rows:
                _save_table_to_disk(_log_dir, f"eval_table_{ds_name}", step, columns, rows)
            task_scores[ds_name] = avg_score

        if task_scores:
            eval_mean = sum(task_scores.values()) / len(task_scores)
            all_metrics["eval/mean"] = eval_mean
            print(f"    eval_mean={eval_mean:.3f}")

        torch.cuda.empty_cache()
        gc.collect()

    eval_list = eval_names if eval_names is not None else TRAINING_EVALS
    evals_to_run = [e for e in eval_list if not (skip_rot13 and e == "rot13_reconstruction")]
    print(f"  [training_eval] Running {len(evals_to_run)} evals: {', '.join(evals_to_run)}")

    for eval_name in evals_to_run:
        print(f"  [training_eval] Running {eval_name}...")

        try:
            items = load_eval_items_hf(eval_name, eval_dir=eval_dir)
            items = _subsample(items, max_items_per_eval, seed=hash(eval_name))  # fixed seed: same items every step

            # Dispatch to appropriate handler
            if eval_name == "decorative_cot":
                completed = _run_decorative_cot_eval(
                    model, tokenizer, items, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )
            elif eval_name == "sentence_insertion":
                completed = _run_sentence_insertion_eval(
                    model, tokenizer, items, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )
            elif eval_name == "reasoning_termination_riya":
                completed = _run_reasoning_termination_eval(
                    model, tokenizer, items, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )
            elif eval_name == "rot13_reconstruction":
                completed = _run_rot13_eval(
                    model, tokenizer, items, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )
            elif eval_name == "compqa":
                completed = _run_compqa_eval(
                    model, tokenizer, items, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )
            else:
                # Standard binary evals + any new evals from config
                completed = _run_standard_eval(
                    model, tokenizer, items, eval_name, act_layers,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir, eval_batch_size=eval_batch_size,
                )

            if not completed:
                print(f"    No completed items for {eval_name}")
                continue

            # Score based on eval type
            if eval_name == "rot13_reconstruction":
                recon_metrics = _score_reconstruction_metrics(completed)
                if recon_metrics:
                    match_rate = recon_metrics.get("avg_token_match_rate", 0.0)
                    all_metrics[f"eval/{eval_name}_match_rate"] = match_rate
                    all_metrics[f"eval_n/{eval_name}"] = len(completed)
                    if "avg_kl_divergence" in recon_metrics:
                        all_metrics[f"eval/{eval_name}_kl"] = recon_metrics["avg_kl_divergence"]
                    print(f"    {eval_name}: match_rate={match_rate:.3f} (n={len(completed)})")
                # Wandb table for rot13 — show how it's messing up
                try:
                    import wandb
                    rot13_cols = ["id", "question", "rot13_cot", "oracle_output", "clean_cot", "match_rate", "kl"]
                    rot13_table = wandb.Table(columns=rot13_cols)
                    rot13_rows = []
                    for c in completed:
                        row = [
                            c.example_id, c.clean_prompt[:200],
                            (c.test_response or "")[:500], (c.oracle_response or "")[:500],
                            (c.clean_response or "")[:500],
                            round(c.metadata.get("token_match_rate", 0.0), 3),
                            round(c.metadata.get("kl_divergence", 0.0) or 0.0, 3),
                        ]
                        rot13_table.add_data(*row)
                        rot13_rows.append(row)
                    all_metrics[f"eval_table/{eval_name}"] = rot13_table
                    if _log_dir:
                        _save_table_to_disk(_log_dir, f"eval_table_{eval_name}", step, rot13_cols, rot13_rows)
                except Exception:
                    pass
            elif eval_name == "sentence_insertion":
                si_metrics = _score_sentence_insertion(completed)
                if si_metrics:
                    acc = si_metrics.get("accuracy", 0.0)
                    all_metrics[f"eval/{eval_name}_acc"] = acc
                    all_metrics[f"eval_n/{eval_name}"] = si_metrics.get("n", len(completed))
                    print(f"    {eval_name}: acc={acc:.3f} (n={si_metrics.get('n', len(completed))})")
            elif eval_name == "forced_answer_entropy_riya":
                # Score as top-1 answer prediction accuracy
                correct = 0
                total = 0
                for c in completed:
                    if not c.oracle_response:
                        continue
                    # Extract letter from oracle response
                    pred_letter = None
                    for ch in c.oracle_response.upper():
                        if ch in "ABCD":
                            pred_letter = ch
                            break
                    gt_letter = c.correct_answer.upper() if c.correct_answer else None
                    if pred_letter and gt_letter:
                        total += 1
                        if pred_letter == gt_letter:
                            correct += 1
                if total > 0:
                    acc = correct / total
                    all_metrics[f"eval/{eval_name}_acc"] = acc
                    all_metrics[f"eval_n/{eval_name}"] = total
                    print(f"    {eval_name}: top1_acc={acc:.3f} (n={total})")
            elif eval_name == "compqa":
                # Score via aggregate token F1 from per-item metadata
                f1_scores = [c.metadata.get("token_f1", 0.0) for c in completed if c.oracle_response]
                if f1_scores:
                    avg_f1 = sum(f1_scores) / len(f1_scores)
                    all_metrics[f"eval/{eval_name}_token_f1"] = avg_f1
                    all_metrics[f"eval_n/{eval_name}"] = len(f1_scores)
                    print(f"    {eval_name}: token_f1={avg_f1:.3f} (n={len(f1_scores)})")
            else:
                # Binary evals
                binary_metrics = _score_binary_eval(eval_name, completed, max_score=max_items_per_eval)
                all_metrics.update(binary_metrics)
                if binary_metrics:
                    acc_key = f"eval/{eval_name}_acc"
                    if acc_key in binary_metrics:
                        print(f"    {eval_name}: acc={binary_metrics[acc_key]:.3f} (n={binary_metrics.get(f'eval_n/{eval_name}', 0)})")

            # Log a sample oracle response for qualitative inspection
            for c in completed:
                if c.oracle_response:
                    all_metrics[f"eval/{eval_name}_sample_oracle"] = c.oracle_response[:200]
                    all_metrics[f"eval/{eval_name}_sample_gt"] = c.ground_truth_label
                    break

            # Wandb table for all evals
            try:
                import wandb
                if eval_name != "rot13_reconstruction":  # rot13 has its own table above
                    cols = ["id", "question", "oracle_output", "ground_truth", "correct"]
                    table = wandb.Table(columns=cols)
                    table_rows = []
                    parsing_cfg = EVAL_PARSING.get(eval_name)
                    for c in completed:
                        if not c.oracle_response:
                            continue
                        gt = c.ground_truth_label
                        oracle = c.oracle_response[:300]
                        is_correct = "?"
                        if gt and gt not in {"indeterminate", "pending_manual", "pending_multi_run"}:
                            if parsing_cfg and parsing_cfg["positive_keywords"]:
                                pred = parse_oracle_binary(oracle, parsing_cfg["positive_keywords"], parsing_cfg["negative_keywords"])
                                if pred is not None:
                                    pred_label = parsing_cfg["positive_label"] if pred == "positive" else parsing_cfg["negative_label"]
                                    is_correct = "yes" if pred_label == gt else "no"
                                else:
                                    is_correct = "?"
                            else:
                                is_correct = "yes" if gt.lower() in oracle.lower() else "no"
                        row = [c.example_id, (c.test_prompt or c.clean_prompt or "")[:200], oracle, gt or "", is_correct]
                        table.add_data(*row)
                        table_rows.append(row)
                    if len(table.data) > 0:
                        all_metrics[f"eval_table/{eval_name}"] = table
                    if _log_dir and table_rows:
                        _save_table_to_disk(_log_dir, f"eval_table_{eval_name}", step, cols, table_rows)
            except Exception:
                pass

        except Exception as e:
            print(f"  [training_eval] ERROR running {eval_name}: {e}")
            traceback.print_exc()
            continue

        # Free memory between evals
        torch.cuda.empty_cache()

    # Compute overall accuracy across binary evals
    acc_keys = [k for k in all_metrics if k.endswith("_acc")]
    if acc_keys:
        acc_values = [all_metrics[k] for k in acc_keys]
        all_metrics["eval/mean_acc"] = sum(acc_values) / len(acc_values)

    # Restore training state
    if was_training:
        model.train()
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  [training_eval] Completed. {len(all_metrics)} metrics logged.")
    return all_metrics
