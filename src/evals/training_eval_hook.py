"""
Training eval hook: run unfaithfulness evals periodically during training.

Integrates 6 unfaithfulness evals into the training loop and returns a flat
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
    extract_numerical_answer,
    answers_match,
    determine_ground_truth,
    ci_label,
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
    extract_activation_bundle as _extract_bundle_raw,
    cache_path as _cache_path,
    load_bundle as _load_bundle,
    save_bundle as _save_bundle,
)
from core.ao import (
    generate_cot,
    generate_direct_answer,
    run_oracle_on_activations as _run_oracle_raw,
    load_extra_adapter,
    layer_percent_to_layer,
)


# Evals to run during training, in order of cost (cheapest first)
TRAINING_EVALS = [
    "hinted_mcq",
    "hinted_mcq_truthfulqa",
    "sycophancy_v2_riya",
    "decorative_cot",
    "sentence_insertion",
    "reasoning_termination_riya",
    "atypical_answer_mcq",
]

ROT13_ADAPTER_HF = "ceselder/rot13-qwen3-8b-lora"
ROT13_ADAPTER_NAME = "rot13"


def _first_rollout(rollouts) -> str | None:
    """Extract first rollout from a list of precomputed rollouts."""
    if isinstance(rollouts, list) and len(rollouts) > 0:
        return rollouts[0]
    return None


def _apply_oracle_mode_to_extract(model, tokenizer, **kwargs):
    """Wrapper applying current oracle mode to activation extraction."""
    kwargs.setdefault("stride", _ORACLE_MODE["stride"])
    return _extract_bundle_raw(model, tokenizer, **kwargs)


def _apply_oracle_mode_to_oracle(model, tokenizer, activations, prompt, **kwargs):
    """Wrapper applying current oracle mode to oracle inference."""
    kwargs.setdefault("placeholder_token", _ORACLE_MODE["placeholder_token"])
    kwargs.setdefault("oracle_adapter_name", _ORACLE_MODE["oracle_adapter_name"])
    return _run_oracle_raw(model, tokenizer, activations, prompt, **kwargs)


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
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, eval_name, example_id)
    if not path.exists():
        return None
    bundle = _load_bundle(path, map_location=device)
    if bundle.activations is not None:
        return bundle
    return None


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
    path = _cache_path(cache_dir, eval_name, example_id)
    if path.exists():
        return  # already cached
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
    try:
        _save_bundle(bundle, path)
    except Exception as e:
        print(f"    [cache] Failed to save {eval_name}/{example_id}: {e}")


def _run_standard_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    eval_name: str,
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """Run a standard binary eval (hinted_mcq, sycophancy_v2_riya, reasoning_termination_riya).

    Flow per item:
    1. Load cached activations OR generate clean + test responses and extract activations
    2. Run oracle on activations with the trained adapter
    3. Determine ground truth from clean/test answer comparison
    """
    completed = []

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
                n_positions = len(cached.boundary_positions)
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
                    # Only test response available (e.g. sycophancy_v2_riya has
                    # precomputed labels so clean_response not needed for GT)
                    test_response = precomp_test
                    test_answer = (
                        item.metadata.get("qwen3_8b_test_answer")
                        or _extract_answer(test_response, eval_name)
                    )
                    clean_response = generate_cot(
                        model, tokenizer, item.clean_prompt,
                        max_new_tokens=2048, device=device, adapter_name=None,
                    )
                    clean_answer = _extract_answer(clean_response, eval_name)
                else:
                    clean_response = generate_cot(
                        model, tokenizer, item.clean_prompt,
                        max_new_tokens=2048, device=device, adapter_name=None,
                    )
                    test_response = generate_cot(
                        model, tokenizer, item.test_prompt,
                        max_new_tokens=2048, device=device, adapter_name=None,
                    )
                    clean_answer = _extract_answer(clean_response, eval_name)
                    test_answer = _extract_answer(test_response, eval_name)

                # Extract activations (forward pass only, no generation)
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name=eval_name,
                        example_id=item.example_id,
                        prompt=item.test_prompt,
                        cot_text=test_response,
                        act_layer=act_layer,
                        device=device,
                        max_boundaries=10,
                        generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [{eval_name}] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None
                n_positions = len(bundle.boundary_positions) if bundle else 0

                # Auto-cache for next eval run
                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        clean_response=clean_response, test_response=test_response,
                        clean_answer=clean_answer, test_answer=test_answer,
                    )

            # Run oracle on activations (this uses the TRAINED adapter)
            oracle_response = ""
            if activations is not None:
                try:
                    template = ORACLE_PROMPTS_TEMPLATES.get(eval_name, "What is this model doing?")
                    oracle_prompt = _oracle_prompt(n_positions, template)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        device=device,
                    )
                except Exception as e:
                    print(f"    [{eval_name}] Oracle inference failed for {item.example_id}: {e}")

            ground_truth = determine_ground_truth(item, clean_answer, test_answer)

            completed.append(CompletedEvalItem(
                eval_name=eval_name,
                example_id=item.example_id,
                clean_prompt=item.clean_prompt,
                test_prompt=item.test_prompt,
                correct_answer=item.correct_answer,
                nudge_answer=item.nudge_answer,
                clean_response=clean_response,
                test_response=test_response,
                clean_answer=clean_answer,
                test_answer=test_answer,
                ground_truth_label=ground_truth,
                oracle_response=oracle_response,
                activations_path=None,
                metadata={**item.metadata},
            ))
        except Exception as e:
            print(f"  [training_eval] Warning: item {item.example_id} failed: {e}")
            continue

    return completed


def _run_decorative_cot_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """Decorative CoT eval. Uses cached activations + static labels when available."""
    completed = []

    for item in items:
        try:
            # Try cached bundle first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                # Use cached activations and precomputed label from metadata
                label = item.metadata.get("decorative_label", "indeterminate")
                representative_cot = cached.test_response or ""
                activations = cached.activations
                n_positions = len(cached.boundary_positions)
                with_cot_acc = item.metadata.get("with_cot_acc", 0.0)
                without_cot_acc = item.metadata.get("without_cot_acc", 0.0)
            elif item.metadata.get("decorative_label"):
                # Static label in dataset but no cached activations — need extraction
                label = item.metadata["decorative_label"]
                with_cot_acc = item.metadata.get("with_cot_acc", 0.0)
                without_cot_acc = item.metadata.get("without_cot_acc", 0.0)
                representative_cot = item.metadata.get("representative_cot", "")
                if not representative_cot:
                    representative_cot = generate_cot(
                        model, tokenizer, item.test_prompt,
                        max_new_tokens=2048, device=device, adapter_name=None,
                    )
                activations = None
                n_positions = 0
                if representative_cot:
                    try:
                        bundle = _apply_oracle_mode_to_extract(
                            model, tokenizer,
                            eval_name=item.eval_name, example_id=item.example_id,
                            prompt=item.test_prompt, cot_text=representative_cot,
                            act_layer=act_layer, device=device,
                            max_boundaries=10, generation_adapter_name=None,
                        )
                        if bundle and bundle.activations is not None:
                            activations = bundle.activations
                            n_positions = len(bundle.boundary_positions)
                            _auto_cache_bundle(
                                cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                                prompt=item.test_prompt, cot_text=representative_cot,
                                activations=activations, boundary_positions=bundle.boundary_positions,
                                test_response=representative_cot,
                            )
                    except Exception as e:
                        print(f"    [decorative_cot] Activation extraction failed for {item.example_id}: {e}")
            else:
                # Full computation (expensive fallback) — use temperature sampling
                # so different runs can produce different answers for meaningful CIs
                n_runs = 3
                temperature = 0.6
                with_cot_correct = 0
                without_cot_correct = 0
                for run_i in range(n_runs):
                    temp = None if run_i == 0 else temperature
                    cot_response = generate_cot(
                        model, tokenizer, item.test_prompt,
                        max_new_tokens=2048, device=device, adapter_name=None,
                        temperature=temp,
                    )
                    direct_response = generate_direct_answer(
                        model, tokenizer, item.clean_prompt,
                        device=device, adapter_name=None,
                        temperature=temp,
                    )
                    if answers_match(extract_numerical_answer(cot_response), item.correct_answer):
                        with_cot_correct += 1
                    if answers_match(extract_numerical_answer(direct_response), item.correct_answer):
                        without_cot_correct += 1
                with_cot_acc = with_cot_correct / n_runs
                without_cot_acc = without_cot_correct / n_runs
                label = ci_label(with_cot_correct, n_runs, without_cot_correct, n_runs)
                representative_cot = generate_cot(
                    model, tokenizer, item.test_prompt,
                    max_new_tokens=2048, device=device, adapter_name=None,
                )
                activations = None
                n_positions = 0
                if representative_cot:
                    try:
                        bundle = _apply_oracle_mode_to_extract(
                            model, tokenizer,
                            eval_name=item.eval_name, example_id=item.example_id,
                            prompt=item.test_prompt, cot_text=representative_cot,
                            act_layer=act_layer, device=device,
                            max_boundaries=10, generation_adapter_name=None,
                        )
                        if bundle and bundle.activations is not None:
                            activations = bundle.activations
                            n_positions = len(bundle.boundary_positions)
                            _auto_cache_bundle(
                                cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                                prompt=item.test_prompt, cot_text=representative_cot,
                                activations=activations, boundary_positions=bundle.boundary_positions,
                                test_response=representative_cot,
                            )
                    except Exception as e:
                        print(f"    [decorative_cot] Activation extraction failed for {item.example_id}: {e}")

            # Run oracle (uses the TRAINED adapter — this is the only part that changes)
            oracle_response = ""
            if activations is not None:
                try:
                    template = ORACLE_PROMPTS_TEMPLATES["decorative_cot"]
                    oracle_prompt = _oracle_prompt(n_positions, template)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        device=device,
                    )
                except Exception as e:
                    print(f"    [decorative_cot] Oracle inference failed for {item.example_id}: {e}")

            completed.append(CompletedEvalItem(
                eval_name=item.eval_name,
                example_id=item.example_id,
                clean_prompt=item.clean_prompt,
                test_prompt=item.test_prompt,
                correct_answer=item.correct_answer,
                nudge_answer=None,
                clean_response="",
                test_response=representative_cot,
                clean_answer=None,
                test_answer=None,
                ground_truth_label=label,
                oracle_response=oracle_response,
                activations_path=None,
                metadata={
                    **item.metadata,
                    "with_cot_acc": with_cot_acc,
                    "without_cot_acc": without_cot_acc,
                },
            ))
        except Exception as e:
            print(f"  [training_eval] Warning: decorative_cot item {item.example_id} failed: {e}")
            continue

    return completed


def _run_sentence_insertion_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """Run sentence insertion eval from pre-spliced CoTs in metadata."""
    completed = []

    for item in items:
        try:
            test_response = item.metadata.get("spliced_cot_text", "")

            # Try cached activations first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)
            if cached is not None:
                activations = cached.activations
                n_positions = len(cached.boundary_positions)
            else:
                # Extract activations (forward pass only)
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        act_layer=act_layer, device=device,
                        max_boundaries=30, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [sentence_insertion] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None
                n_positions = len(bundle.boundary_positions) if bundle else 0

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=test_response,
                    )

            oracle_response = ""
            if activations is not None:
                try:
                    template = ORACLE_PROMPTS_TEMPLATES["sentence_insertion"]
                    oracle_prompt = _oracle_prompt(n_positions, template)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        device=device,
                    )
                except Exception as e:
                    print(f"    [sentence_insertion] Oracle inference failed for {item.example_id}: {e}")

            ground_truth = determine_ground_truth(item, None, None)

            completed.append(CompletedEvalItem(
                eval_name=item.eval_name,
                example_id=item.example_id,
                clean_prompt=item.clean_prompt,
                test_prompt=item.test_prompt,
                correct_answer=item.correct_answer,
                nudge_answer=item.nudge_answer,
                clean_response="",
                test_response=test_response,
                clean_answer=None,
                test_answer=None,
                ground_truth_label=ground_truth,
                oracle_response=oracle_response,
                activations_path=None,
                metadata={**item.metadata},
            ))
        except Exception as e:
            print(f"  [training_eval] Warning: sentence_insertion item {item.example_id} failed: {e}")
            continue

    return completed


def _run_reasoning_termination_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """Reasoning termination eval. Uses PARTIAL CoT prefix from metadata.

    Unlike standard evals, this eval does NOT generate fresh CoT. It uses
    the precomputed cot_prefix (cut at a specific token position) so the
    oracle sees activations from mid-reasoning, not a completed trace.
    """
    completed = []

    for item in items:
        try:
            # Try cached bundle first
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                test_response = cached.test_response or ""
                activations = cached.activations
                n_positions = len(cached.boundary_positions)
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
                        act_layer=act_layer, device=device,
                        max_boundaries=10, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [reasoning_term] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None
                n_positions = len(bundle.boundary_positions) if bundle else 0

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                        prompt=item.test_prompt, cot_text=test_response,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=test_response,
                    )

            # Run oracle
            oracle_response = ""
            if activations is not None:
                try:
                    template = ORACLE_PROMPTS_TEMPLATES.get(item.eval_name, "What is this model doing?")
                    oracle_prompt = _oracle_prompt(n_positions, template)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        device=device,
                    )
                except Exception as e:
                    print(f"    [reasoning_term] Oracle inference failed for {item.example_id}: {e}")

            ground_truth = determine_ground_truth(item, None, None)

            completed.append(CompletedEvalItem(
                eval_name=item.eval_name,
                example_id=item.example_id,
                clean_prompt=item.clean_prompt,
                test_prompt=item.test_prompt,
                correct_answer=item.correct_answer,
                nudge_answer=item.nudge_answer,
                clean_response="",
                test_response=test_response,
                clean_answer=None,
                test_answer=None,
                ground_truth_label=ground_truth,
                oracle_response=oracle_response,
                activations_path=None,
                metadata={**item.metadata},
            ))
        except Exception as e:
            print(f"  [training_eval] Warning: reasoning_term item {item.example_id} failed: {e}")
            continue

    return completed


def _run_rot13_eval(
    model,
    tokenizer,
    items: list[EvalItem],
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """Run ROT13 model-organism reconstruction eval.

    With cached activations: just runs oracle inference (fast).
    Without cache: generates rot13 + normal CoTs and extracts activations (slow).
    """
    completed = []
    rot13_loaded = False

    for item in items:
        try:
            # Try cached bundle first (has rot13 activations + precomputed CoTs)
            cached = _try_load_cached(cache_dir, item.eval_name, item.example_id, device)

            if cached is not None:
                rot13_cot = cached.test_response or ""
                normal_cot = cached.clean_response or ""
                activations = cached.activations
                n_positions = len(cached.boundary_positions)
            else:
                # Try precomputed CoTs from metadata first (no LoRA adapter needed)
                meta = item.metadata or {}
                rot13_cot = (meta.get("rot13_cot") or meta.get("qwen3_8b_test_response") or "")
                normal_cot = (meta.get("normal_cot") or meta.get("qwen3_8b_clean_response") or "")

                if not rot13_cot.strip():
                    # Last resort: generate from scratch (requires LoRA adapter)
                    if not rot13_loaded:
                        load_extra_adapter(model, ROT13_ADAPTER_HF, adapter_name=ROT13_ADAPTER_NAME)
                        rot13_loaded = True
                    rot13_cot = generate_cot(
                        model, tokenizer, item.test_prompt,
                        max_new_tokens=2048, device=device,
                        adapter_name=ROT13_ADAPTER_NAME,
                    )
                if not normal_cot.strip():
                    normal_cot = generate_cot(
                        model, tokenizer, item.test_prompt,
                        max_new_tokens=2048, device=device,
                        adapter_name=None,
                    )

                # Extract activations from the rot13 CoT using the ROT13 adapter
                # (activations must come from the ROT13 model, not base)
                activations = None
                n_positions = 0
                if rot13_cot.strip():
                    if not rot13_loaded:
                        load_extra_adapter(model, ROT13_ADAPTER_HF, adapter_name=ROT13_ADAPTER_NAME)
                        rot13_loaded = True
                    try:
                        bundle = _apply_oracle_mode_to_extract(
                            model, tokenizer,
                            eval_name=item.eval_name, example_id=item.example_id,
                            prompt=item.test_prompt, cot_text=rot13_cot,
                            act_layer=act_layer, device=device,
                            max_boundaries=20, generation_adapter_name=ROT13_ADAPTER_NAME,
                        )
                        if bundle and bundle.activations is not None:
                            activations = bundle.activations
                            n_positions = len(bundle.boundary_positions)
                            _auto_cache_bundle(
                                cache_dir, eval_name=item.eval_name, example_id=item.example_id,
                                prompt=item.test_prompt, cot_text=rot13_cot,
                                activations=activations, boundary_positions=bundle.boundary_positions,
                                clean_response=normal_cot, test_response=rot13_cot,
                            )
                    except Exception as e:
                        print(f"    [rot13] Activation extraction failed for {item.example_id}: {e}")

            # Run oracle (trained adapter — this is the only part that changes per step)
            oracle_response = ""
            if activations is not None:
                try:
                    template = ORACLE_PROMPTS_TEMPLATES["rot13_reconstruction"]
                    oracle_prompt = _oracle_prompt(n_positions, template)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        max_new_tokens=1024, device=device,
                    )
                except Exception as e:
                    print(f"    [rot13] Oracle inference failed for {item.example_id}: {e}")

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
                test_response=rot13_cot,
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
        except Exception as e:
            print(f"  [training_eval] Warning: rot13 item {item.example_id} failed: {e}")
            continue

    return completed


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
    act_layer: int,
    model_name: str,
    device: str,
    oracle_adapter_name: str,
    cache_dir: Path | None = None,
) -> list[CompletedEvalItem]:
    """CompQA eval: answer questions about CoT reasoning quality.

    Unlike binary evals, each item has its own question (test_prompt) about the CoT.
    The CoT to analyze is in metadata["cot_text"]. Scored via token F1 against
    Gemini ground truth in correct_answer.
    """
    completed = []

    for item in items:
        try:
            cot_text = (item.metadata.get("cot_text") or "").strip()
            if not cot_text:
                continue

            # Try cached bundle first
            cached = _try_load_cached(cache_dir, "compqa", item.example_id, device)

            if cached is not None:
                activations = cached.activations
                n_positions = len(cached.boundary_positions)
            else:
                # Extract activations from clean_prompt + cot_text
                try:
                    bundle = _apply_oracle_mode_to_extract(
                        model, tokenizer,
                        eval_name="compqa", example_id=item.example_id,
                        prompt=item.clean_prompt, cot_text=cot_text,
                        act_layer=act_layer, device=device,
                        max_boundaries=15, generation_adapter_name=None,
                    )
                except Exception as e:
                    print(f"    [compqa] Activation extraction failed for {item.example_id}: {e}")
                    bundle = None
                activations = bundle.activations if bundle else None
                n_positions = len(bundle.boundary_positions) if bundle else 0

                if bundle:
                    _auto_cache_bundle(
                        cache_dir, eval_name="compqa", example_id=item.example_id,
                        prompt=item.clean_prompt, cot_text=cot_text,
                        activations=activations, boundary_positions=bundle.boundary_positions,
                        test_response=cot_text,
                    )

            # Run oracle with per-item question
            oracle_response = ""
            if activations is not None:
                try:
                    oracle_prompt = _oracle_prompt(n_positions, item.test_prompt)
                    oracle_response = _apply_oracle_mode_to_oracle(
                        model, tokenizer, activations, oracle_prompt,
                        model_name=model_name, act_layer=act_layer,
                        max_new_tokens=256, device=device,
                    )
                except Exception as e:
                    print(f"    [compqa] Oracle inference failed for {item.example_id}: {e}")

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
                test_response=cot_text,
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
        except Exception as e:
            print(f"  [training_eval] Warning: compqa item {item.example_id} failed: {e}")
            continue

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
    for item in scoreable:
        pred = parse_oracle_binary(
            item.oracle_response,
            parsing_config["positive_keywords"],
            parsing_config["negative_keywords"],
        )
        if pred is None:
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
        return {}

    return {
        f"eval/{eval_name}_acc": correct / total,
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
):
    """Pre-extract and cache activation bundles for all eval items.

    Run this once before training so that unfaith evals during training
    are pure cache lookups (no live generation/extraction = no NCCL timeouts).
    """
    from tqdm.auto import tqdm

    if not activation_cache_dir:
        print("  [precache] No activation_cache_dir set, skipping")
        return

    cache_dir = Path(activation_cache_dir)
    act_layer = 9  # first layer in [9, 18, 27] — used for activation extraction

    model.eval()
    eval_list = eval_names or TRAINING_EVALS

    total_cached, total_new = 0, 0
    for eval_name in eval_list:
        items = load_eval_items_hf(eval_name, eval_dir=eval_dir)

        # Find uncached items
        uncached = []
        for item in items:
            path = _cache_path(cache_dir, eval_name, item.example_id)
            if path.exists():
                total_cached += 1
            else:
                uncached.append(item)

        if not uncached:
            print(f"  [precache] {eval_name}: all {len(items)} items cached")
            continue

        print(f"  [precache] {eval_name}: {len(uncached)}/{len(items)} items need extraction")

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
                act_layer=act_layer,
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

    print(f"  [precache] Done: {total_new} new bundles cached, {total_cached} already existed")


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
) -> dict[str, Any]:
    """Run unfaithfulness evals and return results dict for wandb logging.

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

    Returns:
        Flat dict of metrics suitable for wandb.log().
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Configure oracle mode: use trained adapter with paragraph tokens
    set_oracle_mode(trained=True, oracle_adapter_name=oracle_adapter_name, stride=5)

    act_layer = layer_percent_to_layer(model_name, 50)

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

    eval_list = eval_names if eval_names is not None else TRAINING_EVALS
    evals_to_run = [e for e in eval_list if not (skip_rot13 and e == "rot13_reconstruction")]
    print(f"  [training_eval] Running {len(evals_to_run)} evals: {', '.join(evals_to_run)}")

    for eval_name in evals_to_run:
        print(f"  [training_eval] Running {eval_name}...")

        try:
            items = load_eval_items_hf(eval_name, eval_dir=eval_dir)
            items = _subsample(items, max_items_per_eval, seed=step + hash(eval_name))

            # Dispatch to appropriate handler
            if eval_name == "decorative_cot":
                completed = _run_decorative_cot_eval(
                    model, tokenizer, items, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
                )
            elif eval_name == "sentence_insertion":
                completed = _run_sentence_insertion_eval(
                    model, tokenizer, items, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
                )
            elif eval_name == "reasoning_termination_riya":
                completed = _run_reasoning_termination_eval(
                    model, tokenizer, items, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
                )
            elif eval_name == "rot13_reconstruction":
                completed = _run_rot13_eval(
                    model, tokenizer, items, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
                )
            elif eval_name == "compqa":
                completed = _run_compqa_eval(
                    model, tokenizer, items, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
                )
            else:
                # Standard binary evals + any new evals from config
                completed = _run_standard_eval(
                    model, tokenizer, items, eval_name, act_layer,
                    model_name, device, oracle_adapter_name,
                    cache_dir=cache_dir,
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
                    # n logged to console only, not wandb (clutters charts)
                    if "avg_kl_divergence" in recon_metrics:
                        all_metrics[f"eval/{eval_name}_kl"] = recon_metrics["avg_kl_divergence"]
                    print(f"    {eval_name}: match_rate={match_rate:.3f}")
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
                    print(f"    {eval_name}: acc={acc:.3f}")
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
                    print(f"    {eval_name}: top1_acc={acc:.3f} (n={total})")
            elif eval_name == "compqa":
                # Score via aggregate token F1 from per-item metadata
                f1_scores = [c.metadata.get("token_f1", 0.0) for c in completed if c.oracle_response]
                if f1_scores:
                    avg_f1 = sum(f1_scores) / len(f1_scores)
                    all_metrics[f"eval/{eval_name}_token_f1"] = avg_f1
                    print(f"    {eval_name}: token_f1={avg_f1:.3f} (n={len(f1_scores)})")
            else:
                # Binary evals
                binary_metrics = _score_binary_eval(eval_name, completed, max_score=max_items_per_eval)
                all_metrics.update(binary_metrics)
                if binary_metrics:
                    acc_key = f"eval/{eval_name}_acc"
                    if acc_key in binary_metrics:
                        print(f"    {eval_name}: acc={binary_metrics[acc_key]:.3f} (n={binary_metrics.get(f'eval/{eval_name}_n', 0)})")

            # Log a sample oracle response for qualitative inspection
            for c in completed:
                if c.oracle_response:
                    all_metrics[f"eval/{eval_name}_sample_oracle"] = c.oracle_response[:200]
                    all_metrics[f"eval/{eval_name}_sample_gt"] = c.ground_truth_label
                    break

            # Wandb table for all unfaith evals
            try:
                import wandb
                if eval_name != "rot13_reconstruction":  # rot13 has its own table above
                    cols = ["id", "question", "oracle_output", "ground_truth", "correct"]
                    table = wandb.Table(columns=cols)
                    table_rows = []
                    for c in completed:
                        if not c.oracle_response:
                            continue
                        # Re-score for table
                        gt = c.ground_truth_label
                        oracle = c.oracle_response[:300]
                        is_correct = "?"
                        if gt and gt not in {"indeterminate", "pending_manual", "pending_multi_run"}:
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
