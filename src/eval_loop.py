"""
Unified eval loop for the CoT Oracle.

Replaces training_eval_hook.py (~2000 lines), score_oracle.py, and run_evals.py.

All 11 tasks share the same flow:
  load test split → materialize activations → oracle generate → score → wandb metrics

Four scoring functions cover all tasks with zero per-task special-casing.
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path
from typing import Any

import torch

from tasks import TASKS, TaskDef, ScoringMode, get_eval_tasks
from data_loading import load_task_data
from evals.common import parse_oracle_binary


# ── Scoring functions ──

def _score_binary(
    task_def: TaskDef,
    predictions: list[str],
    targets: list[str],
) -> dict[str, float]:
    """Score binary classification via keyword parsing → accuracy."""
    if not predictions:
        return {"accuracy": 0.0, "n": 0}

    correct = 0
    total = 0
    unparsed = 0

    for pred_text, target in zip(predictions, targets):
        parsed = parse_oracle_binary(
            pred_text,
            list(task_def.positive_keywords),
            list(task_def.negative_keywords),
        )
        if parsed is None:
            unparsed += 1
            continue

        pred_label = (
            task_def.positive_label if parsed == "positive"
            else task_def.negative_label
        )
        total += 1
        if pred_label == target:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": acc,
        "n": total,
        "unparsed": unparsed,
        "unparsed_rate": unparsed / len(predictions) if predictions else 0.0,
    }


def _score_token_f1(
    predictions: list[str],
    targets: list[str],
) -> dict[str, float]:
    """Word-level F1 between prediction and target."""
    if not predictions:
        return {"token_f1": 0.0, "n": 0}

    f1_scores = []
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.lower().split()
        target_tokens = target.lower().split()

        if not target_tokens:
            f1_scores.append(1.0 if not pred_tokens else 0.0)
            continue

        pred_set = set(pred_tokens)
        target_set = set(target_tokens)

        if not pred_set and not target_set:
            f1_scores.append(1.0)
            continue
        if not pred_set or not target_set:
            f1_scores.append(0.0)
            continue

        common = pred_set & target_set
        precision = len(common) / len(pred_set)
        recall = len(common) / len(target_set)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return {
        "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "n": len(f1_scores),
    }


def _score_step_accuracy(
    predictions: list[str],
    targets: list[str],
) -> dict[str, float]:
    """Parse step number from prediction, allow off-by-1. 'none' detection for clean items."""
    if not predictions:
        return {"step_accuracy": 0.0, "n": 0}

    correct = 0
    total = 0

    for pred_text, target in zip(predictions, targets):
        total += 1
        pred_lower = pred_text.lower().strip()
        target_lower = target.lower().strip()

        # Handle "none" / "no insertion" case
        if target_lower in ("none", "no insertion", "-1"):
            if any(w in pred_lower for w in ("none", "no insertion", "no step", "clean")):
                correct += 1
            continue

        # Extract step number from prediction
        pred_nums = re.findall(r'\b(\d+)\b', pred_lower)
        target_nums = re.findall(r'\b(\d+)\b', target_lower)

        if not target_nums:
            continue

        target_step = int(target_nums[0])

        if pred_nums:
            pred_step = int(pred_nums[0])
            # Off-by-1 tolerance
            if abs(pred_step - target_step) <= 1:
                correct += 1

    return {
        "step_accuracy": correct / total if total > 0 else 0.0,
        "n": total,
    }


def _score_token_match(
    predictions: list[str],
    targets: list[str],
    tokenizer=None,
) -> dict[str, float]:
    """Token-level match rate for reconstruction tasks.

    If tokenizer is provided, uses token IDs for matching.
    Otherwise falls back to word-level matching.
    """
    if not predictions:
        return {"token_match_rate": 0.0, "n": 0}

    match_rates = []
    for pred, target in zip(predictions, targets):
        if tokenizer is not None:
            pred_ids = tokenizer.encode(pred, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
        else:
            pred_ids = pred.lower().split()
            target_ids = target.lower().split()

        if not target_ids:
            match_rates.append(1.0 if not pred_ids else 0.0)
            continue

        # Count matching tokens up to min length
        matches = sum(
            1 for p, t in zip(pred_ids, target_ids) if p == t
        )
        match_rates.append(matches / len(target_ids))

    return {
        "token_match_rate": sum(match_rates) / len(match_rates) if match_rates else 0.0,
        "n": len(match_rates),
    }


# ── Score dispatcher ──

def score_task(
    task_def: TaskDef,
    predictions: list[str],
    targets: list[str],
    tokenizer=None,
) -> dict[str, float]:
    """Score any task via its ScoringMode. Returns {metric: value, n: count}."""
    if task_def.scoring == ScoringMode.BINARY:
        return _score_binary(task_def, predictions, targets)
    elif task_def.scoring == ScoringMode.TOKEN_F1:
        return _score_token_f1(predictions, targets)
    elif task_def.scoring == ScoringMode.STEP_ACCURACY:
        return _score_step_accuracy(predictions, targets)
    elif task_def.scoring == ScoringMode.TOKEN_MATCH:
        return _score_token_match(predictions, targets, tokenizer)
    else:
        raise ValueError(f"Unknown scoring mode: {task_def.scoring}")


# ── Batched oracle generation ──

# Imports from AO that are needed for generation.
# These are deferred to avoid import errors when AO isn't on path yet.
_AO_IMPORTS_LOADED = False
_ao_modules: dict[str, Any] = {}


def _ensure_ao_imports():
    global _AO_IMPORTS_LOADED, _ao_modules
    if _AO_IMPORTS_LOADED:
        return
    from core.ao import (
        collect_activations_multiple_layers,
        get_batched_steering_hook,
        get_hf_submodule,
        add_hook,
        _active_adapter_name,
        SPECIAL_TOKEN,
    )
    from nl_probes.utils.activation_utils import (
        collect_activations_multiple_layers as _cam,
    )
    _ao_modules["collect_activations_multiple_layers"] = collect_activations_multiple_layers
    _ao_modules["get_batched_steering_hook"] = get_batched_steering_hook
    _ao_modules["get_hf_submodule"] = get_hf_submodule
    _ao_modules["add_hook"] = add_hook
    _ao_modules["_active_adapter_name"] = _active_adapter_name
    _ao_modules["SPECIAL_TOKEN"] = SPECIAL_TOKEN
    _AO_IMPORTS_LOADED = True


def _materialize_activations(
    model,
    tokenizer,
    items: list[dict],
    layers: list[int],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Extract activation vectors from context_input_ids at context_positions.

    Returns list of activation tensors, one per item. Each tensor has shape
    [total_positions, D] where total_positions = K * len(layers).
    """
    _ensure_ao_imports()
    collect_activations_multiple_layers = _ao_modules["collect_activations_multiple_layers"]
    get_hf_submodule = _ao_modules["get_hf_submodule"]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    contexts = [item["context_input_ids"] for item in items]
    all_positions = [item["context_positions"] for item in items]
    max_len = max(len(c) for c in contexts)

    # Pad and prepare tensors
    input_ids_list = []
    attn_masks_list = []
    left_offsets = []

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_list.append(
            torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device)
        )
        attn_masks_list.append(
            torch.tensor(
                [False] * pad_len + [True] * len(c), dtype=torch.bool, device=device
            )
        )
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attn_masks_list, dim=0),
    }

    submodules = {
        layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers
    }

    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    # Extract per-item activation vectors
    result = []
    N = len(layers)
    for b in range(len(items)):
        positions = all_positions[b]
        K = len(positions) // N

        vectors_parts = []
        for li, layer in enumerate(layers):
            acts_BLD = acts_by_layer[layer]
            chunk_positions = positions[li * K : (li + 1) * K]
            adjusted = [p + left_offsets[b] for p in chunk_positions]
            layer_vecs = acts_BLD[b, adjusted, :]  # [K, D]
            vectors_parts.append(layer_vecs)

        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()
        result.append(vectors)

    # Clean up
    del acts_by_layer, inputs_BL
    torch.cuda.empty_cache()

    return result


def _batched_oracle_generate(
    model,
    tokenizer,
    items: list[tuple[torch.Tensor, str]],
    layers: list[int],
    device: str = "cuda",
    injection_layer: int = 1,
    max_new_tokens: int = 64,
    eval_batch_size: int = 8,
    oracle_adapter_name: str | None = "default",
) -> list[str]:
    """Batched oracle generation with per-item activation steering.

    Args:
        items: List of (activations [K_i, D], oracle_prompt_text) tuples.
        layers: Layer list for prefix format (e.g. [9, 18, 27]).
        device: Target device.
        injection_layer: Layer to inject at (default 1).
        max_new_tokens: Max tokens to generate per item.
        eval_batch_size: Mini-batch size for generation.
        oracle_adapter_name: Adapter name to use for generation.

    Returns:
        List of oracle response strings, one per input item.
    """
    if not items:
        return []

    _ensure_ao_imports()
    get_batched_steering_hook = _ao_modules["get_batched_steering_hook"]
    get_hf_submodule = _ao_modules["get_hf_submodule"]
    add_hook = _ao_modules["add_hook"]
    _active_adapter_name = _ao_modules["_active_adapter_name"]
    SPECIAL_TOKEN = _ao_modules["SPECIAL_TOKEN"]

    eval_batch_size = max(1, int(eval_batch_size))
    dtype = torch.bfloat16
    ph_token = SPECIAL_TOKEN

    layer_list = list(layers) if len(layers) > 1 else layers
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Expected single token for '{ph_token}', got {len(ph_id)}"
    ph_id = ph_id[0]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Phase 1: Tokenize all items and find placeholder positions
    all_input_ids: list[list[int]] = []
    all_ph_positions: list[list[int]] = []

    for activations, oracle_prompt in items:
        num_positions = activations.shape[0]
        N = len(layer_list)
        K = num_positions // N
        assert K * N == num_positions, f"num_positions={num_positions} not divisible by {N} layers"
        parts = [f"L{l}:" + ph_token * K for l in layer_list]
        prefix = " ".join(parts) + "\n"
        full_prompt = prefix + oracle_prompt

        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)

        positions = [i for i, tid in enumerate(input_ids) if tid == ph_id][:num_positions]
        assert len(positions) == num_positions, (
            f"Found {len(positions)} placeholder positions, expected {num_positions}"
        )

        all_input_ids.append(input_ids)
        all_ph_positions.append(positions)

    # Phase 2: Set adapter
    previous_adapter = _active_adapter_name(model)
    if oracle_adapter_name is not None:
        model.set_adapter(oracle_adapter_name)

    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)
    was_training = model.training
    model.eval()

    # Phase 3: Generate in mini-batches with length bucketing
    sorted_indices = sorted(range(len(items)), key=lambda i: len(all_input_ids[i]))
    all_responses: list[str] = [""] * len(items)

    try:
        for group_start in range(0, len(sorted_indices), eval_batch_size):
            initial_indices = sorted_indices[group_start:group_start + eval_batch_size]
            pending_groups: list[list[int]] = [initial_indices]

            while pending_groups:
                batch_indices = pending_groups.pop(0)
                try:
                    batch_ids = [all_input_ids[i] for i in batch_indices]
                    batch_pre_pad_pos = [all_ph_positions[i] for i in batch_indices]
                    batch_acts = [items[i][0] for i in batch_indices]

                    max_len = max(len(ids) for ids in batch_ids)
                    padded_ids = []
                    attention_masks = []
                    batch_padded_positions = []

                    for j, ids in enumerate(batch_ids):
                        pad_len = max_len - len(ids)
                        padded_ids.append([pad_id] * pad_len + ids)
                        attention_masks.append([0] * pad_len + [1] * len(ids))
                        batch_padded_positions.append(
                            [p + pad_len for p in batch_pre_pad_pos[j]]
                        )

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

                    for j, item_idx in enumerate(batch_indices):
                        generated = outputs[j][max_len:]
                        all_responses[item_idx] = tokenizer.decode(
                            generated, skip_special_tokens=True,
                        )

                except Exception as e:
                    msg = str(e).lower()
                    is_oom = "out of memory" in msg or "cuda oom" in msg
                    if is_oom and len(batch_indices) > 1:
                        mid = len(batch_indices) // 2
                        pending_groups.insert(0, batch_indices[mid:])
                        pending_groups.insert(0, batch_indices[:mid])
                        torch.cuda.empty_cache()
                        continue
                    print(f"    [eval] Mini-batch of {len(batch_indices)} failed: {e}")
                    if is_oom:
                        torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
        if (previous_adapter
                and previous_adapter in getattr(model, "peft_config", {})
                and previous_adapter != oracle_adapter_name):
            model.set_adapter(previous_adapter)

    return all_responses


# ── Main eval entry point ──

def run_eval(
    model,
    tokenizer,
    task_names: list[str] | None = None,
    max_items: int = 30,
    eval_batch_size: int = 4,
    device: str = "cuda",
    layers: list[int] | None = None,
    injection_layer: int = 1,
    oracle_adapter_name: str = "default",
    skip_rot13: bool = False,
    activation_extract_batch_size: int = 4,
) -> dict[str, float]:
    """Run eval for all (or specified) tasks.

    For each task: load test split → materialize activations → oracle generate → score.
    Returns flat metrics dict for wandb.log().
    """
    if layers is None:
        layers = [9, 18, 27]

    all_tasks = get_eval_tasks()
    if task_names is not None:
        tasks_to_eval = {k: all_tasks[k] for k in task_names if k in all_tasks}
    else:
        tasks_to_eval = all_tasks

    metrics: dict[str, float] = {}

    for task_name, task_def in tasks_to_eval.items():
        if skip_rot13 and task_def.needs_rot13_adapter:
            continue

        t0 = time.time()
        try:
            result = _eval_single_task(
                model=model,
                tokenizer=tokenizer,
                task_name=task_name,
                task_def=task_def,
                max_items=max_items,
                eval_batch_size=eval_batch_size,
                device=device,
                layers=layers,
                injection_layer=injection_layer,
                oracle_adapter_name=oracle_adapter_name,
                activation_extract_batch_size=activation_extract_batch_size,
            )
            elapsed = time.time() - t0

            # Flatten into wandb-compatible keys
            primary_metric = _primary_metric_name(task_def.scoring)
            if primary_metric in result:
                metrics[f"eval/{task_name}"] = result[primary_metric]
            metrics[f"eval_n/{task_name}"] = result.get("n", 0)
            if "unparsed_rate" in result:
                metrics[f"eval_n/{task_name}_parse_fail"] = result["unparsed_rate"]
            metrics[f"eval_time/{task_name}"] = elapsed

            print(
                f"  [eval] {task_name}: "
                f"{primary_metric}={result.get(primary_metric, 0):.3f} "
                f"(n={result.get('n', 0)}, {elapsed:.1f}s)"
            )

        except Exception as e:
            print(f"  [eval] {task_name} FAILED: {e}")
            metrics[f"eval/{task_name}_error"] = 1.0

        gc.collect()
        torch.cuda.empty_cache()

    return metrics


def _eval_single_task(
    model,
    tokenizer,
    task_name: str,
    task_def: TaskDef,
    max_items: int,
    eval_batch_size: int,
    device: str,
    layers: list[int],
    injection_layer: int,
    oracle_adapter_name: str,
    activation_extract_batch_size: int,
) -> dict[str, float]:
    """Eval a single task: load → materialize → generate → score."""
    # Load test data
    test_data = load_task_data(task_name, split="test", n=max_items)
    if not test_data:
        return {"n": 0}

    # Materialize activations in mini-batches
    all_activations: list[torch.Tensor] = []
    for start in range(0, len(test_data), activation_extract_batch_size):
        chunk = test_data[start:start + activation_extract_batch_size]
        chunk_acts = _materialize_activations(
            model, tokenizer, chunk, layers=layers, device=device,
        )
        all_activations.extend(chunk_acts)

    # Build (activations, prompt) pairs for oracle generation
    oracle_items = [
        (act, item["prompt"])
        for act, item in zip(all_activations, test_data)
    ]

    # Generate oracle responses
    predictions = _batched_oracle_generate(
        model=model,
        tokenizer=tokenizer,
        items=oracle_items,
        layers=layers,
        device=device,
        injection_layer=injection_layer,
        max_new_tokens=task_def.max_new_tokens,
        eval_batch_size=eval_batch_size,
        oracle_adapter_name=oracle_adapter_name,
    )

    # Extract targets
    targets = [item["target_response"] for item in test_data]

    # Score
    return score_task(task_def, predictions, targets, tokenizer)


def _primary_metric_name(scoring: ScoringMode) -> str:
    """Map scoring mode to its primary metric key."""
    return {
        ScoringMode.BINARY: "accuracy",
        ScoringMode.TOKEN_F1: "token_f1",
        ScoringMode.STEP_ACCURACY: "step_accuracy",
        ScoringMode.TOKEN_MATCH: "token_match_rate",
    }[scoring]
