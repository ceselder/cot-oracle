"""
Shared infrastructure for open-ended AO evals.

Each eval provides:
  - Dataset loading + prompt building (eval-specific)
  - score_fn(results, metadata) -> scored_results
  - metrics_fn(scored_results) -> metrics dict

This module handles the boilerplate: adapter loading, run_verbalizer loop,
result saving, cleanup, and the __main__ block.
"""

import json
import os
import random
import re
from dataclasses import asdict
from typing import Any, Callable

import matplotlib
import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import AObench.base_experiment as base_experiment
from AObench.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
)
from AObench.configs.sft_config import SelfInterpTrainingConfig
from AObench.utils.common import load_model, load_tokenizer
from AObench.utils.dataset_utils import BinaryFeatureResult

# Standard verbalizer LoRAs for running evals. Not used as a default —
# callers must pass verbalizer_lora_paths explicitly.
STANDARD_VERBALIZER_LORAS = [
    "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B",
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
]

# Type aliases for the eval-specific callables
ScoreFn = Callable[[list[VerbalizerResults], list[dict[str, Any]]], list[dict[str, Any]]]
MetricsFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
BinaryMetricsFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
PrintSampleFn = Callable[[list[dict[str, Any]]], None] | None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def ensure_default_adapter(model: AutoModelForCausalLM) -> None:
    """Ensure model has a 'default' adapter (needed for adapter switching)."""
    if not hasattr(model, "peft_config") or "default" not in model.peft_config:
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")


def extract_yes_no(response: str) -> str | None:
    """Extract yes/no from AO response (first word)."""
    text = response.strip().lower()
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    if first_word in ("yes", "no"):
        return first_word
    return None


def default_layer_combination(training_config: SelfInterpTrainingConfig) -> list[int]:
    """Select which layer combination to use from a training config.

    Prefers multi-layer [25, 50, 75] if available, falls back to single-layer [50].
    """
    combos = training_config.layer_combinations

    # Multi-layer models trained on 25/50/75% layers (current standard)
    if [25, 50, 75] in combos:
        return [25, 50, 75]

    # Single-layer models trained on 50% layer only (older checkpoints, e.g. past_lens_addition)
    if [50] in combos:
        return [50]

    raise ValueError(
        f"No recognized layer combination found in {combos}. "
        f"Expected [25, 50, 75] (multi-layer) or [50] (single-layer)."
    )


def build_verbalizer_eval_config(
    model_name: str,
    training_config: SelfInterpTrainingConfig,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    selected_layer_combination: list[int] | None = None,
) -> base_experiment.VerbalizerEvalConfig:
    """Build a VerbalizerEvalConfig from an AO training config.

    training_config is required — layer_combinations and selected_layer_combination
    are read from it to ensure the eval uses the same layers the AO was trained on.
    """
    layer_combinations = training_config.layer_combinations
    if selected_layer_combination is None:
        selected_layer_combination = default_layer_combination(training_config)
    assert selected_layer_combination in layer_combinations, (
        f"selected_layer_combination {selected_layer_combination} not in "
        f"training config layer_combinations {layer_combinations}"
    )

    return base_experiment.VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        eval_batch_size=eval_batch_size,
        verbalizer_generation_kwargs=generation_kwargs,
        layer_combinations=layer_combinations,
        selected_layer_combination=selected_layer_combination,
    )


def _print_metrics(metrics: dict[str, Any]) -> None:
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")


def _load_adapter_and_build_config(
    model: AutoModelForCausalLM,
    verbalizer_entry: str,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
) -> tuple[str, base_experiment.VerbalizerEvalConfig]:
    """Load a verbalizer adapter and build its eval config."""
    sanitized_name, training_config = base_experiment.load_oracle_adapter(model, verbalizer_entry)
    config = build_verbalizer_eval_config(
        model_name=model_name,
        training_config=training_config,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
    )
    base_experiment.assert_training_config_matches_verbalizer_eval_config(config, training_config)
    return sanitized_name, config


def get_first_ao_response(result: VerbalizerResults) -> str | None:
    """Extract the first AO response from a VerbalizerResults."""
    if not result.responses:
        return None
    return result.responses[0]


def run_default_eval(
    *,
    eval_name: str,
    run_eval_fn: Callable[..., dict[str, Any]],
    run_eval_kwargs: dict[str, Any],
    model_name: str,
) -> None:
    """
    Standard __main__ boilerplate: set seeds, load model, run eval, print summary.

    run_eval_fn is called with (model_name, model, tokenizer, device, **run_eval_kwargs).
    It should accept those as keyword arguments and return the summary dict.

    Callers must provide verbalizer_lora_paths in run_eval_kwargs — no silent
    defaults are applied.
    """
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Set default output_dir if not provided
    if "output_dir" not in run_eval_kwargs:
        output_dir = f"experiments/{eval_name}_eval_results/{model_name_str}"
        os.makedirs(output_dir, exist_ok=True)
        run_eval_kwargs["output_dir"] = output_dir

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()

    summary = run_eval_fn(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **run_eval_kwargs,
    )

    print(f"\n=== {eval_name.replace('_', ' ').title()} Eval Summary ===")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Generation eval loop
# ---------------------------------------------------------------------------


def run_verbalizer_generation_eval_loop(
    *,
    eval_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    prompt_infos: list[VerbalizerInputInfo],
    entry_metadata: list[dict[str, Any]],
    score_fn: ScoreFn,
    metrics_fn: MetricsFn,
    num_entries: int,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    print_sample_fn: PrintSampleFn = None,
    extra_output_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run generation-based eval loop.

    Iterates verbalizer LoRAs, runs run_verbalizer (text generation),
    scores with score_fn, computes metrics, saves results.
    """
    ensure_default_adapter(model)
    model.eval()

    all_scored_results: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_verbalizer_name, loop_config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        print(f"Running {eval_name} eval with verbalizer: {verbalizer_entry}")
        verbalizer_key = verbalizer_entry.split("/")[-1]
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")

        results = base_experiment.run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_verbalizer_name,
            target_lora_path=None,
            config=loop_config,
            device=device,
        )

        scored_results = score_fn(results, entry_metadata)
        metrics: dict[str, Any] | None = None
        if scored_results:
            metrics = metrics_fn(scored_results)
            metrics_by_verbalizer[verbalizer_key] = metrics
            print(f"\n  Metrics for {verbalizer_key}:")
            _print_metrics(metrics)

            if print_sample_fn is not None:
                print_sample_fn(scored_results)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{eval_name}_{lora_name}.json")
            output_data: dict[str, Any] = {
                "config": asdict(loop_config),
                "verbalizer": verbalizer_entry,
                "num_entries": num_entries,
                "scored_results": scored_results,
                "metrics": metrics if scored_results else None,
                "verbalizer_results": [asdict(r) for r in results],
            }
            if extra_output_data:
                output_data.update(extra_output_data)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved results to {output_path}")

        all_scored_results.extend(scored_results)

        if sanitized_verbalizer_name is not None:
            if sanitized_verbalizer_name in model.peft_config:
                model.delete_adapter(sanitized_verbalizer_name)

    overall_metrics = metrics_fn(all_scored_results) if all_scored_results else {}
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_entries": num_entries,
        "num_scored": len(all_scored_results),
    }


# ---------------------------------------------------------------------------
# Binary (logit) scoring eval loop
# ---------------------------------------------------------------------------

YES_NO_CANDIDATE_VARIANTS: dict[str, list[str]] = {
    "yes": ["yes", " yes", "Yes", " Yes", "YES", " YES", "\nyes", "\nYes", "\nYES"],
    "no": ["no", " no", "No", " No", "NO", " NO", "\nno", "\nNo", "\nNO"],
}


def build_yes_no_candidate_token_groups(tokenizer: AutoTokenizer) -> dict[str, list[int]]:
    """Collect single-token yes/no variants for first-token AO scoring."""
    token_groups: dict[str, list[int]] = {}
    for label, variants in YES_NO_CANDIDATE_VARIANTS.items():
        token_ids: list[int] = []
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in token_ids:
                token_ids.append(int(ids[0]))
        if not token_ids:
            raise ValueError(f"Tokenizer had no single-token variants for label '{label}'")
        token_groups[label] = token_ids
    return token_groups


def describe_candidate_token_groups(
    tokenizer: AutoTokenizer,
    candidate_token_groups: dict[str, list[int]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        label: [
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
            for token_id in token_ids
        ]
        for label, token_ids in candidate_token_groups.items()
    }


def compute_roc_curve_data(
    labels: list[int],
    scores: list[float],
) -> dict[str, Any] | None:
    if not labels:
        return None

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = 1 + threshold_indices - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_indices]]

    tpr = tps / positives
    fpr = fps / negatives
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.0+ renamed trapz
    auc = float(_trapz(tpr, fpr))

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc,
        "positives": positives,
        "negatives": negatives,
    }


def score_binary_yes_no_results(
    results: list[BinaryFeatureResult],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    for result, meta in zip(results, entry_metadata, strict=True):
        ground_truth = result.meta_info["ground_truth"].strip().lower()
        if ground_truth not in ("yes", "no"):
            continue

        yes_score = float(result.candidate_scores["yes"])
        no_score = float(result.candidate_scores["no"])
        margin = yes_score - no_score
        predicted_answer = "yes" if margin >= 0 else "no"

        scored.append({
            **meta,
            "act_key": result.meta_info["act_key"],
            "ground_truth": ground_truth,
            "binary_label": 1 if ground_truth == "yes" else 0,
            "yes_score": yes_score,
            "no_score": no_score,
            "margin_yes_minus_no": margin,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == ground_truth,
            "argmax_token_id": result.argmax_token_id,
            "argmax_token_text": result.argmax_token_text,
            "argmax_logit": result.argmax_logit,
        })

    return scored


def _compute_binary_metric_block(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_results)
    metrics: dict[str, Any] = {
        "total": total,
        "correct_at_zero": sum(1 for r in scored_results if r["is_correct"]),
        "accuracy_at_zero": (
            sum(1 for r in scored_results if r["is_correct"]) / total if total > 0 else 0.0
        ),
    }

    if total > 0:
        positive_rows = [r for r in scored_results if r["binary_label"] == 1]
        negative_rows = [r for r in scored_results if r["binary_label"] == 0]
        metrics["positive_rate"] = len(positive_rows) / total
        metrics["mean_margin_when_yes"] = (
            sum(r["margin_yes_minus_no"] for r in positive_rows) / len(positive_rows)
            if positive_rows else 0.0
        )
        metrics["mean_margin_when_no"] = (
            sum(r["margin_yes_minus_no"] for r in negative_rows) / len(negative_rows)
            if negative_rows else 0.0
        )

    roc_data = compute_roc_curve_data(
        labels=[int(r["binary_label"]) for r in scored_results],
        scores=[float(r["margin_yes_minus_no"]) for r in scored_results],
    )
    if roc_data is not None:
        metrics["roc_auc"] = float(roc_data["auc"])
        metrics["num_positive"] = int(roc_data["positives"])
        metrics["num_negative"] = int(roc_data["negatives"])

    return metrics


MAX_GROUP_VALUES = 20


def _append_group_metrics(
    metrics: dict[str, Any],
    scored_results: list[dict[str, Any]],
    field_name: str,
    prefix: str,
) -> None:
    field_values = sorted({r[field_name] for r in scored_results if field_name in r and r[field_name] is not None})
    if not field_values or len(field_values) > MAX_GROUP_VALUES:
        return

    for field_value in field_values:
        subset = [r for r in scored_results if r.get(field_name) == field_value]
        subset_metrics = _compute_binary_metric_block(subset)
        for key, value in subset_metrics.items():
            metrics[f"{prefix}_{field_value}_{key}"] = value


def compute_binary_yes_no_metrics(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _compute_binary_metric_block(scored_results)
    _append_group_metrics(metrics, scored_results, "prompt_name", "prompt")
    _append_group_metrics(metrics, scored_results, "condition", "cond")
    return metrics


def save_binary_yes_no_roc_plot(
    scored_results: list[dict[str, Any]],
    output_path: str,
    title: str,
) -> str | None:
    overall_curve = compute_roc_curve_data(
        labels=[int(r["binary_label"]) for r in scored_results],
        scores=[float(r["margin_yes_minus_no"]) for r in scored_results],
    )
    if overall_curve is None:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(
        overall_curve["fpr"],
        overall_curve["tpr"],
        label=f"overall (AUC={overall_curve['auc']:.3f})",
        linewidth=2.5,
        color="black",
    )

    prompt_names = sorted({r["prompt_name"] for r in scored_results if "prompt_name" in r})
    if len(prompt_names) > 1:
        for prompt_name in prompt_names:
            subset = [r for r in scored_results if r.get("prompt_name") == prompt_name]
            prompt_curve = compute_roc_curve_data(
                labels=[int(r["binary_label"]) for r in subset],
                scores=[float(r["margin_yes_minus_no"]) for r in subset],
            )
            if prompt_curve is None:
                continue
            plt.plot(
                prompt_curve["fpr"],
                prompt_curve["tpr"],
                label=f"{prompt_name} (AUC={prompt_curve['auc']:.3f})",
                linewidth=1.8,
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.5, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def run_verbalizer_binary_eval_loop(
    *,
    eval_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    prompt_infos: list[VerbalizerInputInfo],
    entry_metadata: list[dict[str, Any]],
    num_entries: int,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    binary_metrics_fn: BinaryMetricsFn | None = None,
) -> dict[str, Any]:
    """Run binary (logit) scoring eval loop.

    Iterates verbalizer LoRAs, runs run_verbalizer_binary_score,
    scores with score_binary_yes_no_results, computes metrics, saves ROC plots.
    """
    ensure_default_adapter(model)
    model.eval()

    candidate_token_groups = build_yes_no_candidate_token_groups(tokenizer)
    candidate_group_info = describe_candidate_token_groups(tokenizer, candidate_token_groups)

    all_scored_results: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
    plot_paths_by_verbalizer: dict[str, str] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_verbalizer_name, loop_config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        assert len(loop_config.activation_input_types) == 1, (
            f"Binary eval requires exactly one activation_input_type, "
            f"got {loop_config.activation_input_types}. Multiple act types produce "
            f"multiple results per entry, breaking the 1:1 mapping with entry_metadata."
        )

        print(f"Running {eval_name} binary eval with verbalizer: {verbalizer_entry}")
        verbalizer_key = verbalizer_entry.split("/")[-1]
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_verbalizer_name,
            target_lora_path=None,
            config=loop_config,
            device=device,
            candidate_token_groups=candidate_token_groups,
        )

        scored_results = score_binary_yes_no_results(binary_results, entry_metadata)
        metrics: dict[str, Any] | None = None
        plot_path: str | None = None

        if scored_results:
            _metrics_fn = binary_metrics_fn or compute_binary_yes_no_metrics
            metrics = _metrics_fn(scored_results)
            metrics_by_verbalizer[verbalizer_key] = metrics
            print(f"\n  Binary score metrics for {verbalizer_key}:")
            _print_metrics(metrics)

            if output_dir is not None:
                plot_path = save_binary_yes_no_roc_plot(
                    scored_results,
                    output_path=os.path.join(output_dir, f"{eval_name}_{lora_name}_roc_auc.png"),
                    title=f"{eval_name} - {verbalizer_key}",
                )
                if plot_path is not None:
                    plot_paths_by_verbalizer[verbalizer_key] = plot_path
                    print(f"  Saved ROC curve to {plot_path}")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{eval_name}_binary_{lora_name}.json")
            output_data: dict[str, Any] = {
                "config": asdict(loop_config),
                "verbalizer": verbalizer_entry,
                "num_entries": num_entries,
                "binary_score_candidate_groups": candidate_group_info,
                "binary_scored_results": scored_results,
                "binary_score_metrics": metrics if scored_results else None,
                "binary_roc_plot_path": plot_path,
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved binary results to {output_path}")

        all_scored_results.extend(scored_results)

        if sanitized_verbalizer_name is not None:
            if sanitized_verbalizer_name in model.peft_config:
                model.delete_adapter(sanitized_verbalizer_name)

    _overall_metrics_fn = binary_metrics_fn or compute_binary_yes_no_metrics
    overall_metrics = _overall_metrics_fn(all_scored_results) if all_scored_results else {}
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_scored": len(all_scored_results),
        "candidate_token_groups": candidate_group_info,
        "plot_paths_by_verbalizer": plot_paths_by_verbalizer,
    }
