"""
Generate comparison report: bar charts + summary table for AObench eval results.

Usage:
    python -m AObench.utils.report results_dir/
    python -m AObench.utils.report results_dir/ --output report.html

Reads the per-eval JSON files produced by run_all.py and generates a visual
comparison across verbalizer LoRAs.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BOOTSTRAP_REPS = 600


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 3:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Extract metrics from eval summaries
# ---------------------------------------------------------------------------

# Maps eval name -> (metric_key, display_name, higher_is_better)
EVAL_METRIC_MAP: dict[str, tuple[str, str, bool]] = {
    "number_prediction": ("matches_model_answer_rate", "Model Match Rate", True),
    "mmlu_prediction": ("roc_auc", "ROC AUC", True),
    "backtracking": ("mean_correctness", "Mean Correctness (1-5)", True),
    "backtracking_mc": ("accuracy", "MC Accuracy", True),
    "missing_info": ("roc_auc", "ROC AUC", True),
    "sycophancy": ("roc_auc", "ROC AUC", True),
    "system_prompt_qa_hidden": ("mean_correctness", "Mean Correctness (1-5)", True),
    "system_prompt_qa_latentqa": ("mean_correctness", "Mean Correctness (1-5)", True),
    "taboo": ("accuracy", "Accuracy", True),
    "personaqa": ("accuracy", "Accuracy", True),
    "vagueness": ("specificity_rate", "Non-Vagueness", True),
    "domain_confusion": ("domain_correct_specific_rate", "Domain Accuracy", True),
    "activation_sensitivity": ("activation_sensitivity", "Activation Sensitivity", True),
    "hallucination_1pos": ("correct_rate", "Correct Rate (1 pos)", True),
    "hallucination_5pos": ("correct_rate", "Correct Rate (5 pos)", True),
    "hallucination_20pos": ("correct_rate", "Correct Rate (20 pos)", True),
}

# Chance baselines for each eval
CHANCE_BASELINES: dict[str, float] = {
    "backtracking_mc": 0.25,
    "mmlu_prediction": 0.50,
    "missing_info": 0.50,
    "sycophancy": 0.50,
}
SCALE_1_5_EVALS = {"backtracking", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"}


def extract_verbalizer_metric(
    summary: dict[str, Any],
    eval_name: str,
) -> dict[str, float]:
    """Extract the primary metric per verbalizer from an eval summary."""
    metric_key, _, _ = EVAL_METRIC_MAP.get(eval_name, ("accuracy", "Accuracy", True))

    result: dict[str, float] = {}

    # Handle mode_results (mmlu_prediction, sycophancy have nested modes)
    # Average the metric across modes per verbalizer
    if "mode_results" in summary:
        per_verb_values: dict[str, list[float]] = {}
        for mode_name, mode_result in summary.get("mode_results", {}).items():
            mbv = mode_result.get("metrics_by_verbalizer", {})
            for verb_key, metrics in mbv.items():
                if isinstance(metrics, dict) and metric_key in metrics:
                    per_verb_values.setdefault(verb_key, []).append(metrics[metric_key])
        for verb_key, vals in per_verb_values.items():
            result[verb_key] = sum(vals) / len(vals)
        return result

    # Standard: metrics_by_verbalizer
    mbv = summary.get("metrics_by_verbalizer", {})
    for verb_key, metrics in mbv.items():
        if isinstance(metrics, dict) and metric_key in metrics:
            result[verb_key] = metrics[metric_key]

    # mc_results_by_verbalizer (backtracking_mc)
    mc_mbv = summary.get("mc_results_by_verbalizer", {})
    for verb_key, metrics in mc_mbv.items():
        if isinstance(metrics, dict) and metric_key in metrics:
            result[verb_key] = metrics[metric_key]

    return result


def _slug_verbalizer_name(name: str) -> str:
    return name.split("/")[-1].replace("/", "_").replace(".", "_")


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(set(labels)) < 2:
        return None
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_indices]
    fps = 1 + threshold_indices - tps
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    tpr = tps / positives
    fpr = fps / negatives
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tpr, fpr))


def _primary_metric_from_rows(eval_name: str, rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    if eval_name == "number_prediction":
        return sum(1 for r in rows if r.get("matches_model_answer")) / len(rows)
    if eval_name in {"mmlu_prediction", "missing_info", "sycophancy"}:
        auc = _binary_auc(
            [int(r["binary_label"]) for r in rows],
            [float(r["margin_yes_minus_no"]) for r in rows],
        )
        return auc
    if eval_name == "backtracking_mc":
        return sum(1 for r in rows if r.get("is_correct")) / len(rows)
    if eval_name in {"backtracking", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"}:
        values = [float(r["correctness"]) for r in rows if "correctness" in r]
        return sum(values) / len(values) if values else None
    if eval_name == "vagueness":
        return sum(1 for r in rows if r.get("category") == "specific_correct") / len(rows)
    if eval_name == "domain_confusion":
        return sum(1 for r in rows if r.get("category") == "domain_correct_specific") / len(rows)
    if eval_name.startswith("hallucination_"):
        return sum(1 for r in rows if r.get("category") == "correct") / len(rows)
    if eval_name == "activation_sensitivity":
        return sum(1 for r in rows if r.get("category") == "divergent_meaningful") / len(rows)
    return None


def _bootstrap_metric_samples(
    eval_name: str,
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, Any] | None:
    base_metric = _primary_metric_from_rows(eval_name, rows)
    if base_metric is None:
        return None
    if len(rows) < 2:
        return {
            "mean": float(base_metric),
            "lo": float(base_metric),
            "hi": float(base_metric),
            "samples": np.full(n_bootstrap, float(base_metric), dtype=np.float64),
        }

    rng = np.random.default_rng(seed)
    n = len(rows)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_rows = [rows[j] for j in idx]
        metric = _primary_metric_from_rows(eval_name, sample_rows)
        samples[i] = float(base_metric if metric is None or math.isnan(metric) else metric)
    alpha = 0.025
    return {
        "mean": float(base_metric),
        "lo": float(np.percentile(samples, 100 * alpha)),
        "hi": float(np.percentile(samples, 100 * (1 - alpha))),
        "samples": samples,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _raw_payloads_for_eval(
    results_path: Path,
    eval_name: str,
    verbalizer_name: str,
) -> list[dict[str, Any]]:
    slug = _slug_verbalizer_name(verbalizer_name)
    eval_dir = results_path / eval_name

    if eval_name == "mmlu_prediction":
        paths = [
            eval_dir / "pre_answer" / f"mmlu_prediction_binary_{slug}.json",
            eval_dir / "post_answer" / f"mmlu_prediction_binary_{slug}.json",
        ]
    elif eval_name == "sycophancy":
        paths = [
            eval_dir / "no_cot" / f"sycophancy_no_cot_binary_{slug}.json",
            eval_dir / "cot" / f"sycophancy_cot_binary_{slug}.json",
        ]
    elif eval_name == "backtracking_mc":
        paths = [eval_dir / f"backtracking_mc_{slug}.json"]
    elif eval_name in {"mmlu_prediction", "missing_info"}:
        paths = [eval_dir / f"{eval_name}_binary_{slug}.json"]
    elif eval_name in {"number_prediction", "backtracking", "vagueness", "domain_confusion", "activation_sensitivity"} or eval_name.startswith("hallucination_") or eval_name.startswith("system_prompt_qa_"):
        paths = [eval_dir / f"{eval_name}_{slug}.json"]
    else:
        paths = [eval_dir / f"{eval_name}_{slug}.json"]

    payloads = []
    for path in paths:
        payload = _load_json(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def compute_eval_intervals(
    results_path: Path,
    eval_results: dict[str, dict[str, float]],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
) -> dict[str, dict[str, dict[str, Any]]]:
    eval_intervals: dict[str, dict[str, dict[str, Any]]] = {}

    for eval_name, verb_metrics in eval_results.items():
        verb_intervals: dict[str, dict[str, Any]] = {}
        for verbalizer_name, summary_metric in verb_metrics.items():
            payloads = _raw_payloads_for_eval(results_path, eval_name, verbalizer_name)
            if not payloads:
                verb_intervals[verbalizer_name] = {
                    "mean": float(summary_metric),
                    "lo": float(summary_metric),
                    "hi": float(summary_metric),
                    "samples": np.full(n_bootstrap, float(summary_metric), dtype=np.float64),
                    "source": "summary_only",
                }
                continue

            per_mode_boots: list[np.ndarray] = []
            per_mode_means: list[float] = []
            per_mode_los: list[float] = []
            per_mode_his: list[float] = []
            for payload in payloads:
                rows = payload.get("binary_scored_results")
                if rows is None:
                    rows = payload.get("scored_results")
                if rows is None:
                    rows = payload.get("pairs")
                boot = _bootstrap_metric_samples(eval_name, rows or [], n_bootstrap=n_bootstrap)
                if boot is None:
                    continue
                per_mode_boots.append(np.asarray(boot["samples"], dtype=np.float64))
                per_mode_means.append(float(boot["mean"]))
                per_mode_los.append(float(boot["lo"]))
                per_mode_his.append(float(boot["hi"]))

            if not per_mode_boots:
                verb_intervals[verbalizer_name] = {
                    "mean": float(summary_metric),
                    "lo": float(summary_metric),
                    "hi": float(summary_metric),
                    "samples": np.full(n_bootstrap, float(summary_metric), dtype=np.float64),
                    "source": "summary_only",
                }
                continue

            stacked = np.stack(per_mode_boots, axis=0)
            combined_samples = stacked.mean(axis=0)
            verb_intervals[verbalizer_name] = {
                "mean": float(np.mean(per_mode_means)),
                "lo": float(np.percentile(combined_samples, 2.5)),
                "hi": float(np.percentile(combined_samples, 97.5)),
                "samples": combined_samples,
                "source": "raw_outputs",
            }
        if verb_intervals:
            eval_intervals[eval_name] = verb_intervals

    return eval_intervals


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Canonical display names and colors for known checkpoints
DISPLAY_NAMES: dict[str, str] = {
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "Adam's AO",
    "latentqa_cls_past_lens_addition_Qwen3-8B": "Adam's AO",
    "checkpoints_latentqa_cls_on_policy_Qwen3-8B": "Adam's on-policy",
    "latentqa_cls_on_policy_Qwen3-8B": "Adam's on-policy",
    "adam-reupload-qwen3-8b-latentqa-cls-past-lens": "Adam original (66.5M tok)",
    "adam-reupload-qwen3-8b-full-mix-synthetic-qa-v3-replace-lqa": "Adam synth-QA-v3 (tok ?)",
    "cot-oracle-paper-ablation-adam-recipe-1layer": "Paper Adam recipe (17M tok)",
    "cot-oracle-paper-ablation-ours-1layer": "Paper ours 1L (22.5M tok)",
    "cot-oracle-paper-ablation-ours-3layers": "Paper ours 3L (18M tok)",
    "cot-oracle-paper-ablation-ours-3layers-onpolicy-lens-only": "Paper ours 3L on-policy (22.3M tok)",
    "cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO": "Ours final no-DPO (100M tok)",
    "qwen3-8b-final-sprint-checkpoint-no-DPO": "Ours final no-DPO (100M tok)",
    "cot-oracle-grpo-v1": "Ours GRPO",
    "cot-oracle-grpo-step-500": "Ours GRPO step 500 (tok n/a)",
    "checkpoints_Qwen3-8B_full_mix_synthetic_qa_v3_replace_lqa": "Adam's synth-QA-v3",
    "Qwen3-8B_full_mix_synthetic_qa_v3_replace_lqa": "Adam's synth-QA-v3",
}

DISPLAY_COLORS: dict[str, str] = {
    "Adam's AO": "#5C8D4D",
    "Adam's on-policy": "#8BAA5B",
    "Adam's synth-QA-v3": "#D8893D",
    "Adam original (66.5M tok)": "#4E7F4E",
    "Adam synth-QA-v3 (tok ?)": "#B8742F",
    "Paper Adam recipe (17M tok)": "#7C9C59",
    "Paper ours 1L (22.5M tok)": "#5DA5DA",
    "Paper ours 3L (18M tok)": "#2F7FB8",
    "Paper ours 3L on-policy (22.3M tok)": "#1F5C8B",
    "Ours final no-DPO (100M tok)": "#2A6FDF",
    "Ours GRPO": "#E91E63",
    "Ours GRPO step 500 (tok n/a)": "#C73E7C",
}
DISPLAY_ORDER = [
    "Adam original (66.5M tok)",
    "Adam synth-QA-v3 (tok ?)",
    "Paper Adam recipe (17M tok)",
    "Paper ours 1L (22.5M tok)",
    "Paper ours 3L (18M tok)",
    "Paper ours 3L on-policy (22.3M tok)",
    "Ours final no-DPO (100M tok)",
    "Ours GRPO",
    "Ours GRPO step 500 (tok n/a)",
]

EVAL_DISPLAY_NAMES: dict[str, str] = {
    "number_prediction": "Number\nPrediction",
    "mmlu_prediction": "MMLU\nPrediction",
    "backtracking": "Backtracking",
    "backtracking_mc": "Backtracking\nMC",
    "missing_info": "Missing\nInfo",
    "sycophancy": "Sycophancy",
    "system_prompt_qa_hidden": "System Prompt\nHidden",
    "system_prompt_qa_latentqa": "System Prompt\nPersona",
    "taboo": "Taboo",
    "personaqa": "PersonaQA",
    "vagueness": "Vagueness",
    "domain_confusion": "Domain\nConfusion",
    "activation_sensitivity": "Activation\nSensitivity",
    "hallucination_1pos": "Hallucination\n1 pos",
    "hallucination_5pos": "Hallucination\n5 pos",
    "hallucination_20pos": "Hallucination\n20 pos",
}


def shorten_lora_name(name: str) -> str:
    """Shorten a verbalizer LoRA name for display."""
    name = name.split("/")[-1]
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    # Common prefixes to strip
    for prefix in ["checkpoints_", "cot-oracle-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if len(name) > 40:
        name = name[:37] + "..."
    return name


def verbalizer_sort_key(name: str) -> tuple[int, str]:
    display = shorten_lora_name(name)
    if display in DISPLAY_ORDER:
        return (DISPLAY_ORDER.index(display), display)
    return (len(DISPLAY_ORDER), display)


def format_eval_display_name(eval_name: str) -> str:
    return EVAL_DISPLAY_NAMES.get(eval_name, eval_name.replace("_", "\n").title())


def format_metric_value(eval_name: str, value: float) -> str:
    if eval_name in SCALE_1_5_EVALS:
        return f"{value:.1f}"
    return f"{value:.2f}"


def normalize_metric_for_aggregate(eval_name: str, value: float) -> float:
    """Map eval metrics to a roughly comparable scale for aggregate plotting."""
    normalized = np.asarray(value, dtype=np.float64)
    if eval_name in SCALE_1_5_EVALS:
        normalized = (normalized - 1.0) / 4.0

    chance = CHANCE_BASELINES.get(eval_name)
    if chance is not None and chance < 1.0:
        normalized = (normalized - chance) / (1.0 - chance)

    if normalized.ndim == 0:
        return float(normalized)
    return normalized


def compute_aggregate_scores(
    eval_results: dict[str, dict[str, float]],
    eval_intervals: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    per_verbalizer: dict[str, dict[str, float]] = {}
    for eval_name, metrics in eval_results.items():
        for verbalizer_name, metric_value in metrics.items():
            per_verbalizer.setdefault(verbalizer_name, {})[eval_name] = normalize_metric_for_aggregate(
                eval_name,
                metric_value,
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for verbalizer_name, per_eval in per_verbalizer.items():
        if not per_eval:
            continue
        payload: dict[str, Any] = {
            "mean_normalized_score": sum(per_eval.values()) / len(per_eval),
            "num_evals": len(per_eval),
            "per_eval_normalized": per_eval,
        }
        if eval_intervals:
            sample_arrays = []
            for eval_name in per_eval:
                interval = eval_intervals.get(eval_name, {}).get(verbalizer_name)
                if not interval:
                    continue
                samples = np.asarray(interval.get("samples", []), dtype=np.float64)
                if samples.size == 0:
                    continue
                sample_arrays.append(normalize_metric_for_aggregate(eval_name, samples))
            if sample_arrays:
                stacked = np.stack(sample_arrays, axis=0)
                agg_samples = stacked.mean(axis=0)
                payload["ci_lo"] = float(np.percentile(agg_samples, 2.5))
                payload["ci_hi"] = float(np.percentile(agg_samples, 97.5))
        aggregate[verbalizer_name] = payload
    return aggregate


def plot_aggregate_scores(
    aggregate_scores: dict[str, dict[str, Any]],
    output_path: str,
    title: str = "Chance-Adjusted Aggregate Score",
) -> str:
    if not aggregate_scores:
        return output_path

    items = sorted(
        aggregate_scores.items(),
        key=lambda item: item[1]["mean_normalized_score"],
        reverse=True,
    )
    verbalizer_names = [name for name, _ in items]
    values = [item["mean_normalized_score"] for _, item in items]
    labels = [shorten_lora_name(name) for name in verbalizer_names]

    fig_height = max(5.8, 0.68 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))
    colors = [DISPLAY_COLORS.get(label, fallback_colors[i]) for i, label in enumerate(labels)]

    y = np.arange(len(labels))
    xerr = []
    has_err = False
    for _, payload in items:
        lo = payload.get("ci_lo")
        hi = payload.get("ci_hi")
        if lo is None or hi is None:
            xerr.append((0.0, 0.0))
            continue
        has_err = True
        mean = float(payload["mean_normalized_score"])
        xerr.append((mean - float(lo), float(hi) - mean))
    xerr_arr = np.array(xerr, dtype=np.float64).T if has_err else None
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5, xerr=xerr_arr, capsize=3 if has_err else 0)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + (0.015 if value >= 0 else -0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized score (higher is better)")
    ax.set_title(title)
    ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    x_min = min(values)
    x_max = max(values)
    ax.set_xlim(min(-0.05, x_min - 0.05), max(1.0, x_max + 0.08))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_comparison_bar_chart(
    eval_results: dict[str, dict[str, float]],
    output_path: str,
    title: str = "AObench Comparison",
    eval_intervals: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> str:
    """Create a dashboard-style comparison plot for verbalizers across evals.

    eval_results: {eval_name: {verbalizer_name: metric_value}}
    """
    all_verbs: set[str] = set()
    for metrics in eval_results.values():
        all_verbs.update(metrics.keys())
    eval_names = list(eval_results.keys())
    aggregate_scores = compute_aggregate_scores(eval_results, eval_intervals=eval_intervals)

    def _row_sort_key(name: str) -> tuple[float, tuple[int, str]]:
        aggregate = aggregate_scores.get(name, {}).get("mean_normalized_score", float("-inf"))
        return (-float(aggregate), verbalizer_sort_key(name))

    verb_names = sorted(all_verbs, key=_row_sort_key)
    n_evals = len(eval_names)
    n_verbs = len(verb_names)

    if n_evals == 0 or n_verbs == 0:
        return output_path

    normalized_matrix = np.full((n_verbs, n_evals), np.nan, dtype=np.float64)
    raw_matrix = np.full((n_verbs, n_evals), np.nan, dtype=np.float64)
    for row_idx, verb_name in enumerate(verb_names):
        for col_idx, eval_name in enumerate(eval_names):
            value = eval_results.get(eval_name, {}).get(verb_name)
            if value is None:
                continue
            raw_matrix[row_idx, col_idx] = float(value)
            normalized_matrix[row_idx, col_idx] = normalize_metric_for_aggregate(eval_name, float(value))

    labels = [shorten_lora_name(name) for name in verb_names]
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(n_verbs, 1)))
    row_colors = [DISPLAY_COLORS.get(label, fallback_colors[i]) for i, label in enumerate(labels)]

    fig_width = max(13.5, 6.0 + 1.15 * n_evals)
    fig_height = max(6.2, 0.72 * n_verbs + 2.3)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, max(1.8, 0.62 * n_evals)], wspace=0.04)
    ax_agg = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1], sharey=ax_agg)

    y = np.arange(n_verbs)
    aggregate_values = [aggregate_scores.get(name, {}).get("mean_normalized_score", np.nan) for name in verb_names]
    xerr = []
    has_err = False
    for name, value in zip(verb_names, aggregate_values, strict=True):
        payload = aggregate_scores.get(name, {})
        lo = payload.get("ci_lo")
        hi = payload.get("ci_hi")
        if lo is None or hi is None or np.isnan(value):
            xerr.append((0.0, 0.0))
            continue
        has_err = True
        xerr.append((float(value) - float(lo), float(hi) - float(value)))
    xerr_arr = np.array(xerr, dtype=np.float64).T if has_err else None

    bars = ax_agg.barh(
        y,
        aggregate_values,
        color=row_colors,
        edgecolor="white",
        linewidth=0.7,
        xerr=xerr_arr,
        capsize=3 if has_err else 0,
        height=0.74,
    )
    for bar, value in zip(bars, aggregate_values, strict=True):
        if np.isnan(value):
            continue
        ax_agg.text(
            float(value) + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax_agg.axvline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    ax_agg.set_xlim(min(-0.08, np.nanmin(aggregate_values) - 0.06), max(1.0, np.nanmax(aggregate_values) + 0.14))
    ax_agg.set_xlabel("Aggregate normalized score\n(higher is better)", fontsize=10)
    ax_agg.set_title("Overall", fontsize=12, pad=10)
    ax_agg.set_yticks(y)
    ax_agg.set_yticklabels(labels, fontsize=10)
    ax_agg.invert_yaxis()
    ax_agg.xaxis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax_agg.set_axisbelow(True)

    for tick, label in zip(ax_agg.get_yticklabels(), labels, strict=True):
        tick.set_color(DISPLAY_COLORS.get(label, "#1a1a1a"))

    masked = np.ma.masked_invalid(normalized_matrix)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax_heat.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.2,
        vmax=1.0,
    )

    ax_heat.set_xticks(np.arange(n_evals))
    ax_heat.set_xticklabels([format_eval_display_name(name) for name in eval_names], fontsize=10)
    ax_heat.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax_heat.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    ax_heat.set_yticks(y)
    ax_heat.tick_params(axis="y", left=False, labelleft=False)
    ax_heat.set_xticks(np.arange(-0.5, n_evals, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n_verbs, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(n_verbs):
        for col_idx, eval_name in enumerate(eval_names):
            raw_value = raw_matrix[row_idx, col_idx]
            norm_value = normalized_matrix[row_idx, col_idx]
            if np.isnan(raw_value):
                ax_heat.text(
                    col_idx,
                    row_idx,
                    "--",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#7a7a7a",
                )
                continue
            text_color = "white" if (norm_value >= 0.62 or norm_value <= -0.05) else "#1d1d1d"
            ax_heat.text(
                col_idx,
                row_idx,
                format_metric_value(eval_name, float(raw_value)),
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.038, pad=0.03)
    colorbar.set_label("Normalized score (higher is better)", rotation=90, labelpad=12)

    fig.suptitle(title, fontsize=18, y=0.955)
    fig.text(
        0.5,
        0.024,
        "Rows are sorted by aggregate score. Cell color uses a normalized higher-is-better scale. Cell text shows the raw primary metric.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    left_margin = min(0.30, 0.12 + 0.0045 * max(len(label) for label in labels))
    bottom_margin = 0.18 if n_evals >= 8 else 0.15
    fig.subplots_adjust(left=left_margin, right=0.955, top=0.90, bottom=bottom_margin, wspace=0.04)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_per_eval_detail(
    eval_name: str,
    verbalizer_metrics: dict[str, float],
    output_path: str,
    chance_baseline: float | None = None,
    eval_intervals: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create a single-eval bar chart with one bar per verbalizer."""
    metric_info = EVAL_METRIC_MAP.get(eval_name, ("score", "Score", True))
    _, display_name, _ = metric_info

    verb_names = sorted(verbalizer_metrics.keys(), key=verbalizer_sort_key)
    values = list(verbalizer_metrics.values())
    n = len(verb_names)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(n, 1)))
    bar_colors = [
        DISPLAY_COLORS.get(shorten_lora_name(v), fallback_colors[i])
        for i, v in enumerate(verb_names)
    ]

    yerr = None
    if eval_intervals:
        errs = []
        for verb_name, val in zip(verb_names, values, strict=True):
            interval = eval_intervals.get(verb_name)
            if not interval:
                errs.append((0.0, 0.0))
                continue
            errs.append((max(0.0, val - float(interval.get("lo", val))), max(0.0, float(interval.get("hi", val)) - val)))
        yerr = np.array(errs, dtype=np.float64).T

    bars = ax.bar(
        range(n), values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        yerr=yerr,
        capsize=3 if yerr is not None else 0,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    if chance_baseline is not None:
        ax.axhline(y=chance_baseline, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Chance ({chance_baseline})")
        ax.legend()

    ax.set_ylabel(display_name)
    ax.set_title(f"{eval_name.replace('_', ' ').title()}")
    ax.set_xticks(range(n))
    ax.set_xticklabels([shorten_lora_name(v) for v in verb_names], fontsize=8, rotation=30, ha="right")
    ax.set_ylim(0, max(max(values) + 0.15 if values else 0.5, 0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results_dir: str,
    output_dir: str | None = None,
    filter_verbalizers: list[str] | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
) -> None:
    """Read all eval summaries and generate comparison report."""
    results_path = Path(results_dir)
    if output_dir is None:
        output_dir = str(results_path / "report")
    os.makedirs(output_dir, exist_ok=True)

    # Load all summaries
    all_summaries_path = results_path / "all_summaries.json"
    if all_summaries_path.exists():
        all_summaries = json.loads(all_summaries_path.read_text())
    else:
        # Try loading individual summary files
        all_summaries = {}
        for f in results_path.glob("*_summary.json"):
            eval_name = f.stem.replace("_summary", "")
            all_summaries[eval_name] = json.loads(f.read_text())

    if not all_summaries:
        print(f"No eval summaries found in {results_dir}")
        return

    print(f"Found {len(all_summaries)} eval results: {list(all_summaries.keys())}")

    # Extract primary metrics per eval per verbalizer
    eval_results: dict[str, dict[str, float]] = {}
    for eval_name, summary in all_summaries.items():
        metrics = extract_verbalizer_metric(summary, eval_name)
        if metrics:
            # Filter to requested verbalizers
            if filter_verbalizers:
                metrics = {
                    k: v for k, v in metrics.items()
                    if any(f in k for f in filter_verbalizers)
                }
            if metrics:
                eval_results[eval_name] = metrics

    eval_intervals = compute_eval_intervals(results_path, eval_results, n_bootstrap=bootstrap_reps)

    # 1. Overall comparison chart
    comparison_path = os.path.join(output_dir, "comparison.png")
    plot_comparison_bar_chart(eval_results, comparison_path, "AObench Results Comparison", eval_intervals=eval_intervals)
    print(f"Saved comparison chart: {comparison_path}")

    # 1b. Aggregate chance-adjusted ranking
    aggregate_scores = compute_aggregate_scores(eval_results, eval_intervals=eval_intervals)
    aggregate_path = os.path.join(output_dir, "aggregate_scores.png")
    plot_aggregate_scores(aggregate_scores, aggregate_path)
    print(f"Saved aggregate chart: {aggregate_path}")

    # 2. Per-eval detail charts
    for eval_name, verb_metrics in eval_results.items():
        detail_path = os.path.join(output_dir, f"detail_{eval_name}.png")
        chance = CHANCE_BASELINES.get(eval_name)
        plot_per_eval_detail(
            eval_name,
            verb_metrics,
            detail_path,
            chance,
            eval_intervals=eval_intervals.get(eval_name),
        )
        print(f"Saved detail chart: {detail_path}")

    # 3. Summary table (text)
    table_path = os.path.join(output_dir, "summary.txt")
    with open(table_path, "w") as f:
        f.write("AObench Results Summary\n")
        f.write("=" * 80 + "\n\n")

        for eval_name, verb_metrics in sorted(eval_results.items()):
            metric_info = EVAL_METRIC_MAP.get(eval_name, ("score", "Score", True))
            _, display_name, _ = metric_info
            f.write(f"--- {eval_name} ({display_name}) ---\n")
            for verb, val in sorted(verb_metrics.items()):
                interval = eval_intervals.get(eval_name, {}).get(verb)
                if interval:
                    f.write(
                        f"  {shorten_lora_name(verb):45s}  {val:.4f}  "
                        f"[{interval['lo']:.4f}, {interval['hi']:.4f}]\n"
                    )
                else:
                    f.write(f"  {shorten_lora_name(verb):45s}  {val:.4f}\n")
            chance = CHANCE_BASELINES.get(eval_name)
            if chance is not None:
                f.write(f"  {'(chance)':45s}  {chance:.4f}\n")
            f.write("\n")

        if aggregate_scores:
            f.write("--- aggregate (chance-adjusted where applicable) ---\n")
            for verb, payload in sorted(
                aggregate_scores.items(),
                key=lambda item: item[1]["mean_normalized_score"],
                reverse=True,
            ):
                f.write(
                    f"  {shorten_lora_name(verb):45s}  "
                    f"{payload['mean_normalized_score']:.4f}"
                    + (
                        f"  [{payload['ci_lo']:.4f}, {payload['ci_hi']:.4f}]"
                        if "ci_lo" in payload and "ci_hi" in payload else ""
                    )
                    + "  "
                    f"(n={payload['num_evals']})\n"
                )
            f.write("\n")

    print(f"Saved summary: {table_path}")

    # 4. Summary JSON for programmatic use
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    intervals_json_path = os.path.join(output_dir, "eval_intervals.json")
    with open(intervals_json_path, "w") as f:
        json.dump(
            {
                eval_name: {
                    verb: {
                        "mean": payload["mean"],
                        "lo": payload["lo"],
                        "hi": payload["hi"],
                        "source": payload.get("source", "unknown"),
                    }
                    for verb, payload in per_eval.items()
                }
                for eval_name, per_eval in eval_intervals.items()
            },
            f,
            indent=2,
        )

    aggregate_json_path = os.path.join(output_dir, "aggregate_scores.json")
    with open(aggregate_json_path, "w") as f:
        json.dump(aggregate_scores, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AObench comparison report")
    parser.add_argument("results_dir", help="Directory containing eval results (all_summaries.json or *_summary.json)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for report")
    parser.add_argument("--filter", nargs="*", default=None, help="Only include verbalizers containing these substrings")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output, filter_verbalizers=args.filter)


if __name__ == "__main__":
    main()
