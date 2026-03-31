"""
Generate comparison report: bar charts + summary table for AObench eval results.

Usage:
    python -m AObench.report results_dir/
    python -m AObench.report results_dir/ --output report.html

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


def extract_bootstrap_data(
    summary: dict[str, Any],
    eval_name: str,
) -> list[float] | None:
    """Try to extract per-example scores for bootstrap CI."""
    # Look for scored_results or binary_scored_results in the raw output files
    # These may not be in the summary — would need the detailed output files
    return None


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


def normalize_metric_for_aggregate(eval_name: str, value: float) -> float:
    """Map eval metrics to a roughly comparable scale for aggregate plotting."""
    normalized = float(value)
    if eval_name in SCALE_1_5_EVALS:
        normalized = (normalized - 1.0) / 4.0

    chance = CHANCE_BASELINES.get(eval_name)
    if chance is not None and chance < 1.0:
        normalized = (normalized - chance) / (1.0 - chance)

    return normalized


def compute_aggregate_scores(
    eval_results: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    per_verbalizer: dict[str, dict[str, float]] = {}
    for eval_name, metrics in eval_results.items():
        for verbalizer_name, metric_value in metrics.items():
            per_verbalizer.setdefault(verbalizer_name, {})[eval_name] = normalize_metric_for_aggregate(
                eval_name,
                metric_value,
            )

    return {
        verbalizer_name: {
            "mean_normalized_score": sum(per_eval.values()) / len(per_eval),
            "num_evals": len(per_eval),
            "per_eval_normalized": per_eval,
        }
        for verbalizer_name, per_eval in per_verbalizer.items()
        if per_eval
    }


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

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5.5))
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))
    colors = [DISPLAY_COLORS.get(label, fallback_colors[i]) for i, label in enumerate(labels)]

    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
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
    ax.set_xlabel("Mean normalized score (chance-adjusted where applicable)")
    ax.set_title(title)
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
) -> str:
    """Create a grouped bar chart comparing verbalizers across evals.

    eval_results: {eval_name: {verbalizer_name: metric_value}}
    """
    # Collect all unique verbalizer names
    all_verbs: set[str] = set()
    for metrics in eval_results.values():
        all_verbs.update(metrics.keys())
    verb_names = sorted(all_verbs, key=verbalizer_sort_key)

    eval_names = list(eval_results.keys())
    n_evals = len(eval_names)
    n_verbs = len(verb_names)

    if n_evals == 0 or n_verbs == 0:
        return output_path

    fig, ax = plt.subplots(figsize=(max(10, n_evals * 1.5), 6))

    x = np.arange(n_evals)
    bar_width = 0.8 / max(n_verbs, 1)
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(n_verbs, 1)))

    for i, verb_name in enumerate(verb_names):
        display = shorten_lora_name(verb_name)
        color = DISPLAY_COLORS.get(display, fallback_colors[i])
        values = []
        for eval_name in eval_names:
            val = eval_results[eval_name].get(verb_name, 0)
            if eval_name in SCALE_1_5_EVALS:
                val = (val - 1) / 4  # map 1-5 → 0-1
            values.append(val)
        offset = (i - n_verbs / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width * 0.9,
            label=display,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=7, rotation=45,
                )

    # Add chance baselines
    for j, eval_name in enumerate(eval_names):
        if eval_name in CHANCE_BASELINES:
            chance = CHANCE_BASELINES[eval_name]
            ax.plot(
                [j - 0.4, j + 0.4], [chance, chance],
                color="red", linestyle="--", linewidth=1, alpha=0.7,
            )

    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in eval_names], fontsize=8)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
        title="Verbalizer LoRA",
    )
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_per_eval_detail(
    eval_name: str,
    verbalizer_metrics: dict[str, float],
    output_path: str,
    chance_baseline: float | None = None,
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

    bars = ax.bar(
        range(n), values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
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

    # 1. Overall comparison chart
    comparison_path = os.path.join(output_dir, "comparison.png")
    plot_comparison_bar_chart(eval_results, comparison_path, "AObench Results Comparison")
    print(f"Saved comparison chart: {comparison_path}")

    # 1b. Aggregate chance-adjusted ranking
    aggregate_scores = compute_aggregate_scores(eval_results)
    aggregate_path = os.path.join(output_dir, "aggregate_scores.png")
    plot_aggregate_scores(aggregate_scores, aggregate_path)
    print(f"Saved aggregate chart: {aggregate_path}")

    # 2. Per-eval detail charts
    for eval_name, verb_metrics in eval_results.items():
        detail_path = os.path.join(output_dir, f"detail_{eval_name}.png")
        chance = CHANCE_BASELINES.get(eval_name)
        plot_per_eval_detail(eval_name, verb_metrics, detail_path, chance)
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
                    f"{payload['mean_normalized_score']:.4f}  "
                    f"(n={payload['num_evals']})\n"
                )
            f.write("\n")

    print(f"Saved summary: {table_path}")

    # 4. Summary JSON for programmatic use
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    aggregate_json_path = os.path.join(output_dir, "aggregate_scores.json")
    with open(aggregate_json_path, "w") as f:
        json.dump(aggregate_scores, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AObench comparison report")
    parser.add_argument("results_dir", help="Directory containing eval results (all_summaries.json or *_summary.json)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for report")
    parser.add_argument("--filter", nargs="*", default=None, help="Only include verbalizers containing these substrings")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output, filter_verbalizers=args.filter)
