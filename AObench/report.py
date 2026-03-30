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
    "cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO": "Ours (SFT)",
    "qwen3-8b-final-sprint-checkpoint-no-DPO": "Ours (SFT)",
    "cot-oracle-grpo-v1": "Ours (GRPO)",
    "checkpoints_Qwen3-8B_full_mix_synthetic_qa_v3_replace_lqa": "Adam's synth-QA-v3",
    "Qwen3-8B_full_mix_synthetic_qa_v3_replace_lqa": "Adam's synth-QA-v3",
}

DISPLAY_COLORS: dict[str, str] = {
    "Adam's AO": "#4CAF50",
    "Adam's on-policy": "#8BC34A",
    "Adam's synth-QA-v3": "#FF9800",
    "Ours (SFT)": "#2196F3",
    "Ours (GRPO)": "#E91E63",
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
    verb_names = sorted(all_verbs)

    eval_names = list(eval_results.keys())
    n_evals = len(eval_names)
    n_verbs = len(verb_names)

    if n_evals == 0 or n_verbs == 0:
        return output_path

    fig, ax = plt.subplots(figsize=(max(10, n_evals * 1.5), 6))

    x = np.arange(n_evals)
    bar_width = 0.8 / max(n_verbs, 1)
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(n_verbs, 1)))

    # Normalize 1-5 scale metrics to 0-1 for comparison chart
    SCALE_1_5 = {"backtracking", "system_prompt_qa_hidden", "system_prompt_qa_latentqa"}

    for i, verb_name in enumerate(verb_names):
        display = shorten_lora_name(verb_name)
        color = DISPLAY_COLORS.get(display, fallback_colors[i])
        values = []
        for eval_name in eval_names:
            val = eval_results[eval_name].get(verb_name, 0)
            if eval_name in SCALE_1_5:
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

    verb_names = list(verbalizer_metrics.keys())
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

    print(f"Saved summary: {table_path}")

    # 4. Summary JSON for programmatic use
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AObench comparison report")
    parser.add_argument("results_dir", help="Directory containing eval results (all_summaries.json or *_summary.json)")
    parser.add_argument("--output", "-o", default=None, help="Output directory for report")
    parser.add_argument("--filter", nargs="*", default=None, help="Only include verbalizers containing these substrings")
    args = parser.parse_args()

    generate_report(args.results_dir, args.output, filter_verbalizers=args.filter)
