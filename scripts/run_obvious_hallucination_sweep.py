#!/usr/bin/env python3
"""
Run an obvious-hallucination position sweep on the paper checkpoint collection.

This mirrors the domain-confusion sweep but tracks the obvious-hallucination
failure mode directly as a function of activation positions.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench.open_ended_eval.hallucination import _run_hallucination
from AObench.utils.common import load_model, load_tokenizer, timestamped_eval_results_dir
from AObench.utils.report import DISPLAY_COLORS, shorten_lora_name, verbalizer_sort_key

MODEL_NAME = "Qwen/Qwen3-8B"
PAPER_COLLECTION_VERBALIZERS = [
    "ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens",
    "ceselder/adam-reupload-qwen3-8b-full-mix-synthetic-qa-v3-replace-lqa",
    "ceselder/cot-oracle-paper-ablation-adam-recipe-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-3layers",
    "ceselder/cot-oracle-paper-ablation-ours-3layers-onpolicy-lens-only",
    "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO",
    "ceselder/cot-oracle-grpo-step-500",
]
DEFAULT_POSITIONS = [1, 3, 5, 10, 50, 100]
PRIMARY_METRIC = "obvious_hallucination_rate"


def default_output_dir() -> str:
    return timestamped_eval_results_dir("obvious_hallucination_sweep")


def _metric_by_verbalizer(summary: dict[str, Any], metric_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for verbalizer_name, metrics in summary.get("metrics_by_verbalizer", {}).items():
        value = metrics.get(metric_key)
        if isinstance(value, (int, float)):
            result[verbalizer_name] = float(value)
    return result


def plot_sweep(
    *,
    sweep_metrics: dict[int, dict[str, float]],
    positions: list[int],
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    all_verbalizers: set[str] = set()
    for metrics in sweep_metrics.values():
        all_verbalizers.update(metrics.keys())
    verbalizers = sorted(all_verbalizers, key=verbalizer_sort_key)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(len(verbalizers), 1)))

    for i, verbalizer in enumerate(verbalizers):
        xs = []
        ys = []
        for n_positions in positions:
            value = sweep_metrics.get(n_positions, {}).get(verbalizer)
            if value is None:
                continue
            xs.append(n_positions)
            ys.append(value)

        if not xs:
            continue

        display = shorten_lora_name(verbalizer)
        color = DISPLAY_COLORS.get(display, fallback_colors[i])
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=6, label=display, color=color)

    ax.set_xscale("log")
    ax.set_xticks(positions)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(min(positions), max(positions))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Activation Positions")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an obvious-hallucination position sweep")
    parser.add_argument(
        "--verbalizer-lora",
        type=str,
        action="append",
        default=None,
        help="Override the default paper collection checkpoint list. Can be repeated.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="*",
        default=None,
        help="Activation-position counts to evaluate. Default: 1 3 5 10 50 100",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Backbone model used for the AO verbalizers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for summaries and plots.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap on number of hallucination entries.",
    )
    args = parser.parse_args()

    verbalizer_lora_paths = args.verbalizer_lora or list(PAPER_COLLECTION_VERBALIZERS)
    positions = list(args.positions or DEFAULT_POSITIONS)
    output_dir = args.output_dir or default_output_dir()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir, "run_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "positions": positions,
                "metric": PRIMARY_METRIC,
                "verbalizer_lora_paths": verbalizer_lora_paths,
                "max_entries": args.max_entries,
            },
            indent=2,
        )
    )

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    print(f"Loading model: {args.model_name} on {device} with dtype={dtype}")
    model = load_model(args.model_name, dtype)
    model.eval()

    sweep_payload: dict[str, Any] = {
        "metric": PRIMARY_METRIC,
        "positions": positions,
        "verbalizer_lora_paths": verbalizer_lora_paths,
        "summaries_by_position": {},
        "primary_metric_by_position": {},
    }

    for n_positions in positions:
        print(f"\n{'=' * 70}")
        print(f"RUNNING OBVIOUS HALLUCINATION SWEEP @ n_positions={n_positions}")
        print(f"{'=' * 70}")

        position_output_dir = os.path.join(output_dir, f"positions_{n_positions}")
        summary = _run_hallucination(
            n_positions=n_positions,
            eval_name=f"hallucination_{n_positions}pos",
            model_name=args.model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_dir=position_output_dir,
            eval_batch_size=32,
            verbalizer_lora_paths=verbalizer_lora_paths,
            max_entries=args.max_entries,
        )

        sweep_payload["summaries_by_position"][str(n_positions)] = summary
        sweep_payload["primary_metric_by_position"][str(n_positions)] = _metric_by_verbalizer(summary, PRIMARY_METRIC)

        with open(os.path.join(output_dir, f"obvious_hallucination_{n_positions}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    sweep_json_path = os.path.join(output_dir, "obvious_hallucination_sweep.json")
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_payload, f, indent=2)

    primary_metrics = {
        int(k): v for k, v in sweep_payload["primary_metric_by_position"].items()
    }
    plot_sweep(
        sweep_metrics=primary_metrics,
        positions=positions,
        output_path=os.path.join(output_dir, "obvious_hallucination_sweep.png"),
        title="Obvious Hallucination vs Activation Positions",
        ylabel="Obvious Hallucination Rate",
    )

    print(f"Sweep summary saved to {sweep_json_path}")
    print(f"Plot saved to {os.path.join(output_dir, 'obvious_hallucination_sweep.png')}")


if __name__ == "__main__":
    main()
