#!/usr/bin/env python3
"""Evaluate original AO and trained oracle on hallucination & vagueness tasks.

Runs both models single-layer, scores with LLM judge, saves results + plot.
Also breaks down scores by granularity tier.

Usage:
    CUDA_VISIBLE_DEVICES=2 OPENROUTER_API_KEY=... python scripts/eval_hallucination_vagueness.py
"""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "ao_reference"))

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.ao import AO_CHECKPOINTS, choose_attn_implementation
from eval_loop import run_eval, _eval_cache


def load_model(model_name, ao_checkpoint, device):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device, dtype=dtype,
        attn_implementation=choose_attn_implementation(model_name),
    )
    model = PeftModel.from_pretrained(base_model, ao_checkpoint, is_trainable=False)
    model.eval()
    return model, tokenizer


def eval_model(model, tokenizer, task_names, layers, max_items, position_mode, device):
    """Run eval for given tasks/layers, return metrics dict."""
    _eval_cache.clear()
    metrics, traces = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=task_names,
        max_items=max_items,
        eval_batch_size=4,
        device=device,
        layers=layers,
        injection_layer=1,
        oracle_adapter_name="default",
        skip_rot13=True,
        position_mode=position_mode,
    )
    return metrics, traces


def compute_tier_scores(traces: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """From traces, compute per-tier mean scores. Returns {task: {tier: {mean, n}}}."""
    tier_scores = {}
    for task_name, task_traces in traces.items():
        tier_buckets = defaultdict(list)
        for t in task_traces:
            tier = t.get("tier", "unknown")
            score = t.get("judge_score")
            if score is not None:
                tier_buckets[tier].append(float(score))
        tier_scores[task_name] = {
            tier: {"mean": sum(vals) / len(vals), "n": len(vals)}
            for tier, vals in sorted(tier_buckets.items())
        }
    return tier_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--trained-checkpoint", default="ceselder/cot-oracle-v15-stochastic")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="data/hallucination_vagueness_eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = ["hallucination", "vagueness"]
    layers = args.layers

    results = {}
    all_tier_scores = {}  # key -> {task: {tier: {mean, n}}}

    # ── Original AO (one layer at a time) ──
    ao_checkpoint = AO_CHECKPOINTS[args.model]
    model, tokenizer = load_model(args.model, ao_checkpoint, args.device)

    for layer in layers:
        key = f"original_ao_L{layer}"
        print(f"\n{'='*60}")
        print(f"  Original AO — layer {layer}")
        print(f"{'='*60}")

        metrics, traces = eval_model(
            model, tokenizer, task_names, layers=[layer],
            max_items=args.max_items, position_mode="last_5", device=args.device,
        )
        results[key] = {
            "checkpoint": ao_checkpoint,
            "layer": layer,
            "position_mode": "last_5",
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        all_tier_scores[key] = compute_tier_scores(traces)

        for task_name, task_traces in traces.items():
            trace_path = output_dir / f"traces_original_ao_{task_name}_L{layer}.jsonl"
            with open(trace_path, "w") as f:
                for t in task_traces:
                    f.write(json.dumps(t, default=str) + "\n")
            print(f"  Saved {len(task_traces)} traces -> {trace_path}")

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ── Trained Oracle (one layer at a time) ──
    model, tokenizer = load_model(args.model, args.trained_checkpoint, args.device)

    for layer in layers:
        key = f"trained_oracle_L{layer}"
        print(f"\n{'='*60}")
        print(f"  Trained Oracle ({args.trained_checkpoint}) — layer {layer}")
        print(f"{'='*60}")

        metrics, traces = eval_model(
            model, tokenizer, task_names, layers=[layer],
            max_items=args.max_items, position_mode="last_5", device=args.device,
        )
        results[key] = {
            "checkpoint": args.trained_checkpoint,
            "layer": layer,
            "position_mode": "last_5",
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        all_tier_scores[key] = compute_tier_scores(traces)

        for task_name, task_traces in traces.items():
            trace_path = output_dir / f"traces_trained_{task_name}_L{layer}.jsonl"
            with open(trace_path, "w") as f:
                for t in task_traces:
                    f.write(json.dumps(t, default=str) + "\n")
            print(f"  Saved {len(task_traces)} traces -> {trace_path}")

    del model
    torch.cuda.empty_cache()

    # ── Save results JSON (with tier scores) ──
    for key in results:
        results[key]["tier_scores"] = all_tier_scores.get(key, {})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Print tier breakdown ──
    print(f"\n{'='*60}")
    print("  TIER BREAKDOWN")
    print(f"{'='*60}")
    for key in sorted(results.keys()):
        tier_data = results[key].get("tier_scores", {})
        if not tier_data:
            continue
        print(f"\n  {key}:")
        for task, tiers in sorted(tier_data.items()):
            print(f"    {task}:")
            for tier, stats in sorted(tiers.items()):
                print(f"      {tier}: {stats['mean']:.3f} (n={stats['n']})")

    # ── Plots ──
    make_plot(results, output_dir, layers)
    make_tier_plot(results, output_dir, layers)


def make_plot(results, output_dir, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = ["hallucination", "vagueness"]
    model_prefixes = ["original_ao", "trained_oracle"]
    model_labels = ["Original AO", "Trained Oracle (v15)"]
    colors = ["#4878CF", "#E07B54"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    x = np.arange(len(layers))
    width = 0.35

    for ax, task in zip(axes, tasks):
        for i, (prefix, label, color) in enumerate(zip(model_prefixes, model_labels, colors)):
            vals = []
            for layer in layers:
                key = f"{prefix}_L{layer}"
                val = results[key]["metrics"].get(f"eval/{task}", 0.0)
                vals.append(val)
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(task.capitalize(), fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("LLM Judge Score", fontsize=12)
    axes[0].legend(fontsize=10)
    fig.suptitle("Hallucination & Vagueness Eval (single-layer, last 5 positions)", fontsize=14, y=1.02)

    plt.tight_layout()
    plot_path = output_dir / "hallucination_vagueness_eval.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()


def make_tier_plot(results, output_dir, layers):
    """Per-tier breakdown: one subplot per (task, tier), bars = model x layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model_prefixes = ["original_ao", "trained_oracle"]
    model_labels = ["Original AO", "Trained Oracle (v15)"]
    colors = ["#4878CF", "#E07B54"]

    # Collect all tiers per task
    task_tiers = {}
    for key, data in results.items():
        for task, tiers in data.get("tier_scores", {}).items():
            if task not in task_tiers:
                task_tiers[task] = set()
            task_tiers[task].update(tiers.keys())

    if not task_tiers:
        return

    tasks = sorted(task_tiers.keys())
    max_tiers = max(len(v) for v in task_tiers.values())

    fig, axes = plt.subplots(len(tasks), max_tiers, figsize=(4 * max_tiers, 4.5 * len(tasks)), squeeze=False)

    x = np.arange(len(layers))
    width = 0.35

    for row, task in enumerate(tasks):
        tiers = sorted(task_tiers[task])
        for col, tier in enumerate(tiers):
            ax = axes[row][col]
            for i, (prefix, label, color) in enumerate(zip(model_prefixes, model_labels, colors)):
                vals = []
                for layer in layers:
                    key = f"{prefix}_L{layer}"
                    tier_data = results[key].get("tier_scores", {}).get(task, {}).get(tier, {})
                    vals.append(tier_data.get("mean", 0.0))
                bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color, alpha=0.85)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

            ax.set_title(f"{task} / {tier}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", alpha=0.3)

            if col == 0:
                ax.set_ylabel("LLM Judge Score", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for col in range(len(tiers), max_tiers):
            axes[row][col].set_visible(False)

    fig.suptitle("Per-Tier Breakdown (single-layer, last 5 positions)", fontsize=14, y=1.01)
    plt.tight_layout()
    plot_path = output_dir / "tier_breakdown.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Tier plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
