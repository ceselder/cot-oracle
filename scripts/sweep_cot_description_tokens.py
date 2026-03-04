#!/usr/bin/env python3
"""Sweep cot_description metrics vs number of tokens (positions) fed to the oracle."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
AO_REF_DIR = ROOT / "ao_reference"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(AO_REF_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.ao import choose_attn_implementation
from data_loading import load_task_data, prepare_context_ids
from eval_loop import TASKS, _eval_cache, _extract_base_positions, run_eval

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

TASK = "cot_description"
DEFAULT_N_VALUES = [1, 2, 3, 5, 8, 13, 20, 35, 60, 100]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep cot_description metrics vs number of positions fed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--activation-extract-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_logs/cot_description_token_sweep")
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    return parser.parse_args()


def _load_model_and_tokenizer(model_name: str, checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=choose_attn_implementation(model_name),
    )
    model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
    model.eval()
    return model, tokenizer


def _compute_mean_positions_per_n(test_data: list[dict], n_layers: int, n_values: list[int]) -> dict[int | None, float]:
    """Compute mean number of positions per item for last_N modes and 'all'."""
    base_position_counts = []
    for item in test_data:
        ctx_pos = item.get("context_positions", [])
        base_pos = _extract_base_positions(ctx_pos, n_layers)
        base_position_counts.append(len(base_pos))

    results = {}
    for n in n_values:
        results[n] = float(np.mean([min(n, count) for count in base_position_counts]))
    results[None] = float(np.mean(base_position_counts))  # "all" mode
    return results


def _run_single(model, tokenizer, args, position_mode: str) -> dict:
    """Run eval for cot_description with a given position_mode. Returns per-example metrics."""
    metrics, all_traces = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=[TASK],
        max_items=args.max_items,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        layers=args.layers,
        injection_layer=1,
        oracle_adapter_name="default",
        skip_rot13=True,
        activation_extract_batch_size=args.activation_extract_batch_size,
        no_activations=False,
        position_mode=position_mode,
        stochastic_max_k=100,
        eval_position_seed=args.seed,
    )
    traces = all_traces[TASK]
    correctness = [t["judge_correctness"] for t in traces if "judge_correctness" in t]
    specificity = [t["judge_specificity"] for t in traces if "judge_specificity" in t]
    confidence = [t["judge_confidence"] for t in traces if "judge_confidence" in t]
    return {
        "correctness": float(np.mean(correctness)) if correctness else float("nan"),
        "specificity": float(np.mean(specificity)) if specificity else float("nan"),
        "confidence": float(np.mean(confidence)) if confidence else float("nan"),
        "n": len(correctness),
        "_correctness_scores": correctness,
        "_specificity_scores": specificity,
        "_confidence_scores": confidence,
    }


def _plot(rows: list[dict], output_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["mean_positions"])
    xs = [r["mean_positions"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(xs, [r["correctness"] for r in rows], marker="o", color="#1f77b4", label="Correctness", linewidth=2, markersize=6)
    ax.plot(xs, [r["specificity"] for r in rows], marker="s", color="#ff7f0e", label="Specificity", linewidth=2, markersize=6)
    ax.plot(xs, [r["confidence"] for r in rows], marker="^", color="#2ca02c", label="Confidence", linewidth=2, markersize=6)
    ax.set_xlabel("Mean CoT tokens fed to oracle", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("cot_description: LLM judge scores vs tokens fed", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    for row in rows:
        label = row["position_mode"].replace("last_", "")
        ax.annotate(label, xy=(row["mean_positions"], row["correctness"]), xytext=(3, 4), textcoords="offset points", fontsize=7, color="gray")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {output_path}")


def main() -> None:
    args = _parse_args()
    checkpoint = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint)
    n_layers = len(args.layers)

    # Load and tokenize test data once to compute mean_positions analytically
    random.seed(args.seed)
    try:
        probe_data = load_task_data(TASK, split="test", n=args.max_items, shuffle=False)
    except Exception:
        probe_data = []
    if not probe_data:
        probe_data = load_task_data(TASK, split="train", n=args.max_items, shuffle=False)
    prepare_context_ids(probe_data, tokenizer, layers=args.layers)
    print(f"Loaded {len(probe_data)} test items for position counting")

    mean_positions_by_n = _compute_mean_positions_per_n(probe_data, n_layers, args.n_values)
    print("Mean positions per mode:")
    for n_val, mean_pos in sorted(mean_positions_by_n.items(), key=lambda x: (x[0] or 1e9)):
        mode_label = f"last_{n_val}" if n_val is not None else "all"
        print(f"  {mode_label}: {mean_pos:.1f}")

    modes = [(f"last_{n}", n) for n in args.n_values] + [("all", None)]

    rows = []
    for position_mode, n_val in tqdm(modes, desc="position modes"):
        mean_positions = mean_positions_by_n[n_val]
        _eval_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[sweep] position_mode={position_mode}  mean_positions={mean_positions:.1f}")
        t0 = time.time()
        result = _run_single(model, tokenizer, args, position_mode)
        elapsed = time.time() - t0
        row = {
            "position_mode": position_mode,
            "n_val": n_val,
            "mean_positions": mean_positions,
            **{k: v for k, v in result.items() if not k.startswith("_")},
            "elapsed_s": elapsed,
        }
        rows.append({**row, **{k: v for k, v in result.items() if k.startswith("_")}})
        print(f"  correctness={result['correctness']:.3f}  specificity={result['specificity']:.3f}  confidence={result['confidence']:.3f}  n={result['n']}  elapsed={elapsed:.1f}s")

        # Save partial results
        with open(run_dir / "rows.jsonl", "w") as f:
            for r in rows:
                r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(r_clean) + "\n")

    summary = {
        "checkpoint": checkpoint,
        "model": args.model,
        "task": TASK,
        "layers": args.layers,
        "max_items": args.max_items,
        "n_values": args.n_values,
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(rows, run_dir / "cot_description_token_sweep.png")
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
