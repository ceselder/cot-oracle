#!/usr/bin/env python3
"""Sweep cot_description metrics vs number of tokens (positions) fed to the oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from eval_loop import (
    TASKS,
    _eval_cache,
    _extract_base_positions,
    run_eval,
)

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

TASK = "cot_description"
DEFAULT_N_VALUES = [1, 2, 3, 5, 8, 13, 20, 35, 60, 100]


def _cache_key(checkpoint: str, position_mode: str, max_items: int, layers: list[int], seed: int) -> str:
    raw = f"{checkpoint}|{position_mode}|{max_items}|{sorted(layers)}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_load(cache_dir: Path, key: str) -> tuple[dict | None, list | None]:
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None, None
    data = json.loads(p.read_text())
    tp = cache_dir / f"{key}_traces.jsonl"
    traces = [json.loads(line) for line in tp.read_text().splitlines() if line.strip()] if tp.exists() else None
    return data, traces


def _cache_save(cache_dir: Path, key: str, data: dict, traces: list | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(data))
    if traces is not None:
        with open(cache_dir / f"{key}_traces.jsonl", "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep cot_description metrics vs number of positions fed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--activation-extract-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_logs/cot_description_token_sweep")
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    parser.add_argument("--baseline-checkpoint", default=None, help="First extra checkpoint to overlay")
    parser.add_argument("--baseline-label", default="AO baseline")
    parser.add_argument("--baseline-layers", type=int, nargs="+", default=[18], help="Layer override for --baseline-checkpoint (default: [18])")
    parser.add_argument("--precomputed-baseline-rows", default=None, help="Pre-computed baseline_rows.jsonl (skips GPU run for --baseline-checkpoint)")
    parser.add_argument("--extra-baseline-checkpoint", default=None, help="Second extra checkpoint to overlay")
    parser.add_argument("--extra-baseline-label", default="Extra baseline")
    parser.add_argument("--primary-label", default="Base", help="Label for the primary checkpoint subplot")
    parser.add_argument("--rerun", action="store_true", help="Ignore cache and rerun all modes")
    parser.add_argument("--precomputed-rows", default=None, help="Path to existing rows.jsonl to skip primary sweep")
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


def _run_single(model, tokenizer, args, position_mode: str, layers: list[int] | None = None) -> dict:
    """Run eval for cot_description with a given position_mode. Returns per-example metrics."""
    layers = layers or args.layers
    metrics, all_traces = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=[TASK],
        max_items=args.max_items,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        layers=layers,
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
    return {
        "correctness": float(np.mean(correctness)) if correctness else float("nan"),
        "specificity": float(np.mean(specificity)) if specificity else float("nan"),
        "calibration": float(metrics.get(f"eval/{TASK}_calibration", float("nan"))),
        "overconfidence": float(metrics.get(f"eval/{TASK}_overconfidence", float("nan"))),
        "n": len(correctness),
        "_correctness_scores": correctness,
        "_specificity_scores": specificity,
        "_traces": traces,
    }



def _plot_cot_panel(ax, rows: list[dict], label_suffix: str = "", alpha: float = 1.0, linestyle: str = "-") -> None:
    """Plot cot_description metrics onto ax. label_suffix distinguishes baseline vs oracle."""
    rows = sorted(rows, key=lambda r: r["mean_positions"])
    xs = [r["mean_positions"] for r in rows]

    def _plot_if_valid(ys, marker, color, label):
        if any(not np.isnan(y) for y in ys):
            ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2, markersize=5, linestyle=linestyle, alpha=alpha)

    _plot_if_valid([r["correctness"] for r in rows], "o", "#1f77b4", "Correctness\nmean factual match (0–1)")
    _plot_if_valid([r.get("calibration", float("nan")) for r in rows], "v", "#8c564b", "Calibration\nmean |conf − corr|")
    _plot_if_valid([r.get("overconfidence", float("nan")) for r in rows], "^", "#d62728", "Overconfidence\nfrac(conf > corr + 0.25)")


_BASELINE_STYLES = [
    ("--", 0.75),
    (":", 0.75),
    ("-.", 0.65),
]


def _plot(
    rows: list[dict],
    output_path: Path,
    baselines: list[tuple[list[dict], str]] | None = None,
    primary_label: str = "v15 stochastic",
) -> None:
    """One subplot per model, horizontal, sharex+sharey."""
    baselines = baselines or []
    # Order: first baseline (Adam AO), primary (Base), remaining baselines (DPO, ...)
    if baselines:
        all_panels = [baselines[0], (rows, primary_label)] + baselines[1:]
    else:
        all_panels = [(rows, primary_label)]
    n = len(all_panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True, sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (panel_rows, label) in zip(axes, all_panels):
        _plot_cot_panel(ax, panel_rows, linestyle="-")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Mean CoT positions fed (per layer)", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score", fontsize=11)
    fig.suptitle("cot_description: LLM judge scores vs positions fed (Qwen3-8B)", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved plot to {output_path}")


def _sweep_checkpoint(
    ckpt: str, model, tokenizer, args, modes: list[tuple[str, int | None]],
    mean_positions_by_n: dict, cache_dir: Path, label: str = "sweep",
    layers: list[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run (or load from cache) one full position-mode sweep. Returns (rows, all_traces)."""
    layers = layers or args.layers
    rows: list[dict] = []
    all_traces: list[dict] = []

    for position_mode, n_val in tqdm(modes, desc=f"{label} position modes"):
        mean_positions = mean_positions_by_n[n_val]
        key = _cache_key(ckpt, position_mode, args.max_items, layers, args.seed)
        cached_data, cached_traces = (None, None) if args.rerun else _cache_load(cache_dir, key)
        if cached_data is not None:
            print(f"  [{label}] cache hit: {position_mode}")
            rows.append({**cached_data, "mean_positions": mean_positions})
            if cached_traces:
                all_traces.extend(cached_traces)
            continue

        _eval_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[{label}] position_mode={position_mode}  mean_positions={mean_positions:.1f}")
        t0 = time.time()
        result = _run_single(model, tokenizer, args, position_mode, layers=layers)
        elapsed = time.time() - t0
        row = {
            "position_mode": position_mode,
            "n_val": n_val,
            "mean_positions": mean_positions,
            **{k: v for k, v in result.items() if not k.startswith("_")},
            "elapsed_s": elapsed,
        }
        rows.append({**row, **{k: v for k, v in result.items() if k.startswith("_") and k != "_traces"}})
        print(f"  correctness={result['correctness']:.3f}  specificity={result['specificity']:.3f}  overconfidence={result['overconfidence']:.3f}  n={result['n']}  elapsed={elapsed:.1f}s")

        # Annotate traces with eval metadata and cache them
        raw_traces = result.get("_traces", [])
        annotated = [{**t, "eval_name": TASK, "position_mode": position_mode} for t in raw_traces]
        all_traces.extend(annotated)
        _cache_save(cache_dir, key, {k: v for k, v in row.items() if not k.startswith("_")}, traces=annotated)

    return rows, all_traces


def _load_stratified(tokenizer, layers: list[int], n: int, seed: int) -> list[dict]:
    """Load n//2 answerable + n//2 unanswerable cot_description items."""
    import datasets as hf_datasets
    task_def = TASKS[TASK]
    try:
        ds = hf_datasets.load_dataset(task_def.hf_repo, split="test")
    except Exception:
        ds = hf_datasets.load_dataset(task_def.hf_repo, split="train")
    answerable = [dict(row) for row in ds if str(row.get("answerable", "True")) == "True"]
    unanswerable = [dict(row) for row in ds if str(row.get("answerable", "True")) == "False"]
    rng = random.Random(seed)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    half = n // 2
    items = answerable[:half] + unanswerable[:half]
    rng.shuffle(items)
    print(f"  Stratified: {min(half, len(answerable))} answerable + {min(half, len(unanswerable))} unanswerable")
    prepare_context_ids(items, tokenizer, layers=layers)
    return items


def main() -> None:
    args = _parse_args()
    checkpoint = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(os.environ.get("CACHE_DIR", "/tmp")) / "cot_description_sweep"

    # Load tokenizer (always needed for position counting); only load full model if running primary sweep.
    if args.precomputed_rows:
        from transformers import AutoTokenizer as _AutoTokenizer
        tokenizer = _AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = None
    else:
        model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint)

    # Load stratified test data to compute mean_positions
    probe_data = _load_stratified(tokenizer, args.layers, args.max_items, args.seed)
    print(f"Loaded {len(probe_data)} stratified test items")

    mean_positions_by_n = _compute_mean_positions_per_n(probe_data, len(args.layers), args.n_values)
    print("Mean positions per mode:")
    for n_val, mean_pos in sorted(mean_positions_by_n.items(), key=lambda x: (x[0] or 1e9)):
        print(f"  {'last_' + str(n_val) if n_val is not None else 'all'}: {mean_pos:.1f}")

    modes = [(f"last_{n}", n) for n in args.n_values] + [("all", None)]

    if args.precomputed_rows:
        print(f"Loading precomputed rows from {args.precomputed_rows}")
        with open(args.precomputed_rows) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            r.setdefault("calibration", float("nan"))
            r.setdefault("overconfidence", float("nan"))
            key = _cache_key(checkpoint, r["position_mode"], args.max_items, args.layers, args.seed)
            if _cache_load(cache_dir, key)[0] is None:
                _cache_save(cache_dir, key, {k: v for k, v in r.items() if not k.startswith("_")})
        primary_traces: list[dict] = []
    else:
        rows, primary_traces = _sweep_checkpoint(checkpoint, model, tokenizer, args, modes, mean_positions_by_n, cache_dir, label="sweep")

    with open(run_dir / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    if primary_traces:
        with open(run_dir / "traces_primary.jsonl", "w") as f:
            for t in primary_traces:
                f.write(json.dumps(t) + "\n")
        print(f"  Saved {len(primary_traces)} traces → {run_dir}/traces_primary.jsonl")

    # Baselines: (ckpt, label, precomputed_path_or_None, layers_override_or_None)
    _baseline_specs = []
    if args.baseline_checkpoint:
        _baseline_specs.append((args.baseline_checkpoint, args.baseline_label, args.precomputed_baseline_rows, args.baseline_layers))
    if args.extra_baseline_checkpoint:
        _baseline_specs.append((args.extra_baseline_checkpoint, args.extra_baseline_label, None, None))

    baselines_data: list[tuple[list[dict], str]] = []
    for b_idx, (b_ckpt_raw, b_label, b_precomputed, b_layers) in enumerate(_baseline_specs):
        b_layers = b_layers or args.layers
        suffix = f"b{b_idx}"
        safe_label = b_label.lower().replace(" ", "_").replace("/", "_")

        if b_precomputed:
            print(f"\n[{suffix}] Loading precomputed rows from {b_precomputed}")
            with open(b_precomputed) as f:
                b_rows = [json.loads(line) for line in f if line.strip()]
            for r in b_rows:
                r.setdefault("calibration", float("nan"))
                r.setdefault("overconfidence", float("nan"))
            b_traces: list[dict] = []
        else:
            b_ckpt = str(Path(b_ckpt_raw).resolve()) if Path(b_ckpt_raw).exists() else b_ckpt_raw
            print(f"\n[{suffix}] Loading {b_ckpt} (layers={b_layers}) ...")
            b_model, _ = _load_model_and_tokenizer(args.model, b_ckpt)
            b_rows, b_traces = _sweep_checkpoint(b_ckpt, b_model, tokenizer, args, modes, mean_positions_by_n, cache_dir, label=suffix, layers=b_layers)
            del b_model
            gc.collect()
            torch.cuda.empty_cache()

        with open(run_dir / f"baseline_rows_{b_idx}.jsonl", "w") as f:
            for r in b_rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
        if b_traces:
            traces_path = run_dir / f"traces_{safe_label}.jsonl"
            with open(traces_path, "w") as f:
                for t in b_traces:
                    f.write(json.dumps(t) + "\n")
            print(f"  Saved {len(b_traces)} traces → {traces_path}")
        baselines_data.append((b_rows, b_label))

    summary = {
        "checkpoint": checkpoint,
        "model": args.model,
        "task": TASK,
        "layers": args.layers,
        "max_items": args.max_items,
        "n_values": args.n_values,
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "baselines": [{"checkpoint": spec[0], "label": spec[1], "layers": spec[3] or args.layers} for spec in _baseline_specs],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(rows, run_dir / "cot_description_token_sweep.png", baselines=baselines_data or None, primary_label=args.primary_label)
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
