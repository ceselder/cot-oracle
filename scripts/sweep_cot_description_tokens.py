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
from eval_loop import (
    TASKS,
    _eval_cache,
    _extract_base_positions,
    _resample_eval_positions,
    _materialize_activations,
    _batched_oracle_generate,
    run_eval,
)
from qa_judge import (
    QA_GEMINI_SCORE_MODEL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    extract_judge_json,
)

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

TASK = "cot_description"
IF_REPO = "mats-10-sprint-cs-jb/cot-oracle-eval-instruction-following"
DEFAULT_N_VALUES = [1, 2, 3, 5, 8, 13, 20, 35, 60, 100]

IF_JUDGE_SYSTEM = (
    "You evaluate whether an oracle's response follows a specific formatting constraint. "
    "Return ONLY JSON with keys `compliance` and `reason`.\n\n"
    "`compliance` (float 0-1): 1.0 = fully satisfies every aspect of the constraint, "
    "0.5 = partially satisfies it (e.g. right structure but wrong count), "
    "0.0 = completely ignores or violates it.\n\n"
    "`reason`: one sentence, at most 15 words."
)


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
    n_total = len(traces)
    return {
        "correctness": float(np.mean(correctness)) if correctness else float("nan"),
        "specificity": float(np.mean(specificity)) if specificity else float("nan"),
        "confidence": float(np.mean(confidence)) if confidence else float("nan"),
        "retention": len(correctness) / n_total if n_total else float("nan"),
        "n": len(correctness),
        "_correctness_scores": correctness,
        "_specificity_scores": specificity,
        "_confidence_scores": confidence,
    }


def _load_if_data(tokenizer, layers: list[int], max_items: int, seed: int) -> list[dict]:
    """Load instruction-following dataset from HF and tokenize."""
    import datasets as hf_datasets
    try:
        ds = hf_datasets.load_dataset(IF_REPO, split="test")
    except Exception:
        ds = hf_datasets.load_dataset(IF_REPO, split="train")
    items = list(ds)
    random.seed(seed)
    random.shuffle(items)
    items = items[:max_items]
    # context_input_ids already precomputed; prepare_context_ids is a no-op for them
    prepare_context_ids(items, tokenizer, layers=layers)
    return items


def _judge_if_compliance(items: list[dict], predictions: list[str]) -> list[float]:
    """Call LLM judge to score constraint compliance for each prediction."""
    import httpx, re
    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    scores: list[float] = []
    with httpx.Client(timeout=90.0) as client:
        for item, pred in zip(items, predictions):
            body = {
                "model": QA_GEMINI_SCORE_MODEL,
                "messages": [
                    {"role": "system", "content": IF_JUDGE_SYSTEM},
                    {"role": "user", "content": f"Constraint: {item['constraint']}\n\nOracle response:\n{pred}\n\nDoes the oracle response follow the constraint?"},
                ],
                "temperature": 0.0,
                "max_tokens": 80,
            }
            resp = client.post(OPENROUTER_CHAT_COMPLETIONS_URL, json=body, headers=headers)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            try:
                parsed = extract_judge_json(raw)
                scores.append(float(parsed["compliance"]))
            except Exception as e:
                print(f"  [IF judge] WARNING: failed to parse response ({e}): {raw[:200]!r}")
                scores.append(float("nan"))
    return scores


def _run_if_single(
    model, tokenizer, args, position_mode: str, if_data: list[dict]
) -> dict:
    """Run oracle on IF dataset with given position_mode and judge constraint compliance."""
    import copy
    items = copy.deepcopy(if_data)
    _resample_eval_positions(
        items, "instruction_following", args.layers, position_mode,
        stochastic_max_k=100, eval_position_seed=args.seed,
    )
    # Materialize activations in batches
    all_activations = []
    for i in range(0, len(items), args.activation_extract_batch_size):
        batch = items[i : i + args.activation_extract_batch_size]
        acts = _materialize_activations(model, tokenizer, batch, args.layers, args.device)
        all_activations.extend(acts)
    # Generate oracle responses
    gen_items = [(acts, item["prompt"]) for acts, item in zip(all_activations, items)]
    task_def = TASKS[TASK]
    predictions = _batched_oracle_generate(
        model, tokenizer, gen_items, args.layers, args.device,
        injection_layer=1, max_new_tokens=task_def.max_new_tokens,
        eval_batch_size=args.eval_batch_size,
    )
    print(f"    [IF] Judging {len(predictions)} responses...")
    scores = _judge_if_compliance(items, predictions)
    valid = [s for s in scores if not np.isnan(s)]
    return {
        "compliance": float(np.mean(valid)) if valid else float("nan"),
        "n": len(valid),
        "_compliance_scores": scores,
        "_if_predictions": predictions,
        "_if_constraints": [it["constraint"] for it in items],
        "_if_constraint_types": [it["constraint_type"] for it in items],
    }


def _plot(rows: list[dict], output_path: Path, if_rows: list[dict] | None = None) -> None:
    rows = sorted(rows, key=lambda r: r["mean_positions"])
    xs = [r["mean_positions"] for r in rows]

    n_panels = 2 if if_rows else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(xs, [r["correctness"] for r in rows], marker="o", color="#1f77b4", label="Correctness", linewidth=2, markersize=6)
    ax.plot(xs, [r["specificity"] for r in rows], marker="s", color="#ff7f0e", label="Specificity", linewidth=2, markersize=6)
    ax.plot(xs, [r["confidence"] for r in rows], marker="^", color="#2ca02c", label="Confidence", linewidth=2, markersize=6)
    ax.plot(xs, [r["retention"] for r in rows], marker="D", color="#9467bd", label="Retention", linewidth=2, markersize=6, linestyle="--")
    ax.set_xlabel("Mean CoT tokens fed to oracle", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("cot_description: LLM judge scores vs tokens fed", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    for row in rows:
        label = row["position_mode"].replace("last_", "")
        ax.annotate(label, xy=(row["mean_positions"], row["correctness"]), xytext=(3, 4), textcoords="offset points", fontsize=7, color="gray")

    if if_rows:
        if_rows = sorted(if_rows, key=lambda r: r["mean_positions"])
        if_xs = [r["mean_positions"] for r in if_rows]
        ax2 = axes[1]
        ax2.plot(if_xs, [r["compliance"] for r in if_rows], marker="o", color="#d62728", label="Compliance", linewidth=2, markersize=6)
        ax2.set_xlabel("Mean CoT tokens fed to oracle", fontsize=12)
        ax2.set_ylabel("Compliance (0–1)", fontsize=12)
        ax2.set_title("instruction following: constraint compliance vs tokens fed", fontsize=13)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        for row in if_rows:
            label = row["position_mode"].replace("last_", "")
            ax2.annotate(label, xy=(row["mean_positions"], row["compliance"]), xytext=(3, 4), textcoords="offset points", fontsize=7, color="gray")

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

    # Load instruction-following dataset (same CoT rollouts, different prompts)
    try:
        if_data = _load_if_data(tokenizer, args.layers, args.max_items, args.seed)
        print(f"Loaded {len(if_data)} IF items")
    except Exception as e:
        print(f"  WARNING: Could not load IF dataset ({e}), skipping IF sweep")
        if_data = []

    mean_positions_by_n = _compute_mean_positions_per_n(probe_data, n_layers, args.n_values)
    print("Mean positions per mode:")
    for n_val, mean_pos in sorted(mean_positions_by_n.items(), key=lambda x: (x[0] or 1e9)):
        mode_label = f"last_{n_val}" if n_val is not None else "all"
        print(f"  {mode_label}: {mean_pos:.1f}")

    modes = [(f"last_{n}", n) for n in args.n_values] + [("all", None)]

    rows: list[dict] = []
    if_rows: list[dict] = []
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
        print(f"  correctness={result['correctness']:.3f}  specificity={result['specificity']:.3f}  confidence={result['confidence']:.3f}  retention={result['retention']:.3f}  n={result['n']}  elapsed={elapsed:.1f}s")

        if if_data:
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.time()
            if_result = _run_if_single(model, tokenizer, args, position_mode, if_data)
            if_elapsed = time.time() - t1
            if_row = {
                "position_mode": position_mode,
                "n_val": n_val,
                "mean_positions": mean_positions,
                **{k: v for k, v in if_result.items() if not k.startswith("_")},
                "elapsed_s": if_elapsed,
            }
            if_rows.append({**if_row, **{k: v for k, v in if_result.items() if k.startswith("_")}})
            print(f"  [IF] compliance={if_result['compliance']:.3f}  n={if_result['n']}  elapsed={if_elapsed:.1f}s")

        # Save partial results
        with open(run_dir / "rows.jsonl", "w") as f:
            for r in rows:
                r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(r_clean) + "\n")
        if if_rows:
            with open(run_dir / "if_rows.jsonl", "w") as f:
                for r in if_rows:
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
        "if_rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in if_rows],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(rows, run_dir / "cot_description_token_sweep.png", if_rows=if_rows or None)
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
