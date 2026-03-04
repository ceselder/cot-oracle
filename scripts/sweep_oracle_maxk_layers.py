#!/usr/bin/env python3
"""Sweep a trained activation oracle over max_k and layer subsets, then render per-task 3x3 matshows."""

from __future__ import annotations

import argparse
import gc
import json
import os
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
import torch
import wandb
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import cot_utils
from core.ao import choose_attn_implementation
from eval_loop import TASKS, _eval_cache, _primary_metric_name, run_eval

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

DEFAULT_TASKS = [
    "hint_admission",
    "atypical_answer",
    "reasoning_termination",
    "answer_trajectory",
    "futurelens",
    "backtrack_prediction",
    "sycophancy",
    "truthfulqa_hint_verbalized",
    "truthfulqa_hint",
    "sentence_insertion",
]
DEFAULT_MAX_K_VALUES = [2, 10, 100]
DEFAULT_LAYER_SPECS = [("1L", [18]), ("2L", [9, 27]), ("3L", [9, 18, 27])]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep max_k and layer-count settings for a trained activation oracle")
    parser.add_argument("--checkpoint", default="checkpoints/final")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--activation-extract-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--max-k-values", type=int, nargs="+", default=DEFAULT_MAX_K_VALUES)
    parser.add_argument("--trace-samples-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_logs/maxk_layer_grid")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run", default="maxk-layer-grid-3x3")
    return parser.parse_args()


def _resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    return str(path.resolve()) if path.exists() else checkpoint


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


def _layer_label(layer_label: str, layers: list[int]) -> str:
    return f"{layer_label} [{','.join(str(layer) for layer in layers)}]"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _render_task_heatmap(task_name: str, metric_name: str, matrix: list[list[float]], x_labels: list[str], y_labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    im = ax.matshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{task_name} ({metric_name})", fontsize=11)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("max_k")
    ax.set_ylabel("layers fed")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    heatmap_dir = run_dir / "heatmaps"
    cell_dir_root = run_dir / "cells"
    run_dir.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        assert task_name in TASKS, f"Unknown task: {task_name}"
        assert task_name != "rot13_reconstruction", "rot13_reconstruction needs the dedicated ROT13 adapter and is not supported by this sweep"

    model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        config={
            "model": args.model,
            "checkpoint": checkpoint,
            "tasks": args.tasks,
            "max_k_values": args.max_k_values,
            "layer_specs": [{"label": label, "layers": layers} for label, layers in DEFAULT_LAYER_SPECS],
            "max_items": args.max_items,
            "eval_batch_size": args.eval_batch_size,
            "activation_extract_batch_size": args.activation_extract_batch_size,
            "grid_shape": [len(DEFAULT_LAYER_SPECS), len(args.max_k_values)],
            "world_size": 1,
            "baseline_type": "trained_oracle_maxk_layer_grid",
        },
        tags=["baseline", "trained_oracle", "max_k", "layer_grid", "3x3"],
    )
    wandb.define_metric("grid/cell_index")
    wandb.define_metric("*", step_metric="grid/cell_index")

    summary_rows = []
    trace_rows = []
    matrix_scores = {task_name: {} for task_name in args.tasks}
    metric_names = {}
    cell_index = 0
    outer_pbar = tqdm(DEFAULT_LAYER_SPECS, desc="layer grid")
    for layer_label, layers in outer_pbar:
        inner_pbar = tqdm(args.max_k_values, desc=f"{layer_label} max_k", leave=False)
        for max_k in inner_pbar:
            inner_pbar.set_postfix(max_k=max_k)
            cot_utils.CONFIGURED_LAYERS = list(layers)
            _eval_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
            started = time.time()
            metrics, all_traces = run_eval(
                model=model,
                tokenizer=tokenizer,
                task_names=args.tasks,
                max_items=args.max_items,
                eval_batch_size=args.eval_batch_size,
                device=args.device,
                layers=layers,
                injection_layer=1,
                oracle_adapter_name="default",
                skip_rot13=True,
                activation_extract_batch_size=args.activation_extract_batch_size,
                no_activations=False,
                position_mode="stochastic",
                stochastic_max_k=max_k,
                eval_position_seed=args.seed,
            )
            elapsed = time.time() - started
            cell_name = f"maxk_{max_k}__{layer_label}"
            cell_dir = cell_dir_root / cell_name
            per_task_metrics = {}
            wandb_log = {"grid/cell_index": cell_index, "grid/max_k": max_k, "grid/layer_count": len(layers), "grid/layers": ",".join(str(layer) for layer in layers), "grid/elapsed_s": elapsed}
            for task_name in args.tasks:
                task_score_key = f"eval/{task_name}"
                task_n_key = f"eval_n/{task_name}"
                assert task_score_key in metrics, f"Missing primary metric for {task_name} at {cell_name}"
                assert task_n_key in metrics, f"Missing item count for {task_name} at {cell_name}"
                task_def = TASKS[task_name]
                metric_name = _primary_metric_name(task_name, task_def.scoring)
                metric_names[task_name] = metric_name
                score = float(metrics[task_score_key])
                n_items = int(metrics[task_n_key])
                traces = all_traces[task_name]
                task_record = {
                    "task": task_name,
                    "metric_name": metric_name,
                    "score": score,
                    "n": n_items,
                    "max_k": max_k,
                    "layer_label": layer_label,
                    "layers": layers,
                    "elapsed_s": elapsed,
                }
                summary_rows.append(task_record)
                per_task_metrics[task_name] = task_record
                matrix_scores[task_name][(layer_label, max_k)] = score
                wandb_log[f"grid_score/{task_name}"] = score
                wandb_log[f"grid_n/{task_name}"] = n_items
                traces_with_meta = []
                for trace_idx, trace in enumerate(traces):
                    row = {
                        "task": task_name,
                        "metric_name": metric_name,
                        "max_k": max_k,
                        "layer_label": layer_label,
                        "layers": layers,
                        "trace_index": trace_idx,
                        **trace,
                    }
                    traces_with_meta.append(row)
                    if trace_idx < args.trace_samples_per_cell:
                        trace_rows.append({
                            "task": task_name,
                            "max_k": max_k,
                            "layer_label": layer_label,
                            "layers": ",".join(str(layer) for layer in layers),
                            "question": trace["question"][:240],
                            "expected": trace["expected"][:240],
                            "predicted": trace["predicted"][:240],
                            "correct": str(trace["correct"]),
                        })
                _write_jsonl(cell_dir / "traces" / f"{task_name}.jsonl", traces_with_meta)
            _write_json(cell_dir / "metrics.json", {"cell": cell_name, "max_k": max_k, "layer_label": layer_label, "layers": layers, "elapsed_s": elapsed, "tasks": per_task_metrics})
            wandb.log(wandb_log, step=cell_index)
            cell_index += 1

    _write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "model": args.model,
        "tasks": args.tasks,
        "max_k_values": args.max_k_values,
        "layer_specs": [{"label": label, "layers": layers} for label, layers in DEFAULT_LAYER_SPECS],
        "rows": summary_rows,
    })
    _write_jsonl(run_dir / "summary_rows.jsonl", summary_rows)
    _write_jsonl(run_dir / "trace_samples.jsonl", trace_rows)

    summary_table = wandb.Table(columns=["task", "metric_name", "score", "n", "max_k", "layer_label", "layers", "elapsed_s"])
    for row in summary_rows:
        summary_table.add_data(row["task"], row["metric_name"], row["score"], row["n"], row["max_k"], row["layer_label"], ",".join(str(layer) for layer in row["layers"]), row["elapsed_s"])

    trace_table = wandb.Table(columns=["task", "max_k", "layer_label", "layers", "question", "expected", "predicted", "correct"])
    for row in trace_rows:
        trace_table.add_data(row["task"], row["max_k"], row["layer_label"], row["layers"], row["question"], row["expected"], row["predicted"], row["correct"])

    final_log = {"grid/cell_index": cell_index, "grid_summary": summary_table, "trace_samples": trace_table}
    x_labels = [str(value) for value in args.max_k_values]
    y_labels = [_layer_label(label, layers) for label, layers in DEFAULT_LAYER_SPECS]
    for task_name in args.tasks:
        matrix = [[matrix_scores[task_name][(layer_label, max_k)] for max_k in args.max_k_values] for layer_label, _ in DEFAULT_LAYER_SPECS]
        heatmap_path = heatmap_dir / f"{task_name}.png"
        _render_task_heatmap(task_name, metric_names[task_name], matrix, x_labels, y_labels, heatmap_path)
        final_log[f"heatmap/{task_name}"] = wandb.Image(str(heatmap_path))
    wandb.log(final_log, step=cell_index)
    wandb.finish()

    print(f"Saved sweep outputs to {run_dir}")


if __name__ == "__main__":
    main()
