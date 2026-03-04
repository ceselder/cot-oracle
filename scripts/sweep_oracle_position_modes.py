#!/usr/bin/env python3
"""Compare a trained activation oracle across position-selection modes."""

from __future__ import annotations

import argparse
import gc
import json
import math
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
    "chunked_convqa",
    "chunked_compqa",
]
DEFAULT_LAYERS = [9, 18, 27]
MODE_SPECS = [
    {"key": "last_only", "label": "Only Last", "position_mode": "last_only"},
    {"key": "last_5", "label": "Last Five", "position_mode": "last_5"},
    {"key": "mixed", "label": "Mixed", "position_mode": "mixed"},
    {"key": "all", "label": "All Activations", "position_mode": "all"},
    {"key": "random_sentence_count", "label": "Random = #Sentences", "position_mode": "random_sentence_count"},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep a trained activation oracle across position-selection modes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--activation-extract-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--trace-samples-per-mode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_logs/position_mode_compare")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run", default="position-mode-compare")
    parser.add_argument("--overview-cols", type=int, default=3)
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_task_arrays(tasks: list[str], summary_rows: list[dict]) -> tuple[list[str], dict[str, str], dict[str, list[float]]]:
    mode_keys = [spec["key"] for spec in MODE_SPECS]
    metrics_by_task = {}
    scores_by_task = {task: [0.0] * len(mode_keys) for task in tasks}
    for row in summary_rows:
        mode_idx = mode_keys.index(row["mode_key"])
        metrics_by_task[row["task"]] = row["metric_name"]
        scores_by_task[row["task"]][mode_idx] = row["score"]
    return mode_keys, metrics_by_task, scores_by_task


def _render_task_strip(task_name: str, metric_name: str, scores: list[float], output_path: Path) -> None:
    labels = [spec["label"] for spec in MODE_SPECS]
    fig, ax = plt.subplots(figsize=(9.2, 2.2), constrained_layout=True)
    im = ax.matshow([scores], cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{task_name} ({metric_name})", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="left")
    ax.set_yticks([])
    for col_idx, value in enumerate(scores):
        ax.text(col_idx, 0, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_overview(tasks: list[str], summary_rows: list[dict], output_path: Path, n_cols: int) -> None:
    labels = [spec["label"] for spec in MODE_SPECS]
    _, metrics_by_task, scores_by_task = _build_task_arrays(tasks, summary_rows)
    n_tasks = len(tasks)
    n_rows = math.ceil(n_tasks / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 2.8 * n_rows), squeeze=False, constrained_layout=True)
    mappable = None
    for idx, task_name in enumerate(tasks):
        ax = axes[idx // n_cols][idx % n_cols]
        scores = scores_by_task[task_name]
        mappable = ax.matshow([scores], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{task_name} ({metrics_by_task[task_name]})", fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="left")
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8)
        for col_idx, value in enumerate(scores):
            ax.text(col_idx, 0, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=8)
    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle("Activation Oracle Position-Mode Comparison", fontsize=14)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, fraction=0.015, pad=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    strip_dir = run_dir / "heatmaps"
    mode_dir_root = run_dir / "modes"
    run_dir.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        assert task_name in TASKS, f"Unknown task: {task_name}"
        assert task_name != "rot13_reconstruction", "rot13_reconstruction needs the dedicated ROT13 adapter and is not supported by this sweep"

    layers = list(args.layers)
    model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        config={
            "model": args.model,
            "checkpoint": checkpoint,
            "tasks": args.tasks,
            "layers": layers,
            "mode_specs": MODE_SPECS,
            "max_items": args.max_items,
            "eval_batch_size": args.eval_batch_size,
            "activation_extract_batch_size": args.activation_extract_batch_size,
            "world_size": 1,
            "baseline_type": "trained_oracle_position_mode_compare",
        },
        tags=["baseline", "trained_oracle", "position_modes"],
    )
    wandb.define_metric("mode/index")
    wandb.define_metric("*", step_metric="mode/index")

    summary_rows = []
    trace_rows = []
    cell_index = 0
    cot_utils.CONFIGURED_LAYERS = layers
    for mode_spec in tqdm(MODE_SPECS, desc="position modes"):
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
            position_mode=mode_spec["position_mode"],
            stochastic_max_k=100,
            eval_position_seed=args.seed,
        )
        elapsed = time.time() - started
        mode_dir = mode_dir_root / mode_spec["key"]
        per_task_metrics = {}
        wandb_log = {
            "mode/index": cell_index,
            "mode/key": mode_spec["key"],
            "mode/label": mode_spec["label"],
            "mode/elapsed_s": elapsed,
        }
        for task_name in args.tasks:
            task_score_key = f"eval/{task_name}"
            task_n_key = f"eval_n/{task_name}"
            assert task_score_key in metrics, f"Missing primary metric for {task_name} at {mode_spec['key']}"
            assert task_n_key in metrics, f"Missing item count for {task_name} at {mode_spec['key']}"
            task_def = TASKS[task_name]
            metric_name = _primary_metric_name(task_name, task_def.scoring)
            score = float(metrics[task_score_key])
            n_items = int(metrics[task_n_key])
            task_record = {
                "task": task_name,
                "metric_name": metric_name,
                "score": score,
                "n": n_items,
                "mode_key": mode_spec["key"],
                "mode_label": mode_spec["label"],
                "position_mode": mode_spec["position_mode"],
                "layers": layers,
                "elapsed_s": elapsed,
            }
            summary_rows.append(task_record)
            per_task_metrics[task_name] = task_record
            wandb_log[f"score/{task_name}"] = score
            wandb_log[f"n/{task_name}"] = n_items
            traces = all_traces[task_name]
            traces_with_meta = []
            for trace_idx, trace in enumerate(traces):
                row = {
                    "task": task_name,
                    "metric_name": metric_name,
                    "mode_key": mode_spec["key"],
                    "mode_label": mode_spec["label"],
                    "trace_index": trace_idx,
                    **trace,
                }
                traces_with_meta.append(row)
                if trace_idx < args.trace_samples_per_mode:
                    trace_rows.append({
                        "task": task_name,
                        "mode_key": mode_spec["key"],
                        "mode_label": mode_spec["label"],
                        "question": trace["question"][:240],
                        "expected": trace["expected"][:240],
                        "predicted": trace["predicted"][:240],
                        "correct": str(trace["correct"]),
                    })
            _write_jsonl(mode_dir / "traces" / f"{task_name}.jsonl", traces_with_meta)
        _write_json(mode_dir / "metrics.json", {"mode": mode_spec, "layers": layers, "elapsed_s": elapsed, "tasks": per_task_metrics})
        wandb.log(wandb_log, step=cell_index)
        cell_index += 1

    _write_json(run_dir / "summary.json", {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "model": args.model,
        "tasks": args.tasks,
        "layers": layers,
        "mode_specs": MODE_SPECS,
        "rows": summary_rows,
    })
    _write_jsonl(run_dir / "summary_rows.jsonl", summary_rows)
    _write_jsonl(run_dir / "trace_samples.jsonl", trace_rows)

    summary_table = wandb.Table(columns=["task", "metric_name", "score", "n", "mode_key", "mode_label", "elapsed_s"])
    for row in summary_rows:
        summary_table.add_data(row["task"], row["metric_name"], row["score"], row["n"], row["mode_key"], row["mode_label"], row["elapsed_s"])

    trace_table = wandb.Table(columns=["task", "mode_key", "mode_label", "question", "expected", "predicted", "correct"])
    for row in trace_rows:
        trace_table.add_data(row["task"], row["mode_key"], row["mode_label"], row["question"], row["expected"], row["predicted"], row["correct"])

    mode_keys, metrics_by_task, scores_by_task = _build_task_arrays(args.tasks, summary_rows)
    final_log = {"mode/index": cell_index, "summary_table": summary_table, "trace_samples": trace_table}
    for task_name in args.tasks:
        strip_path = strip_dir / f"{task_name}.png"
        _render_task_strip(task_name, metrics_by_task[task_name], scores_by_task[task_name], strip_path)
        final_log[f"heatmap/{task_name}"] = wandb.Image(str(strip_path))
    overview_path = run_dir / "all_tasks_overview.png"
    _render_overview(args.tasks, summary_rows, overview_path, args.overview_cols)
    final_log["overview"] = wandb.Image(str(overview_path))
    wandb.log(final_log, step=cell_index)
    wandb.finish()

    print(f"Saved sweep outputs to {run_dir}")


if __name__ == "__main__":
    main()
