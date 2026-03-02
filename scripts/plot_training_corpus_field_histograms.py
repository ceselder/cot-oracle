#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loading import load_readout_task_data, load_task_data, prepare_context_ids
from tasks import get_trainable_tasks


PLACEHOLDER_TOKEN = " ?"
READOUT_TASKS = {
    "futurelens_cot",
    "futurelens_fineweb",
    "pastlens_cot",
    "pastlens_fineweb",
    "reconstruction_cot",
    "reconstruction_fineweb",
}
FIELDS = ("question", "cot_field", "oracle_prefix", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-task training-corpus field length histograms.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output", default="eval_logs/training_corpus_field_histograms.png")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def enabled_train_tasks(config: dict) -> list[tuple[str, int]]:
    trainable = get_trainable_tasks()
    task_items = []
    for task_name, task_cfg in config["tasks"].items():
        if task_name not in trainable:
            continue
        n = task_cfg.get("n", 0)
        if n > 0:
            task_items.append((task_name, n))
    if not task_items:
        raise RuntimeError("No enabled training tasks found in config.")
    return task_items


def load_items_for_task(
    task_name: str,
    n: int,
    split: str,
    tokenizer,
    stride: int | str,
    layers: list[int],
    seed: int,
    model_name: str,
) -> list[dict]:
    if task_name in READOUT_TASKS:
        items = load_readout_task_data(
            task_name=task_name,
            tokenizer=tokenizer,
            n=n,
            split=split,
            stride=stride,
            layers=layers,
            seed=seed,
            model_name=model_name,
        )
    else:
        items = load_task_data(task_name, split=split, n=n, shuffle=False)
        prepare_context_ids(items, tokenizer, stride=stride, layers=layers)
    filtered = [item for item in items if item.get("context_input_ids")]
    if not filtered:
        raise RuntimeError(f"Task {task_name!r} produced 0 items with context_input_ids.")
    dropped = len(items) - len(filtered)
    if dropped > 0:
        print(f"  [{task_name}] Dropped {dropped} items without context_input_ids")
    return filtered


def token_lengths(tokenizer, texts: list[str], batch_size: int, desc: str) -> list[int]:
    lengths: list[int] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def oracle_prefix_lengths(tokenizer, num_positions: list[int]) -> list[int]:
    cache: dict[int, int] = {}
    lengths: list[int] = []
    for n in num_positions:
        if n not in cache:
            prefix = "Activations: " + PLACEHOLDER_TOKEN * n + "\n"
            cache[n] = len(tokenizer.encode(prefix, add_special_tokens=False))
        lengths.append(cache[n])
    return lengths


def summarize(lengths: list[int]) -> dict[str, float | int]:
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def plot_histograms(
    per_task_lengths: dict[str, dict[str, list[int]]],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    task_names = list(per_task_lengths)
    n_rows = len(task_names)
    fig, axes = plt.subplots(n_rows, len(FIELDS), figsize=(18, max(2.2 * n_rows, 4.5)), squeeze=False)
    colors = {
        "question": "#3b82f6",
        "cot_field": "#16a34a",
        "oracle_prefix": "#d97706",
        "prompt": "#dc2626",
    }

    for row_idx, task_name in enumerate(task_names):
        for col_idx, field in enumerate(FIELDS):
            ax = axes[row_idx][col_idx]
            values = per_task_lengths[task_name][field]
            max_value = max(values)
            if max_value == 0:
                hist_bins = np.array([-0.5, 0.5])
            else:
                hist_bins = min(bins, max(10, int(np.sqrt(len(values)))))
            ax.hist(values, bins=hist_bins, color=colors[field], alpha=0.85)
            ax.set_xlim(left=0)
            ax.tick_params(axis="both", labelsize=7)
            if row_idx == 0:
                ax.set_title(field, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(task_name, fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("tokens", fontsize=8)
            stats = summarize(values)
            ax.text(
                0.98,
                0.96,
                f"n={stats['count']}\np50={stats['p50']:.0f}\np95={stats['p95']:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    fig.suptitle("Training Corpus Field Lengths by Task", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    output_path = ROOT / args.output
    summary_path = output_path.with_suffix(".json")

    config = load_config(config_path)
    task_specs = enabled_train_tasks(config)
    model_name = config["model"]["name"]
    stride = config["activations"]["stride"]
    layers = list(config["activations"]["layers"])
    seed = int(config["training"]["seed"])

    print(f"[hist] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    per_task_lengths: dict[str, dict[str, list[int]]] = {}
    summary: dict[str, object] = {
        "config": str(config_path),
        "output": str(output_path),
        "split": args.split,
        "measure": "tokens",
        "cot_field_measure": "len(context_input_ids)",
        "oracle_prefix_measure": "tokenized length of 'Activations: ' + PLACEHOLDER_TOKEN * num_positions + newline",
        "tasks": [],
    }

    for task_name, n in tqdm(task_specs, desc="tasks"):
        print(f"[hist] Loading {task_name} (n={n})")
        items = load_items_for_task(
            task_name=task_name,
            n=n,
            split=args.split,
            tokenizer=tokenizer,
            stride=stride,
            layers=layers,
            seed=seed,
            model_name=model_name,
        )

        question_texts = [item.get("hinted_prompt") or item.get("question", "") for item in items]
        prompt_texts = [item["prompt"] for item in items]
        cot_lengths = [len(item["context_input_ids"]) for item in items]
        prefix_lengths = oracle_prefix_lengths(tokenizer, [item["num_positions"] for item in items])
        question_lengths = token_lengths(tokenizer, question_texts, args.batch_size, f"{task_name}:question")
        prompt_lengths = token_lengths(tokenizer, prompt_texts, args.batch_size, f"{task_name}:prompt")

        per_task_lengths[task_name] = {
            "question": question_lengths,
            "cot_field": cot_lengths,
            "oracle_prefix": prefix_lengths,
            "prompt": prompt_lengths,
        }
        summary["tasks"].append({
            "task": task_name,
            "n_requested": n,
            "n_loaded": len(items),
            "fields": {field: summarize(per_task_lengths[task_name][field]) for field in FIELDS},
        })

    plot_histograms(per_task_lengths, output_path, bins=args.bins, dpi=args.dpi)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[hist] Wrote plot: {output_path}")
    print(f"[hist] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
