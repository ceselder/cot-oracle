#!/usr/bin/env python3
"""Train mean-concat linear probes, backfill split metrics to wandb, and plot vs v16 checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from tqdm.auto import tqdm, trange

_SRC = Path(__file__).resolve().parent.parent / "src"
_AO_REF = Path(__file__).resolve().parent.parent / "ao_reference"
for p in [str(_SRC), str(_AO_REF)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.ao import load_extra_adapter, load_model_with_ao
from data_loading import load_task_data, prepare_context_ids
from eval_loop import _batched_oracle_generate, _materialize_activations, _parse_atypical, _parse_termination, _parse_trajectory, _resample_eval_positions

LAYERS = [9, 18, 27]
TASKS = ["reasoning_termination", "answer_trajectory", "atypical_answer"]
SPLITS = ["train_1", "train_2", "eval"]
SPLIT_DISPLAY = {"train_1": "train", "train_2": "test", "eval": "eval"}
TASK_MAX_NEW_TOKENS = {"reasoning_termination": 64, "answer_trajectory": 64, "atypical_answer": 64}
CHECKPOINT_ROOT = Path("/ceph/scratch/jbauer/checkpoints/cot_oracle_v16_posembed")
CHECKPOINT_PATHS = {"step_2536": CHECKPOINT_ROOT / "step_2536", "final": CHECKPOINT_ROOT / "final"}
PROBE_RUN_IDS = {
    "reasoning_termination": "1ohxlpxm",
    "answer_trajectory": "vzg2cqdj",
    "atypical_answer": "u7yipkgk",
}


@dataclass
class ProbeModel:
    task_name: str
    is_regression: bool
    labels: list[str]
    mu: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _group_key(item: dict) -> str:
    if "question_id" in item:
        return str(item["question_id"])
    return str(item["question"])


def split_train_80_20(items: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        g = _group_key(item)
        if g not in groups:
            groups[g] = []
        groups[g].append(item)
    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    cut = int(0.8 * len(group_ids))
    train1_ids = set(group_ids[:cut])
    train2_ids = set(group_ids[cut:])
    train_1: list[dict] = []
    train_2: list[dict] = []
    for gid in group_ids:
        if gid in train1_ids:
            train_1.extend(groups[gid])
        else:
            train_2.extend(groups[gid])
    return train_1, train_2


def sample_items(items: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or len(items) <= n:
        return list(items)
    idx = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    idx = idx[:n]
    idx.sort()
    return [items[i] for i in idx]


def balanced_accuracy(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    per_class = []
    for c in classes:
        idx = [i for i, y in enumerate(y_true) if y == c]
        correct = sum(1 for i in idx if y_pred[i] == c)
        per_class.append(correct / len(idx))
    return float(sum(per_class) / len(per_class))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def balanced_accuracy_int(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        idx = y_true == c
        recalls.append(float(np.mean(y_pred[idx] == c)))
    return float(np.mean(recalls))


def bootstrap_std_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, n_boot: int, seed: int) -> float:
    if n_boot <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y_true == c)[0] for c in range(n_classes)]
    vals = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        recalls = []
        for c, idx in enumerate(class_indices):
            sampled = rng.choice(idx, size=len(idx), replace=True)
            recalls.append(float(np.mean(y_pred[sampled] == c)))
        vals[bi] = float(np.mean(recalls))
    return float(np.std(vals, ddof=1))


def bootstrap_std_mae(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int) -> float:
    if n_boot <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    n = len(y_true)
    abs_err = np.abs(y_true - y_pred)
    vals = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[bi] = float(np.mean(abs_err[idx]))
    return float(np.std(vals, ddof=1))


def _task_label_from_parser(task_name: str, text: str) -> str | None:
    if task_name == "reasoning_termination":
        parsed = _parse_termination(text)
        if parsed is None:
            return None
        return "will_terminate" if parsed["label"] == "yes" else "will_continue"
    if task_name == "atypical_answer":
        parsed = _parse_atypical(text)
        if parsed is None:
            return None
        return parsed["label"]
    raise ValueError(task_name)


def _confidence_from_text(text: str) -> float | None:
    parsed = _parse_trajectory(text)
    if parsed is None:
        return None
    if "confidence" in parsed:
        return float(parsed["confidence"]) / 100.0
    match = re.search(r"confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1)) / 100.0


def feature_mean_concat(activation: torch.Tensor, n_layers: int) -> torch.Tensor:
    k = activation.shape[0] // n_layers
    parts = []
    for li in range(n_layers):
        parts.append(activation[li * k : (li + 1) * k, :].float().mean(dim=0))
    return torch.cat(parts, dim=0)


def build_features(activations: list[torch.Tensor], n_layers: int) -> torch.Tensor:
    return torch.stack([feature_mean_concat(act, n_layers) for act in activations], dim=0)


def train_linear_probe_binary(X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, wd: float, batch_size: int, device: str) -> ProbeModel:
    labels = sorted({int(v.item()) for v in y})
    assert labels == [0, 1]
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xs = ((X - mu) / std).to(device)
    ys = y.float().to(device)
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    for _ in trange(epochs, desc="    probe_train", leave=False):
        perm = torch.randperm(Xs.shape[0], device=device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            out = probe(Xs[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(out, ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return ProbeModel(
        task_name="",
        is_regression=False,
        labels=["0", "1"],
        mu=mu.cpu(),
        std=std.cpu(),
        weight=probe.weight.detach().cpu(),
        bias=probe.bias.detach().cpu(),
    )


def train_linear_probe_regression(X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, wd: float, batch_size: int, device: str) -> ProbeModel:
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xs = ((X - mu) / std).to(device)
    ys = y.float().to(device)
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    for _ in trange(epochs, desc="    probe_train", leave=False):
        perm = torch.randperm(Xs.shape[0], device=device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            out = probe(Xs[idx]).squeeze(-1)
            loss = F.mse_loss(out, ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return ProbeModel(
        task_name="",
        is_regression=True,
        labels=[],
        mu=mu.cpu(),
        std=std.cpu(),
        weight=probe.weight.detach().cpu(),
        bias=probe.bias.detach().cpu(),
    )


def run_probe(model: ProbeModel, X: torch.Tensor) -> torch.Tensor:
    Xs = (X - model.mu) / model.std
    logits = Xs @ model.weight.t() + model.bias
    if model.is_regression:
        return logits.squeeze(-1)
    return (logits.squeeze(-1) > 0).long()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _compute_split_source_metadata(seed: int, max_train_total: int, max_eval: int) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], dict[str, dict[str, int]]]:
    held_out_sources_by_task: dict[str, list[str]] = {}
    split_sources_by_task: dict[str, dict[str, list[str]]] = {}
    split_sizes: dict[str, dict[str, int]] = {}

    for task_name in TASKS:
        train_items_all = load_task_data(task_name, split="train", n=None, shuffle=False)
        eval_items_all = load_task_data(task_name, split="test", n=None, shuffle=False)
        train_1_all, train_2_all = split_train_80_20(train_items_all, seed=seed)

        if max_train_total > 0:
            n_train_1 = int(round(max_train_total * 0.8))
            n_train_2 = max_train_total - n_train_1
            train_1 = sample_items(train_1_all, n_train_1, seed=seed + 11)
            train_2 = sample_items(train_2_all, n_train_2, seed=seed + 12)
        else:
            train_1 = train_1_all
            train_2 = train_2_all

        if max_eval > 0:
            eval_items = sample_items(eval_items_all, max_eval, seed=seed + 13)
        else:
            eval_items = eval_items_all

        split_items = {"train_1": train_1, "train_2": train_2, "eval": eval_items}
        split_sources_by_task[task_name] = {k: sorted({item["source"] for item in v}) for k, v in split_items.items()}
        split_sizes[task_name] = {k: len(v) for k, v in split_items.items()}

        train_sources = {item["source"] for item in train_items_all}
        eval_sources = {item["source"] for item in eval_items_all}
        held_out_sources_by_task[task_name] = sorted(eval_sources - train_sources)

    return held_out_sources_by_task, split_sources_by_task, split_sizes


def _plot_results(results: dict, plot_path: Path, ckpt_labels: list[str]) -> None:
    plt.rcParams["figure.autolayout"] = False
    method_labels = ["probe"] + [f"v16_{c}" for c in ckpt_labels]
    method_colors = {
        "probe": "C0",
        "v16_step_2536": "C3",
        "v16_final": "C1",
    }
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4), layout="constrained")
    split_keys = ["train_1", "train_2", "eval"]
    split_labels = [SPLIT_DISPLAY[s] for s in split_keys]
    x = np.array([0.0, 0.58, 1.8], dtype=np.float64)

    for ax, task_name in zip(axes, TASKS):
        width = 0.34 / len(method_labels)
        offsets = np.linspace(-(len(method_labels) - 1) * width / 2, (len(method_labels) - 1) * width / 2, len(method_labels))

        metric_key = "balanced_accuracy" if task_name in ("reasoning_termination", "atypical_answer") else "mae"
        for mi, method in enumerate(method_labels):
            vals = []
            stds = []
            for split_name in split_keys:
                if method == "probe":
                    vals.append(results["probe"][task_name][split_name][metric_key])
                    stds.append(results["probe"][task_name][split_name]["bootstrap_std"])
                else:
                    ck = method.replace("v16_", "")
                    vals.append(results["v16"][ck][task_name][split_name][metric_key])
                    stds.append(results["v16"][ck][task_name][split_name]["bootstrap_std"])
            bar_color = method_colors.get(method, "#4b5563")
            ax.bar(
                x + offsets[mi],
                vals,
                width=width,
                color=bar_color,
                label=method,
                alpha=0.96,
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 1.1},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)
        if metric_key == "balanced_accuracy":
            ax.set_ylim(0.35, 1.04)
            ax.set_ylabel("Balanced Accuracy")
        else:
            ymax = max(results["probe"][task_name][s][metric_key] + results["probe"][task_name][s]["bootstrap_std"] for s in split_keys)
            for ck in ckpt_labels:
                ymax = max(ymax, *(results["v16"][ck][task_name][s][metric_key] + results["v16"][ck][task_name][s]["bootstrap_std"] for s in split_keys))
            ax.set_ylim(0.0, ymax * 1.3 + 0.01)
            ax.set_ylabel("MAE (lower is better)")
        ax.set_title(task_name)
        split_sources = results["split_sources"][task_name]
        for xi, split_name in zip(x, split_keys):
            dataset_text = "\n".join(textwrap.wrap(", ".join(split_sources[split_name]), width=26))
            ax.text(xi, -0.12, dataset_text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Qwen/Qwen3-8B: v16_posembed vs mean-concat linear probes on train/test/eval (bootstrap std)", y=1.04)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-total", type=int, default=250, help="Cap before 80/20 split accounting. 0 = full.")
    parser.add_argument("--max-eval", type=int, default=100, help="Cap eval split size. 0 = full.")
    parser.add_argument("--bootstrap-samples", type=int, default=500, help="Bootstrap resamples per split metric.")
    parser.add_argument("--position-mode", default="end_rdm_stc")
    parser.add_argument("--stochastic-max-k", type=int, default=100)
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=120)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--probe-wd", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--checkpoints", nargs="+", default=["final"])
    parser.add_argument("--output-dir", default="logs/probe_v16_backfill/20260306_v16_probe_splits")
    parser.add_argument("--plot-path", default="plots/v16_posembed_probe_train1_train2_eval.png")
    parser.add_argument("--results-path", default="", help="Existing results.json path (used by --plot-only).")
    parser.add_argument("--plot-only", action="store_true", help="Skip training/eval and only render plot from results.json.")
    parser.add_argument("--no-wandb-backfill", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path.home() / ".env")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results_path = Path(args.results_path) if args.results_path else output_dir / "results.json"
        with open(results_path) as f:
            results = json.load(f)
        held_out_sources_by_task, split_sources_by_task, split_sizes = _compute_split_source_metadata(
            seed=args.seed,
            max_train_total=args.max_train_total,
            max_eval=args.max_eval,
        )
        results["held_out_eval_sources"] = held_out_sources_by_task
        results["split_sources"] = split_sources_by_task
        results["split_sizes"] = split_sizes
        _plot_results(results=results, plot_path=Path(args.plot_path), ckpt_labels=args.checkpoints)
        print(f"Saved plot: {args.plot_path}")
        return

    print("Loading Qwen + AO runtime...")
    model, tokenizer = load_model_with_ao("Qwen/Qwen3-8B", device=args.device)

    sent_delim_ids: set[int] = set()
    for pattern in [".", ".\n", ".\n\n"]:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if len(ids) == 1:
            sent_delim_ids.add(ids[0])

    split_items_by_task: dict[str, dict[str, list[dict]]] = {}
    split_acts_by_task: dict[str, dict[str, list[torch.Tensor]]] = {}
    held_out_sources_by_task: dict[str, list[str]] = {}
    split_sources_by_task: dict[str, dict[str, list[str]]] = {}
    split_sizes: dict[str, dict[str, int]] = {}

    print("\nPreparing splits + activations...")
    for task_name in TASKS:
        print(f"\n=== {task_name} ===")
        train_items_all = load_task_data(task_name, split="train", n=None, shuffle=False)
        eval_items_all = load_task_data(task_name, split="test", n=None, shuffle=False)

        train_1_all, train_2_all = split_train_80_20(train_items_all, seed=args.seed)

        if args.max_train_total > 0:
            n_train_1 = int(round(args.max_train_total * 0.8))
            n_train_2 = args.max_train_total - n_train_1
            train_1 = sample_items(train_1_all, n_train_1, seed=args.seed + 11)
            train_2 = sample_items(train_2_all, n_train_2, seed=args.seed + 12)
        else:
            train_1 = train_1_all
            train_2 = train_2_all

        if args.max_eval > 0:
            eval_items = sample_items(eval_items_all, args.max_eval, seed=args.seed + 13)
        else:
            eval_items = eval_items_all

        train_sources = {item["source"] for item in train_items_all}
        eval_sources = {item["source"] for item in eval_items_all}
        held_out_sources = sorted(eval_sources - train_sources)
        held_out_sources_by_task[task_name] = held_out_sources

        splits = {"train_1": train_1, "train_2": train_2, "eval": eval_items}
        split_items_by_task[task_name] = splits
        split_sizes[task_name] = {k: len(v) for k, v in splits.items()}
        split_sources_by_task[task_name] = {k: sorted({item["source"] for item in v}) for k, v in splits.items()}
        print(f"  split sizes: {split_sizes[task_name]}")
        print(f"  held-out eval sources: {held_out_sources}")
        print(f"  split sources: {split_sources_by_task[task_name]}")

        split_acts_by_task[task_name] = {}
        for split_name, items in splits.items():
            prepare_context_ids(items, tokenizer, layers=LAYERS)
            _resample_eval_positions(
                test_data=items,
                task_name=task_name,
                layers=LAYERS,
                position_mode=args.position_mode,
                stochastic_max_k=args.stochastic_max_k,
                eval_position_seed=args.seed,
                sentence_delim_ids=sent_delim_ids,
            )
            all_acts: list[torch.Tensor] = []
            for start in tqdm(range(0, len(items), args.extract_batch_size), desc=f"  acts {task_name}/{split_name}", leave=False):
                chunk = items[start : start + args.extract_batch_size]
                chunk_acts = _materialize_activations(model, tokenizer, chunk, layers=LAYERS, device=args.device)
                all_acts.extend([act.cpu() for act in chunk_acts])
            split_acts_by_task[task_name][split_name] = all_acts

    results: dict[str, dict] = {
        "probe": {},
        "v16": {},
        "held_out_eval_sources": held_out_sources_by_task,
        "split_sources": split_sources_by_task,
        "split_sizes": split_sizes,
    }

    print("\nTraining probes + scoring splits...")
    for task_name in TASKS:
        acts = split_acts_by_task[task_name]
        items = split_items_by_task[task_name]
        X_train = build_features(acts["train_1"], n_layers=len(LAYERS))
        X_train2 = build_features(acts["train_2"], n_layers=len(LAYERS))
        X_eval = build_features(acts["eval"], n_layers=len(LAYERS))

        task_result: dict[str, dict[str, float]] = {}

        if task_name in ("reasoning_termination", "atypical_answer"):
            class_labels = sorted({item["label"] for item in items["train_1"] + items["train_2"] + items["eval"]})
            label_to_idx = {label: i for i, label in enumerate(class_labels)}
            idx_to_label = {i: label for label, i in label_to_idx.items()}
            y_train = torch.tensor([label_to_idx[item["label"]] for item in items["train_1"]], dtype=torch.long)
            probe = train_linear_probe_binary(X_train, y_train, epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, batch_size=args.probe_batch_size, device=args.device)
            probe.task_name = task_name
            probe.labels = class_labels

            for split_name, X_split in [("train_1", X_train), ("train_2", X_train2), ("eval", X_eval)]:
                pred_idx = run_probe(probe, X_split).tolist()
                pred_labels = [idx_to_label[i] for i in pred_idx]
                true_labels = [item["label"] for item in items[split_name]]
                true_idx = np.array([label_to_idx[y] for y in true_labels], dtype=np.int64)
                pred_idx_np = np.array(pred_idx, dtype=np.int64)
                score = balanced_accuracy_int(true_idx, pred_idx_np, len(class_labels))
                std = bootstrap_std_balanced_accuracy(
                    y_true=true_idx,
                    y_pred=pred_idx_np,
                    n_classes=len(class_labels),
                    n_boot=args.bootstrap_samples,
                    seed=args.seed + 1000 * TASKS.index(task_name) + 100 * SPLITS.index(split_name) + 1,
                )
                task_result[split_name] = {"balanced_accuracy": score, "bootstrap_std": std}
                rows = []
                for item, pred in zip(items[split_name], pred_labels):
                    rows.append({
                        "task": task_name,
                        "method": "probe_mean_concat",
                        "split": split_name,
                        "question": item["question"],
                        "target": item["label"],
                        "prediction": pred,
                    })
                write_jsonl(pred_dir / f"probe_mean_concat_{task_name}_{split_name}.jsonl", rows)

        if task_name == "answer_trajectory":
            y_train = torch.tensor([float(item["confidence"]) for item in items["train_1"]], dtype=torch.float32)
            probe = train_linear_probe_regression(X_train, y_train, epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, batch_size=args.probe_batch_size, device=args.device)
            probe.task_name = task_name
            for split_name, X_split in [("train_1", X_train), ("train_2", X_train2), ("eval", X_eval)]:
                pred_conf = run_probe(probe, X_split).clamp_(0.0, 1.0).tolist()
                true_conf = [float(item["confidence"]) for item in items[split_name]]
                true_np = np.array(true_conf, dtype=np.float64)
                pred_np = np.array(pred_conf, dtype=np.float64)
                score = float(np.mean(np.abs(true_np - pred_np)))
                std = bootstrap_std_mae(
                    y_true=true_np,
                    y_pred=pred_np,
                    n_boot=args.bootstrap_samples,
                    seed=args.seed + 1000 * TASKS.index(task_name) + 100 * SPLITS.index(split_name) + 2,
                )
                task_result[split_name] = {"mae": score, "bootstrap_std": std}
                rows = []
                for item, pred in zip(items[split_name], pred_conf):
                    rows.append({
                        "task": task_name,
                        "method": "probe_mean_concat",
                        "split": split_name,
                        "question": item["question"],
                        "target_confidence": float(item["confidence"]),
                        "prediction_confidence": float(pred),
                    })
                write_jsonl(pred_dir / f"probe_mean_concat_{task_name}_{split_name}.jsonl", rows)

        results["probe"][task_name] = task_result

    print("\nScoring v16 checkpoints on same splits...")
    ckpt_labels = args.checkpoints
    for ckpt_label in args.checkpoints:
        ckpt_path = CHECKPOINT_PATHS[ckpt_label]
        adapter_name = f"v16_{ckpt_label}"
        load_extra_adapter(model, str(ckpt_path), adapter_name=adapter_name)
        ckpt_result: dict[str, dict[str, dict[str, float]]] = {}

        for task_name in TASKS:
            splits = split_items_by_task[task_name]
            split_acts = split_acts_by_task[task_name]
            task_result: dict[str, dict[str, float]] = {}
            ckpt_idx = ckpt_labels.index(ckpt_label)
            for split_name in SPLITS:
                items = splits[split_name]
                acts = split_acts[split_name]
                oracle_items = [(act, item["prompt"]) for act, item in zip(acts, items)]
                predictions = _batched_oracle_generate(
                    model=model,
                    tokenizer=tokenizer,
                    items=oracle_items,
                    layers=LAYERS,
                    device=args.device,
                    injection_layer=1,
                    max_new_tokens=TASK_MAX_NEW_TOKENS[task_name],
                    eval_batch_size=args.eval_batch_size,
                    oracle_adapter_name=adapter_name,
                )

                if task_name in ("reasoning_termination", "atypical_answer"):
                    classes = sorted({item["label"] for item in items})
                    label_to_idx = {c: i for i, c in enumerate(classes)}
                    true_labels = [item["label"] for item in items]
                    pred_labels = []
                    for pred in predictions:
                        parsed = _task_label_from_parser(task_name, pred)
                        if parsed is None:
                            parsed = "__unparsed__"
                        pred_labels.append(parsed)
                    true_idx = np.array([label_to_idx[y] for y in true_labels], dtype=np.int64)
                    pred_idx = np.array([label_to_idx[p] if p in label_to_idx else -1 for p in pred_labels], dtype=np.int64)
                    score = balanced_accuracy_int(true_idx, pred_idx, len(classes))
                    std = bootstrap_std_balanced_accuracy(
                        y_true=true_idx,
                        y_pred=pred_idx,
                        n_classes=len(classes),
                        n_boot=args.bootstrap_samples,
                        seed=args.seed + 10000 * (ckpt_idx + 1) + 1000 * TASKS.index(task_name) + 100 * SPLITS.index(split_name) + 3,
                    )
                    task_result[split_name] = {"balanced_accuracy": score, "bootstrap_std": std}
                    rows = []
                    for item, pred_text, pred_label in zip(items, predictions, pred_labels):
                        rows.append({
                            "task": task_name,
                            "method": f"v16_{ckpt_label}",
                            "split": split_name,
                            "question": item["question"],
                            "target": item["label"],
                            "prediction_text": pred_text,
                            "prediction_label": pred_label,
                        })
                    write_jsonl(pred_dir / f"v16_{ckpt_label}_{task_name}_{split_name}.jsonl", rows)

                if task_name == "answer_trajectory":
                    true_conf = [float(item["confidence"]) for item in items]
                    pred_conf = []
                    for pred in predictions:
                        conf = _confidence_from_text(pred)
                        if conf is None:
                            conf = 0.0
                        pred_conf.append(conf)
                    true_np = np.array(true_conf, dtype=np.float64)
                    pred_np = np.array(pred_conf, dtype=np.float64)
                    score = float(np.mean(np.abs(true_np - pred_np)))
                    std = bootstrap_std_mae(
                        y_true=true_np,
                        y_pred=pred_np,
                        n_boot=args.bootstrap_samples,
                        seed=args.seed + 10000 * (ckpt_idx + 1) + 1000 * TASKS.index(task_name) + 100 * SPLITS.index(split_name) + 4,
                    )
                    task_result[split_name] = {"mae": score, "bootstrap_std": std}
                    rows = []
                    for item, pred_text, pred in zip(items, predictions, pred_conf):
                        rows.append({
                            "task": task_name,
                            "method": f"v16_{ckpt_label}",
                            "split": split_name,
                            "question": item["question"],
                            "target_confidence": float(item["confidence"]),
                            "prediction_text": pred_text,
                            "prediction_confidence": float(pred),
                        })
                    write_jsonl(pred_dir / f"v16_{ckpt_label}_{task_name}_{split_name}.jsonl", rows)

            ckpt_result[task_name] = task_result
        results["v16"][ckpt_label] = ckpt_result

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    if not args.no_wandb_backfill:
        print("\nBackfilling probe runs on wandb...")
        for task_name in TASKS:
            run_id = PROBE_RUN_IDS[task_name]
            run = wandb.init(project="cot_oracle", entity="MATS10-CS-JB", id=run_id, resume="allow")
            payload = {"train/samples_seen": 1000, "split/max_train_total": args.max_train_total, "split/max_eval": args.max_eval}
            metric_key = "balanced_accuracy" if task_name in ("reasoning_termination", "atypical_answer") else "mae"
            payload[f"probe/{task_name}/train_1/{metric_key}"] = results["probe"][task_name]["train_1"][metric_key]
            payload[f"probe/{task_name}/train_2/{metric_key}"] = results["probe"][task_name]["train_2"][metric_key]
            payload[f"probe/{task_name}/eval/{metric_key}"] = results["probe"][task_name]["eval"][metric_key]
            payload[f"probe/{task_name}/train_1/{metric_key}_bootstrap_std"] = results["probe"][task_name]["train_1"]["bootstrap_std"]
            payload[f"probe/{task_name}/train_2/{metric_key}_bootstrap_std"] = results["probe"][task_name]["train_2"]["bootstrap_std"]
            payload[f"probe/{task_name}/eval/{metric_key}_bootstrap_std"] = results["probe"][task_name]["eval"]["bootstrap_std"]
            payload[f"probe/{task_name}/held_out_eval_sources"] = ",".join(results["held_out_eval_sources"][task_name])
            run.log(payload, step=1000)
            run.summary[f"backfill/{task_name}/train_1/{metric_key}"] = results["probe"][task_name]["train_1"][metric_key]
            run.summary[f"backfill/{task_name}/train_2/{metric_key}"] = results["probe"][task_name]["train_2"][metric_key]
            run.summary[f"backfill/{task_name}/eval/{metric_key}"] = results["probe"][task_name]["eval"][metric_key]
            run.summary[f"backfill/{task_name}/train_1/{metric_key}_bootstrap_std"] = results["probe"][task_name]["train_1"]["bootstrap_std"]
            run.summary[f"backfill/{task_name}/train_2/{metric_key}_bootstrap_std"] = results["probe"][task_name]["train_2"]["bootstrap_std"]
            run.summary[f"backfill/{task_name}/eval/{metric_key}_bootstrap_std"] = results["probe"][task_name]["eval"]["bootstrap_std"]
            run.summary[f"backfill/{task_name}/held_out_eval_sources"] = ",".join(results["held_out_eval_sources"][task_name])
            run.finish()

    print("\nPlotting...")
    _plot_results(results=results, plot_path=Path(args.plot_path), ckpt_labels=args.checkpoints)
    print(f"Saved plot: {args.plot_path}")


if __name__ == "__main__":
    main()
