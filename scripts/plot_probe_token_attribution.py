#!/usr/bin/env python3
"""Plot per-token attribution scores from trained linear probes.

For each task's trained linear probe, computes per-token scores by projecting
each position's activation through the probe's weight direction (class 1 - class 0).
Produces:
  1. Per-example heatmaps of token attributions (top N examples)
  2. Aggregate histogram of high-scoring token types across examples

Usage:
    python scripts/plot_probe_token_attribution.py --task sycophancy
    python scripts/plot_probe_token_attribution.py  # all 3 tasks
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.parse_math'] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "baselines"))
sys.path.insert(0, str(_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

FAST_CACHE_DIR = Path(os.environ["FAST_CACHE_DIR"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])
ACT_CACHE = FAST_CACHE_DIR / "qwen_probe_acts_s1_full"
CKPT_DIR = CACHE_DIR / "checkpoints" / "qwen_attention_probe"
PLOT_DIR = _ROOT / "plots" / "probe_attribution"

LAYERS = [9, 18, 27]
D_MODEL = 4096
BINARY_LABELS = ["independent", "influenced"]

TASKS = {
    "hint_admission": {
        "hf_repo_train": "mats-10-sprint-cs-jb/qwen3-8b-hint-admission-rollouts",
        "hf_repo_test": "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned",
        "label_map": {"hint_used_correct": "influenced", "hint_used_wrong": "influenced", "hint_resisted": "independent"},
        "prompt_field": "hinted_prompt",
    },
    "sycophancy": {
        "hf_repo": "mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned",
        "label_map": {"sycophantic": "influenced", "non_sycophantic": "independent"},
        "prompt_field": "hinted_prompt",
    },
    "truthfulqa_hint": {
        "hf_repo": "mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-unverbalized-cleaned",
        "label_map": {"hint_used_correct": "influenced", "hint_used_wrong": "influenced", "hint_resisted": "independent"},
        "prompt_field": "hinted_prompt",
    },
}


def load_probe(task_name: str, probe_type: str = "linear") -> torch.nn.Module:
    """Load trained probe checkpoint."""
    from train_qwen_probe import LinearConcatProbe
    ckpt_path = CKPT_DIR / f"{probe_type}_{task_name}" / "model.pt"
    model = LinearConcatProbe(LAYERS, d_model=D_MODEL, n_outputs=2, pooling="mean")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_probe_direction(model) -> torch.Tensor:
    """Extract the classification direction: w[influenced] - w[independent]."""
    w = model.linear.weight.data  # [2, 3*D]
    # class 1 = influenced, class 0 = independent
    direction = w[1] - w[0]  # [3*D]
    return direction.float()


def compute_token_scores(acts: dict[int, torch.Tensor], direction: torch.Tensor) -> torch.Tensor:
    """Compute per-token attribution scores by projecting onto probe direction.

    For mean-pool probes, each token's contribution to the mean is proportional to
    its projection onto the direction. Returns [K] scores.
    """
    K = acts[LAYERS[0]].shape[0]
    # Concat layers for each position: [K, 3*D]
    concat = torch.cat([acts[l].float() for l in LAYERS], dim=-1)  # [K, 3*D]
    scores = concat @ direction  # [K]
    return scores


def load_test_items(task_name: str, max_items: int = 200):
    """Load test items with cached activations and tokenizer-decoded tokens."""
    from datasets import load_dataset
    from train_qwen_probe import _process_split

    cfg = TASKS[task_name]
    if "hf_repo" in cfg:
        ds = load_dataset(cfg["hf_repo"])
        test_items = _process_split(ds["test"], cfg, task_name)
    else:
        test_split = load_dataset(cfg["hf_repo_test"], split="test")
        test_items = _process_split(test_split, cfg, task_name)

    # Load cached activations
    cache_dir = ACT_CACHE / task_name / "test"
    loaded = []
    for item in test_items[:max_items]:
        cache_path = cache_dir / f"{item['example_id']}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            item["acts"] = {k: v for k, v in cached.items() if isinstance(k, int)}
            item["token_ids"] = cached["token_ids"]
            loaded.append(item)
    return loaded


def plot_example_heatmaps(task_name: str, items: list[dict], direction: torch.Tensor,
                          tokenizer, n_examples: int = 10):
    """Plot per-token attribution heatmaps for top examples."""
    out_dir = PLOT_DIR / task_name / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items[:n_examples]):
        scores = compute_token_scores(item["acts"], direction)
        tokens = tokenizer.convert_ids_to_tokens(item["token_ids"].tolist())
        K = len(tokens)

        fig, ax = plt.subplots(figsize=(max(12, K * 0.15), 3))
        colors = plt.cm.RdBu_r((scores.numpy() - scores.min().item()) /
                                 max(scores.max().item() - scores.min().item(), 1e-8))

        for i, (tok, score) in enumerate(zip(tokens, scores)):
            ax.bar(i, score.item(), color=colors[i], width=0.9)

        # Mark top-5 tokens
        top5_idx = scores.abs().topk(min(5, K)).indices
        for ti in top5_idx:
            ax.annotate(tokens[ti], (ti.item(), scores[ti].item()),
                       fontsize=6, rotation=45, ha="center", va="bottom")

        ax.set_xlabel("Token position in CoT")
        ax.set_ylabel("Attribution score (→ influenced)")
        ax.set_title(f"{task_name} | {item['example_id']} | true={item['label']}")
        ax.axhline(y=0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{idx:03d}_{item['example_id']}.png", dpi=150)
        plt.close(fig)

    print(f"  Saved {n_examples} heatmaps to {out_dir}")


def plot_top_token_histogram(task_name: str, items: list[dict], direction: torch.Tensor,
                             tokenizer, top_k_per_example: int = 10):
    """Plot histogram of most frequently high-scoring tokens across all examples."""
    out_dir = PLOT_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    influenced_counter = Counter()
    independent_counter = Counter()

    for item in tqdm(items, desc=f"Computing attributions ({task_name})"):
        scores = compute_token_scores(item["acts"], direction)
        tokens = tokenizer.convert_ids_to_tokens(item["token_ids"].tolist())
        K = len(tokens)
        actual_k = min(top_k_per_example, K)

        # Top tokens pushing toward "influenced"
        top_influenced = scores.topk(actual_k).indices
        for ti in top_influenced:
            influenced_counter[tokens[ti.item()]] += 1

        # Top tokens pushing toward "independent"
        top_independent = (-scores).topk(actual_k).indices
        for ti in top_independent:
            independent_counter[tokens[ti.item()]] += 1

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    n_show = 30
    for ax, counter, label, color in [
        (ax1, influenced_counter, "→ influenced", "firebrick"),
        (ax2, independent_counter, "→ independent", "steelblue"),
    ]:
        most_common = counter.most_common(n_show)
        tokens_list = [t for t, _ in most_common]
        counts = [c for _, c in most_common]
        y_pos = range(len(tokens_list))
        ax.barh(y_pos, counts, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens_list, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (across test examples)")
        ax.set_title(f"Top tokens {label}")

    fig.suptitle(f"{task_name}: Most frequent high-attribution tokens (top-{top_k_per_example}/example)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "top_token_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved histogram to {out_dir / 'top_token_histogram.png'}")

    # Also save as JSON for inspection
    with open(out_dir / "top_tokens.json", "w") as f:
        json.dump({
            "influenced": influenced_counter.most_common(100),
            "independent": independent_counter.most_common(100),
        }, f, indent=2)


def plot_score_by_position(task_name: str, items: list[dict], direction: torch.Tensor):
    """Plot average attribution score as a function of relative position in CoT."""
    out_dir = PLOT_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 50
    influenced_bins = np.zeros(n_bins)
    independent_bins = np.zeros(n_bins)
    influenced_counts = np.zeros(n_bins)
    independent_counts = np.zeros(n_bins)

    for item in items:
        scores = compute_token_scores(item["acts"], direction).numpy()
        K = len(scores)
        # Relative positions [0, 1]
        rel_pos = np.linspace(0, 1, K)
        bin_idx = np.clip((rel_pos * n_bins).astype(int), 0, n_bins - 1)

        if item["label"] == "influenced":
            for i, bi in enumerate(bin_idx):
                influenced_bins[bi] += scores[i]
                influenced_counts[bi] += 1
        else:
            for i, bi in enumerate(bin_idx):
                independent_bins[bi] += scores[i]
                independent_counts[bi] += 1

    influenced_mean = np.divide(influenced_bins, influenced_counts,
                                 where=influenced_counts > 0, out=np.zeros(n_bins))
    independent_mean = np.divide(independent_bins, independent_counts,
                                  where=independent_counts > 0, out=np.zeros(n_bins))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 100, n_bins)
    ax.plot(x, influenced_mean, color="firebrick", label="influenced examples", linewidth=2)
    ax.plot(x, independent_mean, color="steelblue", label="independent examples", linewidth=2)
    ax.fill_between(x, influenced_mean, alpha=0.15, color="firebrick")
    ax.fill_between(x, independent_mean, alpha=0.15, color="steelblue")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Relative position in CoT (%)")
    ax.set_ylabel("Mean attribution score (→ influenced)")
    ax.set_title(f"{task_name}: Attribution score by CoT position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_by_position.png", dpi=150)
    plt.close(fig)
    print(f"  Saved position plot to {out_dir / 'score_by_position.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", default=list(TASKS.keys()))
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--n-heatmaps", type=int, default=10)
    parser.add_argument("--probe-type", default="linear")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    for task_name in args.task:
        print(f"\n{'='*60}")
        print(f"  {task_name}")
        print(f"{'='*60}")

        model = load_probe(task_name, args.probe_type)
        direction = get_probe_direction(model)
        items = load_test_items(task_name, max_items=args.max_items)
        print(f"  Loaded {len(items)} test items")

        plot_top_token_histogram(task_name, items, direction, tokenizer)
        plot_score_by_position(task_name, items, direction)
        plot_example_heatmaps(task_name, items, direction, tokenizer, n_examples=args.n_heatmaps)


if __name__ == "__main__":
    main()
