#!/usr/bin/env python3
"""Quick analysis: are high-attribution tokens disproportionately late in the sequence?"""
import sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.parse_math'] = False
import matplotlib.pyplot as plt

ROOT = Path("/nfs/nhome/live/jbauer/cot-oracle")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

FAST_CACHE_DIR = Path(os.environ["FAST_CACHE_DIR"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])
ACT_CACHE = FAST_CACHE_DIR / "qwen_probe_acts_s1_full"
CKPT_DIR = CACHE_DIR / "checkpoints" / "qwen_attention_probe"
PLOT_DIR = ROOT / "plots" / "probe_attribution"
LAYERS = [9, 18, 27]

from plot_probe_token_attribution import TASKS, load_probe, get_probe_direction, compute_token_scores, load_test_items

TOP_K = 10

for task_name in ["hint_admission", "sycophancy", "truthfulqa_hint"]:
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")

    model = load_probe(task_name)
    direction = get_probe_direction(model)
    items = load_test_items(task_name, max_items=200)

    # For each example, record relative positions of top-k influenced and top-k independent tokens
    influenced_positions = []  # relative positions [0,1] of top-k "influenced" tokens
    independent_positions = []
    all_lengths = []

    for item in items:
        scores = compute_token_scores(item["acts"], direction)
        K = len(scores)
        all_lengths.append(K)
        actual_k = min(TOP_K, K)

        rel_pos = np.linspace(0, 1, K)

        top_infl_idx = scores.topk(actual_k).indices.numpy()
        top_indep_idx = (-scores).topk(actual_k).indices.numpy()

        influenced_positions.extend(rel_pos[top_infl_idx])
        independent_positions.extend(rel_pos[top_indep_idx])

    influenced_positions = np.array(influenced_positions)
    independent_positions = np.array(independent_positions)

    # Statistics
    print(f"  Avg sequence length: {np.mean(all_lengths):.0f} tokens")
    print(f"  Top-{TOP_K} → influenced:  mean rel pos = {influenced_positions.mean():.3f}, median = {np.median(influenced_positions):.3f}")
    print(f"  Top-{TOP_K} → independent: mean rel pos = {independent_positions.mean():.3f}, median = {np.median(independent_positions):.3f}")
    print(f"  (Uniform baseline: mean=0.500, median=0.500)")

    # Fraction in last 25% of sequence
    frac_infl_last25 = (influenced_positions > 0.75).mean()
    frac_indep_last25 = (independent_positions > 0.75).mean()
    frac_infl_last10 = (influenced_positions > 0.90).mean()
    frac_indep_last10 = (independent_positions > 0.90).mean()
    print(f"  Fraction in last 25%:  influenced={frac_infl_last25:.3f}, independent={frac_indep_last25:.3f} (uniform=0.250)")
    print(f"  Fraction in last 10%:  influenced={frac_infl_last10:.3f}, independent={frac_indep_last10:.3f} (uniform=0.100)")

    # Plot histogram of relative positions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = np.linspace(0, 1, 21)

    ax1.hist(influenced_positions, bins=bins, color="firebrick", alpha=0.7, density=True)
    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax1.set_xlabel("Relative position in CoT")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Top-{TOP_K} → influenced tokens")
    ax1.legend()

    ax2.hist(independent_positions, bins=bins, color="steelblue", alpha=0.7, density=True)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax2.set_xlabel("Relative position in CoT")
    ax2.set_title(f"Top-{TOP_K} → independent tokens")
    ax2.legend()

    fig.suptitle(f"{task_name}: Position distribution of top-{TOP_K} attribution tokens", fontsize=13)
    fig.tight_layout()
    out_path = PLOT_DIR / task_name / "top_token_position_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved to {out_path}")

