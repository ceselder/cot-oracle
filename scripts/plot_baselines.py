#!/usr/bin/env python3
"""Plot baseline comparison: linear probes vs LLM monitor vs chance."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Load data ──
with open("data/probe_baseline_results_v2.json") as f:
    probes = json.load(f)
with open("data/llm_monitor_baselines.json") as f:
    llm = json.load(f)

# ── Classification datasets (shared between probes & LLM monitor) ──
datasets = [
    ("hint_admission", "Hint\nAdmission"),
    ("atypical_answer", "Atypical\nAnswer"),
    ("decorative_cot", "Decorative\nCoT"),
    ("sycophancy", "Sycophancy"),
    ("truthfulqa_verb", "TruthfulQA\n(Verbalized)"),
    ("truthfulqa_unverb", "TruthfulQA\n(Unverbalized)"),
]

# ── Extract best probe per dataset ──
layers = ["L9", "L14", "L18", "L23", "L27"]

probe_best = {}
probe_best_layer = {}
for key, label in datasets:
    d = probes[key]
    best_acc = 0
    best_name = ""
    # Check last-token per layer
    for layer in layers:
        acc = d.get(f"last_linear_{layer}", {}).get("balanced_accuracy", 0)
        if acc > best_acc:
            best_acc = acc
            best_name = f"last_{layer}"
    # Check concat
    acc = d.get("last_linear_concat", {}).get("balanced_accuracy", 0)
    if acc > best_acc:
        best_acc = acc
        best_name = "last_concat"
    # Check mean-pool per layer
    for layer in layers:
        acc = d.get(f"mean_linear_{layer}", {}).get("balanced_accuracy", 0)
        if acc > best_acc:
            best_acc = acc
            best_name = f"mean_{layer}"
    acc = d.get("mean_linear_concat", {}).get("balanced_accuracy", 0)
    if acc > best_acc:
        best_acc = acc
        best_name = "mean_concat"
    probe_best[key] = best_acc
    probe_best_layer[key] = best_name

llm_scores = {key: llm[key]["balanced_accuracy"] for key, _ in datasets}

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.3

bars_probe = ax.bar(x - width/2, [probe_best[k] for k, _ in datasets],
                     width, label="Linear Probe (best)", color="#4C72B0", zorder=3)
bars_llm = ax.bar(x + width/2, [llm_scores[k] for k, _ in datasets],
                   width, label="LLM Monitor (Gemini 3 Flash)", color="#DD8452", zorder=3)

# Chance line
ax.axhline(y=0.5, color="#888888", linestyle="--", linewidth=1, label="Chance", zorder=2)

# Labels on bars
for bar in bars_probe:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4C72B0")
for bar in bars_llm:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#DD8452")

# Probe layer annotation
for i, (key, _) in enumerate(datasets):
    layer = probe_best_layer[key]
    ax.text(x[i] - width/2, 0.48, layer.replace("last_", "").replace("mean_", "μ"),
            ha="center", va="top", fontsize=6.5, color="#4C72B0", style="italic")

ax.set_ylabel("Balanced Accuracy", fontsize=12)
ax.set_title("Baseline Comparison: Linear Probes vs LLM Monitor", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in datasets], fontsize=9)
ax.set_ylim(0.4, 0.85)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/baseline_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to data/baseline_comparison.png")

# ── Print summary table ──
print("\n" + "=" * 80)
print(f"{'Dataset':<25} {'Best Probe':>12} {'Layer':>12} {'LLM Monitor':>12} {'Winner':>10}")
print("-" * 80)
for key, label in datasets:
    p = probe_best[key]
    l = llm_scores[key]
    layer = probe_best_layer[key]
    winner = "Probe" if p > l else "LLM" if l > p else "Tie"
    print(f"{key:<25} {p:>12.3f} {layer:>12} {l:>12.3f} {winner:>10}")
print("=" * 80)
