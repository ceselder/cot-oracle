#!/usr/bin/env python3
"""Plot baseline comparison: CoT Oracle vs Linear Probe vs Attention Probe vs LLM Monitor."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Load data ──
with open("data/probe_baseline_results_v2.json") as f:
    probes_v2 = json.load(f)
with open("data/probe_baseline_results_quick.json") as f:
    probes_quick = json.load(f)
with open("data/llm_monitor_baselines.json") as f:
    llm = json.load(f)

# ── Datasets (classification only — answer_trajectory is regression/MAE) ──
datasets = [
    ("hint_admission",       "Hint\nAdmission"),
    ("atypical_answer",      "Atypical\nAnswer"),
    ("decorative_cot",       "Decorative\nCoT"),
    ("sycophancy",           "Sycophancy"),
    ("truthfulqa_verb",      "TruthfulQA\n(Verbalized)"),
    ("truthfulqa_unverb",    "TruthfulQA\n(Unverbalized)"),
    ("backtrack_prediction", "Backtrack\nPrediction"),
    ("correctness",          "Correctness"),
]

# ── Merge probe dicts ──
all_probes = {**probes_v2, **probes_quick}

# ── Extract best linear probe per dataset ──
layers = ["L9", "L14", "L18", "L21", "L23", "L27"]

def best_linear(d):
    best_acc, best_name = 0, ""
    for pfx in ["mean_linear_", "last_linear_"]:
        for layer in layers:
            acc = d.get(f"{pfx}{layer}", {}).get("balanced_accuracy", 0)
            if acc > best_acc:
                best_acc, best_name = acc, f"{pfx}{layer}"
        acc = d.get(f"{pfx}concat", {}).get("balanced_accuracy", 0)
        if acc > best_acc:
            best_acc, best_name = acc, f"{pfx}concat"
    return best_acc, best_name

def best_attn(d):
    best_acc, best_name = 0, ""
    for k, v in d.items():
        if k.startswith("attn_"):
            acc = v.get("balanced_accuracy", 0)
            if acc > best_acc:
                best_acc, best_name = acc, k
    return best_acc if best_acc > 0 else None, best_name

# ── Collect per-dataset scores ──
linear_scores = {}
linear_labels = {}
attn_scores = {}
llm_scores = {}
oracle_scores = {}  # TODO: fill in when oracle results exist

for key, label in datasets:
    d = all_probes[key]
    lp, ln = best_linear(d)
    linear_scores[key] = lp
    linear_labels[key] = ln
    ap, an = best_attn(d)
    attn_scores[key] = ap  # None if no attention probe run
    llm_scores[key] = llm[key]["balanced_accuracy"] if key in llm else None
    oracle_scores[key] = None  # placeholder

# ── Plot ──
fig, ax = plt.subplots(figsize=(14, 6.5))

x = np.arange(len(datasets))
n_bars = 4
width = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

colors = {
    "oracle":  "#2ca02c",   # green — ours
    "linear":  "#4C72B0",   # blue
    "attn":    "#9467bd",   # purple
    "llm":     "#DD8452",   # orange
}

# Gather bar data (None → 0 height, hatched)
oracle_vals = [oracle_scores[k] for k, _ in datasets]
linear_vals = [linear_scores[k] for k, _ in datasets]
attn_vals   = [attn_scores[k] for k, _ in datasets]
llm_vals    = [llm_scores[k] for k, _ in datasets]

def draw_bars(positions, values, color, label, zorder=3):
    """Draw bars, using hatch pattern for None (missing) values."""
    present = [v if v is not None else 0 for v in values]
    bars = ax.bar(positions, present, width, color=color, zorder=zorder, label=label)
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val is None:
            bar.set_height(0.52)  # small stub above chance
            bar.set_color("white")
            bar.set_edgecolor(color)
            bar.set_linewidth(1.5)
            bar.set_hatch("///")
            bar.set_alpha(0.6)
            ax.text(bar.get_x() + bar.get_width()/2, 0.53, "?",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=color, zorder=4)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=color, zorder=4)
    return bars

draw_bars(x + offsets[0], oracle_vals, colors["oracle"], "CoT Oracle (Ours)")
draw_bars(x + offsets[1], linear_vals, colors["linear"], "Linear Probe (best)")
draw_bars(x + offsets[2], attn_vals,   colors["attn"],   "Attention Probe")
draw_bars(x + offsets[3], llm_vals,    colors["llm"],    "LLM Monitor (Gemini 3 Flash)")

# Chance line
ax.axhline(y=0.5, color="#888888", linestyle="--", linewidth=1, label="Chance", zorder=2)

# Probe layer annotation (under the linear probe bars)
for i, (key, _) in enumerate(datasets):
    layer = linear_labels[key].replace("last_linear_", "").replace("mean_linear_", "μ")
    ax.text(x[i] + offsets[1], 0.48, layer,
            ha="center", va="top", fontsize=5.5, color=colors["linear"], style="italic")

# Correctness footnote
ax.annotate("*sees answer", xy=(x[7] + offsets[3], 0.981), xytext=(x[7] + offsets[3] + 0.15, 0.93),
            fontsize=6, color=colors["llm"], style="italic",
            arrowprops=dict(arrowstyle="-", color=colors["llm"], lw=0.5))

ax.set_ylabel("Balanced Accuracy", fontsize=12)
ax.set_title("Baseline Comparison Across Datasets", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in datasets], fontsize=8.5)
ax.set_ylim(0.42, 1.05)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.3, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("data/baseline_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to data/baseline_comparison.png")

# ── Print summary table ──
print("\n" + "=" * 95)
print(f"{'Dataset':<22} {'CoT Oracle':>11} {'Linear Probe':>13} {'Attn Probe':>11} {'LLM Monitor':>12} {'Winner':>10}")
print("-" * 95)
for key, label in datasets:
    o = oracle_scores[key]
    l = linear_scores[key]
    a = attn_scores[key]
    m = llm_scores[key]
    os_ = f"{o:.3f}" if o is not None else "?"
    ls_ = f"{l:.3f}"
    as_ = f"{a:.3f}" if a is not None else "?"
    ms_ = f"{m:.3f}" if m is not None else "?"
    # Winner among available scores
    available = {"Linear": l}
    if a is not None: available["Attn"] = a
    if m is not None: available["LLM"] = m
    if o is not None: available["Oracle"] = o
    winner = max(available, key=available.get)
    print(f"{key:<22} {os_:>11} {ls_:>13} ({linear_labels[key].split('_')[-1]:>6}) {as_:>11} {ms_:>12} {winner:>10}")
print("=" * 95)
