#!/usr/bin/env python3
"""Plot probe OOD generalization: hint_admission ↔ truthfulqa_hint transfer."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams["font.family"] = "DejaVu Sans"

with open("data/probe_baseline_results_v3.json") as f:
    d = json.load(f)

# ── Data ──
layers = ["L9", "L18", "L27"]
layer_labels = ["Layer 9\n(25%)", "Layer 18\n(50%)", "Layer 27\n(75%)"]

# In-domain scores
ha_in = [d["hint_admission"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]
tq_in = [d["truthfulqa_hint"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]

# Cross-domain scores
ha_to_tq = [d["cross_hint_admission_to_truthfulqa_hint"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]
tq_to_ha = [d["cross_truthfulqa_hint_to_hint_admission"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]

# LLM monitor scores (constant across layers — black-box, no layer concept)
llm_ha = 0.847  # hint_admission
llm_tq = 0.746  # truthfulqa_hint

# ── Compute summary stats ──
retentions_ha_tq = [ha_to_tq[i] / tq_in[i] for i in range(len(layers))]
retentions_tq_ha = [tq_to_ha[i] / ha_in[i] for i in range(len(layers))]

# ── Plot ──
fig = plt.figure(figsize=(14, 5.5))

# Three panels: left = test on TQ, middle = test on HA, right = summary
gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 2], wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2])

x = np.arange(len(layers))
width = 0.3

colors = {
    "in_domain": "#4C72B0",
    "transfer":  "#2ca02c",
    "llm":       "#DD8452",
    "retention":  "#7f7f7f",
}

# ── Left panel: test on TruthfulQA Hint ──
bars1 = ax1.bar(x - width/2, tq_in, width, color=colors["in_domain"],
                label="In-domain", zorder=3)
bars2 = ax1.bar(x + width/2, ha_to_tq, width, color=colors["transfer"],
                label="OOD transfer", zorder=3)
ax1.axhline(y=llm_tq, color=colors["llm"], linestyle="--", linewidth=2,
            label=f"LLM Monitor ({llm_tq:.2f})", zorder=2)
ax1.axhline(y=0.5, color="#888888", linestyle=":", linewidth=1, zorder=1)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Retention labels on transfer bars
for i in range(len(layers)):
    ret = retentions_ha_tq[i] * 100
    ax1.text(x[i] + width/2, ha_to_tq[i] - 0.025, f"{ret:.0f}%",
            ha="center", va="top", fontsize=8, color=colors["transfer"],
            fontweight="bold", style="italic")

# Winner highlight
for i in range(len(layers)):
    best_bar = bars1[i] if tq_in[i] >= ha_to_tq[i] else bars2[i]
    best_bar.set_edgecolor("black")
    best_bar.set_linewidth(2)

ax1.set_title("Eval: TruthfulQA Hint", fontsize=12, fontweight="bold")
ax1.set_ylabel("Balanced Accuracy", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(layer_labels, fontsize=9)
ax1.set_ylim(0.42, 1.05)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Subtitle: train source
ax1.text(0.5, -0.18, "Train: Hint Admission (OOD) vs TruthfulQA (in-domain)",
         ha="center", transform=ax1.transAxes, fontsize=8, color="#666")

# ── Middle panel: test on Hint Admission ──
bars3 = ax2.bar(x - width/2, ha_in, width, color=colors["in_domain"],
                label="In-domain", zorder=3)
bars4 = ax2.bar(x + width/2, tq_to_ha, width, color=colors["transfer"],
                label="OOD transfer", zorder=3)
ax2.axhline(y=llm_ha, color=colors["llm"], linestyle="--", linewidth=2,
            label=f"LLM Monitor ({llm_ha:.2f})", zorder=2)
ax2.axhline(y=0.5, color="#888888", linestyle=":", linewidth=1, zorder=1)

for bars in [bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for i in range(len(layers)):
    ret = retentions_tq_ha[i] * 100
    ax2.text(x[i] + width/2, tq_to_ha[i] - 0.025, f"{ret:.0f}%",
            ha="center", va="top", fontsize=8, color=colors["transfer"],
            fontweight="bold", style="italic")

for i in range(len(layers)):
    best_bar = bars3[i] if ha_in[i] >= tq_to_ha[i] else bars4[i]
    best_bar.set_edgecolor("black")
    best_bar.set_linewidth(2)

ax2.set_title("Eval: Hint Admission", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(layer_labels, fontsize=9)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.setp(ax2.get_yticklabels(), visible=False)

ax2.text(0.5, -0.18, "Train: TruthfulQA (OOD) vs Hint Admission (in-domain)",
         ha="center", transform=ax2.transAxes, fontsize=8, color="#666")

# ── Right panel: retention summary ──
directions = ["HA \u2192 TQ", "TQ \u2192 HA"]
mean_ret = [np.mean(retentions_ha_tq) * 100, np.mean(retentions_tq_ha) * 100]
min_ret = [min(retentions_ha_tq) * 100, min(retentions_tq_ha) * 100]
max_ret = [max(retentions_ha_tq) * 100, max(retentions_tq_ha) * 100]

y_pos = np.arange(len(directions))
bar_h = 0.5

# Horizontal bars for mean retention
hbars = ax3.barh(y_pos, mean_ret, bar_h, color=[colors["transfer"], colors["transfer"]],
                 zorder=3, alpha=0.85)

# Error bars showing min-max range
for i in range(len(directions)):
    ax3.plot([min_ret[i], max_ret[i]], [y_pos[i], y_pos[i]],
             color="black", linewidth=2, zorder=4)
    ax3.plot([min_ret[i]], [y_pos[i]], "k|", markersize=10, zorder=4)
    ax3.plot([max_ret[i]], [y_pos[i]], "k|", markersize=10, zorder=4)

# Value labels
for i, bar in enumerate(hbars):
    w = bar.get_width()
    ax3.text(w + 1, bar.get_y() + bar.get_height()/2,
             f"{mean_ret[i]:.0f}%", ha="left", va="center",
             fontsize=11, fontweight="bold", color=colors["transfer"])

# Beats LLM?
beats_labels = []
best_ha_tq = max(ha_to_tq)
best_tq_ha = max(tq_to_ha)
for i, (best_transfer, llm_val, name) in enumerate([
    (best_ha_tq, llm_tq, "TQ"),
    (best_tq_ha, llm_ha, "HA"),
]):
    if best_transfer > llm_val:
        beats_labels.append(f"Beats LLM ({best_transfer:.2f} > {llm_val:.2f})")
    else:
        beats_labels.append(f"Below LLM ({best_transfer:.2f} < {llm_val:.2f})")

for i, label in enumerate(beats_labels):
    color = colors["transfer"] if "Beats" in label else colors["llm"]
    ax3.text(50, y_pos[i] - 0.32, label,
             ha="center", va="top", fontsize=7.5, color=color, style="italic")

ax3.axvline(x=100, color="#888888", linestyle=":", linewidth=1, zorder=1)
ax3.set_xlim(50, 110)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(directions, fontsize=11, fontweight="bold")
ax3.set_xlabel("Retention (%)", fontsize=10)
ax3.set_title("OOD Retention", fontsize=12, fontweight="bold")
ax3.grid(axis="x", alpha=0.3, zorder=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.suptitle("Probe Out-of-Distribution Generalization: Hint Admission \u2194 TruthfulQA Hint",
             fontsize=13, fontweight="bold", y=1.02)
plt.savefig("data/cross_eval_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to data/cross_eval_comparison.png")

# ── Print summary ──
print("\nTransfer retention (% of in-domain performance):")
for i, l in enumerate(layers):
    ret1 = retentions_ha_tq[i] * 100
    ret2 = retentions_tq_ha[i] * 100
    print(f"  {l}: HA\u2192TQ {ha_to_tq[i]:.3f}/{tq_in[i]:.3f} = {ret1:.0f}%   TQ\u2192HA {tq_to_ha[i]:.3f}/{ha_in[i]:.3f} = {ret2:.0f}%")

print(f"\nMean retention: HA\u2192TQ {np.mean(retentions_ha_tq)*100:.0f}%  TQ\u2192HA {np.mean(retentions_tq_ha)*100:.0f}%")

print(f"\nOOD transfer probe vs LLM monitor:")
print(f"  HA\u2192TQ: best={best_ha_tq:.3f} vs LLM={llm_tq:.3f}  {'BEATS LLM' if best_ha_tq > llm_tq else 'below LLM'}")
print(f"  TQ\u2192HA: best={best_tq_ha:.3f} vs LLM={llm_ha:.3f}  {'BEATS LLM' if best_tq_ha > llm_ha else 'below LLM'}")
