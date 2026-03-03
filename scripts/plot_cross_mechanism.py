#!/usr/bin/env python3
"""Plot cross-mechanism probe transfer: same-mechanism vs cross-mechanism."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Load data ──
with open("data/probe_baseline_results_v3.json") as f:
    v3 = json.load(f)

with open("data/probe_cross_mechanism_results.json") as f:
    xm = json.load(f)

layers = ["L9", "L18", "L27"]
layer_labels = ["Layer 9\n(25%)", "Layer 18\n(50%)", "Layer 27\n(75%)"]

# In-domain scores (best of last-position)
ha_in = [v3["hint_admission"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]
tq_in = [v3["truthfulqa_hint"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]
sy_in = [xm["sycophancy"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]

# Same-mechanism transfer: HA → TQ (both hint tasks)
ha_to_tq = [v3["cross_hint_admission_to_truthfulqa_hint"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]

# Cross-mechanism transfer: HA → Sycophancy (different mechanism)
ha_to_sy = [xm["xmech_hint_admission_to_sycophancy"][f"last_linear_{l}"]["balanced_accuracy"] for l in layers]

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                gridspec_kw={"width_ratios": [3, 2]})

x = np.arange(len(layers))
width = 0.22

# Left panel: per-layer comparison
colors = {
    "in_ha":   "#4C72B0",  # blue
    "in_tq":   "#6baed6",  # light blue
    "same":    "#2ca02c",  # green
    "cross":   "#d62728",  # red
}

bars_ha = ax1.bar(x - 1.5*width, ha_in, width, color=colors["in_ha"],
                  label="In-domain: Hint Admission", zorder=3)
bars_tq = ax1.bar(x - 0.5*width, tq_in, width, color=colors["in_tq"],
                  label="In-domain: TruthfulQA Hint", zorder=3)
bars_same = ax1.bar(x + 0.5*width, ha_to_tq, width, color=colors["same"],
                    label="Same-mechanism: HA → TQ", zorder=3)
bars_cross = ax1.bar(x + 1.5*width, ha_to_sy, width, color=colors["cross"],
                     label="Cross-mechanism: HA → Sycophancy", zorder=3)

ax1.axhline(y=0.5, color="#888888", linestyle="--", linewidth=1, label="Chance", zorder=1)

# Value labels
for bars in [bars_ha, bars_tq, bars_same, bars_cross]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# Fade the cross-mechanism bars to emphasize the collapse
for bar in bars_cross:
    bar.set_edgecolor("#d62728")
    bar.set_linewidth(1.5)

ax1.set_title("Probe Transfer: Same vs Cross Mechanism", fontsize=12, fontweight="bold")
ax1.set_ylabel("Balanced Accuracy", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(layer_labels, fontsize=9)
ax1.set_ylim(0.35, 1.05)
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right panel: retention summary
# Same-mechanism retention (HA→TQ vs TQ in-domain)
same_ret = [ha_to_tq[i] / tq_in[i] * 100 for i in range(len(layers))]
# Cross-mechanism retention (HA→Syc vs Syc in-domain)
cross_ret = [ha_to_sy[i] / sy_in[i] * 100 for i in range(len(layers))]

categories = ["Same-mechanism\n(HA → TQ)", "Cross-mechanism\n(HA → Sycophancy)"]
means = [np.mean(same_ret), np.mean(cross_ret)]
mins = [min(same_ret), min(cross_ret)]
maxs = [max(same_ret), max(cross_ret)]

y_pos = np.arange(len(categories))
bar_h = 0.5

hbars = ax2.barh(y_pos, means, bar_h,
                 color=[colors["same"], colors["cross"]],
                 zorder=3, alpha=0.85)

# Min-max range bars
for i in range(len(categories)):
    ax2.plot([mins[i], maxs[i]], [y_pos[i], y_pos[i]],
             color="black", linewidth=2, zorder=4)
    ax2.plot([mins[i]], [y_pos[i]], "k|", markersize=10, zorder=4)
    ax2.plot([maxs[i]], [y_pos[i]], "k|", markersize=10, zorder=4)

# Value labels
for i, bar in enumerate(hbars):
    w = bar.get_width()
    ax2.text(w + 2, bar.get_y() + bar.get_height()/2,
             f"{means[i]:.0f}%", ha="left", va="center",
             fontsize=13, fontweight="bold",
             color=colors["same"] if i == 0 else colors["cross"])

ax2.axvline(x=50, color="#888888", linestyle="--", linewidth=1, zorder=1, label="Chance")
ax2.axvline(x=100, color="#888888", linestyle=":", linewidth=1, zorder=1)

ax2.set_xlim(0, 115)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(categories, fontsize=10, fontweight="bold")
ax2.set_xlabel("Retention (% of in-domain)", fontsize=10)
ax2.set_title("Transfer Retention", fontsize=12, fontweight="bold")
ax2.grid(axis="x", alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Linear Probes Are Mechanism-Specific, Not General Unfaithfulness Detectors",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/cross_mechanism_transfer.png", dpi=150, bbox_inches="tight")
print("Saved to data/cross_mechanism_transfer.png")

# ── Summary ──
print(f"\nSame-mechanism retention (HA → TQ): {np.mean(same_ret):.0f}% (range {min(same_ret):.0f}-{max(same_ret):.0f}%)")
print(f"Cross-mechanism retention (HA → Syc): {np.mean(cross_ret):.0f}% (range {min(cross_ret):.0f}-{max(cross_ret):.0f}%)")
print(f"\nPer-layer cross-mechanism (HA → Sycophancy):")
for i, l in enumerate(layers):
    print(f"  {l}: {ha_to_sy[i]:.3f} (in-domain syc: {sy_in[i]:.3f}, retention: {cross_ret[i]:.0f}%)")
