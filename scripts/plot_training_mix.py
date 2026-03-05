"""
Diagram comparing Original AO vs Our Current Training Mix.
Shows both sequence counts and token counts (measured from training_data_cache).
Uses layout="constrained". Saves to plots/training_mix.png.

Average inp tokens per sequence are measured from /ceph/scratch/jbauer/training_data_cache/:
  cot_hint_admission:            ~330  (avg of two cache files)
  cot_atypical_answer:            206
  cot_reasoning_termination:      300
  cot_correctness:                329
  cot_decorative:                 154
  cot_backtrack_prediction:       414
  cot_sycophancy:                 218
  cot_next_step  (futurelens):    344
  cot_past_step  (pastlens):      345
  cot_answer_trajectory:          257
  chunked_convqa:                 387
  fineweb_futurelens:             181
  fineweb_pastlens:               179
  latentqa (weighted avg):        181
  classification (sst2/ag/snli):   57
  PastLens AO (est from fineweb): 200
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "context_pred": "#4A90D9",
    "open_qa":      "#8E44AD",
    "classification":"#27AE60",
    "binary_cot":   "#E67E22",
    "cot_traj":     "#E74C3C",
    "fineweb":      "#5DADE2",
}
LABELS = {
    "context_pred":   "Context Prediction (PastLens / FutureLens)",
    "open_qa":        "Open-ended QA (LatentQA / Chunked QA / SQA)",
    "classification": "Classification Benchmarks",
    "binary_cot":     "Binary CoT Analysis (novel tasks)",
    "cot_traj":       "CoT Trajectory",
    "fineweb":        "FineWeb Readout",
}

# Each entry: (display_label, n_sequences_effective, category, avg_inp_tokens)
ORIGINAL_AO = [
    ("PastLens (single token)",   100_000, "context_pred",   200),
    ("PastLens (multi-token)",    100_000, "context_pred",   200),
    ("LatentQA",                  100_000, "open_qa",        181),
    ("Classification\n(7 datasets × 6K × 2 variants)",
                                   84_000, "classification",  70),
]

OUR_MIX = [
    # Binary CoT analysis
    ("Hint Admission",          3_000 * 2,  "binary_cot",   330),
    ("Atypical Answer",        20_000 * 1,  "binary_cot",   206),
    ("Reasoning Termination",  12_000 * 2,  "binary_cot",   300),
    ("Correctness",             7_800 * 2,  "binary_cot",   329),
    ("Decorative CoT",          4_100 * 2,  "binary_cot",   154),
    ("Backtrack Prediction",   12_000 * 2,  "binary_cot",   414),
    ("Sycophancy",             10_000 * 2,  "binary_cot",   218),
    ("TruthfulQA Hint",         3_800 * 2,  "binary_cot",   330),
    # CoT trajectory / generative
    ("Answer Trajectory",      30_000 * 1,  "cot_traj",     257),
    ("Resampling Importance",     468 * 4,  "cot_traj",     400),
    # Context prediction (CoT)
    ("FutureLens",             20_000 * 1,  "context_pred", 344),
    ("PastLens",               20_000 * 1,  "context_pred", 345),
    # Open QA
    ("Chunked Conv QA",        27_000 * 1,  "open_qa",      387),
    ("Chunked Comp QA",        30_000 * 1,  "open_qa",      387),
    ("SQA",                     8_500 * 2,  "open_qa",      180),
    # FineWeb readout
    ("FutureLens (FineWeb)",   20_000,      "fineweb",      181),
    ("PastLens (FineWeb)",     20_000,      "fineweb",      179),
    # Classification
    ("Classification\n(SST2 / AGNews / SNLI)",
                               15_000,      "classification", 57),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def aggregate(data, metric="seq"):
    """Sum sequences (metric='seq') or tokens (metric='tok') per category."""
    totals: dict[str, int] = {}
    for _, n, cat, avg_tok in data:
        val = n if metric == "seq" else n * avg_tok
        totals[cat] = totals.get(cat, 0) + val
    return totals


CAT_ORDER = ["context_pred", "fineweb", "open_qa", "classification", "binary_cot", "cot_traj"]


def stacked_bar(ax, totals: dict[str, int], y: float, bar_h: float,
                scale: float = 1.0, fmt: str = "{:.0f}K", thresh: float = 8_000,
                unit: str = "total"):
    """Draw one horizontal stacked bar; values are divided by scale for display."""
    x = 0
    for cat in CAT_ORDER:
        n = totals.get(cat, 0)
        if n == 0:
            continue
        ax.barh(y, n, left=x, height=bar_h,
                color=COLORS[cat], edgecolor="white", linewidth=0.6)
        if n > thresh:
            label = fmt.format(n / scale)
            ax.text(x + n / 2, y, label, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        x += n
    total = sum(totals.values())
    ax.text(x + ax.get_xlim()[1] * 0.01, y,
            fmt.format(total / scale) + f" {unit}",
            va="center", fontsize=8.5, color="#333", fontweight="semibold")


def per_task_bars(ax, data, y_start: float, bar_h: float = 0.38, gap: float = 0.44,
                  metric: str = "seq", scale: float = 1.0, fmt: str = "{:.1f}K"):
    y = y_start
    for (label, n, cat, avg_tok) in data:
        val = n if metric == "seq" else n * avg_tok
        ax.barh(y, val, height=bar_h, color=COLORS[cat],
                edgecolor="white", linewidth=0.5, alpha=0.92)
        display_label = label.replace("\n", " ")
        ax.text(-ax.get_xlim()[1] * 0.03, y, display_label,
                ha="right", va="center", fontsize=7.2, color="#333")
        if val >= 1_500 * scale:
            ax.text(val + ax.get_xlim()[1] * 0.015, y, fmt.format(val / scale),
                    va="center", fontsize=6.8, color="#555")
        y -= gap
    return y


# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 2,
    figsize=(15, 15),
    layout="constrained",
    gridspec_kw={"height_ratios": [1, 1, 3], "width_ratios": [1, 1]},
)
fig.patch.set_facecolor("#FAFAFA")

# Unpack axes
ax_seq_orig, ax_seq_ours = axes[0]       # row 0: sequence summary bars
ax_tok_orig, ax_tok_ours = axes[1]       # row 1: token summary bars
ax_detail_orig, ax_detail_ours = axes[2] # row 2: per-task detail

# ── Row 0: Sequence summary ──────────────────────────────────────────────────
for ax, dataset, title in [
    (ax_seq_orig, ORIGINAL_AO, "Original AO — sequences"),
    (ax_seq_ours, OUR_MIX,     "Our Mix — sequences"),
]:
    totals = aggregate(dataset, "seq")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 440_000)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#ddd", linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    stacked_bar(ax, totals, y=1.0, bar_h=0.55, scale=1_000,
                fmt="{:.0f}K", thresh=8_000, unit="seqs")

# ── Row 1: Token summary ─────────────────────────────────────────────────────
tok_max = max(
    sum(aggregate(ORIGINAL_AO, "tok").values()),
    sum(aggregate(OUR_MIX,     "tok").values()),
) * 1.18

for ax, dataset, title in [
    (ax_tok_orig, ORIGINAL_AO, "Original AO — oracle input tokens"),
    (ax_tok_ours, OUR_MIX,     "Our Mix — oracle input tokens"),
]:
    totals = aggregate(dataset, "tok")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, tok_max)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#ddd", linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    stacked_bar(ax, totals, y=1.0, bar_h=0.55, scale=1e6,
                fmt="{:.1f}M", thresh=tok_max * 0.04, unit="tokens")

# ── Row 2: Per-task detail ───────────────────────────────────────────────────
for ax, dataset, title in [
    (ax_detail_orig, ORIGINAL_AO, "Original AO — per dataset (sequences)"),
    (ax_detail_ours, OUR_MIX,     "Our Mix — per task (sequences, eff. after ×epochs)"),
]:
    n_rows = len(dataset)
    y_top = (n_rows - 1) * 0.44
    # set x-lim first so per_task_bars can use it for label offsets (sequences)
    x_max = max(n for _, n, _, avg in dataset)
    ax.set_xlim(-x_max * 0.38, x_max * 1.22)
    ax.set_ylim(-0.4, y_top + 0.4)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{max(x,0)/1000:.0f}K"))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    ax.set_xlabel("Sequences", fontsize=8)
    per_task_bars(ax, dataset, y_top, metric="seq", scale=1_000, fmt="{:.1f}K")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Training Mix: Original AO vs Our CoT Oracle\n"
    "Sequence Counts & Oracle Input Token Counts by Task Category",
    fontsize=13, fontweight="bold", color="#222",
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=COLORS[k], label=LABELS[k])
    for k in CAT_ORDER
]
fig.legend(
    handles=legend_patches,
    loc="outside lower center",
    ncol=3,
    fontsize=8.5,
    frameon=True,
    framealpha=0.9,
    edgecolor="#ccc",
)

out_path = Path(__file__).parent.parent / "plots" / "training_mix.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved → {out_path}")
plt.close()
