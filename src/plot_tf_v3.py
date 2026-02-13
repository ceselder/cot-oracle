"""Plot AO distance accuracy results (teacher forcing, 3 conditions)."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

with open("/home/celeste/Documents/side-projects/cot-oracle/results/ao_distance_tf_v3.json") as f:
    data = json.load(f)

curves = data["curves"]
distances = list(range(1, 51))

# Color scheme
COLORS = {
    "baseline": "#888888",
    "AO 50%": "#2196F3",
    "AO 25+50+75%": "#E91E63",
}

def get_vals(name, metric):
    return [curves[name][metric][str(d)] for d in distances]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("AO Token Prediction Accuracy vs Distance (Teacher Forcing, n=200)\nQwen3-8B + PastLens AO", fontsize=14, fontweight="bold")

sources = ["FineWeb", "CoT"]
metrics = [("top1", "Top-1 Accuracy (%)"), ("top5", "Top-5 Accuracy (%)"), ("log_prob", "Mean Log Probability")]

for row, source in enumerate(sources):
    for col, (metric, ylabel) in enumerate(metrics):
        ax = axes[row, col]

        baseline_key = f"{source} baseline"
        ao50_key = f"{source} AO 50%"
        ao3_key = f"{source} AO 25+50+75%"

        baseline_vals = get_vals(baseline_key, metric)
        ao50_vals = get_vals(ao50_key, metric)
        ao3_vals = get_vals(ao3_key, metric)

        ax.plot(distances, baseline_vals, color=COLORS["baseline"], linewidth=2, label="Baseline (no activation)", alpha=0.7)
        ax.plot(distances, ao50_vals, color=COLORS["AO 50%"], linewidth=2, label="AO 50% (1 position)")
        ax.plot(distances, ao3_vals, color=COLORS["AO 25+50+75%"], linewidth=2, label="AO 25+50+75% (3 positions)")

        ax.set_xlabel("Distance (tokens ahead)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{source} — {ylabel.split('(')[0].strip()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if metric != "log_prob":
            ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("/home/celeste/Documents/side-projects/cot-oracle/results/ao_distance_tf_v3.png", dpi=150, bbox_inches="tight")
print("Saved main plot")

# Now plot the DELTA over baseline (the real signal)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("AO Improvement Over Baseline (Teacher Forcing)\nDelta = AO accuracy - baseline accuracy", fontsize=13, fontweight="bold")

for col, source in enumerate(sources):
    ax = axes2[col]

    baseline = np.array(get_vals(f"{source} baseline", "top1"))
    ao50 = np.array(get_vals(f"{source} AO 50%", "top1"))
    ao3 = np.array(get_vals(f"{source} AO 25+50+75%", "top1"))

    delta_50 = ao50 - baseline
    delta_3 = ao3 - baseline

    ax.plot(distances, delta_50, color=COLORS["AO 50%"], linewidth=2, label="AO 50% - baseline")
    ax.plot(distances, delta_3, color=COLORS["AO 25+50+75%"], linewidth=2, label="AO 25+50+75% - baseline")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Distance (tokens ahead)")
    ax.set_ylabel("Delta Top-1 Accuracy (pp)")
    ax.set_title(f"{source}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate d=1
    ax.annotate(f"d=1: +{delta_50[0]:.1f}pp", xy=(1, delta_50[0]), fontsize=9,
                xytext=(5, delta_50[0]+3), arrowprops=dict(arrowstyle="->", color=COLORS["AO 50%"]),
                color=COLORS["AO 50%"])
    ax.annotate(f"d=1: +{delta_3[0]:.1f}pp", xy=(1, delta_3[0]), fontsize=9,
                xytext=(5, delta_3[0]-5), arrowprops=dict(arrowstyle="->", color=COLORS["AO 25+50+75%"]),
                color=COLORS["AO 25+50+75%"])

plt.tight_layout()
plt.savefig("/home/celeste/Documents/side-projects/cot-oracle/results/ao_distance_tf_v3_delta.png", dpi=150, bbox_inches="tight")
print("Saved delta plot")

# Print summary table
print("\n=== KEY NUMBERS ===")
for source in sources:
    baseline = get_vals(f"{source} baseline", "top1")
    ao50 = get_vals(f"{source} AO 50%", "top1")
    ao3 = get_vals(f"{source} AO 25+50+75%", "top1")

    print(f"\n{source}:")
    for d in [1, 2, 3, 5, 10, 20, 30, 50]:
        i = d - 1
        print(f"  d={d:2d}: baseline={baseline[i]:5.1f}%  AO50={ao50[i]:5.1f}% (+{ao50[i]-baseline[i]:+.1f})  AO3={ao3[i]:5.1f}% (+{ao3[i]-baseline[i]:+.1f})")
