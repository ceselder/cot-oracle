"""Generate averaged 5-line plot and stats from v5 verdict oscillation results.

v5: Trained oracle uses L18 only, Adam's AO uses 3 layers.
Plus sentence-only variants for both.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("eval_logs/verdict_oscillation_v5")
plots_dir = Path("plots/verdict_oscillation")
plots_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "results.json") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions\n")

# Collect scores
keys = ["model_scores", "oracle_scores", "adam_scores", "oracle_sent_scores", "adam_sent_scores"]
all_scores = {k: [] for k in keys}

for r in results:
    for k in keys:
        all_scores[k].append(r[k])

# Flatten for correlations
def flatten_paired(a_lists, b_lists):
    fa, fb = [], []
    for a, b in zip(a_lists, b_lists):
        n = min(len(a), len(b))
        fa.extend(a[:n])
        fb.extend(b[:n])
    return np.array(fa), np.array(fb)

model_flat, oracle_flat = flatten_paired(all_scores["model_scores"], all_scores["oracle_scores"])
_, adam_flat = flatten_paired(all_scores["model_scores"], all_scores["adam_scores"])
_, oracle_sent_flat = flatten_paired(all_scores["model_scores"], all_scores["oracle_sent_scores"])
_, adam_sent_flat = flatten_paired(all_scores["model_scores"], all_scores["adam_sent_scores"])

corrs = {
    "Trained (L18, cumul)": np.corrcoef(model_flat, oracle_flat)[0, 1],
    "Adam's AO (3L, cumul)": np.corrcoef(model_flat, adam_flat)[0, 1],
    "Trained (L18, sent-only)": np.corrcoef(model_flat, oracle_sent_flat)[0, 1],
    "Adam's AO (3L, sent-only)": np.corrcoef(model_flat, adam_sent_flat)[0, 1],
}

# Final verdict agreements
def count_agreements(a_lists, b_lists):
    agree = 0
    total = 0
    for a, b in zip(a_lists, b_lists):
        if a and b:
            agree += int((a[-1] > 0.5) == (b[-1] > 0.5))
            total += 1
    return agree, total

print("=== V5 Statistics (swapped layers + sentence-only) ===")
for name, r in corrs.items():
    agree, total = count_agreements(all_scores["model_scores"],
        all_scores["oracle_scores"] if "Trained" in name and "cumul" in name else
        all_scores["adam_scores"] if "Adam" in name and "cumul" in name else
        all_scores["oracle_sent_scores"] if "Trained" in name else
        all_scores["adam_sent_scores"])
    print(f"  {name:30s}: r={r:.3f}, final agree={agree}/{total}")

print(f"\n  Model avg P(guilty):          {model_flat.mean():.3f}")
print(f"  Trained (L18, cumul) avg:      {oracle_flat.mean():.3f}")
print(f"  Adam's AO (3L, cumul) avg:     {adam_flat.mean():.3f}")
print(f"  Trained (L18, sent-only) avg:  {oracle_sent_flat.mean():.3f}")
print(f"  Adam's AO (3L, sent-only) avg: {adam_sent_flat.mean():.3f}")

# Averaged plot
n_points = 50
score_keys = ["model_scores", "oracle_scores", "adam_scores", "oracle_sent_scores", "adam_sent_scores"]
interp = {k: np.zeros(n_points) for k in score_keys}

for i in range(len(results)):
    n = min(len(all_scores[k][i]) for k in score_keys)
    if n < 2:
        continue
    x_orig = np.linspace(0, 1, n)
    x_interp = np.linspace(0, 1, n_points)
    for k in score_keys:
        interp[k] += np.interp(x_interp, x_orig, all_scores[k][i][:n])

for k in score_keys:
    interp[k] /= len(results)

fig, ax = plt.subplots(figsize=(14, 7))
x = np.linspace(0, 100, n_points)

ax.plot(x, interp["model_scores"], 'b-', linewidth=2.5, label='Model Verdict')
ax.plot(x, interp["oracle_scores"], 'r-', linewidth=2, label=f'Trained L18 cumul (r={corrs["Trained (L18, cumul)"]:.3f})')
ax.plot(x, interp["adam_scores"], 'g-', linewidth=2, label=f'Adam 3L cumul (r={corrs["Adam\'s AO (3L, cumul)"]:.3f})')
ax.plot(x, interp["oracle_sent_scores"], 'r--', linewidth=1.5, alpha=0.6,
        label=f'Trained L18 sent-only (r={corrs["Trained (L18, sent-only)"]:.3f})')
ax.plot(x, interp["adam_sent_scores"], 'g--', linewidth=1.5, alpha=0.6,
        label=f'Adam 3L sent-only (r={corrs["Adam\'s AO (3L, sent-only)"]:.3f})')

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('CoT Progress (%)', fontsize=12)
ax.set_ylabel('P(guilty)', fontsize=12)
ax.set_title(f'V5: Swapped Layers + Sentence-Only (n={len(results)} questions)', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "averaged_v5.png", dpi=150)
plt.savefig(results_dir / "averaged_v5.png", dpi=150)
print(f"\nSaved averaged plot to {plots_dir / 'averaged_v5.png'}")

# Copy per-question plots
import shutil
for f in results_dir.glob("q*_oscillation.png"):
    shutil.copy2(f, plots_dir / f"v5_{f.name}")
print("Copied per-question plots to plots/verdict_oscillation/")
