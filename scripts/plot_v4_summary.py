"""Generate averaged 3-line plot and stats from v4 verdict oscillation results."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("eval_logs/verdict_oscillation_v4")
plots_dir = Path("plots/verdict_oscillation")
plots_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "results.json") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions\n")

# Collect all scores
all_model = []
all_oracle = []
all_adam = []
agreements_oracle = 0
agreements_adam = 0
agreements_model_oracle_final = 0
total = 0

for r in results:
    ms = r["model_scores"]
    os_ = r["oracle_scores"]
    adam = r["adam_scores"]
    all_model.append(ms)
    all_oracle.append(os_)
    all_adam.append(adam)

    # Final verdict agreement
    model_guilty = ms[-1] > 0.5
    oracle_guilty = os_[-1] > 0.5
    adam_guilty = adam[-1] > 0.5
    if model_guilty == oracle_guilty:
        agreements_model_oracle_final += 1
    if model_guilty == adam_guilty:
        agreements_adam += 1
    total += 1

# Pearson correlations (flatten all sentence-level scores)
flat_model = []
flat_oracle = []
flat_adam = []
for ms, os_, adam in zip(all_model, all_oracle, all_adam):
    n = min(len(ms), len(os_), len(adam))
    flat_model.extend(ms[:n])
    flat_oracle.extend(os_[:n])
    flat_adam.extend(adam[:n])

flat_model = np.array(flat_model)
flat_oracle = np.array(flat_oracle)
flat_adam = np.array(flat_adam)

corr_oracle = np.corrcoef(flat_model, flat_oracle)[0, 1]
corr_adam = np.corrcoef(flat_model, flat_adam)[0, 1]

print("=== V4 Statistics ===")
print(f"Trained Oracle vs Model: r={corr_oracle:.3f}, final agreement={agreements_model_oracle_final}/{total}")
print(f"Adam's AO vs Model:      r={corr_adam:.3f}, final agreement={agreements_adam}/{total}")
print(f"Model avg P(guilty):     {flat_model.mean():.3f}")
print(f"Trained Oracle avg:      {flat_oracle.mean():.3f}")
print(f"Adam's AO avg:           {flat_adam.mean():.3f}")

# Averaged plot: interpolate all questions to normalized [0, 1] x-axis
n_points = 50
interp_model = np.zeros(n_points)
interp_oracle = np.zeros(n_points)
interp_adam = np.zeros(n_points)

for ms, os_, adam in zip(all_model, all_oracle, all_adam):
    n = min(len(ms), len(os_), len(adam))
    x_orig = np.linspace(0, 1, n)
    x_interp = np.linspace(0, 1, n_points)
    interp_model += np.interp(x_interp, x_orig, ms[:n])
    interp_oracle += np.interp(x_interp, x_orig, os_[:n])
    interp_adam += np.interp(x_interp, x_orig, adam[:n])

interp_model /= len(results)
interp_oracle /= len(results)
interp_adam /= len(results)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.linspace(0, 100, n_points)
ax.plot(x, interp_model, 'b-', linewidth=2, label='Model Verdict')
ax.plot(x, interp_oracle, 'r-', linewidth=2, label='Trained Oracle')
ax.plot(x, interp_adam, 'g-', linewidth=2, label="Adam's AO")
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('CoT Progress (%)', fontsize=12)
ax.set_ylabel('P(guilty)', fontsize=12)
ax.set_title(f'Average Verdict Oscillation (n={len(results)} questions)\n'
             f'Trained Oracle r={corr_oracle:.3f} | Adam AO r={corr_adam:.3f}', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "averaged_v4.png", dpi=150)
plt.savefig(results_dir / "averaged_v4.png", dpi=150)
print(f"\nSaved averaged plot to {plots_dir / 'averaged_v4.png'}")

# Also copy per-question plots to plots dir
import shutil
for f in results_dir.glob("q*_oscillation.png"):
    shutil.copy2(f, plots_dir / f"v4_{f.name}")
shutil.copy2(results_dir / "summary.png", plots_dir / "v4_summary.png")
print("Copied per-question plots to plots/verdict_oscillation/")
