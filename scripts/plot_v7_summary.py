"""Generate stats and averaged plots from v7 verdict oscillation (2 prompt variants)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("eval_logs/verdict_oscillation_v7")
plots_dir = Path("plots/verdict_oscillation")
plots_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "results.json") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions\n")

prompt_names = ["yesno", "cot_aware"]

def flatten_paired(a_lists, b_lists):
    fa, fb = [], []
    for a, b in zip(a_lists, b_lists):
        n = min(len(a), len(b))
        fa.extend(a[:n])
        fb.extend(b[:n])
    return np.array(fa), np.array(fb)

def count_agreements(a_lists, b_lists):
    agree = total = 0
    for a, b in zip(a_lists, b_lists):
        if a and b:
            agree += int((a[-1] > 0.5) == (b[-1] > 0.5))
            total += 1
    return agree, total

model_lists = [r["model_scores"] for r in results]

print("=== V7 Statistics (yes/no prompts, normal layer configs) ===\n")
all_corrs = {}
for pname in prompt_names:
    oracle_lists = [r[f"oracle_{pname}"] for r in results]
    adam_lists = [r[f"adam_{pname}"] for r in results]

    mf, of = flatten_paired(model_lists, oracle_lists)
    _, af = flatten_paired(model_lists, adam_lists)

    r_oracle = np.corrcoef(mf, of)[0, 1]
    r_adam = np.corrcoef(mf, af)[0, 1]
    agree_o, total_o = count_agreements(model_lists, oracle_lists)
    agree_a, total_a = count_agreements(model_lists, adam_lists)

    all_corrs[f"Trained ({pname})"] = r_oracle
    all_corrs[f"Adam ({pname})"] = r_adam

    print(f"  Prompt: {pname}")
    print(f"    Trained oracle: r={r_oracle:.3f}, final agree={agree_o}/{total_o}, avg={of.mean():.3f}")
    print(f"    Adam's AO:      r={r_adam:.3f}, final agree={agree_a}/{total_a}, avg={af.mean():.3f}")
    print()

print(f"  Model avg P(guilty): {mf.mean():.3f}")

# Generate averaged plot per prompt
for pname in prompt_names:
    oracle_lists = [r[f"oracle_{pname}"] for r in results]
    adam_lists = [r[f"adam_{pname}"] for r in results]

    n_points = 50
    interp_m = np.zeros(n_points)
    interp_o = np.zeros(n_points)
    interp_a = np.zeros(n_points)
    count = 0

    for ms, os_, ads in zip(model_lists, oracle_lists, adam_lists):
        n = min(len(ms), len(os_), len(ads))
        if n < 2:
            continue
        x_orig = np.linspace(0, 1, n)
        x_interp = np.linspace(0, 1, n_points)
        interp_m += np.interp(x_interp, x_orig, ms[:n])
        interp_o += np.interp(x_interp, x_orig, os_[:n])
        interp_a += np.interp(x_interp, x_orig, ads[:n])
        count += 1

    interp_m /= count
    interp_o /= count
    interp_a /= count

    r_o = all_corrs[f"Trained ({pname})"]
    r_a = all_corrs[f"Adam ({pname})"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0, 100, n_points)
    ax.plot(x, interp_m, 'b-', linewidth=2.5, label='Model Verdict')
    ax.plot(x, interp_o, 'r-', linewidth=2, label=f'Trained Oracle (r={r_o:.3f})')
    ax.plot(x, interp_a, 'g-', linewidth=2, label=f"Adam's AO (r={r_a:.3f})")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('CoT Progress (%)', fontsize=12)
    ax.set_ylabel('P(guilty)', fontsize=12)
    ax.set_title(f'V7 Averaged — Prompt: "{pname}" (n={count})', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"averaged_v7_{pname}.png", dpi=150)
    plt.close()
    print(f"Saved {plots_dir / f'averaged_v7_{pname}.png'}")

# Combined 4-line plot
fig, ax = plt.subplots(figsize=(14, 7))
x = np.linspace(0, 100, n_points)

# Recompute for combined
for pname, style_o, style_a in [("yesno", "r-", "g-"), ("cot_aware", "r--", "g--")]:
    oracle_lists = [r[f"oracle_{pname}"] for r in results]
    adam_lists = [r[f"adam_{pname}"] for r in results]
    interp_o = np.zeros(n_points)
    interp_a = np.zeros(n_points)
    interp_m = np.zeros(n_points)
    count = 0
    for ms, os_, ads in zip(model_lists, oracle_lists, adam_lists):
        n = min(len(ms), len(os_), len(ads))
        if n < 2:
            continue
        x_orig = np.linspace(0, 1, n)
        x_interp = np.linspace(0, 1, n_points)
        interp_m += np.interp(x_interp, x_orig, ms[:n])
        interp_o += np.interp(x_interp, x_orig, os_[:n])
        interp_a += np.interp(x_interp, x_orig, ads[:n])
        count += 1
    interp_m /= count
    interp_o /= count
    interp_a /= count

    r_o = all_corrs[f"Trained ({pname})"]
    r_a = all_corrs[f"Adam ({pname})"]
    lw = 2 if pname == "yesno" else 1.5
    alpha = 1.0 if pname == "yesno" else 0.7
    ax.plot(x, interp_o, style_o, linewidth=lw, alpha=alpha,
            label=f'Trained/{pname} (r={r_o:.3f})')
    ax.plot(x, interp_a, style_a, linewidth=lw, alpha=alpha,
            label=f'Adam/{pname} (r={r_a:.3f})')

ax.plot(x, interp_m, 'b-', linewidth=2.5, label='Model Verdict')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('CoT Progress (%)', fontsize=12)
ax.set_ylabel('P(guilty)', fontsize=12)
ax.set_title(f'V7: Yes/No Prompt Comparison (n={count})', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / "averaged_v7_combined.png", dpi=150)
print(f"Saved {plots_dir / 'averaged_v7_combined.png'}")

# Copy per-question plots
import shutil
for f in results_dir.glob("q*.png"):
    shutil.copy2(f, plots_dir / f"v7_{f.name}")
print("Copied per-question plots")
