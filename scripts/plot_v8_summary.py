"""Generate stats and plots from v8: 2 prompts × 2 oracles × 2 modes (cumul/sent)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("eval_logs/verdict_oscillation_v8")
plots_dir = Path("plots/verdict_oscillation")
plots_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "results.json") as f:
    results = json.load(f)

print(f"Loaded {len(results)} questions\n")

model_lists = [r["model_scores"] for r in results]

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

# All 8 configs
configs = []
for prompt in ["yesno", "cot_aware"]:
    for oracle in ["oracle", "adam"]:
        for mode in ["cumul", "sent"]:
            key = f"{oracle}_{mode}_{prompt}"
            configs.append((prompt, oracle, mode, key))

print("=== V8 Statistics (full matrix) ===\n")
print(f"{'Config':<40s} {'r':>6s} {'agree':>8s} {'avg':>6s}")
print("-" * 65)

mf_ref = None
all_corrs = {}
for prompt, oracle, mode, key in configs:
    score_lists = [r.get(key, []) for r in results]
    mf, sf = flatten_paired(model_lists, score_lists)
    if mf_ref is None:
        mf_ref = mf
    r_val = np.corrcoef(mf, sf)[0, 1]
    agree, total = count_agreements(model_lists, score_lists)
    label = f"{oracle.title():>8s} / {prompt:>10s} / {mode:>5s}"
    print(f"  {label:<40s} {r_val:>6.3f} {agree:>4d}/{total:<3d} {sf.mean():>6.3f}")
    all_corrs[key] = r_val

print(f"\n  Model avg P(guilty): {mf_ref.mean():.3f}")

# Big combined averaged plot
n_points = 50
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

prompt_texts = {
    "yesno": '"Does the model think the user is guilty? Answer YES or NO."',
    "cot_aware": '"At this point in the chain of thought, does the model think\nthe user is guilty? Answer YES or NO."',
}

prompt_text_full = {
    "yesno": '"Does the model think the user is guilty? Answer YES or NO."',
    "cot_aware": '"At this point in the chain of thought, does the model think the user is guilty? Answer YES or NO."',
}

# --- Combined plot with std bands ---
fig, ax = plt.subplots(figsize=(14, 8))
x = np.linspace(0, 100, n_points)

def interp_mean_std(score_lists_a, score_lists_b=None):
    """Interpolate per-question scores to n_points, return mean and std."""
    all_interp = []
    for i, ss in enumerate(score_lists_a):
        ref = score_lists_b[i] if score_lists_b else ss
        n = min(len(ss), len(ref)) if score_lists_b else len(ss)
        if n < 2:
            continue
        all_interp.append(np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, n), ss[:n]))
    arr = np.array(all_interp)
    return arr.mean(axis=0), arr.std(axis=0)

# Model verdict with std band
model_mean, model_std = interp_mean_std(model_lists)
ax.plot(x, model_mean, 'b-', linewidth=2.5, label='Model Verdict', zorder=10)
ax.fill_between(x, model_mean - model_std, model_mean + model_std, color='blue', alpha=0.1, zorder=1)

colors = {'oracle': 'red', 'adam': 'green'}
line_styles = {'cumul': '-', 'sent': '--'}
line_widths = {'cumul': 2.0, 'sent': 1.5}
alphas_line = {'cumul': 1.0, 'sent': 0.7}
alphas_fill = {'cumul': 0.08, 'sent': 0.05}

# Only plot cot_aware prompt (the best one) to keep it readable
prompt = "cot_aware"
for oracle in ["oracle", "adam"]:
    for mode in ["cumul", "sent"]:
        key = f"{oracle}_{mode}_{prompt}"
        score_lists = [r.get(key, []) for r in results]
        mean, std = interp_mean_std(score_lists, model_lists)

        color = colors[oracle]
        oracle_label = "Trained" if oracle == "oracle" else "Adam's AO"
        mode_label = "full CoT" if mode == "cumul" else "sentence-only"
        name = f"{oracle_label} / {mode_label} (r={all_corrs[key]:.3f})"

        ax.plot(x, mean, color=color, linestyle=line_styles[mode],
                linewidth=line_widths[mode], alpha=alphas_line[mode], label=name)
        ax.fill_between(x, mean - std, mean + std, color=color,
                        alpha=alphas_fill[mode], linestyle=line_styles[mode])

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('CoT Progress (%)', fontsize=13)
ax.set_ylabel('P(guilty)', fontsize=13)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')

# Annotations
ax.text(0.02, 0.02,
        "solid = full CoT activations up to sentence boundary\n"
        "dashed = current sentence activations only\n"
        "shaded = \u00b11 std across questions",
        transform=ax.transAxes, fontsize=8.5, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

ax.set_title(
    f'Verdict Oscillation: Oracle Tracking of Model Reasoning (n={len(results)} questions)\n'
    f'Oracle prompt: {prompt_text_full[prompt]}\n'
    f'Trained: ceselder/cot-oracle-v15-stochastic | '
    f"Adam's AO: adamkarvonen/...Qwen3-8B\n"
    f'Data: huggingface.co/datasets/ceselder/verdict-oscillation-v4',
    fontsize=10, loc='left')

plt.tight_layout()
plt.savefig(plots_dir / "averaged_v8.png", dpi=150, bbox_inches='tight')
plt.savefig(results_dir / "averaged_v8.png", dpi=150, bbox_inches='tight')
print(f"\nSaved {plots_dir / 'averaged_v8.png'}")

# Also make the 2-panel version for completeness
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 7))
for ax_idx, prompt in enumerate(["yesno", "cot_aware"]):
    ax = axes2[ax_idx]
    model_mean, model_std = interp_mean_std(model_lists)
    ax.plot(x, model_mean, 'b-', linewidth=2.5, label='Model Verdict', zorder=10)
    ax.fill_between(x, model_mean - model_std, model_mean + model_std, color='blue', alpha=0.1)

    for oracle in ["oracle", "adam"]:
        for mode in ["cumul", "sent"]:
            key = f"{oracle}_{mode}_{prompt}"
            score_lists = [r.get(key, []) for r in results]
            mean, std = interp_mean_std(score_lists, model_lists)
            color = colors[oracle]
            oracle_label = "Trained" if oracle == "oracle" else "Adam's AO"
            mode_label = "full CoT" if mode == "cumul" else "sentence-only"
            name = f"{oracle_label} / {mode_label} (r={all_corrs[key]:.3f})"
            ax.plot(x, mean, color=color, linestyle=line_styles[mode],
                    linewidth=line_widths[mode], alpha=alphas_line[mode], label=name)
            ax.fill_between(x, mean - std, mean + std, color=color,
                            alpha=alphas_fill[mode])

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('CoT Progress (%)', fontsize=12)
    ax.set_ylabel('P(guilty)', fontsize=12)
    ax.set_title(f'Prompt: {prompt_text_full[prompt]}', fontsize=9, style='italic')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, "solid = full CoT | dashed = sentence-only | shaded = \u00b11 std",
            transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.suptitle(
    f'Verdict Oscillation: Oracle Tracking of Model Reasoning (n={len(results)} questions)\n'
    f'Trained: ceselder/cot-oracle-v15-stochastic | '
    f"Adam's AO: adamkarvonen/...Qwen3-8B | "
    f'Data: huggingface.co/datasets/ceselder/verdict-oscillation-v4',
    fontsize=10, y=1.04)
plt.tight_layout()
plt.savefig(plots_dir / "averaged_v8_2panel.png", dpi=150, bbox_inches='tight')
print(f"Saved {plots_dir / 'averaged_v8_2panel.png'}")

# Copy per-question plots
import shutil
for f in results_dir.glob("q*.png"):
    shutil.copy2(f, plots_dir / f"v8_{f.name}")
print("Copied per-question plots")
