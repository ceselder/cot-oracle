"""Illustrative plot of how positional embedding injection works at layer 1.

6 rows:
1. Subject model's CoT (the oracle never sees this text — only its activations)
2. Raw sinusoidal PE vectors at extraction positions (unit-normalized, own color scale)
3. Oracle prompt: raw token residual stream (after embed + Layer 0)
4. Activation steering vectors injected at ? placeholders (norm-matched)
5. Scaled position encoding: α · ‖v_act‖ · PE  (own color scale)
6. Combined: token residual + steering + position encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(42)

# ── Subject model's CoT tokens (Row 1) ──────────────────────────────────
cot_tokens = [
    "<|im_start|>", "user", "\n",
    "What", "is", "6", "×", "7", "?",
    "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n",
    "Let", "me", "think", "step", "by", "step", ".",
    "First", ",", "6", "×", "7", "means",
    "adding", "6", "seven", "times", ".",
    "6", "+", "6", "=", "12", ",",
    "12", "+", "6", "=", "18", ",",
    "18", "+", "6", "=", "24", ",",
    "24", "+", "6", "=", "30", ",",
    "30", "+", "6", "=", "36", ",",
    "36", "+", "6", "=", "42", ".",
    "So", "the", "answer", "is", "42", ".",
]
cot_len = len(cot_tokens)

# Prompt region vs CoT region in subject model input
prompt_end = 14  # tokens 0-13 are the user prompt + chat template
cot_region = list(range(prompt_end, cot_len))  # tokens 14+ are CoT

# Stride-5 extraction positions (uneven at boundaries)
stride = 5
extraction_positions = list(range(cot_region[0], cot_region[-1], stride)) + [cot_region[-1]]
n_extracted = len(extraction_positions)

# ── Oracle prompt tokens (Rows 3-6) ─────────────────────────────────────
n_pos_per_layer = n_extracted
layers = [9, 18, 27]

oracle_tokens = ["<|im_start|>", "user", "\n"]
placeholder_mask = [False, False, False]
for i, layer in enumerate(layers):
    if i > 0:
        oracle_tokens.append(" ")
        placeholder_mask.append(False)
    oracle_tokens.append(f"L{layer}:")
    placeholder_mask.append(False)
    for _ in range(n_pos_per_layer):
        oracle_tokens.append("?")
        placeholder_mask.append(True)
oracle_tokens += [".", "\n"]
placeholder_mask += [False, False]
task_tokens = ["Did", "the", "model", "use", "a", "hint", "in", "its", "reasoning", "?"]
oracle_tokens += task_tokens
placeholder_mask += [False] * len(task_tokens)
oracle_tokens += ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
placeholder_mask += [False] * 5

oracle_len = len(oracle_tokens)
placeholder_positions = [i for i, m in enumerate(placeholder_mask) if m]

# ── Dimensions ───────────────────────────────────────────────────────────
d_model = 4096
half_d = d_model // 2

# ── Row 1: Subject model residual stream (full CoT) ─────────────────────
cot_resid = np.random.randn(cot_len, d_model) * 0.8
for i in range(cot_len):
    cot_resid[i, (i * 7) % d_model] += 1.8
    cot_resid[i, (i * 13 + 3) % d_model] -= 1.2

# ── Row 2: Raw sinusoidal PE for ALL CoT positions (standard Vaswani) ───
# PE(pos, 2i)   = sin(pos / 10000^(2i/d))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# Using raw integer positions (0, 1, 2, ...) — standard Vaswani, not normalized
cot_total = len(cot_region)
pe_raw_cot = np.zeros((cot_len, d_model))
freqs = np.exp(np.arange(half_d) * -(math.log(10000.0) / half_d))
for pos in range(prompt_end, cot_len):
    raw_pos = pos - prompt_end  # integer position within CoT: 0, 1, 2, ...
    angles = raw_pos * freqs
    pe = np.concatenate([np.sin(angles), np.cos(angles)])
    pe_raw_cot[pos] = pe / np.linalg.norm(pe)  # unit-normalized

# ── Row 3: Oracle token residual stream ──────────────────────────────────
oracle_resid = np.random.randn(oracle_len, d_model) * 0.8
for i in range(oracle_len):
    oracle_resid[i, (i * 7) % d_model] += 1.5
    oracle_resid[i, (i * 11 + 5) % d_model] -= 1.0
token_norm = np.mean(np.linalg.norm(oracle_resid, axis=-1))

# ── Row 4: Activation steering at ? positions ───────────────────────────
raw_acts = np.random.randn(len(placeholder_positions), d_model) * 1.2
steering_coeff = 1.0
acts_only = np.zeros((oracle_len, d_model))
for idx, pos in enumerate(placeholder_positions):
    orig_norm = np.linalg.norm(oracle_resid[pos])
    normed = raw_acts[idx] / np.linalg.norm(raw_acts[idx])
    acts_only[pos] = normed * orig_norm * steering_coeff
act_norm = np.mean(np.linalg.norm(acts_only[placeholder_positions], axis=-1))

# ── Row 5: Scaled position encoding at ? positions ──────────────────────
# v_pe = alpha * ||v_act|| * PE_norm(raw_pos)
alpha = 0.1
pe_scaled = np.zeros((oracle_len, d_model))
freqs = np.exp(np.arange(half_d) * -(math.log(10000.0) / half_d))
for idx, pos in enumerate(placeholder_positions):
    source_pos = extraction_positions[idx % n_pos_per_layer]
    raw_pos = source_pos - prompt_end  # integer position within CoT
    angles = raw_pos * freqs
    pe = np.concatenate([np.sin(angles), np.cos(angles)])
    pe = pe / np.linalg.norm(pe)
    v_norm = np.linalg.norm(acts_only[pos])
    pe_scaled[pos] = alpha * v_norm * pe
pe_norm = np.mean(np.linalg.norm(pe_scaled[placeholder_positions], axis=-1))

# ── Row 6: Combined ─────────────────────────────────────────────────────
combined = oracle_resid.copy()
for pos in placeholder_positions:
    combined[pos] = oracle_resid[pos] + acts_only[pos] + pe_scaled[pos]

# ── Plotting ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(22, 16), layout="constrained")
cmap = "RdBu_r"

# Global vmax for rows that share the activation scale (rows 1, 3, 4, 6)
vmax_global = max(np.abs(cot_resid).max(), np.abs(oracle_resid).max(),
                  np.abs(acts_only).max(), np.abs(combined).max()) * 0.85

# PE rows get their own scale so structure is visible
vmax_pe_raw = np.abs(pe_raw_cot).max() * 1.05  # unit-normalized → ~0.09
vmax_pe_scaled = np.abs(pe_scaled[placeholder_positions]).max() * 1.05

def style_cot_ax(ax):
    ax.set_xticks(range(cot_len))
    ax.set_xticklabels(cot_tokens, rotation=60, ha="right", fontsize=5)
    for j, t in enumerate(cot_tokens):
        if t.startswith("<|"):
            ax.get_xticklabels()[j].set_color("gray")
            ax.get_xticklabels()[j].set_fontsize(4.5)
    ax.set_yticks([0, half_d, d_model - 1])
    ax.set_yticklabels(["0", f"{half_d}", f"{d_model-1}"], fontsize=7)
    ax.set_ylabel("hidden dim", fontsize=8)
    # Mark extraction positions
    for ep in extraction_positions:
        ax.axvline(ep, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
    # Shade prompt region
    ax.axvspan(-0.5, prompt_end - 0.5, alpha=0.08, color="gray")

def style_oracle_ax(ax):
    ax.set_xticks(range(oracle_len))
    ax.set_xticklabels(oracle_tokens, rotation=60, ha="right", fontsize=5)
    for j, t in enumerate(oracle_tokens):
        if t in ("L9:", "L18:", "L27:"):
            ax.get_xticklabels()[j].set_fontweight("bold")
            ax.get_xticklabels()[j].set_fontsize(7)
            ax.get_xticklabels()[j].set_color("darkblue")
        elif t.startswith("<|"):
            ax.get_xticklabels()[j].set_color("gray")
            ax.get_xticklabels()[j].set_fontsize(4.5)
    ax.set_yticks([0, half_d, d_model - 1])
    ax.set_yticklabels(["0", f"{half_d}", f"{d_model-1}"], fontsize=7)
    ax.set_ylabel("hidden dim", fontsize=8)

# --- Row 1: Subject model CoT ---
ax = axes[0]
im1 = ax.imshow(cot_resid.T, aspect="auto", cmap=cmap, vmin=-vmax_global, vmax=vmax_global, interpolation="nearest")
ax.set_title("Subject model's CoT  (oracle never sees this text — only reads activations from it)",
             fontsize=10, fontweight="bold", loc="left")
style_cot_ax(ax)
mid_cot = (cot_region[0] + cot_region[-1]) / 2
ax.text(mid_cot, -2.5, "cot_text  (activations extracted at dashed positions, stride≈5)",
        ha="center", fontsize=7, color="black", fontstyle="italic")

# --- Row 2: Raw sinusoidal PE (unit-normalized, own color scale) ---
ax = axes[1]
im2 = ax.imshow(pe_raw_cot.T, aspect="auto", cmap=cmap, vmin=-vmax_pe_raw, vmax=vmax_pe_raw, interpolation="nearest")
ax.set_title("PE(source_pos / cot_length)  —  unit-normalized sinusoidal encoding  (‖PE‖ = 1.0,  own color scale)",
             fontsize=10, fontweight="bold", loc="left")
style_cot_ax(ax)

# --- Row 3: Oracle token residual ---
ax = axes[2]
ax.imshow(oracle_resid.T, aspect="auto", cmap=cmap, vmin=-vmax_global, vmax=vmax_global, interpolation="nearest")
ax.set_title(f"Oracle prompt — token residual only  (mean ‖·‖ ≈ {token_norm:.1f},  weight: 1.0×)",
             fontsize=10, fontweight="bold", loc="left")
style_oracle_ax(ax)

# --- Row 4: Activation steering ---
ax = axes[3]
ax.imshow(acts_only.T, aspect="auto", cmap=cmap, vmin=-vmax_global, vmax=vmax_global, interpolation="nearest")
ax.set_title(f"Activation steering at ? placeholders  (norm-matched,  mean ‖·‖ ≈ {act_norm:.1f},  weight: ~{act_norm/token_norm:.1f}×)",
             fontsize=10, fontweight="bold", loc="left")
style_oracle_ax(ax)
# Source position annotations
for idx in range(min(n_pos_per_layer, len(extraction_positions))):
    pos = placeholder_positions[idx]
    src = extraction_positions[idx]
    cot_tok = cot_tokens[src] if src < len(cot_tokens) else "?"
    ax.annotate(f"pos {src} \"{cot_tok}\"", (pos, d_model - 1), (pos, d_model + 6),
                fontsize=4, ha="center", va="top", color="darkblue",
                arrowprops=dict(arrowstyle="-", color="darkblue", lw=0.4))

# --- Row 5: Scaled PE (own color scale) ---
ax = axes[4]
im5 = ax.imshow(pe_scaled.T, aspect="auto", cmap=cmap, vmin=-vmax_pe_scaled, vmax=vmax_pe_scaled, interpolation="nearest")
ax.set_title(f"α · ‖v_act‖ · PE(src_pos/cot_len)  —  α={alpha},  mean ‖·‖ ≈ {pe_norm:.2f},  weight: ~{pe_norm/token_norm:.2f}×  (own color scale)",
             fontsize=10, fontweight="bold", loc="left")
style_oracle_ax(ax)

# --- Row 6: Combined ---
ax = axes[5]
ax.imshow(combined.T, aspect="auto", cmap=cmap, vmin=-vmax_global, vmax=vmax_global, interpolation="nearest")
ax.set_title("Combined  →  modified Layer 1 residual stream fed to oracle",
             fontsize=10, fontweight="bold", loc="left")
style_oracle_ax(ax)
ax.set_xlabel("Oracle prompt tokens  (oracle_prefix: layer labels + ? placeholders  |  task prompt)", fontsize=9)

# Mark boundaries on oracle rows
prefix_start = 3
task_end_pos = prefix_start
for i in range(len(layers)):
    task_end_pos += n_pos_per_layer + 1 + (1 if i > 0 else 0)
task_end_pos += 1  # .\n
task_end_pos += len(task_tokens)
for ax in axes[2:]:
    ax.axvline(prefix_start - 0.5, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(task_end_pos - 0.5, color="black", linewidth=0.8, alpha=0.5)

# Colorbars — one for global scale, one for PE scale
cbar1 = fig.colorbar(im1, ax=[axes[0], axes[2], axes[3], axes[5]], shrink=0.6, pad=0.015, label="activation value")
cbar1.ax.tick_params(labelsize=7)
cbar2 = fig.colorbar(im2, ax=[axes[1], axes[4]], shrink=0.6, pad=0.015, label="PE value (own scale)")
cbar2.ax.tick_params(labelsize=7)

fig.suptitle("Activation Oracle: Injection at Layer 1 with Positional Encoding\n"
             "v_combined = v_act + α · ‖v_act‖ · PE(source_pos / cot_length),   then   resid += normalize(v_combined) · ‖resid_orig‖",
             fontsize=12, fontweight="bold", y=1.01)

out = "data/position_embedding_illustration.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
