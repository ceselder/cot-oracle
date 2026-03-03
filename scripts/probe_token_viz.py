#!/usr/bin/env python3
"""
Per-token probe visualization: train a linear probe, then show which tokens
it fires on across example CoTs.

Outputs an HTML file with tokens colored by probe score (red = positive class,
blue = negative class), plus a text summary.

Usage:
    python scripts/probe_token_viz.py --dataset hint_admission --layer 18
    python scripts/probe_token_viz.py --dataset sycophancy --layer 18 --n-examples 20
"""

import argparse
import gc
import html
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import EarlyStopException, get_hf_submodule

DATASETS = {
    "hint_admission": "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned",
    "sycophancy": "mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned",
    "truthfulqa_hint": "mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-cleaned",
    "decorative_cot": "mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned",
    "reasoning_termination": "mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned",
    "atypical_answer": "mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned",
    "correctness": "mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned",
    "backtrack_prediction": "mats-10-sprint-cs-jb/cot-oracle-backtrack-prediction-cleaned",
}

# Merge multi-class hint labels to binary
HINT_BINARIZE = {
    "hint_used_correct": "hint_used",
    "hint_used_wrong": "hint_used",
    "hint_resisted": "hint_resisted",
}


def extract_all_token_acts(model, input_ids, attn_mask, layer, device):
    """Extract activations at ALL positions for a single layer."""
    submodule = get_hf_submodule(model, layer)
    result = {}

    def hook_fn(module, _inp, out):
        raw = out[0] if isinstance(out, tuple) else out
        result["acts"] = raw.detach().cpu().float()
        raise EarlyStopException()

    handle = submodule.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
    except EarlyStopException:
        pass
    finally:
        handle.remove()

    return result["acts"]  # [B, seq_len, d_model]


def tokenize_row(row, tokenizer):
    """Tokenize a dataset row. Returns dict or None."""
    cot_text = (row.get("cot_text") or "").strip()
    prompt = row.get("hinted_prompt") or row.get("question") or row.get("prompt_text") or ""
    if not cot_text or not prompt:
        return None

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = formatted + cot_text

    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)

    return {
        "input_ids": all_ids,
        "prompt_len": len(prompt_ids),
        "seq_len": len(all_ids),
    }


def train_linear_probe(X_train, y_train, n_classes, device="cuda", epochs=1000, lr=0.01):
    """Train a logistic regression probe. Returns (weight, bias, mu, std)."""
    mu = X_train.mean(0, keepdim=True)
    std = X_train.std(0, keepdim=True).clamp(min=1e-6)
    X_normed = (X_train - mu) / std

    d_in = X_normed.shape[1]
    w = torch.zeros(n_classes, d_in, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    torch.nn.init.xavier_normal_(w)

    X_d = X_normed.to(device)
    y_d = y_train.to(device)

    optimizer = torch.optim.AdamW([w, b], lr=lr, weight_decay=0.01)
    best_loss = float("inf")
    patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        logits = X_d @ w.T + b
        loss = F.cross_entropy(logits, y_d.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            patience_counter = 0
            best_w = w.detach().clone()
            best_b = b.detach().clone()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_w.cpu(), best_b.cpu(), mu.cpu(), std.cpu()


def score_tokens(acts, weight, bias, mu, std):
    """Project per-token activations onto probe direction.

    Returns [seq_len] scores. Higher = more class_1 (positive class).
    """
    # Binary direction: class_1 - class_0
    w = weight[1] - weight[0]  # [d_in]
    b = (bias[1] - bias[0]).item()
    acts_normed = (acts - mu) / std
    scores = (acts_normed @ w) + b
    return scores


def generate_html(examples, output_path, dataset_name, layer, class_names):
    """Generate HTML visualization with tokens colored by probe score."""
    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Probe Token Viz: {dataset_name} L{layer}</title>
<style>
body {{ font-family: 'Menlo', 'Consolas', monospace; font-size: 13px;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #fff; }}
h2 {{ color: #ccc; margin-top: 2em; border-bottom: 1px solid #444; padding-bottom: 5px; }}
.meta {{ color: #888; font-size: 11px; margin-bottom: 10px; }}
.cot {{ line-height: 1.8; word-wrap: break-word; background: #16213e; padding: 15px;
        border-radius: 8px; margin-bottom: 10px; }}
.prompt {{ color: #666; font-size: 11px; margin-bottom: 5px; max-height: 100px;
           overflow: hidden; background: #0f3460; padding: 10px; border-radius: 5px; }}
.token {{ padding: 1px 0; border-radius: 2px; }}
.legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.legend-swatch {{ width: 40px; height: 16px; border-radius: 3px; }}
</style></head><body>
<h1>Per-Token Probe Scores: {dataset_name} (Layer {layer})</h1>
<p>Positive class = <strong>{class_names[1]}</strong> | Negative class = <strong>{class_names[0]}</strong></p>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background: rgba(255,60,60,0.8)"></div>Strong {class_names[1]}</div>
  <div class="legend-item"><div class="legend-swatch" style="background: rgba(255,60,60,0.3)"></div>Weak {class_names[1]}</div>
  <div class="legend-item"><div class="legend-swatch" style="background: rgba(200,200,200,0.1)"></div>Neutral</div>
  <div class="legend-item"><div class="legend-swatch" style="background: rgba(60,120,255,0.3)"></div>Weak {class_names[0]}</div>
  <div class="legend-item"><div class="legend-swatch" style="background: rgba(60,120,255,0.8)"></div>Strong {class_names[0]}</div>
</div>
"""]

    for i, ex in enumerate(examples):
        label = ex["label"]
        tokens = ex["tokens"]
        scores = ex["scores"]

        # Normalize scores to [-1, 1] using soft clipping (tanh on z-scored)
        s = np.array(scores)
        s_mean, s_std = s.mean(), max(s.std(), 1e-6)
        s_norm = np.tanh((s - s_mean) / (2 * s_std))  # soft clip

        html_parts.append(f'<h2>Example {i+1} — Label: {label}</h2>')

        # Show prompt snippet
        prompt_text = ex.get("prompt", "")[:300]
        html_parts.append(f'<div class="prompt">{html.escape(prompt_text)}...</div>')

        html_parts.append('<div class="cot">')

        # Find top-5 scoring tokens
        cot_start = ex["prompt_len"]
        cot_scores = s[cot_start:]
        if len(cot_scores) > 0:
            top_indices = np.argsort(np.abs(cot_scores))[-5:][::-1]

        for j, (tok, score_val, norm_val) in enumerate(zip(tokens, scores, s_norm)):
            if j < cot_start:
                continue  # skip prompt tokens

            # Color: positive = red, negative = blue
            if norm_val > 0.05:
                r, g, b_c = 255, 60, 60
                alpha = min(0.85, abs(norm_val) * 0.85)
            elif norm_val < -0.05:
                r, g, b_c = 60, 120, 255
                alpha = min(0.85, abs(norm_val) * 0.85)
            else:
                r, g, b_c = 200, 200, 200
                alpha = 0.05

            tok_escaped = html.escape(tok).replace("\n", "<br>")
            style = f"background: rgba({r},{g},{b_c},{alpha:.2f})"

            # Bold the top scorers
            cot_idx = j - cot_start
            if len(cot_scores) > 0 and cot_idx in top_indices:
                style += "; font-weight: bold; text-decoration: underline"

            html_parts.append(f'<span class="token" style="{style}" '
                            f'title="score={score_val:.3f}">{tok_escaped}</span>')

        html_parts.append('</div>')

        # Top scoring tokens summary
        if len(cot_scores) > 0:
            cot_tokens = tokens[cot_start:]
            pos_top = np.argsort(cot_scores)[-5:][::-1]
            neg_top = np.argsort(cot_scores)[:5]
            html_parts.append('<div class="meta">Top positive: ')
            html_parts.append(', '.join(
                f'"{html.escape(cot_tokens[k])}" ({cot_scores[k]:.2f})'
                for k in pos_top if k < len(cot_tokens)
            ))
            html_parts.append('<br>Top negative: ')
            html_parts.append(', '.join(
                f'"{html.escape(cot_tokens[k])}" ({cot_scores[k]:.2f})'
                for k in neg_top if k < len(cot_tokens)
            ))
            html_parts.append('</div>')

    html_parts.append("</body></html>")
    output_path.write_text("".join(html_parts))
    print(f"Saved HTML visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-examples", type=int, default=20,
                        help="Number of test examples to visualize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Skip examples longer than this")
    parser.add_argument("--output-dir", default="data/probe_viz")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride for training data (per-token viz always uses stride=1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_name = args.dataset
    hf_repo = DATASETS[ds_name]
    layer = args.layer

    # ── Load data ──
    print(f"Loading {hf_repo}...")
    train_ds = load_dataset(hf_repo, split="train")
    test_ds = load_dataset(hf_repo, split="test")

    # ── Load model ──
    print("Loading Qwen3-8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    # ── Phase 1: Extract training activations (stride=5, last-position pooling) ──
    print(f"\nExtracting training activations (n={args.n_train}, stride={args.stride})...")
    train_vecs = []
    train_labels = []
    label_set = set()

    for i, row in enumerate(tqdm(train_ds, total=min(args.n_train, len(train_ds)))):
        if i >= args.n_train:
            break
        row = dict(row)
        tok = tokenize_row(row, tokenizer)
        if tok is None:
            continue
        if tok["seq_len"] > args.max_seq_len:
            continue

        # Binarize label
        label = row["label"]
        if ds_name in ("hint_admission", "truthfulqa_hint"):
            label = HINT_BINARIZE.get(label, label)
        label_set.add(label)

        # Extract activation at last CoT position only (for training)
        input_ids = torch.tensor([tok["input_ids"]])
        attn_mask = torch.ones_like(input_ids)
        acts = extract_all_token_acts(model, input_ids, attn_mask, layer, args.device)
        # acts: [1, seq_len, d_model]

        # Use last CoT position (stride-based would be more data-efficient
        # but last-pos is simplest and matches our baseline)
        last_pos = tok["seq_len"] - 1
        train_vecs.append(acts[0, last_pos, :])
        train_labels.append(label)

    all_labels = sorted(label_set)
    label2idx = {l: i for i, l in enumerate(all_labels)}
    print(f"  Labels: {all_labels}")
    print(f"  Train size: {len(train_vecs)}")
    from collections import Counter
    print(f"  Distribution: {Counter(train_labels)}")

    X_train = torch.stack(train_vecs)
    y_train = torch.tensor([label2idx[l] for l in train_labels])

    # ── Phase 2: Train linear probe ──
    print(f"\nTraining linear probe (layer {layer})...")
    weight, bias, mu, std = train_linear_probe(
        X_train, y_train, len(all_labels), device=args.device
    )
    print(f"  Probe trained: weight shape {weight.shape}")

    # Quick sanity: eval on training data
    X_normed = (X_train - mu) / std
    logits = X_normed @ weight.T + bias
    preds = logits.argmax(dim=1)
    train_acc = (preds == y_train).float().mean().item()
    print(f"  Train accuracy: {train_acc:.3f}")

    # Free training data
    del X_train, train_vecs
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3: Per-token scoring on test examples ──
    print(f"\nScoring {args.n_examples} test examples per-token...")

    examples = []
    n_pos = sum(1 for row in test_ds if (
        HINT_BINARIZE.get(row["label"], row["label"]) if ds_name in ("hint_admission", "truthfulqa_hint")
        else row["label"]
    ) == all_labels[1])
    n_neg = len(test_ds) - n_pos

    # Get balanced examples: half positive, half negative
    pos_examples = []
    neg_examples = []
    for row in test_ds:
        row = dict(row)
        label = row["label"]
        if ds_name in ("hint_admission", "truthfulqa_hint"):
            label = HINT_BINARIZE.get(label, label)
        if label == all_labels[1] and len(pos_examples) < args.n_examples // 2:
            pos_examples.append(row)
        elif label == all_labels[0] and len(neg_examples) < args.n_examples // 2:
            neg_examples.append(row)
        if len(pos_examples) >= args.n_examples // 2 and len(neg_examples) >= args.n_examples // 2:
            break

    selected = pos_examples + neg_examples

    for row in tqdm(selected, desc="Per-token scoring"):
        tok = tokenize_row(row, tokenizer)
        if tok is None:
            continue
        if tok["seq_len"] > args.max_seq_len:
            continue

        label = row["label"]
        if ds_name in ("hint_admission", "truthfulqa_hint"):
            label = HINT_BINARIZE.get(label, label)

        input_ids = torch.tensor([tok["input_ids"]])
        attn_mask = torch.ones_like(input_ids)
        acts = extract_all_token_acts(model, input_ids, attn_mask, layer, args.device)
        # acts: [1, seq_len, d_model]

        per_token_scores = score_tokens(acts[0], weight, bias, mu, std)

        # Decode each token for display
        tokens = [tokenizer.decode([tid]) for tid in tok["input_ids"]]

        examples.append({
            "label": label,
            "tokens": tokens,
            "scores": per_token_scores.tolist(),
            "prompt_len": tok["prompt_len"],
            "prompt": row.get("hinted_prompt") or row.get("question") or "",
        })

    # ── Phase 4: Generate HTML ──
    html_path = output_dir / f"{ds_name}_L{layer}_token_viz.html"
    generate_html(examples, html_path, ds_name, layer, all_labels)

    # ── Phase 5: Save raw scores as JSONL ──
    jsonl_path = output_dir / f"{ds_name}_L{layer}_token_scores.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "label": ex["label"],
                "prompt_len": ex["prompt_len"],
                "scores": ex["scores"],
                "tokens": ex["tokens"],
            }) + "\n")
    print(f"Saved raw scores to {jsonl_path}")

    # ── Summary stats ──
    print(f"\n{'='*60}")
    print(f"Summary: {ds_name} probe (layer {layer})")
    print(f"{'='*60}")
    for label in all_labels:
        label_examples = [ex for ex in examples if ex["label"] == label]
        if not label_examples:
            continue
        all_cot_scores = []
        for ex in label_examples:
            cot_scores = ex["scores"][ex["prompt_len"]:]
            all_cot_scores.extend(cot_scores)
        s = np.array(all_cot_scores)
        print(f"  {label} ({len(label_examples)} examples):")
        print(f"    Mean score: {s.mean():.3f}, Std: {s.std():.3f}")
        print(f"    Min: {s.min():.3f}, Max: {s.max():.3f}")

    print(f"\nVisualization: {html_path}")
    print("Open in browser or use `python -m http.server 8080` from the output dir")


if __name__ == "__main__":
    main()
