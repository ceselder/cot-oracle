#!/usr/bin/env python3
"""
Run linear + attention probe baselines on all cleaned datasets.

For each binary/multiclass dataset:
  - Linear probe at layer 9 (25%), 18 (50%), 27 (75%), concat all
  - Attention probe over all 3 layers

For answer_trajectory:
  - Regression probe predicting confidence → reports MAE

Requires GPU (Qwen3-8B forward pass for activation extraction).

Usage:
    python scripts/run_probe_baselines.py
    python scripts/run_probe_baselines.py --max-train 2000 --max-test 500
    python scripts/run_probe_baselines.py --datasets correctness sycophancy
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import EarlyStopException, get_hf_submodule, using_adapter
from cot_utils import get_cot_stride_positions

LAYERS = [9, 18, 27]
STRIDE = 5
SEED = 42

# All cleaned datasets: (short_name, hf_repo, task_type, label_field_or_regression_target)
DATASETS = {
    "hint_admission": (
        "mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned",
        "multiclass", "label",
    ),
    "atypical_answer": (
        "mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned",
        "binary", "label",
    ),
    "reasoning_termination": (
        "mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned",
        "binary", "label",
    ),
    "correctness": (
        "mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned",
        "binary", "label",
    ),
    "decorative_cot": (
        "mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned",
        "binary", "label",
    ),
    "sycophancy": (
        "mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned",
        "binary", "label",
    ),
    "truthfulqa_verb": (
        "mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-verbalized",
        "multiclass", "label",
    ),
    "truthfulqa_unverb": (
        "mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-unverbalized",
        "multiclass", "label",
    ),
    "answer_trajectory": (
        "mats-10-sprint-cs-jb/cot-oracle-answer-trajectory-cleaned",
        "regression", "confidence",
    ),
}


# ── Activation extraction ──

def extract_activations(model, input_ids, attn_mask, positions, layers):
    """Single forward pass → {layer: [K, D]} on CPU."""
    submodules = {l: get_hf_submodule(model, l) for l in layers}
    max_layer = max(layers)
    acts = {}
    mod_to_layer = {id(s): l for l, s in submodules.items()}

    def hook_fn(module, _inp, out):
        layer = mod_to_layer[id(module)]
        raw = out[0] if isinstance(out, tuple) else out
        acts[layer] = raw[0, positions, :].detach().cpu()
        if layer == max_layer:
            raise EarlyStopException()

    handles = [s.register_forward_hook(hook_fn) for s in submodules.values()]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)
    except EarlyStopException:
        pass
    finally:
        for h in handles:
            h.remove()
    return acts


def build_full_text(tokenizer, prompt, cot_text):
    """Build full text: chat-formatted prompt + CoT."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    return formatted + cot_text


def process_dataset(
    ds_name, hf_repo, model, tokenizer, device,
    max_train=2000, max_test=500,
):
    """Load dataset, extract activations. Returns (train_acts, test_acts) where each
    is list of (acts_by_layer: {layer: [K,D]}, row_dict)."""
    print(f"\n  Loading {ds_name} from {hf_repo}...")
    try:
        ds = load_dataset(hf_repo)
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None

    splits = {}
    for split_name, max_n in [("train", max_train), ("test", max_test)]:
        if split_name not in ds:
            print(f"    No {split_name} split, skipping")
            continue
        split = ds[split_name]
        if max_n and len(split) > max_n:
            split = split.shuffle(seed=SEED).select(range(max_n))
        splits[split_name] = split

    if "test" not in splits:
        return None, None

    model.eval()
    results = {}
    for split_name, split in splits.items():
        items = []
        skipped = 0
        for row in tqdm(split, desc=f"    {ds_name}/{split_name}", leave=False):
            cot_text = (row.get("cot_text") or "").strip()
            prompt = row.get("hinted_prompt") or row.get("question") or ""
            if not cot_text or not prompt:
                skipped += 1
                continue

            full_text = build_full_text(tokenizer, prompt, cot_text)
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            all_ids = tokenizer.encode(full_text, add_special_tokens=False)
            positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=STRIDE)

            tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            seq_len = tok_out["input_ids"].shape[1]
            positions = [p for p in positions if p < seq_len]
            if len(positions) < 2:
                skipped += 1
                continue

            with using_adapter(model, None):
                acts = extract_activations(
                    model,
                    tok_out["input_ids"].to(device),
                    tok_out["attention_mask"].to(device),
                    positions, LAYERS,
                )
            items.append((acts, dict(row)))

        results[split_name] = items
        if skipped:
            print(f"    {split_name}: {len(items)} ok, {skipped} skipped")

    return results.get("train", []), results.get("test", [])


# ── Pooling ──

def pool_mean(acts_by_layer, layers):
    """Mean-pool each layer → concat → [D*n_layers]."""
    return torch.cat([acts_by_layer[l].mean(dim=0) for l in layers], dim=-1)


def pool_per_layer(acts_by_layer, layers):
    """Mean-pool each layer separately → [n_layers, D]."""
    return torch.stack([acts_by_layer[l].mean(dim=0) for l in layers])


# ── Probes ──

def standardize(X_tr, X_te):
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / std, (X_te - mu) / std


def train_linear_classifier(X_tr, y_tr, X_te, n_classes, *, lr=0.01, epochs=100, wd=1e-4, device="cuda"):
    X_tr_s, X_te_s = standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device)
    probe = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(X_tr_s.shape[0], device=device)
        for start in range(0, len(perm), 512):
            idx = perm[start:start+512]
            opt.zero_grad(set_to_none=True)
            loss_fn(probe(X_tr_s[idx]), y_tr_d[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        return probe(X_te_s).argmax(1).cpu()


def train_linear_regressor(X_tr, y_tr, X_te, *, lr=0.01, epochs=200, wd=1e-4, device="cuda"):
    X_tr_s, X_te_s = standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device).float()
    probe = nn.Linear(X_tr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(X_tr_s.shape[0], device=device)
        for start in range(0, len(perm), 512):
            idx = perm[start:start+512]
            opt.zero_grad(set_to_none=True)
            loss_fn(probe(X_tr_s[idx]).squeeze(-1), y_tr_d[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        return probe(X_te_s).squeeze(-1).cpu()


class AttentionProbe(nn.Module):
    def __init__(self, d_model, n_layers, n_heads=4, hidden_dim=256, n_outputs=1):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim)
        self.pos_embed = nn.Embedding(n_layers, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_outputs))

    def forward(self, x):
        h = self.proj(x) + self.pos_embed(torch.arange(x.shape[1], device=x.device))
        h, w = self.attn(h, h, h)
        return self.head(h.mean(dim=1)), w


def train_attention_classifier(X_tr, y_tr, X_te, n_classes, *, device="cuda"):
    """X: [N, n_layers, D]. Returns predictions + layer importance."""
    X_tr_d, X_te_d = X_tr.to(device), X_te.to(device)
    y_tr_d = y_tr.to(device)
    n_layers, d_model = X_tr.shape[1], X_tr.shape[2]

    model = AttentionProbe(d_model, n_layers, n_outputs=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_loss, patience, best_state = float("inf"), 0, None
    for ep in range(80):
        model.train()
        perm = torch.randperm(len(X_tr_d), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start:start+128]
            opt.zero_grad(set_to_none=True)
            logits, _ = model(X_tr_d[idx])
            loss_fn(logits, y_tr_d[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_tr_d)[0], y_tr_d).item()
        if train_loss < best_loss:
            best_loss = train_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, attn_w = model(X_te_d)
        layer_imp = attn_w.mean(dim=0).sum(dim=0).cpu()  # [n_layers]
    return preds.argmax(1).cpu(), {f"L{LAYERS[i]}": layer_imp[i].item() for i in range(n_layers)}


def train_attention_regressor(X_tr, y_tr, X_te, *, device="cuda"):
    """Regression version of attention probe. Returns predictions + layer importance."""
    X_tr_d, X_te_d = X_tr.to(device), X_te.to(device)
    y_tr_d = y_tr.to(device).float()
    n_layers, d_model = X_tr.shape[1], X_tr.shape[2]

    model = AttentionProbe(d_model, n_layers, n_outputs=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss, patience, best_state = float("inf"), 0, None
    for ep in range(120):
        model.train()
        perm = torch.randperm(len(X_tr_d), device=device)
        for start in range(0, len(perm), 128):
            idx = perm[start:start+128]
            opt.zero_grad(set_to_none=True)
            out, _ = model(X_tr_d[idx])
            loss_fn(out.squeeze(-1), y_tr_d[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_tr_d)[0].squeeze(-1), y_tr_d).item()
        if train_loss < best_loss:
            best_loss = train_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, attn_w = model(X_te_d)
        layer_imp = attn_w.mean(dim=0).sum(dim=0).cpu()
    return preds.squeeze(-1).cpu(), {f"L{LAYERS[i]}": layer_imp[i].item() for i in range(n_layers)}


# ── Scoring ──

def accuracy(preds, gt):
    return sum(p == g for p, g in zip(preds, gt)) / len(gt)


def balanced_accuracy(preds, gt):
    classes = sorted(set(gt))
    per_class = []
    for c in classes:
        mask = [g == c for g in gt]
        if sum(mask) == 0:
            continue
        correct = sum(p == g for p, g, m in zip(preds, gt, mask) if m)
        per_class.append(correct / sum(mask))
    return sum(per_class) / len(per_class)


def mae(preds, gt):
    return sum(abs(p - g) for p, g in zip(preds, gt)) / len(gt)


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--datasets", nargs="+", default=None, help="Subset of datasets to run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="data/probe_baseline_results.json")
    args = parser.parse_args()

    datasets_to_run = args.datasets or list(DATASETS.keys())

    # Load model
    print("Loading Qwen3-8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    print("  Model loaded.\n")

    all_results = {}
    layer_configs = [(f"L{l}", [l]) for l in LAYERS] + [("concat", LAYERS)]

    for ds_name in datasets_to_run:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue

        hf_repo, task_type, target_field = DATASETS[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({task_type})")
        print(f"{'='*60}")

        t0 = time.time()
        train_items, test_items = process_dataset(
            ds_name, hf_repo, model, tokenizer, args.device,
            max_train=args.max_train, max_test=args.max_test,
        )
        extract_time = time.time() - t0

        if not train_items or not test_items:
            print(f"  Skipping {ds_name}: no data")
            continue

        print(f"  Extracted: {len(train_items)} train, {len(test_items)} test ({extract_time:.0f}s)")

        ds_results = {"n_train": len(train_items), "n_test": len(test_items)}

        if task_type in ("binary", "multiclass"):
            # Build labels
            all_labels = sorted(set(row[target_field] for _, row in train_items + test_items))
            label2idx = {l: i for i, l in enumerate(all_labels)}
            n_classes = len(all_labels)
            y_train = torch.tensor([label2idx[row[target_field]] for _, row in train_items])
            y_test_raw = [row[target_field] for _, row in test_items]

            print(f"  Classes: {all_labels}")
            print(f"  Train dist: {dict(zip(*torch.unique(y_train, return_counts=True)))}")

            # Linear probes per layer config
            for config_name, layer_list in layer_configs:
                X_train = torch.stack([pool_mean(acts, layer_list) for acts, _ in train_items])
                X_test = torch.stack([pool_mean(acts, layer_list) for acts, _ in test_items])

                preds_idx = train_linear_classifier(
                    X_train, y_train, X_test, n_classes, device=args.device,
                )
                preds = [all_labels[p.item()] for p in preds_idx]
                acc = accuracy(preds, y_test_raw)
                bal_acc = balanced_accuracy(preds, y_test_raw)
                ds_results[f"linear_{config_name}"] = {"accuracy": acc, "balanced_accuracy": bal_acc}
                print(f"  Linear {config_name:>7s}: acc={acc:.3f}  bal_acc={bal_acc:.3f}")

            # Attention probe
            X_train_layers = torch.stack([pool_per_layer(acts, LAYERS) for acts, _ in train_items])
            X_test_layers = torch.stack([pool_per_layer(acts, LAYERS) for acts, _ in test_items])
            preds_idx, layer_imp = train_attention_classifier(
                X_train_layers, y_train, X_test_layers, n_classes, device=args.device,
            )
            preds = [all_labels[p.item()] for p in preds_idx]
            acc = accuracy(preds, y_test_raw)
            bal_acc = balanced_accuracy(preds, y_test_raw)
            ds_results["attention"] = {"accuracy": acc, "balanced_accuracy": bal_acc, "layer_importance": layer_imp}
            print(f"  Attention      : acc={acc:.3f}  bal_acc={bal_acc:.3f}  imp={layer_imp}")

        elif task_type == "regression":
            # Regression: predict confidence
            y_train = torch.tensor([row[target_field] for _, row in train_items]).float()
            y_test = torch.tensor([row[target_field] for _, row in test_items]).float()

            print(f"  Target stats — train: mean={y_train.mean():.3f} std={y_train.std():.3f}")
            print(f"                 test:  mean={y_test.mean():.3f} std={y_test.std():.3f}")

            # Baseline: always predict mean
            mean_pred = y_train.mean().item()
            baseline_mae = mae([mean_pred] * len(y_test), y_test.tolist())
            ds_results["baseline_mean_mae"] = baseline_mae
            print(f"  Baseline (mean): MAE={baseline_mae:.4f}")

            # Linear probes per layer config
            for config_name, layer_list in layer_configs:
                X_train = torch.stack([pool_mean(acts, layer_list) for acts, _ in train_items])
                X_test = torch.stack([pool_mean(acts, layer_list) for acts, _ in test_items])

                preds = train_linear_regressor(X_train, y_train, X_test, device=args.device)
                m = mae(preds.tolist(), y_test.tolist())
                ds_results[f"linear_{config_name}"] = {"mae": m}
                print(f"  Linear {config_name:>7s}: MAE={m:.4f}")

            # Attention probe (regression)
            X_train_layers = torch.stack([pool_per_layer(acts, LAYERS) for acts, _ in train_items])
            X_test_layers = torch.stack([pool_per_layer(acts, LAYERS) for acts, _ in test_items])
            preds, layer_imp = train_attention_regressor(
                X_train_layers, y_train, X_test_layers, device=args.device,
            )
            m = mae(preds.tolist(), y_test.tolist())
            ds_results["attention"] = {"mae": m, "layer_importance": layer_imp}
            print(f"  Attention      : MAE={m:.4f}  imp={layer_imp}")

        all_results[ds_name] = ds_results

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")

    # Binary/multiclass table
    binary_ds = [d for d in datasets_to_run if d in all_results and DATASETS[d][1] != "regression"]
    if binary_ds:
        header = f"{'Dataset':<25s} {'L9':>7s} {'L18':>7s} {'L27':>7s} {'Concat':>7s} {'Attn':>7s}  N_tr  N_te"
        print(f"\nClassification (balanced accuracy):")
        print(header)
        print("-" * len(header))
        for ds_name in binary_ds:
            r = all_results[ds_name]
            vals = []
            for key in ["linear_L9", "linear_L18", "linear_L27", "linear_concat", "attention"]:
                v = r.get(key, {}).get("balanced_accuracy", 0)
                vals.append(f"{v:.3f}")
            print(f"{ds_name:<25s} {'  '.join(vals)}  {r['n_train']:>5d} {r['n_test']:>5d}")

    # Regression table
    reg_ds = [d for d in datasets_to_run if d in all_results and DATASETS[d][1] == "regression"]
    if reg_ds:
        print(f"\nRegression (MAE, lower is better):")
        header = f"{'Dataset':<25s} {'Mean':>7s} {'L9':>7s} {'L18':>7s} {'L27':>7s} {'Concat':>7s} {'Attn':>7s}  N_tr  N_te"
        print(header)
        print("-" * len(header))
        for ds_name in reg_ds:
            r = all_results[ds_name]
            base = f"{r.get('baseline_mean_mae', 0):.4f}"
            vals = [base]
            for key in ["linear_L9", "linear_L18", "linear_L27", "linear_concat", "attention"]:
                v = r.get(key, {}).get("mae", 0)
                vals.append(f"{v:.4f}")
            print(f"{ds_name:<25s} {'  '.join(vals)}  {r['n_train']:>5d} {r['n_test']:>5d}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
