#!/usr/bin/env python3
"""Train linear and attention probes on Qwen3-8B activations for all binary evals.

For each binary eval dataset:
  - 5-fold stratified CV (or train/test split where available)
  - Linear probes per-layer + concat, attention probe

Also:
  - hint_admission (15K): 80/20 split, then cross-dataset transfer to all other evals
  - hinted_mcq_truthfulqa: uses its own train/test split

Usage (on GPU machine):
    PYTHONUNBUFFERED=1 python scripts/train_hint_probes.py 2>&1 | tee probe_run.log

    # Skip activation extraction (use cached):
    PYTHONUNBUFFERED=1 python scripts/train_hint_probes.py --skip-extraction

    # CPU-only (probes only, requires cached activations):
    python scripts/train_hint_probes.py --skip-extraction --device cpu
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cot_utils import get_cot_stride_positions
from evals.activation_cache import build_full_text_from_prompt_and_cot

# ── Constants ──

LAYERS = [9, 18, 27]
STRIDE = 5
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
CACHE_DIR = _ROOT / "data" / "probe_cache"
RESULTS_DIR = _ROOT / "data" / "probe_results"

# Probe hyperparameters
LINEAR_LR = 0.01
LINEAR_EPOCHS = 100
LINEAR_WD = 1e-4
ATTN_HEADS = 4
ATTN_HIDDEN = 256
ATTN_LR = 1e-3
ATTN_EPOCHS = 50
ATTN_PATIENCE = 10

# HF org
HF_ORG = "mats-10-sprint-cs-jb"


# ============================================================
# Phase 1: Dataset loaders
# ============================================================

# Each loader returns list of {prompt, cot_text, label, example_id}
# For datasets with train/test splits, returns (train, test) tuple

def load_hint_admission() -> list[dict]:
    """15K items, 3-class → binary. 50/50 balanced."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/qwen3-8b-hint-admission-rollouts", split="train")
    label_map = {"hint_used_correct": "influenced", "hint_used_wrong": "influenced",
                 "hint_resisted": "independent"}
    items = []
    for i, row in enumerate(ds):
        bl = label_map.get(row["label"])
        if bl is None:
            continue
        items.append({"prompt": row["hinted_prompt"], "cot_text": row["cot_text"],
                       "label": bl, "example_id": f"hint_admission_{i:05d}"})
    _print_dist("hint_admission", items)
    return items


def _load_truthfulqa_split(repo_suffix: str) -> tuple[list[dict], list[dict]]:
    """Load new TruthfulQA hint-admission datasets (per-rollout format).
    Maps 3-class labels to binary: hint_used_correct/wrong → influenced, hint_resisted → independent.
    """
    from datasets import load_dataset
    label_map = {"hint_used_correct": "influenced", "hint_used_wrong": "influenced",
                 "hint_resisted": "independent"}

    def _parse(ds, tag):
        items = []
        for row in ds:
            bl = label_map.get(row["label"])
            if bl is None:
                continue
            items.append({"prompt": row["hinted_prompt"], "cot_text": row["cot_text"],
                           "label": bl,
                           "example_id": f"{row['question_id']}_r{row['rollout_idx']}"})
        return items

    repo = f"{HF_ORG}/cot-oracle-eval-hinted-mcq-truthfulqa-{repo_suffix}"
    ds_tr = load_dataset(repo, split="train")
    ds_te = load_dataset(repo, split="test")
    train_items, test_items = _parse(ds_tr, "train"), _parse(ds_te, "test")
    _print_dist(f"truthfulqa_{repo_suffix}_train", train_items)
    _print_dist(f"truthfulqa_{repo_suffix}_test", test_items)
    return train_items, test_items


def load_truthfulqa_unverbalized() -> tuple[list[dict], list[dict]]:
    """TruthfulQA unverbalized: 10941 train / 100 test."""
    return _load_truthfulqa_split("unverbalized")


def load_truthfulqa_verbalized() -> tuple[list[dict], list[dict]]:
    """TruthfulQA verbalized: 4280 train / 100 test."""
    return _load_truthfulqa_split("verbalized")


def load_atypical_answer_riya() -> list[dict]:
    """100 items, majority/minority (50/50)."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/cot-oracle-eval-atypical-answer-riya", split="train")
    items = []
    for row in ds:
        cot = row.get("meta_cot_text") or ""
        if not cot.strip():
            continue
        items.append({"prompt": row["test_prompt"], "cot_text": cot,
                       "label": row["correct_answer"], "example_id": row["example_id"]})
    _print_dist("atypical_answer_riya", items)
    return items


def load_cybercrime_ood() -> list[dict]:
    """100 items, benign/cybercrime (50/50)."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/cot-oracle-eval-cybercrime-ood", split="train")
    items = []
    for row in ds:
        cot = row.get("meta_qwen3_8b_test_response") or ""
        if not cot.strip():
            continue
        items.append({"prompt": row["test_prompt"], "cot_text": cot,
                       "label": row["correct_answer"], "example_id": row["example_id"]})
    _print_dist("cybercrime_ood", items)
    return items


def load_reasoning_termination_riya() -> list[dict]:
    """84 items, will_continue/will_terminate (42/42). Uses partial CoT prefixes."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/cot-oracle-eval-reasoning-termination-riya", split="train")
    items = []
    for row in ds:
        cot = row.get("meta_cot_prefix") or ""
        if not cot.strip():
            continue
        items.append({"prompt": row["test_prompt"], "cot_text": cot,
                       "label": row["correct_answer"], "example_id": row["example_id"]})
    _print_dist("reasoning_termination_riya", items)
    return items


def load_sentence_insertion() -> list[dict]:
    """100 items, insertion/no_insertion (50/50)."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/cot-oracle-eval-sentence-insertion", split="train")
    items = []
    for row in ds:
        cot = row.get("meta_spliced_cot_text") or ""
        if not cot.strip():
            continue
        label = "insertion" if row["meta_is_insertion"] else "no_insertion"
        items.append({"prompt": row["test_prompt"], "cot_text": cot,
                       "label": label, "example_id": row["example_id"]})
    _print_dist("sentence_insertion", items)
    return items


def load_sycophancy_v2_riya() -> list[dict]:
    """100 items, sycophantic/non_sycophantic (50/50)."""
    from datasets import load_dataset
    ds = load_dataset(f"{HF_ORG}/cot-oracle-eval-sycophancy-v2-riya", split="train")
    label_map = {"non_sycophantic": "independent", "low_sycophantic": "influenced",
                 "high_sycophantic": "influenced", "sycophantic": "influenced"}
    items = []
    for row in ds:
        cot = row.get("meta_qwen3_8b_test_response") or row.get("meta_representative_response") or ""
        if not cot.strip():
            continue
        bl = label_map.get(row.get("meta_label", ""))
        if bl is None:
            continue
        items.append({"prompt": row["test_prompt"], "cot_text": cot,
                       "label": bl, "example_id": row["example_id"]})
    _print_dist("sycophancy_v2_riya", items)
    return items


def _print_dist(name: str, items: list[dict]) -> None:
    dist = Counter(item["label"] for item in items)
    print(f"  {name}: {len(items)} items, labels: {dict(dist)}")


# ============================================================
# Phase 2: Activation extraction (cached)
# ============================================================

def _extract_multilayer_activations(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    positions: list[int], layers: list[int],
) -> dict[int, torch.Tensor]:
    """Single forward pass extracting activations at positions from multiple layers.
    Returns {layer: [K, D]} on CPU.
    """
    from core.ao import get_hf_submodule, EarlyStopException

    submodules = {l: get_hf_submodule(model, l) for l in layers}
    max_layer = max(layers)
    acts: dict[int, torch.Tensor] = {}
    mod_to_layer = {id(s): l for l, s in submodules.items()}

    def hook_fn(module, _inputs, outputs):
        layer = mod_to_layer[id(module)]
        raw = outputs[0] if isinstance(outputs, tuple) else outputs
        acts[layer] = raw[0, positions, :].detach().cpu()
        if layer == max_layer:
            raise EarlyStopException()

    handles = [s.register_forward_hook(hook_fn) for s in submodules.values()]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except EarlyStopException:
        pass
    finally:
        for h in handles:
            h.remove()
    return acts


def extract_and_cache(
    items: list[dict], dataset_name: str, model, tokenizer, device: str = "cuda",
) -> list[dict]:
    """Extract multi-layer activations, mean-pool per layer, cache to disk."""
    cache_subdir = CACHE_DIR / dataset_name
    cache_subdir.mkdir(parents=True, exist_ok=True)

    successful, skipped = [], 0
    for item in tqdm(items, desc=f"Extracting {dataset_name}"):
        cache_path = cache_subdir / f"{item['example_id']}.pt"

        if cache_path.exists():
            try:
                item["pooled_acts"] = torch.load(cache_path, map_location="cpu", weights_only=True)
                successful.append(item)
                continue
            except Exception:
                cache_path.unlink(missing_ok=True)

        full_text = build_full_text_from_prompt_and_cot(tokenizer, item["prompt"], item["cot_text"])
        messages = [{"role": "user", "content": item["prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=STRIDE)

        if len(positions) < 2:
            skipped += 1
            continue

        tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_tensor = tok_out["input_ids"].to(device)
        attn_mask = tok_out["attention_mask"].to(device)
        seq_len = input_tensor.shape[1]
        positions = [p for p in positions if p < seq_len]
        if len(positions) < 2:
            skipped += 1
            continue

        acts_by_layer = _extract_multilayer_activations(model, input_tensor, attn_mask, positions, LAYERS)
        pooled = {layer: acts.mean(dim=0) for layer, acts in acts_by_layer.items()}
        torch.save(pooled, cache_path)
        item["pooled_acts"] = pooled
        successful.append(item)

    print(f"  {dataset_name}: {len(successful)} extracted, {skipped} skipped")
    return successful


def load_cached_activations(items: list[dict], dataset_name: str) -> list[dict]:
    """Load cached activations from disk."""
    cache_subdir = CACHE_DIR / dataset_name
    successful, missing = [], 0
    for item in items:
        cache_path = cache_subdir / f"{item['example_id']}.pt"
        if cache_path.exists():
            try:
                item["pooled_acts"] = torch.load(cache_path, map_location="cpu", weights_only=True)
                successful.append(item)
            except Exception:
                missing += 1
        else:
            missing += 1
    print(f"  {dataset_name}: {len(successful)} loaded from cache, {missing} missing")
    return successful


# ============================================================
# Phase 3: Probes
# ============================================================

def _standardize(X_tr, X_te):
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / std, (X_te - mu) / std, mu, std


def _build_features(items, layer_list):
    return torch.stack([torch.cat([it["pooled_acts"][l] for l in layer_list], dim=-1) for it in items]).float()


def _build_layer_features_3d(items, layers):
    return torch.stack([torch.stack([it["pooled_acts"][l] for l in layers]) for it in items]).float()


def _labels_to_tensor(items):
    labels_unique = sorted(set(it["label"] for it in items))
    l2i = {l: i for i, l in enumerate(labels_unique)}
    return torch.tensor([l2i[it["label"]] for it in items]), labels_unique, l2i


def _compute_class_weights(y, n_classes):
    counts = torch.bincount(y, minlength=n_classes).float()
    return counts.sum() / (n_classes * counts.clamp_min(1))


def train_linear_probe(X_tr, y_tr, X_te, n_classes, *, lr=LINEAR_LR, epochs=LINEAR_EPOCHS,
                        weight_decay=LINEAR_WD, class_weights=None, device="cuda"):
    """Train linear probe. Returns (test_preds, probe, mu, std)."""
    X_tr_s, X_te_s, mu, std = _standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device)
    probe = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(X_tr_s.shape[0], device=device)
        for start in range(0, X_tr_s.shape[0], 512):
            idx = perm[start:start + 512]
            opt.zero_grad(set_to_none=True)
            loss_fn(probe(X_tr_s[idx]), y_tr_d[idx]).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_te_s).argmax(1).cpu()
    return preds, probe, mu.cpu(), std.cpu()


class AttentionProbe(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, hidden_dim, n_outputs):
        super().__init__()
        self.layer_proj = nn.Linear(d_model, hidden_dim)
        self.layer_pos_embed = nn.Embedding(n_layers, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_outputs))

    def forward(self, x):
        h = self.layer_proj(x)
        h = h + self.layer_pos_embed(torch.arange(x.shape[1], device=x.device))
        h, w = self.attention(h, h, h)
        return self.head(h.mean(dim=1)), w


def train_attention_probe(X_tr, y_tr, X_te, n_classes, *, n_heads=ATTN_HEADS,
                           hidden_dim=ATTN_HIDDEN, lr=ATTN_LR, epochs=ATTN_EPOCHS,
                           patience=ATTN_PATIENCE, class_weights=None, device="cuda"):
    """Train attention probe. Returns (test_preds, attn_weights)."""
    X_tr_d, X_te_d = X_tr.to(device), X_te.to(device)
    y_tr_d = y_tr.to(device)

    model = AttentionProbe(X_tr.shape[2], X_tr.shape[1], n_heads, hidden_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_loss, patience_ctr, best_state = float("inf"), 0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(X_tr_d.shape[0], device=device)
        for start in range(0, X_tr_d.shape[0], 128):
            idx = perm[start:start + 128]
            opt.zero_grad(set_to_none=True)
            logits, _ = model(X_tr_d[idx])
            loss_fn(logits, y_tr_d[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tl = loss_fn(model(X_tr_d)[0], y_tr_d).item()
        if tl < best_loss:
            best_loss, patience_ctr = tl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds, w = model(X_te_d)
        return preds.argmax(1).cpu(), w.cpu()


def compute_metrics(preds, labels, labels_unique):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
    pred_strs = [labels_unique[p] for p in preds.tolist()]
    gt_strs = [labels_unique[g] for g in labels.tolist()]
    acc = accuracy_score(gt_strs, pred_strs)
    bal_acc = balanced_accuracy_score(gt_strs, pred_strs)
    prec, rec, f1, sup = precision_recall_fscore_support(
        gt_strs, pred_strs, labels=labels_unique, zero_division=0)
    return {
        "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4), "n_items": len(preds),
        "per_class": {label: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                               "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                      for i, label in enumerate(labels_unique)},
    }


# ============================================================
# Runners
# ============================================================

def run_kfold_probes(items: list[dict], dataset_name: str, *, n_folds: int = 5,
                     class_weights: torch.Tensor | None = None, device: str = "cuda") -> dict:
    """Run 5-fold stratified CV with linear probes (per-layer + concat) and attention probe."""
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name} — {len(items)} items, {n_folds}-fold CV")
    print(f"{'=' * 60}")

    y_all, labels_unique, l2i = _labels_to_tensor(items)
    n_classes = len(labels_unique)

    # Ensure we have enough items per class for k-fold
    min_class = min(Counter(y_all.tolist()).values())
    actual_folds = min(n_folds, min_class)
    if actual_folds < 2:
        print(f"  SKIP: min class count {min_class} too small for CV")
        return {"skipped": True, "reason": f"min_class={min_class}"}

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    layer_configs = [(f"layer_{l}", [l]) for l in LAYERS] + [("concat_all", LAYERS)]

    results = {}

    # Linear probes
    for layer_name, layer_list in layer_configs:
        X_all = _build_features(items, layer_list)
        all_preds = torch.zeros(len(items), dtype=torch.long)

        for train_idx, test_idx in skf.split(X_all.numpy(), y_all.numpy()):
            train_idx_t = torch.tensor(train_idx)
            test_idx_t = torch.tensor(test_idx)
            preds, _, _, _ = train_linear_probe(
                X_all[train_idx_t], y_all[train_idx_t], X_all[test_idx_t], n_classes,
                class_weights=class_weights, device=device)
            all_preds[test_idx_t] = preds

        metrics = compute_metrics(all_preds, y_all, labels_unique)
        results[f"linear_{layer_name}"] = metrics
        print(f"  linear_{layer_name}: acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}")

    # Attention probe
    X_all_3d = _build_layer_features_3d(items, LAYERS)
    all_preds_attn = torch.zeros(len(items), dtype=torch.long)
    all_attn_weights = []

    for train_idx, test_idx in skf.split(X_all_3d.numpy().reshape(len(items), -1), y_all.numpy()):
        train_idx_t = torch.tensor(train_idx)
        test_idx_t = torch.tensor(test_idx)
        preds, w = train_attention_probe(
            X_all_3d[train_idx_t], y_all[train_idx_t], X_all_3d[test_idx_t], n_classes,
            class_weights=class_weights, device=device)
        all_preds_attn[test_idx_t] = preds
        all_attn_weights.append(w.mean(dim=0))

    attn_metrics = compute_metrics(all_preds_attn, y_all, labels_unique)
    if all_attn_weights:
        mean_attn = torch.stack(all_attn_weights).mean(dim=0)
        attn_per_layer = mean_attn.sum(dim=0)
        attn_metrics["layer_importance"] = {
            f"layer_{LAYERS[i]}": round(attn_per_layer[i].item(), 4) for i in range(len(LAYERS))}
    results["attention_probe"] = attn_metrics
    print(f"  attention_probe: acc={attn_metrics['accuracy']:.3f}, bal_acc={attn_metrics['balanced_accuracy']:.3f}")

    return results


def run_train_test_probes(train_items: list[dict], test_items: list[dict], dataset_name: str,
                          *, device: str = "cuda") -> dict:
    """Train on explicit train split, eval on test split."""
    print(f"\n{'=' * 60}")
    print(f"  {dataset_name} — train {len(train_items)}, test {len(test_items)}")
    print(f"{'=' * 60}")

    y_tr, labels_unique, l2i = _labels_to_tensor(train_items)
    y_te = torch.tensor([l2i[it["label"]] for it in test_items])
    n_classes = len(labels_unique)
    class_weights = _compute_class_weights(y_tr, n_classes)
    print(f"  Classes: {labels_unique}, weights: {dict(zip(labels_unique, class_weights.tolist()))}")

    results = {}
    layer_configs = [(f"layer_{l}", [l]) for l in LAYERS] + [("concat_all", LAYERS)]

    for layer_name, layer_list in layer_configs:
        X_tr = _build_features(train_items, layer_list)
        X_te = _build_features(test_items, layer_list)
        preds, _, _, _ = train_linear_probe(X_tr, y_tr, X_te, n_classes,
                                             class_weights=class_weights, device=device)
        metrics = compute_metrics(preds, y_te, labels_unique)
        results[f"linear_{layer_name}"] = metrics
        print(f"  linear_{layer_name}: acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}")

    X_tr_3d = _build_layer_features_3d(train_items, LAYERS)
    X_te_3d = _build_layer_features_3d(test_items, LAYERS)
    preds, w = train_attention_probe(X_tr_3d, y_tr, X_te_3d, n_classes,
                                      class_weights=class_weights, device=device)
    attn_metrics = compute_metrics(preds, y_te, labels_unique)
    mean_attn = w.mean(dim=0)
    attn_per_layer = mean_attn.sum(dim=0)
    attn_metrics["layer_importance"] = {
        f"layer_{LAYERS[i]}": round(attn_per_layer[i].item(), 4) for i in range(len(LAYERS))}
    results["attention_probe"] = attn_metrics
    print(f"  attention_probe: acc={attn_metrics['accuracy']:.3f}, bal_acc={attn_metrics['balanced_accuracy']:.3f}")

    return results


def run_hint_admission_split(items: list[dict], device: str) -> tuple[dict, dict]:
    """hint_admission 80/20 split. Returns (results, trained_probes_for_transfer)."""
    print(f"\n{'=' * 60}")
    print(f"  hint_admission — 80/20 split ({len(items)} items)")
    print(f"{'=' * 60}")

    rng = random.Random(42)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    split_idx = int(0.8 * len(items))
    train_items = [items[i] for i in indices[:split_idx]]
    test_items = [items[i] for i in indices[split_idx:]]

    y_tr, labels_unique, l2i = _labels_to_tensor(train_items)
    y_te = torch.tensor([l2i[it["label"]] for it in test_items])
    n_classes = len(labels_unique)
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}, Classes: {labels_unique}")

    results = {}
    trained_probes = {}
    layer_configs = [(f"layer_{l}", [l]) for l in LAYERS] + [("concat_all", LAYERS)]

    for layer_name, layer_list in layer_configs:
        X_tr = _build_features(train_items, layer_list)
        X_te = _build_features(test_items, layer_list)
        preds, probe, mu, std = train_linear_probe(X_tr, y_tr, X_te, n_classes, device=device)
        metrics = compute_metrics(preds, y_te, labels_unique)
        results[f"linear_{layer_name}"] = metrics
        trained_probes[f"linear_{layer_name}"] = (probe, mu, std, l2i, labels_unique)
        print(f"  linear_{layer_name}: acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}")

    X_tr_3d = _build_layer_features_3d(train_items, LAYERS)
    X_te_3d = _build_layer_features_3d(test_items, LAYERS)
    preds, w = train_attention_probe(X_tr_3d, y_tr, X_te_3d, n_classes, device=device)
    attn_metrics = compute_metrics(preds, y_te, labels_unique)
    mean_attn = w.mean(dim=0)
    attn_per_layer = mean_attn.sum(dim=0)
    attn_metrics["layer_importance"] = {
        f"layer_{LAYERS[i]}": round(attn_per_layer[i].item(), 4) for i in range(len(LAYERS))}
    results["attention_probe"] = attn_metrics
    print(f"  attention_probe: acc={attn_metrics['accuracy']:.3f}, bal_acc={attn_metrics['balanced_accuracy']:.3f}")

    return results, trained_probes


def run_transfer(target_items: list[dict], target_name: str, trained_probes: dict,
                 device: str) -> dict:
    """Apply hint_admission-trained probes to a different dataset."""
    results = {}
    for probe_name, (probe, mu, std, l2i, labels_unique) in trained_probes.items():
        # Filter target items to labels that exist in hint_admission
        filtered = [it for it in target_items if it["label"] in l2i]
        if len(filtered) < 5:
            results[probe_name] = {"skipped": True, "reason": f"only {len(filtered)} matching labels"}
            continue

        if "concat_all" in probe_name:
            layer_list = LAYERS
        else:
            layer_list = [int(probe_name.split("_")[-1])]

        X_te = _build_features(filtered, layer_list)
        y_te = torch.tensor([l2i[it["label"]] for it in filtered])

        probe_device = next(probe.parameters()).device
        X_te_s = (X_te.to(probe_device) - mu.to(probe_device)) / std.to(probe_device)
        probe.eval()
        with torch.no_grad():
            preds = probe(X_te_s).argmax(1).cpu()

        metrics = compute_metrics(preds, y_te, labels_unique)
        results[probe_name] = metrics
    return results


# ============================================================
# Summary
# ============================================================

def print_summary(all_results: dict) -> None:
    print("\n" + "=" * 90)
    print("FULL RESULTS SUMMARY")
    print("=" * 90)

    header = f"{'Dataset':<30} {'Probe':<25} {'Acc':>6} {'Bal.Acc':>8} {'N':>6}"
    print(header)
    print("-" * len(header))

    for ds_name, ds_results in sorted(all_results.items()):
        if isinstance(ds_results, dict) and "skipped" not in ds_results:
            for probe_name, metrics in ds_results.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    acc = metrics["accuracy"]
                    bal_acc = metrics["balanced_accuracy"]
                    n = metrics["n_items"]
                    print(f"  {ds_name:<28} {probe_name:<25} {acc:>6.3f} {bal_acc:>8.3f} {n:>6}")
            print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train probes on all binary eval datasets")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Use cached activations only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Phase 1: Load all datasets ──
    print("Phase 1: Loading datasets from HuggingFace...")
    ha_items = load_hint_admission()
    tqa_unverb_train, tqa_unverb_test = load_truthfulqa_unverbalized()
    tqa_verb_train, tqa_verb_test = load_truthfulqa_verbalized()
    atypical_items = load_atypical_answer_riya()
    cybercrime_items = load_cybercrime_ood()
    termination_items = load_reasoning_termination_riya()
    insertion_items = load_sentence_insertion()
    sycophancy_items = load_sycophancy_v2_riya()

    # All small eval datasets: (name, items, cache_key)
    small_evals = [
        ("atypical_answer_riya", atypical_items, "atypical_answer_riya"),
        ("cybercrime_ood", cybercrime_items, "cybercrime_ood"),
        ("reasoning_termination_riya", termination_items, "reasoning_termination_riya"),
        ("sentence_insertion", insertion_items, "sentence_insertion"),
        ("sycophancy_v2_riya", sycophancy_items, "sycophancy_v2_riya"),
    ]

    # ── Phase 2: Extract activations ──
    print("\nPhase 2: Extracting activations...")
    model, tokenizer = None, None

    if not args.skip_extraction:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from core.ao import choose_attn_implementation

        print(f"  Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation=choose_attn_implementation(MODEL_NAME))
        model.eval()

    def _extract_or_load(items, cache_key):
        if args.skip_extraction:
            return load_cached_activations(items, cache_key)
        return extract_and_cache(items, cache_key, model, tokenizer, device=args.device)

    ha_items = _extract_or_load(ha_items, "hint_admission")
    tqa_unverb_train = _extract_or_load(tqa_unverb_train, "truthfulqa_unverbalized")
    tqa_unverb_test = _extract_or_load(tqa_unverb_test, "truthfulqa_unverbalized")
    tqa_verb_train = _extract_or_load(tqa_verb_train, "truthfulqa_verbalized")
    tqa_verb_test = _extract_or_load(tqa_verb_test, "truthfulqa_verbalized")

    for i, (name, items, cache_key) in enumerate(small_evals):
        small_evals[i] = (name, _extract_or_load(items, cache_key), cache_key)

    if model is not None:
        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Phase 3: Train probes ──
    print(f"\nPhase 3: Training probes on {args.device}...")
    all_results = {}

    # hint_admission 80/20
    if ha_items:
        ha_results, trained_probes = run_hint_admission_split(ha_items, device=args.device)
        all_results["hint_admission_80_20"] = ha_results
    else:
        trained_probes = {}

    # truthfulqa unverbalized train/test
    if tqa_unverb_train and tqa_unverb_test:
        all_results["truthfulqa_unverbalized"] = run_train_test_probes(
            tqa_unverb_train, tqa_unverb_test, "truthfulqa_unverbalized", device=args.device)

    # truthfulqa verbalized train/test
    if tqa_verb_train and tqa_verb_test:
        all_results["truthfulqa_verbalized"] = run_train_test_probes(
            tqa_verb_train, tqa_verb_test, "truthfulqa_verbalized", device=args.device)

    # All small evals: 5-fold CV
    for name, items, _ in small_evals:
        if items:
            all_results[f"{name}_5fold"] = run_kfold_probes(items, name, device=args.device)

    # Cross-dataset transfer: hint_admission → each eval
    if trained_probes:
        print(f"\n{'=' * 60}")
        print("  CROSS-DATASET TRANSFER (hint_admission → *)")
        print(f"{'=' * 60}")

        # Transfer to truthfulqa (both splits)
        for suffix, tr_items, te_items in [("unverbalized", tqa_unverb_train, tqa_unverb_test),
                                            ("verbalized", tqa_verb_train, tqa_verb_test)]:
            all_tqa = tr_items + te_items
            if all_tqa:
                transfer = run_transfer(all_tqa, f"truthfulqa_{suffix}", trained_probes, args.device)
                all_results[f"transfer_ha→truthfulqa_{suffix}"] = transfer
                best = max((m.get("balanced_accuracy", 0) for m in transfer.values() if isinstance(m, dict) and "balanced_accuracy" in m), default=0)
                print(f"  → truthfulqa_{suffix}: best bal_acc={best:.3f}")

        # Transfer to sycophancy (also has influenced/independent labels)
        for name, items, _ in small_evals:
            if not items:
                continue
            transfer = run_transfer(items, name, trained_probes, args.device)
            all_results[f"transfer_ha→{name}"] = transfer
            valid_metrics = [m for m in transfer.values() if isinstance(m, dict) and "balanced_accuracy" in m]
            if valid_metrics:
                best = max(m["balanced_accuracy"] for m in valid_metrics)
                print(f"  → {name}: best bal_acc={best:.3f}")
            else:
                print(f"  → {name}: no matching labels")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "hint_probe_results_all.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
