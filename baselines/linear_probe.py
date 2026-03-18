"""Baseline: Linear probe on activations.

Unified API: accepts test_data + activations from eval_comprehensive.
"""

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from activation_utils import split_activations_by_layer, pool_activations


def _standardize(X_tr: torch.Tensor, X_te: torch.Tensor):
    """Zero-mean unit-variance standardization. Returns (X_tr_s, X_te_s, mu, std)."""
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / std, (X_te - mu) / std, mu, std


def _train_classifier_fold(X_tr, y_tr, X_te, n_classes, *, lr, epochs, weight_decay, device):
    """Train linear classifier on one fold. Returns predictions on test set."""
    X_tr_s, X_te_s, _, _ = _standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device)

    probe = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

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
        return probe(X_te_s).argmax(1).cpu()


def _extract_label(target_response: str, task_def) -> str:
    """Extract binary label from target_response using task_def keywords."""
    lower = target_response.lower()
    for kw in task_def.positive_keywords:
        if kw.lower() in lower:
            return task_def.positive_label or "positive"
    for kw in task_def.negative_keywords:
        if kw.lower() in lower:
            return task_def.negative_label or "negative"
    # Fallback: check if target_response IS the label
    if target_response.strip() in (task_def.positive_label, task_def.negative_label):
        return target_response.strip()
    return task_def.negative_label or "negative"


def _build_features(activations_list: list[torch.Tensor], layers: list[int], pooling: str) -> torch.Tensor:
    """Build [N, D'] feature matrix from unified [nK, D] activations."""
    feats = []
    for acts in activations_list:
        by_layer = split_activations_by_layer(acts, layers)
        feats.append(torch.cat([pool_activations(by_layer[l], pooling) for l in layers], dim=-1))
    return torch.stack(feats).float()


def run_linear_probe(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    *,
    k_folds: int = 5,
    lr: float = 0.01,
    epochs: int = 100,
    weight_decay: float = 0.0001,
    pooling: str = "mean",
    device: str = "cuda",
) -> list[str]:
    """Run linear probe via k-fold CV. Returns list[str] raw predictions.

    Only supports BINARY tasks; returns ["?"] * N for non-binary.
    """
    from tasks import ScoringMode
    if task_def.scoring != ScoringMode.BINARY:
        return ["?"] * len(test_data)

    # Extract labels
    labels = [_extract_label(d["target_response"], task_def) for d in test_data]
    labels_unique = sorted(set(labels))
    if len(labels_unique) < 2:
        return ["?"] * len(test_data)
    label_to_idx = {l: i for i, l in enumerate(labels_unique)}
    n_classes = len(labels_unique)

    # Run per-layer + concat, pick best
    layer_configs = [(f"layer_{l}", [l]) for l in layers] + [("concat_all", layers)]
    best_acc = -1
    best_preds = None

    for layer_name, layer_list in tqdm(layer_configs, desc="Linear probe layers"):
        X_all = _build_features(activations, layer_list, pooling)
        y_all = torch.tensor([label_to_idx[l] for l in labels])

        skf = StratifiedKFold(n_splits=min(k_folds, len(test_data)), shuffle=True, random_state=42)
        all_preds = [None] * len(test_data)

        for train_idx, test_idx in skf.split(X_all.numpy(), y_all.numpy()):
            preds = _train_classifier_fold(
                X_all[train_idx], y_all[train_idx], X_all[test_idx], n_classes,
                lr=lr, epochs=epochs, weight_decay=weight_decay, device=device,
            )
            for i, idx in enumerate(test_idx):
                all_preds[idx] = labels_unique[preds[i].item()]

        acc = sum(1 for p, l in zip(all_preds, labels) if p == l) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds

    return best_preds
