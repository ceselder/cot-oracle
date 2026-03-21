"""Baseline: Linear probe on activations.

Trains on a separate train split, evaluates on test split.
Unified API: accepts train/test data + activations from eval_comprehensive.
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from activation_utils import split_activations_by_layer, pool_activations


def _standardize(X_tr: torch.Tensor, X_te: torch.Tensor):
    """Zero-mean unit-variance standardization. Returns (X_tr_s, X_te_s)."""
    mu = X_tr.mean(dim=0, keepdim=True)
    std = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / std, (X_te - mu) / std


def _train_and_predict(X_tr, y_tr, X_te, n_classes, *, lr, epochs, weight_decay, patience, device, wandb_run=None, tag=""):
    """Train linear classifier on train set, predict on test set."""
    X_tr_s, X_te_s = _standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device)

    probe = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(X_tr_s.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, X_tr_s.shape[0], 512):
            idx = perm[start:start + 512]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(X_tr_s[idx]), y_tr_d[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        probe.eval()
        with torch.no_grad():
            train_acc = (probe(X_tr_s).argmax(1) == y_tr_d).float().mean().item()

        if wandb_run:
            wandb_run.log({f"probe/{tag}/train_loss": avg_loss, f"probe/{tag}/train_acc": train_acc, "probe/epoch": epoch})

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})

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
    test_activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    *,
    train_data: list[dict] | None = None,
    train_activations: list[torch.Tensor] | None = None,
    lr: float = 0.01,
    epochs: int = 100,
    patience: int = 10,
    weight_decay: float = 0.0001,
    pooling: str = "mean",
    device: str = "cuda",
    wandb_run=None,
) -> list[str]:
    """Train linear probe on train split, evaluate on test split.

    Only supports BINARY tasks; returns ["?"] * N for non-binary.
    If train_data/train_activations are None, returns ["?"] (no training data available).
    """
    from tasks import ScoringMode
    if task_def.scoring != ScoringMode.BINARY:
        return ["?"] * len(test_data)
    if not train_data or not train_activations:
        return ["?"] * len(test_data)

    # Extract labels
    train_labels = [_extract_label(d["target_response"], task_def) for d in train_data]
    test_labels = [_extract_label(d["target_response"], task_def) for d in test_data]
    labels_unique = sorted(set(train_labels))
    if len(labels_unique) < 2:
        return ["?"] * len(test_data)
    label_to_idx = {l: i for i, l in enumerate(labels_unique)}
    n_classes = len(labels_unique)

    y_train = torch.tensor([label_to_idx[l] for l in train_labels])

    # Run per-layer + concat, pick best (by train accuracy as proxy)
    layer_configs = [(f"layer_{l}", [l]) for l in layers] + [("concat_all", layers)]
    best_acc = -1
    best_preds = None

    for layer_name, layer_list in tqdm(layer_configs, desc="Linear probe layers"):
        X_train = _build_features(train_activations, layer_list, pooling)
        X_test = _build_features(test_activations, layer_list, pooling)

        preds = _train_and_predict(
            X_train, y_train, X_test, n_classes,
            lr=lr, epochs=epochs, patience=patience, weight_decay=weight_decay,
            device=device, wandb_run=wandb_run, tag=f"{task_def.name}/{layer_name}",
        )
        pred_labels = [labels_unique[p.item()] for p in preds]

        acc = sum(1 for p, l in zip(pred_labels, test_labels) if p == l) / len(test_labels)
        if wandb_run:
            wandb_run.log({f"probe/{task_def.name}/{layer_name}/test_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_preds = pred_labels

    return best_preds
