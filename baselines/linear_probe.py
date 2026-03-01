"""Baseline 1: Linear probe on activations."""

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from shared import BaselineInput, pool_activations
from scoring import EVAL_TYPES, score_binary, score_ranking


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


def _train_regressor_fold(X_tr, y_tr, X_te, *, lr, epochs, weight_decay, device):
    """Train linear regressor on one fold. Returns predictions on test set."""
    X_tr_s, X_te_s, _, _ = _standardize(X_tr.to(device), X_te.to(device))
    y_tr_d = y_tr.to(device).float()

    probe = nn.Linear(X_tr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        loss_fn(probe(X_tr_s).squeeze(-1), y_tr_d).backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        return probe(X_te_s).squeeze(-1).cpu()


def _build_features(inputs: list[BaselineInput], layer_list: list[int], pooling: str) -> torch.Tensor:
    """Build [N, D'] feature matrix."""
    return torch.stack([
        torch.cat([pool_activations(inp.activations_by_layer[l], pooling) for l in layer_list], dim=-1)
        for inp in inputs
    ]).float()


def run_linear_probe(
    inputs: list[BaselineInput], *,
    layers: list[int], k_folds: int = 5, lr: float = 0.01,
    epochs: int = 100, weight_decay: float = 0.0001,
    pooling: str = "mean", device: str = "cuda",
    test_inputs: list[BaselineInput] | None = None,
) -> dict:
    eval_name = inputs[0].eval_name
    eval_type = EVAL_TYPES[eval_name]

    if eval_type == "generation":
        return {"skipped": True, "reason": "linear probe cannot do generation"}

    layer_configs = [(f"layer_{l}", [l]) for l in layers] + [("concat_all", layers)]
    all_results = {}
    traces = []

    for layer_name, layer_list in tqdm(layer_configs, desc="Linear probe layers"):
        all_inputs = inputs + test_inputs if test_inputs else inputs
        labels_unique = sorted(set(inp.ground_truth_label for inp in all_inputs))
        label_to_idx = {l: i for i, l in enumerate(labels_unique)}
        n_classes = len(labels_unique)

        if eval_type in ("binary", "multiclass"):
            if test_inputs is not None:
                # Train/test split mode — no k-fold
                X_train = _build_features(inputs, layer_list, pooling)
                X_test = _build_features(test_inputs, layer_list, pooling)
                y_train = torch.tensor([label_to_idx[inp.ground_truth_label] for inp in inputs])

                preds = _train_classifier_fold(
                    X_train, y_train, X_test, n_classes,
                    lr=lr, epochs=epochs, weight_decay=weight_decay, device=device,
                )
                all_preds = [labels_unique[p.item()] for p in preds]
                gt_labels = [inp.ground_truth_label for inp in test_inputs]
                metrics = score_binary(all_preds, gt_labels)
                all_results[layer_name] = metrics

                for i in range(min(10, len(test_inputs))):
                    traces.append({
                        "layer": layer_name, "example_id": test_inputs[i].example_id,
                        "prediction": all_preds[i], "ground_truth": test_inputs[i].ground_truth_label,
                    })
            else:
                # K-fold CV mode (existing behavior)
                X_all = _build_features(inputs, layer_list, pooling)
                y_all = torch.tensor([label_to_idx[inp.ground_truth_label] for inp in inputs])

                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                all_preds = [None] * len(inputs)

                for train_idx, test_idx in skf.split(X_all.numpy(), y_all.numpy()):
                    preds = _train_classifier_fold(
                        X_all[train_idx], y_all[train_idx], X_all[test_idx], n_classes,
                        lr=lr, epochs=epochs, weight_decay=weight_decay, device=device,
                    )
                    for i, idx in enumerate(test_idx):
                        all_preds[idx] = labels_unique[preds[i].item()]

                gt_labels = [inp.ground_truth_label for inp in inputs]
                metrics = score_binary(all_preds, gt_labels)
                all_results[layer_name] = metrics

                for i in range(min(10, len(inputs))):
                    traces.append({
                        "layer": layer_name, "example_id": inputs[i].example_id,
                        "prediction": all_preds[i], "ground_truth": inputs[i].ground_truth_label,
                    })

        elif eval_type == "ranking":
            # Per-chunk regression: predict importance scores
            # Each item has chunk_position_indices mapping chunks to activation positions
            items_with_chunks = [
                inp for inp in inputs
                if inp.metadata.get("chunk_position_indices") and inp.metadata.get("importance_scores")
            ]
            if not items_with_chunks:
                all_results[layer_name] = {"skipped": True, "reason": "no ranking data"}
                continue

            kf = KFold(n_splits=min(k_folds, len(items_with_chunks)), shuffle=True, random_state=42)
            all_pred_scores = [None] * len(items_with_chunks)
            all_gt_scores = [inp.metadata["importance_scores"] for inp in items_with_chunks]

            for train_idx, test_idx in kf.split(range(len(items_with_chunks))):
                # Flatten training chunks into (activation, score) pairs
                X_tr_list, y_tr_list = [], []
                for i in train_idx:
                    inp = items_with_chunks[i]
                    chunk_indices = inp.metadata["chunk_position_indices"]
                    scores = inp.metadata["importance_scores"]
                    for ci, score in zip(chunk_indices, scores):
                        feat = torch.cat([inp.activations_by_layer[l][ci] for l in layer_list], dim=-1)
                        X_tr_list.append(feat)
                        y_tr_list.append(score)

                X_tr = torch.stack(X_tr_list).float()
                y_tr = torch.tensor(y_tr_list).float()

                # Predict per-chunk scores for test items
                for i in test_idx:
                    inp = items_with_chunks[i]
                    chunk_indices = inp.metadata["chunk_position_indices"]
                    X_te_list = [
                        torch.cat([inp.activations_by_layer[l][ci] for l in layer_list], dim=-1)
                        for ci in chunk_indices
                    ]
                    X_te = torch.stack(X_te_list).float()
                    preds = _train_regressor_fold(X_tr, y_tr, X_te, lr=lr, epochs=epochs, weight_decay=weight_decay, device=device)
                    all_pred_scores[i] = preds.tolist()

            pred_filtered = [p for p in all_pred_scores if p is not None]
            gt_filtered = [g for p, g in zip(all_pred_scores, all_gt_scores) if p is not None]
            metrics = score_ranking(pred_filtered, gt_filtered)
            all_results[layer_name] = metrics

    return {"per_layer": all_results, "traces": traces, "n_items": len(inputs)}
