"""Baseline 2: All-layer attention probe."""

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from shared import BaselineInput, pool_activations
from scoring import EVAL_TYPES, score_binary, score_ranking


class AttentionProbe(nn.Module):
    """Attention over layer-pooled activations for classification/regression."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, hidden_dim: int, n_outputs: int):
        super().__init__()
        self.layer_proj = nn.Linear(d_model, hidden_dim)
        self.layer_pos_embed = nn.Embedding(n_layers, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_outputs))

    def forward(self, x: torch.Tensor):
        """x: [B, n_layers, D] -> logits: [B, n_outputs], attn_weights: [B, n_layers, n_layers]."""
        h = self.layer_proj(x)  # [B, n_layers, hidden_dim]
        pos_ids = torch.arange(x.shape[1], device=x.device)
        h = h + self.layer_pos_embed(pos_ids)
        h, attn_weights = self.attention(h, h, h)
        h = h.mean(dim=1)  # [B, hidden_dim]
        return self.head(h), attn_weights


def _build_layer_features(inputs: list[BaselineInput], layers: list[int], pooling: str = "mean") -> torch.Tensor:
    """Build [N, n_layers, D] feature tensor by mean-pooling each layer's stride activations."""
    feats = []
    for inp in inputs:
        layer_vecs = [pool_activations(inp.activations_by_layer[l], pooling) for l in layers]
        feats.append(torch.stack(layer_vecs))  # [n_layers, D]
    return torch.stack(feats).float()  # [N, n_layers, D]


def _train_attention_probe_fold(
    X_tr, y_tr, X_te, *, n_heads, hidden_dim, n_outputs, lr, epochs, patience,
    device, is_regression=False,
):
    """Train attention probe on one fold. Returns test predictions and attention weights."""
    X_tr_d, X_te_d = X_tr.to(device), X_te.to(device)
    n_layers = X_tr.shape[1]
    d_model = X_tr.shape[2]

    model = AttentionProbe(d_model, n_layers, n_heads, hidden_dim, n_outputs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    y_tr_d = y_tr.to(device).float() if is_regression else y_tr.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(X_tr_d.shape[0], device=device)
        for start in range(0, X_tr_d.shape[0], 128):
            idx = perm[start:start + 128]
            opt.zero_grad(set_to_none=True)
            logits, _ = model(X_tr_d[idx])
            target = y_tr_d[idx]
            if is_regression:
                loss = loss_fn(logits.squeeze(-1), target)
            else:
                loss = loss_fn(logits, target)
            loss.backward()
            opt.step()

        # Validation loss on test set for early stopping
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_te_d)
            if is_regression:
                val_loss = nn.MSELoss()(val_logits.squeeze(-1), X_te_d.new_zeros(X_te_d.shape[0])).item()
            else:
                # Use training loss as proxy (no labels for test during training)
                train_logits, _ = model(X_tr_d)
                val_loss = loss_fn(train_logits, y_tr_d).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(X_te_d)
        if is_regression:
            return preds.squeeze(-1).cpu(), attn_weights.cpu()
        return preds.argmax(1).cpu(), attn_weights.cpu()


def run_attention_probe(
    inputs: list[BaselineInput], *,
    n_layers: int = 36, k_folds: int = 5, n_heads: int = 4,
    hidden_dim: int = 256, lr: float = 0.001, epochs: int = 50,
    patience: int = 10, device: str = "cuda",
) -> dict:
    eval_name = inputs[0].eval_name
    eval_type = EVAL_TYPES[eval_name]

    if eval_type == "generation":
        return {"skipped": True, "reason": "attention probe cannot do generation"}

    # Use all available layers (sorted)
    available_layers = sorted(set().union(*(inp.activations_by_layer.keys() for inp in inputs)))
    layers_to_use = available_layers[:n_layers]
    actual_n_layers = len(layers_to_use)

    X_all = _build_layer_features(inputs, layers_to_use)  # [N, n_layers, D]
    d_model = X_all.shape[2]

    traces = []
    all_attn_weights = []

    if eval_type == "binary":
        labels_unique = sorted(set(inp.ground_truth_label for inp in inputs))
        label_to_idx = {l: i for i, l in enumerate(labels_unique)}
        y_all = torch.tensor([label_to_idx[inp.ground_truth_label] for inp in inputs])
        n_classes = len(labels_unique)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_preds = [None] * len(inputs)

        for train_idx, test_idx in tqdm(list(skf.split(X_all.numpy(), y_all.numpy())), desc="Attention probe folds"):
            preds, attn_w = _train_attention_probe_fold(
                X_all[train_idx], y_all[train_idx], X_all[test_idx],
                n_heads=n_heads, hidden_dim=hidden_dim, n_outputs=n_classes,
                lr=lr, epochs=epochs, patience=patience, device=device,
            )
            all_attn_weights.append(attn_w.mean(dim=0))  # avg over test items
            for i, idx in enumerate(test_idx):
                all_preds[idx] = labels_unique[preds[i].item()]

        gt_labels = [inp.ground_truth_label for inp in inputs]
        metrics = score_binary(all_preds, gt_labels)

        for i in range(min(10, len(inputs))):
            traces.append({
                "example_id": inputs[i].example_id,
                "prediction": all_preds[i], "ground_truth": inputs[i].ground_truth_label,
            })

    elif eval_type == "ranking":
        items_with_chunks = [
            inp for inp in inputs
            if inp.metadata.get("chunk_position_indices") and inp.metadata.get("importance_scores")
        ]
        if not items_with_chunks:
            return {"skipped": True, "reason": "no ranking data"}

        # For ranking, use full activation as features, predict per-chunk scores
        # Simplified: use mean-pooled all-layer feature for chunk-level regression
        kf = KFold(n_splits=min(k_folds, len(items_with_chunks)), shuffle=True, random_state=42)
        all_pred_scores = [None] * len(items_with_chunks)
        all_gt_scores = [inp.metadata["importance_scores"] for inp in items_with_chunks]

        for train_idx, test_idx in kf.split(range(len(items_with_chunks))):
            # Flatten training chunks
            X_tr_list, y_tr_list = [], []
            for i in train_idx:
                inp = items_with_chunks[i]
                for ci, score in zip(inp.metadata["chunk_position_indices"], inp.metadata["importance_scores"]):
                    layer_acts = [inp.activations_by_layer[l][ci] for l in layers_to_use]
                    X_tr_list.append(torch.stack(layer_acts))  # [n_layers, D]
                    y_tr_list.append(score)

            X_tr = torch.stack(X_tr_list).float()  # [M, n_layers, D]
            y_tr = torch.tensor(y_tr_list).float()

            # Predict per-chunk scores for test items
            for i in test_idx:
                inp = items_with_chunks[i]
                chunk_indices = inp.metadata["chunk_position_indices"]
                X_te_list = [
                    torch.stack([inp.activations_by_layer[l][ci] for l in layers_to_use])
                    for ci in chunk_indices
                ]
                X_te = torch.stack(X_te_list).float()
                preds, _ = _train_attention_probe_fold(
                    X_tr, y_tr, X_te,
                    n_heads=n_heads, hidden_dim=hidden_dim, n_outputs=1,
                    lr=lr, epochs=epochs, patience=patience, device=device,
                    is_regression=True,
                )
                all_pred_scores[i] = preds.tolist()

        pred_filtered = [p for p in all_pred_scores if p is not None]
        gt_filtered = [g for p, g in zip(all_pred_scores, all_gt_scores) if p is not None]
        metrics = score_ranking(pred_filtered, gt_filtered)

    # Compute mean attention weights across folds for interpretability
    mean_attn = torch.stack(all_attn_weights).mean(dim=0) if all_attn_weights else None
    layer_importance = {}
    if mean_attn is not None:
        # Sum of attention received by each layer position
        attn_per_layer = mean_attn.sum(dim=0)  # [n_layers]
        for i, l in enumerate(layers_to_use):
            layer_importance[f"layer_{l}"] = attn_per_layer[i].item()

    return {
        "metrics": metrics,
        "layer_importance": layer_importance,
        "layers_used": layers_to_use,
        "traces": traces,
        "n_items": len(inputs),
    }
