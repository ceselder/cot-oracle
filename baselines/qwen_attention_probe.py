"""Baseline: Qwen-architecture attention probe over raw position sequences.

Unlike attention_probe.py (which mean-pools positions first, then attends over layers),
this probe operates on the raw per-layer position sequences using Qwen3-8B dimensions:
32 attention heads, head_dim=128, hidden_size=4096, SwiGLU MLP (4096→12288→4096).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm.auto import tqdm

from shared import BaselineInput
from scoring import EVAL_TYPES, score_binary, score_ranking


class QwenSwiGLU(nn.Module):
    """SwiGLU MLP matching Qwen3-8B architecture (fresh weights)."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenAttentionProbeLayer(nn.Module):
    """One per-layer module: SwiGLU MLP → multi-head self-attention over positions → mean-pool."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288, n_heads: int = 32):
        super().__init__()
        self.mlp = QwenSwiGLU(hidden_size, intermediate_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, K, D] → [B, D]. key_padding_mask: [B, K], True = padding."""
        h = self.mlp(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = ~key_padding_mask  # True for real positions
            h = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return h


class QwenAttentionProbe(nn.Module):
    """Full probe: L independent QwenAttentionProbeLayer + learned layer readout + classification head."""

    def __init__(self, layers: list[int], hidden_size: int = 4096, intermediate_size: int = 12288,
                 n_heads: int = 32, n_outputs: int = 2):
        super().__init__()
        self.layers = layers
        self.layer_modules = nn.ModuleList([
            QwenAttentionProbeLayer(hidden_size, intermediate_size, n_heads)
            for _ in layers
        ])
        self.layer_weights = nn.Parameter(torch.ones(len(layers)) / len(layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, n_outputs))

    def forward(self, inputs: list[dict[int, torch.Tensor]]) -> torch.Tensor:
        """inputs: list of B dicts {layer_idx: [K_i, D]}. Returns logits [B, n_outputs]."""
        device = self.layer_weights.device
        B = len(inputs)
        layer_outputs = []

        for li, layer_idx in enumerate(self.layers):
            acts_list = [inp[layer_idx] for inp in inputs]
            K_max = max(a.shape[0] for a in acts_list)
            D = acts_list[0].shape[1]

            padded = torch.zeros(B, K_max, D, device=device)
            mask = torch.ones(B, K_max, dtype=torch.bool, device=device)
            for i, a in enumerate(acts_list):
                K_i = a.shape[0]
                padded[i, :K_i] = a.to(device)
                mask[i, :K_i] = False

            layer_outputs.append(self.layer_modules[li](padded, key_padding_mask=mask))

        stacked = torch.stack(layer_outputs, dim=1)  # [B, L, D]
        weights = torch.softmax(self.layer_weights, dim=0)
        aggregated = (stacked * weights[None, :, None]).sum(dim=1)  # [B, D]
        return self.head(aggregated)


def _train_qwen_probe_fold(
    train_acts: list[dict[int, torch.Tensor]], y_train: torch.Tensor,
    test_acts: list[dict[int, torch.Tensor]], *,
    layers: list[int], n_outputs: int, lr: float, epochs: int,
    patience: int, batch_size: int, device: str, is_regression: bool = False,
) -> torch.Tensor:
    """Train QwenAttentionProbe on one fold. Returns test predictions."""
    model = QwenAttentionProbe(layers, n_outputs=n_outputs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    y_train_d = y_train.to(device).float() if is_regression else y_train.to(device)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        indices = torch.randperm(len(train_acts))
        for start in range(0, len(train_acts), batch_size):
            idx = indices[start:start + batch_size]
            batch_acts = [train_acts[i] for i in idx]
            batch_y = y_train_d[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(batch_acts)
            if is_regression:
                loss = loss_fn(logits.squeeze(-1), batch_y)
            else:
                loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()

        # Early stopping on training loss (no test labels during training)
        model.eval()
        with torch.no_grad():
            all_logits = []
            for start in range(0, len(train_acts), batch_size):
                batch = train_acts[start:start + batch_size]
                all_logits.append(model(batch))
            all_logits = torch.cat(all_logits)
            if is_regression:
                train_loss = loss_fn(all_logits.squeeze(-1), y_train_d).item()
            else:
                train_loss = loss_fn(all_logits, y_train_d).item()

        if train_loss < best_loss:
            best_loss = train_loss
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
        all_preds = []
        for start in range(0, len(test_acts), batch_size):
            batch = test_acts[start:start + batch_size]
            all_preds.append(model(batch))
        all_preds = torch.cat(all_preds)
        if is_regression:
            return all_preds.squeeze(-1).cpu()
        return all_preds.argmax(1).cpu()


def _extract_acts(inputs: list[BaselineInput], layers: list[int]) -> list[dict[int, torch.Tensor]]:
    """Extract per-example activation dicts for the given layers."""
    return [{l: inp.activations_by_layer[l] for l in layers} for inp in inputs]


def run_qwen_attention_probe(
    inputs: list[BaselineInput], *,
    layers: list[int] = (9, 18, 27), k_folds: int = 5,
    lr: float = 0.0001, epochs: int = 50, patience: int = 10,
    batch_size: int = 32, device: str = "cuda",
    test_inputs: list[BaselineInput] | None = None,
) -> dict:
    eval_name = inputs[0].eval_name
    eval_type = EVAL_TYPES[eval_name]

    if eval_type == "generation":
        return {"skipped": True, "reason": "qwen attention probe cannot do generation"}

    # Filter to available layers
    all_items = inputs + test_inputs if test_inputs else inputs
    available = set.intersection(*(set(inp.activations_by_layer.keys()) for inp in all_items))
    layers = [l for l in layers if l in available]

    traces = []

    if eval_type in ("binary", "multiclass"):
        labels_unique = sorted(set(inp.ground_truth_label for inp in all_items))
        label_to_idx = {l: i for i, l in enumerate(labels_unique)}
        n_classes = len(labels_unique)

        if test_inputs is not None:
            train_acts = _extract_acts(inputs, layers)
            test_acts = _extract_acts(test_inputs, layers)
            y_train = torch.tensor([label_to_idx[inp.ground_truth_label] for inp in inputs])

            preds = _train_qwen_probe_fold(
                train_acts, y_train, test_acts,
                layers=layers, n_outputs=n_classes, lr=lr, epochs=epochs,
                patience=patience, batch_size=batch_size, device=device,
            )
            all_preds = [labels_unique[p.item()] for p in preds]
            gt_labels = [inp.ground_truth_label for inp in test_inputs]
            metrics = score_binary(all_preds, gt_labels)

            for i in range(min(10, len(test_inputs))):
                traces.append({
                    "example_id": test_inputs[i].example_id,
                    "prediction": all_preds[i], "ground_truth": test_inputs[i].ground_truth_label,
                })
        else:
            all_acts = _extract_acts(inputs, layers)
            y_all = torch.tensor([label_to_idx[inp.ground_truth_label] for inp in inputs])
            all_preds = [None] * len(inputs)

            for train_idx, test_idx in tqdm(list(StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42).split(range(len(inputs)), y_all.numpy())), desc="Qwen attn probe folds"):
                train_acts = [all_acts[i] for i in train_idx]
                test_acts_fold = [all_acts[i] for i in test_idx]
                y_tr = y_all[train_idx]

                preds = _train_qwen_probe_fold(
                    train_acts, y_tr, test_acts_fold,
                    layers=layers, n_outputs=n_classes, lr=lr, epochs=epochs,
                    patience=patience, batch_size=batch_size, device=device,
                )
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

        kf = KFold(n_splits=min(k_folds, len(items_with_chunks)), shuffle=True, random_state=42)
        all_pred_scores = [None] * len(items_with_chunks)
        all_gt_scores = [inp.metadata["importance_scores"] for inp in items_with_chunks]

        for train_idx, test_idx in kf.split(range(len(items_with_chunks))):
            # Flatten training chunks: each chunk → single-position activation dict
            train_acts, y_tr_list = [], []
            for i in train_idx:
                inp = items_with_chunks[i]
                for ci, score in zip(inp.metadata["chunk_position_indices"], inp.metadata["importance_scores"]):
                    train_acts.append({l: inp.activations_by_layer[l][ci:ci+1] for l in layers})
                    y_tr_list.append(score)

            y_tr = torch.tensor(y_tr_list, dtype=torch.float32)

            for i in test_idx:
                inp = items_with_chunks[i]
                test_acts = [
                    {l: inp.activations_by_layer[l][ci:ci+1] for l in layers}
                    for ci in inp.metadata["chunk_position_indices"]
                ]
                preds = _train_qwen_probe_fold(
                    train_acts, y_tr, test_acts,
                    layers=layers, n_outputs=1, lr=lr, epochs=epochs,
                    patience=patience, batch_size=batch_size, device=device,
                    is_regression=True,
                )
                all_pred_scores[i] = preds.tolist()

        pred_filtered = [p for p in all_pred_scores if p is not None]
        gt_filtered = [g for p, g in zip(all_pred_scores, all_gt_scores) if p is not None]
        metrics = score_ranking(pred_filtered, gt_filtered)

    return {
        "metrics": metrics,
        "layers_used": layers,
        "traces": traces,
        "n_items": len(inputs),
    }
