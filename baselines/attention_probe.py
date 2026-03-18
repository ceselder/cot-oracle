"""Qwen-architecture attention probe: joint attention over positions AND layers.

Per-layer SwiGLU MLPs + learned layer embeddings → concatenate all layers into a
single sequence → joint multi-head self-attention (32 heads) → masked mean-pool →
LayerNorm + Linear classification head.

Unified API: accepts test_data + activations from eval_comprehensive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from activation_utils import split_activations_by_layer


class QwenSwiGLU(nn.Module):
    """SwiGLU MLP matching Qwen3-8B architecture (fresh weights)."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 12288):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _subsample_positions(acts: torch.Tensor, max_k: int) -> torch.Tensor:
    """Subsample [K, D] → [max_k, D] uniformly, always keeping the last position."""
    K = acts.shape[0]
    if K <= max_k:
        return acts
    idx = torch.linspace(0, K - 2, max_k - 1).long()
    idx = torch.cat([idx, torch.tensor([K - 1])])
    return acts[idx]


class QwenAttentionProbe(nn.Module):
    """Joint position-layer probe with self-attention."""

    def __init__(self, layers: list[int], hidden_size: int = 4096, intermediate_size: int = 12288,
                 n_heads: int = 32, n_outputs: int = 2, max_positions_per_layer: int = 200):
        super().__init__()
        self.layers = layers
        self.max_positions_per_layer = max_positions_per_layer
        self.layer_mlps = nn.ModuleList([QwenSwiGLU(hidden_size, intermediate_size) for _ in layers])
        self.layer_embedding = nn.Embedding(len(layers), hidden_size)
        self.joint_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, n_outputs))

    def forward(self, inputs: list[dict[int, torch.Tensor]]) -> torch.Tensor:
        device = self.layer_embedding.weight.device
        dtype = self.layer_embedding.weight.dtype
        B = len(inputs)

        all_seqs = []
        all_masks = []

        for li, layer_idx in enumerate(self.layers):
            acts_list = [_subsample_positions(inp[layer_idx], self.max_positions_per_layer) for inp in inputs]
            K_max = max(a.shape[0] for a in acts_list)
            D = acts_list[0].shape[1]

            padded = torch.zeros(B, K_max, D, device=device, dtype=dtype)
            mask = torch.ones(B, K_max, dtype=torch.bool, device=device)
            for i, a in enumerate(acts_list):
                K_i = a.shape[0]
                padded[i, :K_i] = a.to(device=device, dtype=dtype)
                mask[i, :K_i] = False

            h = self.layer_mlps[li](padded)
            h = h + self.layer_embedding(torch.tensor(li, device=device))
            all_seqs.append(h)
            all_masks.append(mask)

        joint_seq = torch.cat(all_seqs, dim=1)
        joint_mask = torch.cat(all_masks, dim=1)
        joint_seq, _ = self.joint_attn(joint_seq, joint_seq, joint_seq, key_padding_mask=joint_mask)

        valid = ~joint_mask
        pooled = (joint_seq * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def _train_qwen_probe_fold(
    train_acts, y_train, test_acts, *,
    layers, n_outputs, lr, epochs, patience, batch_size, device,
):
    """Train QwenAttentionProbe on one fold. Returns test predictions."""
    model = QwenAttentionProbe(layers, n_outputs=n_outputs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y_train_d = y_train.to(device)

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
            loss_fn(model(batch_acts), batch_y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            all_logits = []
            for start in range(0, len(train_acts), batch_size):
                all_logits.append(model(train_acts[start:start + batch_size]))
            train_loss = loss_fn(torch.cat(all_logits), y_train_d).item()

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
            all_preds.append(model(test_acts[start:start + batch_size]))
        return torch.cat(all_preds).argmax(1).cpu()


def run_qwen_attention_probe(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    *,
    k_folds: int = 5,
    lr: float = 0.0001,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[str]:
    """Run Qwen attention probe via k-fold CV. Returns list[str] predictions.

    Only supports BINARY tasks; returns ["?"] * N for non-binary.
    """
    from tasks import ScoringMode
    if task_def.scoring != ScoringMode.BINARY:
        return ["?"] * len(test_data)

    from linear_probe import _extract_label

    labels = [_extract_label(d["target_response"], task_def) for d in test_data]
    labels_unique = sorted(set(labels))
    if len(labels_unique) < 2:
        return ["?"] * len(test_data)
    label_to_idx = {l: i for i, l in enumerate(labels_unique)}
    n_classes = len(labels_unique)

    # Convert unified [nK, D] to per-example {layer: [K, D]} dicts
    all_acts = [split_activations_by_layer(a, layers) for a in activations]
    y_all = torch.tensor([label_to_idx[l] for l in labels])
    all_preds = [None] * len(test_data)

    skf = StratifiedKFold(n_splits=min(k_folds, len(test_data)), shuffle=True, random_state=42)
    for train_idx, test_idx in tqdm(list(skf.split(range(len(test_data)), y_all.numpy())), desc="Qwen attn probe folds"):
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

    return all_preds
