"""Qwen-architecture attention probe: joint attention over positions AND layers.

Per-layer SwiGLU MLPs + learned layer embeddings → concatenate all layers into a
single sequence → joint multi-head self-attention (32 heads) → masked mean-pool →
LayerNorm + Linear classification head.

Unified API: accepts test_data + activations from eval_comprehensive.
Trains on a separate train split, evaluates on test (matching linear_probe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AttentionProbe(nn.Module):
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

    def forward(self, inputs: list[dict[int, torch.Tensor]], return_attention: bool = False):
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
        joint_seq, attn_weights = self.joint_attn(joint_seq, joint_seq, joint_seq, key_padding_mask=joint_mask, average_attn_weights=True)

        valid = ~joint_mask
        pooled = (joint_seq * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        logits = self.head(pooled)
        if return_attention:
            return logits, attn_weights, valid
        return logits


def _train_attention_probe(
    train_acts, y_train, test_acts, *,
    layers, n_outputs, lr, epochs, patience, val_frac=0.2, batch_size, device,
    wandb_run=None, tag="",
):
    """Train AttentionProbe with val-based early stopping. Returns test predictions."""
    n_val = max(1, int(len(train_acts) * val_frac))
    perm = torch.randperm(len(train_acts))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_acts = [train_acts[i] for i in tr_idx]
    val_acts = [train_acts[i] for i in val_idx]
    y_tr_d = y_train[tr_idx].to(device)
    y_val_d = y_train[val_idx].to(device)

    model = AttentionProbe(layers, n_outputs=n_outputs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(tr_acts))
        for start in range(0, len(tr_acts), batch_size):
            idx = indices[start:start + batch_size]
            batch_acts = [tr_acts[i] for i in idx]
            batch_y = y_tr_d[idx]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(batch_acts), batch_y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_logits = torch.cat([model(tr_acts[s:s + batch_size]) for s in range(0, len(tr_acts), batch_size)])
            val_logits = torch.cat([model(val_acts[s:s + batch_size]) for s in range(0, len(val_acts), batch_size)])
            train_loss = loss_fn(tr_logits, y_tr_d).item()
            val_loss = loss_fn(val_logits, y_val_d).item()
            train_acc = (tr_logits.argmax(1) == y_tr_d).float().mean().item()
            val_acc = (val_logits.argmax(1) == y_val_d).float().mean().item()

        if wandb_run:
            wandb_run.log({
                f"probe/{tag}/train_loss": train_loss, f"probe/{tag}/train_acc": train_acc,
                f"probe/{tag}/val_loss": val_loss, f"probe/{tag}/val_acc": val_acc,
                "probe/epoch": epoch,
            })

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
        all_preds = []
        for start in range(0, len(test_acts), batch_size):
            all_preds.append(model(test_acts[start:start + batch_size]))
        return torch.cat(all_preds).argmax(1).cpu()


def run_attention_probe(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    *,
    train_data: list[dict] | None = None,
    train_activations: list[torch.Tensor] | None = None,
    lr: float = 0.0001,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    wandb_run=None,
) -> list[str]:
    """Train attention probe on train split, evaluate on test. Returns list[str] predictions.

    Only supports BINARY tasks; returns ["?"] * N for non-binary.
    """
    from tasks import ScoringMode
    if task_def.scoring != ScoringMode.BINARY:
        return ["?"] * len(test_data)
    if not train_data or not train_activations:
        return ["?"] * len(test_data)

    from linear_probe import _extract_label

    train_labels = [_extract_label(d["target_response"], task_def) for d in train_data]
    test_labels = [_extract_label(d["target_response"], task_def) for d in test_data]
    labels_unique = sorted(set(train_labels))
    if len(labels_unique) < 2:
        return ["?"] * len(test_data)
    label_to_idx = {l: i for i, l in enumerate(labels_unique)}
    n_classes = len(labels_unique)

    train_acts = [split_activations_by_layer(a, layers) for a in train_activations]
    test_acts = [split_activations_by_layer(a, layers) for a in activations]
    y_train = torch.tensor([label_to_idx[l] for l in train_labels])

    preds = _train_attention_probe(
        train_acts, y_train, test_acts,
        layers=layers, n_outputs=n_classes, lr=lr, epochs=epochs,
        patience=patience, batch_size=batch_size, device=device,
        wandb_run=wandb_run, tag=task_def.name,
    )
    return [labels_unique[p.item()] for p in preds]
