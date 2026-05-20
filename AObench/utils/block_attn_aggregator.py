"""Block Attention Residual aggregator — AObench inference-side copy.

Mirrors `nl_probes/utils/block_attn_aggregator.py` (training side). At eval
time we never call backward, so we don't need the autograd-safety clones —
this version is the plain forward.

Loaded by `base_experiment.load_oracle_adapter` when a `block_aggregator.pt`
sits next to the LoRA adapter, and used by the AObench-side
`materialize_missing_steering_vectors` to compress the model's residual stream
into K block-vectors before injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttnAggregator(nn.Module):
    """K learned per-block softmax-attention compressions over per-layer activations.

    Every per-layer activation is L2-normalized to a unit vector before scoring,
    and the aggregator output is a convex combination of those unit vectors —
    so the entire attention math lives on the unit sphere. The steering hook
    downstream re-scales the direction to ‖orig‖ (norm-matched injection).

    Pure params: K * d_model floats in `Q`. No RMSNorm, no gain.
    """

    def __init__(
        self,
        block_layer_groups: list[list[int]],
        d_model: int,
        eps: float = 1e-6,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.block_layer_groups = [list(g) for g in block_layer_groups]
        self.num_blocks = len(self.block_layer_groups)
        self.d_model = d_model
        self.eps = eps
        # Initialised to random; will be overwritten by load_state_dict.
        self.Q = nn.Parameter(torch.randn(self.num_blocks, d_model) * init_scale)

    def forward(self, layer_acts: dict[int, torch.Tensor]) -> torch.Tensor:
        """layer_acts: dict {layer_idx: (B, P, d)} → (B, K, P, d)."""
        dtype = next(iter(layer_acts.values())).dtype
        B, P, d = next(iter(layer_acts.values())).shape
        assert d == self.d_model, f"d_model mismatch: got {d}, expected {self.d_model}"

        block_outputs = []
        for k, layer_indices in enumerate(self.block_layer_groups):
            stack = torch.stack([layer_acts[l] for l in layer_indices], dim=0)  # (L_k, B, P, d)
            unit = F.normalize(stack.float(), dim=-1, eps=self.eps).to(dtype)
            q_k = self.Q[k].to(dtype)  # (d,)
            logits = torch.einsum("lbpd,d->lbp", unit, q_k) / (self.d_model ** 0.5)
            weights = torch.softmax(logits, dim=0)  # (L_k, B, P)
            agg = torch.einsum("lbp,lbpd->bpd", weights, unit)  # (B, P, d)
            block_outputs.append(agg)
        return torch.stack(block_outputs, dim=0).transpose(0, 1).contiguous()


def load_block_aggregator_from_dir(
    lora_path: str,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> BlockAttnAggregator | None:
    """If `${lora_path}/block_aggregator.pt` exists, load + return the aggregator.

    Returns None if the file is absent (i.e. this is not a block-attn LoRA).
    """
    from pathlib import Path

    p = Path(lora_path) / "block_aggregator.pt"
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]
    block_layer_groups = payload["block_layer_groups"]
    saved_d = payload.get("d_model", d_model)
    assert saved_d == d_model, (
        f"block_aggregator.pt d_model={saved_d} mismatch with model d_model={d_model}"
    )
    agg = BlockAttnAggregator(block_layer_groups=block_layer_groups, d_model=d_model)
    agg.load_state_dict(state_dict, strict=True)
    agg = agg.to(device=device, dtype=dtype).eval()
    for p_ in agg.parameters():
        p_.requires_grad_(False)
    print(
        f"[block-attn AObench] loaded {p}: K={agg.num_blocks} blocks, "
        f"groups={agg.block_layer_groups}"
    )
    return agg
