"""Position encoding for activation vectors.

Multiplexes positional information into activation vectors so the oracle
knows the source position of each activation. RoPE is applied inside
attention to Q/K and not stored in the residual stream, so activations
don't carry an explicit position signal.

Formula: v_combined = v + alpha * ||v|| * pos_embed(position / total_length)

where pos_embed is a fixed sinusoidal encoding producing a unit-normalized
[d_model] vector. Standard Vaswani formula applied to continuous t in [0,1].
"""

import math
import torch
import torch.nn.functional as F


def sinusoidal_position_encoding(positions, total_length, d_model, max_freq=10000.0, device="cuda", dtype=torch.bfloat16):
    """Standard sinusoidal PE on normalized position t = pos/total_length.
    Returns unit-normalized [K, d_model]."""
    t = torch.tensor([p / total_length for p in positions], device=device, dtype=torch.float32)
    half_d = d_model // 2
    freqs = torch.exp(torch.arange(half_d, device=device, dtype=torch.float32) * -(math.log(max_freq) / half_d))
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)  # [K, half_d]
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [K, d_model]
    return F.normalize(pe, dim=-1).to(dtype=dtype)


def apply_position_encoding(vectors, source_positions, total_length, alpha=0.1):
    """v_combined = v + alpha * ||v|| * pe(pos/total)."""
    d_model = vectors.shape[-1]
    pe = sinusoidal_position_encoding(source_positions, total_length, d_model, device=vectors.device, dtype=vectors.dtype)
    norms = vectors.norm(dim=-1, keepdim=True)
    return vectors + alpha * norms * pe
