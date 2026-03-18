"""Shared activation utilities for baselines."""

import torch


def split_activations_by_layer(activations: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """Split unified [nK, D] tensor into {layer: [K, D]}.

    The unified format tiles positions: acts[0:K]=layer0, acts[K:2K]=layer1, etc.
    """
    n_layers = len(layers)
    K = activations.shape[0] // n_layers
    return {layer: activations[i * K:(i + 1) * K] for i, layer in enumerate(layers)}


def pool_activations(acts: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool [K, D] → [D]."""
    if method == "mean": return acts.mean(dim=0)
    elif method == "max": return acts.max(dim=0).values
    elif method == "last": return acts[-1]
    raise ValueError(f"Unknown pooling: {method}")
