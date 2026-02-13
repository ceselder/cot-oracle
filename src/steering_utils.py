"""
Steering utilities for activation injection.

Based on activation_oracles nl_probes/utils/steering_hooks.py
"""

import torch
from torch import Tensor
from contextlib import contextmanager
from typing import Callable


def get_steering_hook(
    vectors: list[Tensor],  # List of [d_model] tensors, one per batch element
    positions: list[list[int]],  # List of position lists, one per batch element
    steering_coefficient: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Callable:
    """
    Create a steering hook for norm-matched activation injection.

    Based on AO's get_hf_activation_steering_hook.

    For each batch element b, injects vectors[b][i] at positions[b][i].

    Injection formula (norm-matched addition):
        steered = normalize(vector) * ||original|| * coefficient
        output = original + steered
    """
    # Pre-normalize vectors
    normed_vectors = []
    for vec_list in vectors:
        if isinstance(vec_list, Tensor) and vec_list.dim() == 1:
            # Single vector
            vec_list = [vec_list]
        normed = [v / (v.norm() + 1e-8) for v in vec_list]
        normed_vectors.append(normed)

    def hook(module, input, output):
        # Handle tuple outputs (common in transformers)
        if isinstance(output, tuple):
            resid = output[0]
            rest = output[1:]
        else:
            resid = output
            rest = None

        # resid shape: [batch, seq, d_model]
        resid = resid.clone()

        for b in range(len(normed_vectors)):
            if b >= resid.shape[0]:
                break

            pos_list = positions[b] if b < len(positions) else []
            vec_list = normed_vectors[b]

            for i, pos in enumerate(pos_list):
                if i >= len(vec_list):
                    break
                if pos >= resid.shape[1]:
                    continue

                # Get original activation and its norm
                orig = resid[b, pos, :]
                orig_norm = orig.norm()

                # Norm-matched steering
                normed_vec = vec_list[i].to(device=resid.device, dtype=resid.dtype)
                steered = normed_vec * orig_norm * steering_coefficient

                # Add to original (not replace)
                resid[b, pos, :] = orig + steered

        if rest is not None:
            return (resid,) + rest
        return resid

    return hook


@contextmanager
def add_hook(module: torch.nn.Module, hook_fn: Callable):
    """
    Context manager for temporarily adding a forward hook.

    Usage:
        with add_hook(model.layers[1], hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def get_layer_module(model, layer_idx: int):
    """
    Get the transformer layer module for different model architectures.

    Handles PEFT-wrapped models and different HF model structures.
    """
    # Try to get the base model if PEFT-wrapped
    if hasattr(model, 'base_model'):
        model = model.base_model

    # Try different common paths
    paths_to_try = [
        lambda m: m.model.layers[layer_idx],  # Llama, Qwen, Mistral
        lambda m: m.transformer.h[layer_idx],  # GPT-2 style
        lambda m: m.model.decoder.layers[layer_idx],  # Some encoder-decoder
        lambda m: m.layers[layer_idx],  # Direct
    ]

    for path_fn in paths_to_try:
        try:
            return path_fn(model)
        except (AttributeError, IndexError):
            continue

    raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def collect_activations(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    layer_idx: int,
    position: int = -1,  # -1 = last token
) -> Tensor:
    """
    Collect activations from a specific layer and position.

    Returns activation at the specified position (default: last token).
    """
    layer_module = get_layer_module(model, layer_idx)

    activations = None

    def hook(module, input, output):
        nonlocal activations
        if isinstance(output, tuple):
            activations = output[0].detach()
        else:
            activations = output.detach()

    handle = layer_module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    # Get activation at specified position
    if position == -1:
        # Last non-padding token for each batch element
        # Find last 1 in attention_mask
        seq_lens = attention_mask.sum(dim=1) - 1  # 0-indexed
        batch_size = activations.shape[0]
        result = torch.stack([
            activations[b, seq_lens[b].item(), :]
            for b in range(batch_size)
        ])
    else:
        result = activations[:, position, :]

    return result  # [batch, d_model]
