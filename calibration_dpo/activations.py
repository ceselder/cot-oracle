"""Activation extraction for calibration DPO.

Follows the materialize_multilayer_steering_vectors pattern from src/train.py.
Extracts activations with adapter disabled, at specified positions and layers.
"""

from __future__ import annotations

import torch
from peft import PeftModel

from core.ao import get_hf_submodule, EarlyStopException


def extract_activations(
    model: PeftModel,
    tokenizer,
    examples: list[dict],
    layers: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract activations for a batch of examples.

    For each example, returns a [K*n_layers, D] tensor where K is the number
    of base positions (before layer replication).

    Args:
        model: PeftModel with adapter(s).
        examples: List of dicts with 'context_input_ids' and 'selected_positions'.
        layers: Layer indices to extract from (e.g. [9, 18, 27]).
        device: Target device.

    Returns:
        List of activation tensors, one per example.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    contexts = [ex["context_input_ids"] for ex in examples]
    max_len = max(len(c) for c in contexts)

    # Left-pad and build attention masks
    input_ids_list = []
    attn_mask_list = []
    left_offsets = []
    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_list.append(
            torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device)
        )
        attn_mask_list.append(
            torch.tensor(
                [False] * pad_len + [True] * len(c), dtype=torch.bool, device=device
            )
        )
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
    }

    # Get submodules for each layer
    submodules = {layer: get_hf_submodule(model, layer) for layer in layers}
    max_layer = max(layers)
    module_to_layer = {submodules[l]: l for l in layers}

    # Collect activations with adapter disabled
    acts_by_layer: dict[int, torch.Tensor] = {}

    def gather_hook(module, _inputs, outputs):
        layer = module_to_layer[module]
        act = outputs[0] if isinstance(outputs, tuple) else outputs
        acts_by_layer[layer] = act.detach()
        if layer == max_layer:
            raise EarlyStopException()

    handles = []
    for layer in layers:
        handles.append(submodules[layer].register_forward_hook(gather_hook))

    was_training = model.training
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad():
            model(**inputs_BL)
    except EarlyStopException:
        pass
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # Extract per-example activation vectors at selected positions
    n_layers = len(layers)
    result = []
    for b, ex in enumerate(examples):
        base_positions = ex["base_positions"]
        K = len(base_positions)

        vectors_parts = []
        for layer in layers:
            acts_BLD = acts_by_layer[layer]
            adjusted = [p + left_offsets[b] for p in base_positions]
            layer_vecs = acts_BLD[b, adjusted, :]  # [K, D]
            vectors_parts.append(layer_vecs)

        # Concatenate across layers: [K*n_layers, D]
        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()
        result.append(vectors)

    return result
