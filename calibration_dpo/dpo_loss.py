"""DPO + SFT loss with dual-adapter steering hooks.

The policy adapter is trained while the reference adapter stays frozen.
Both share the same base model and use the same activation steering hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from peft import PeftModel

from core.ao import add_hook, get_batched_steering_hook, get_hf_submodule


@dataclass
class DPOBatchItem:
    """One DPO training item."""
    chosen_ids: list[int]      # Full input_ids for chosen response
    rejected_ids: list[int]    # Full input_ids for rejected response
    chosen_labels: list[int]   # Label mask (-100 for prompt, token ids for response)
    rejected_labels: list[int]
    activations: torch.Tensor  # [K*n_layers, D] steering vectors
    ph_positions: list[int]    # Placeholder positions in the input
    label: str = ""            # Pair type for logging (e.g. "good>refusal", "refusal>bad")


@dataclass
class SFTItem:
    """One SFT training item (ideal response or refusal)."""
    input_ids: list[int]
    labels: list[int]          # -100 for prompt tokens
    activations: torch.Tensor
    ph_positions: list[int]
    weight: float = 1.0        # Loss weight multiplier


def _compute_logprobs(
    model: PeftModel,
    input_ids: torch.Tensor,   # [B, L]
    labels: torch.Tensor,      # [B, L]
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence sum of log-probs for labeled positions.

    Returns: [B] tensor of summed log-probs.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [B, L, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]  # [B, L-1, V]
    shift_labels = labels[:, 1:]      # [B, L-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probs at label positions
    per_token_logprobs = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # [B, L-1]

    # Mask out positions where labels == -100
    mask = (shift_labels != -100).float()
    per_seq = (per_token_logprobs * mask).sum(dim=-1)  # [B]

    return per_seq


def compute_dpo_loss(
    model: PeftModel,
    dpo_items: list[DPOBatchItem],
    injection_layer: int,
    device: torch.device,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO loss using dual adapters (policy vs reference).

    Both forward passes use the same steering hook. The policy adapter is
    active for the policy pass, reference for the reference pass.

    Returns:
        (loss, metrics_dict) where metrics_dict has dpo_loss, reward_margin.
    """
    if not dpo_items:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "dpo_loss": 0.0, "reward_margin": 0.0,
        }

    dtype = torch.bfloat16
    pad_id = 0  # Will be overridden

    # Prepare batched inputs for chosen and rejected
    # Stack chosen and rejected together: [B_chosen, B_rejected]
    B = len(dpo_items)

    all_ids = []
    all_labels = []
    all_acts = []
    all_positions = []

    for item in dpo_items:
        all_ids.append(item.chosen_ids)
        all_labels.append(item.chosen_labels)
        all_acts.append(item.activations)
        all_positions.append(item.ph_positions)

    for item in dpo_items:
        all_ids.append(item.rejected_ids)
        all_labels.append(item.rejected_labels)
        all_acts.append(item.activations)
        all_positions.append(item.ph_positions)

    # Left-pad to max length
    max_len = max(len(ids) for ids in all_ids)
    padded_ids = []
    padded_labels = []
    padded_masks = []
    padded_positions = []

    for i, ids in enumerate(all_ids):
        pad_len = max_len - len(ids)
        padded_ids.append([0] * pad_len + ids)
        padded_labels.append([-100] * pad_len + all_labels[i])
        padded_masks.append([0] * pad_len + [1] * len(ids))
        padded_positions.append([p + pad_len for p in all_positions[i]])

    input_tensor = torch.tensor(padded_ids, device=device, dtype=torch.long)
    label_tensor = torch.tensor(padded_labels, device=device, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, device=device, dtype=torch.long)

    injection_submodule = get_hf_submodule(model, injection_layer)

    hook_fn = get_batched_steering_hook(
        vectors=all_acts,
        positions=padded_positions,
        device=device,
        dtype=dtype,
    )

    # Policy forward pass
    model.set_adapter("policy")
    with add_hook(injection_submodule, hook_fn):
        pi_logprobs = _compute_logprobs(model, input_tensor, label_tensor, mask_tensor)

    pi_chosen = pi_logprobs[:B]
    pi_rejected = pi_logprobs[B:]

    # Reference forward pass (no gradients)
    model.set_adapter("reference")
    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        ref_logprobs = _compute_logprobs(model, input_tensor, label_tensor, mask_tensor)

    ref_chosen = ref_logprobs[:B]
    ref_rejected = ref_logprobs[B:]

    # DPO loss
    log_ratio = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    dpo_loss = -F.logsigmoid(beta * log_ratio).mean()

    reward_margin = (pi_chosen - pi_rejected).mean().item()

    # Switch back to policy for subsequent operations
    model.set_adapter("policy")

    return dpo_loss, {
        "dpo_loss": dpo_loss.item(),
        "reward_margin": reward_margin,
    }


def compute_sft_loss(
    model: PeftModel,
    sft_items: list[SFTItem],
    injection_layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted SFT loss on ideal responses / refusals.

    Returns:
        (loss, metrics_dict) where metrics_dict has sft_loss.
    """
    if not sft_items:
        return torch.tensor(0.0, device=device, requires_grad=True), {"sft_loss": 0.0}

    dtype = torch.bfloat16

    # Left-pad
    max_len = max(len(item.input_ids) for item in sft_items)
    padded_ids = []
    padded_labels = []
    padded_masks = []
    padded_positions = []
    weights = []

    for item in sft_items:
        pad_len = max_len - len(item.input_ids)
        padded_ids.append([0] * pad_len + item.input_ids)
        padded_labels.append([-100] * pad_len + item.labels)
        padded_masks.append([0] * pad_len + [1] * len(item.input_ids))
        padded_positions.append([p + pad_len for p in item.ph_positions])
        weights.append(item.weight)

    input_tensor = torch.tensor(padded_ids, device=device, dtype=torch.long)
    label_tensor = torch.tensor(padded_labels, device=device, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, device=device, dtype=torch.long)
    weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

    injection_submodule = get_hf_submodule(model, injection_layer)

    hook_fn = get_batched_steering_hook(
        vectors=[item.activations for item in sft_items],
        positions=padded_positions,
        device=device,
        dtype=dtype,
    )

    model.set_adapter("policy")
    with add_hook(injection_submodule, hook_fn):
        outputs = model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
        )

    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = label_tensor[:, 1:].contiguous()

    # Per-token cross-entropy
    B, L, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)

    # Per-sequence mean loss
    mask = (shift_labels != -100).float()
    per_seq_loss = (per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    # Weighted mean across sequences
    sft_loss = (per_seq_loss * weight_tensor).sum() / weight_tensor.sum()

    return sft_loss, {"sft_loss": sft_loss.item()}
