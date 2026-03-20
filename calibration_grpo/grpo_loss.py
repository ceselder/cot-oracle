"""DR-GRPO loss: clipped surrogate policy gradient with activation injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import PeftModel

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ao_reference"))

from core.ao import add_hook, get_hf_submodule, get_steering_hook


@dataclass
class GRPOItem:
    input_ids: list[int]        # full prompt + response
    response_start: int         # index where response tokens begin
    activations: torch.Tensor   # [K * n_layers, D] for injection
    ph_positions: list[int]     # placeholder positions in input_ids
    advantage: float
    old_logprob: float          # sum of log-probs under old policy


def _pad_left(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad sequences to max length. Returns (input_ids, attention_mask)."""
    max_len = max(len(s) for s in sequences)
    padded = []
    masks = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append([pad_id] * pad_len + s)
        masks.append([0] * pad_len + [1] * len(s))
    return torch.tensor(padded), torch.tensor(masks)


def _compute_sequence_logprob(
    logits: torch.Tensor,  # [seq_len, vocab]
    input_ids: torch.Tensor,  # [seq_len]
    response_start: int,
    seq_len_original: int,
    pad_offset: int,
) -> torch.Tensor:
    """Compute sum of log-probs over response tokens."""
    # Shift: logits[t] predicts input_ids[t+1]
    start = pad_offset + response_start
    end = pad_offset + seq_len_original
    if start >= end - 1:
        # No response tokens — need a grad-carrying zero
        return logits.sum() * 0.0

    pred_logits = logits[start:end - 1]  # [T-1, vocab]
    target_ids = input_ids[start + 1:end]  # [T-1]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


def compute_old_logprobs(
    model: PeftModel,
    items: list[GRPOItem],
    injection_layer: int,
    device: torch.device,
    pad_id: int = 0,
) -> list[float]:
    """Compute log-probs under current policy (no grad). Used to cache old_logprob."""
    model.eval()
    results = []

    with torch.no_grad():
        for item in items:
            input_tensor = torch.tensor([item.input_ids], device=device)
            attn_mask = torch.ones_like(input_tensor)

            hook_fn = get_steering_hook(
                vectors=item.activations.to(device),
                positions=item.ph_positions,
                device=device,
                dtype=torch.bfloat16,
            )
            injection_sub = get_hf_submodule(model, injection_layer)

            with add_hook(injection_sub, hook_fn):
                outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

            logits = outputs.logits[0].float()
            logprob = _compute_sequence_logprob(
                logits, input_tensor[0],
                response_start=item.response_start,
                seq_len_original=len(item.input_ids),
                pad_offset=0,
            )
            results.append(logprob.item())

    return results


def compute_grpo_loss(
    model: PeftModel,
    items: list[GRPOItem],
    injection_layer: int,
    device: torch.device,
    clip_eps: float = 0.2,
    pad_id: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DR-GRPO clipped surrogate loss.

    For each item:
      ratio = exp(new_logprob - old_logprob)
      clipped_ratio = clip(ratio, 1-eps, 1+eps)
      loss_i = -min(ratio * advantage, clipped_ratio * advantage)
    """
    model.train()

    # Process items one at a time (each has different activations/positions)
    losses = []
    ratios = []
    advantages = []

    for item in items:
        input_tensor = torch.tensor([item.input_ids], device=device)
        attn_mask = torch.ones_like(input_tensor)

        hook_fn = get_steering_hook(
            vectors=item.activations.to(device),
            positions=item.ph_positions,
            device=device,
            dtype=torch.bfloat16,
        )
        injection_sub = get_hf_submodule(model, injection_layer)

        with add_hook(injection_sub, hook_fn):
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

        logits = outputs.logits[0].float()
        new_logprob = _compute_sequence_logprob(
            logits, input_tensor[0],
            response_start=item.response_start,
            seq_len_original=len(item.input_ids),
            pad_offset=0,
        )

        old_lp = torch.tensor(item.old_logprob, device=device, dtype=torch.float32)
        log_ratio = new_logprob - old_lp
        ratio = torch.exp(log_ratio)
        adv = torch.tensor(item.advantage, device=device, dtype=torch.float32)

        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surrogate = torch.min(ratio * adv, clipped_ratio * adv)
        losses.append(-surrogate)
        ratios.append(ratio.item())
        advantages.append(item.advantage)

    loss = torch.stack(losses).mean()

    metrics = {
        "grpo/loss": loss.item(),
        "grpo/mean_ratio": sum(ratios) / len(ratios),
        "grpo/mean_advantage": sum(advantages) / len(advantages),
        "grpo/clip_frac": sum(1 for r in ratios if abs(r - 1.0) > clip_eps) / len(ratios),
    }

    return loss, metrics
