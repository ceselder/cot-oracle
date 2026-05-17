'''Tokenwise GRPO loss with activation injection.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ao_reference'))

from core.ao import add_hook, get_hf_submodule, get_steering_hook


@dataclass
class GRPOItem:
    prompt_ids: list[int]
    response_ids: list[int]
    activations: torch.Tensor
    ph_positions: list[int]
    advantage: float
    old_token_logprobs: list[float]


def _compute_response_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start: int,
    seq_len_original: int,
    pad_offset: int,
) -> torch.Tensor:
    '''Compute per-token log-probs over the response, including the first response token.'''
    start = pad_offset + response_start
    end = pad_offset + seq_len_original

    if start >= end:
        return logits.new_zeros((0,), dtype=torch.float32)
    if start <= 0:
        raise ValueError('response_start must be > 0 to score response tokens')

    pred_logits = logits[start - 1:end - 1]
    target_ids = input_ids[start:end]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)


def compute_old_logprobs(
    model: PeftModel,
    items: list[GRPOItem],
    injection_layer: int,
    device: torch.device,
) -> list[list[float]]:
    '''Cache token log-probs under the sampling policy snapshot.'''
    model.eval()
    results: list[list[float]] = []
    injection_sub = get_hf_submodule(model, injection_layer)

    with torch.no_grad():
        for item in items:
            full_ids = item.prompt_ids + item.response_ids
            input_tensor = torch.tensor([full_ids], device=device)
            attn_mask = torch.ones_like(input_tensor)

            hook_fn = get_steering_hook(
                vectors=item.activations.to(device),
                positions=item.ph_positions,
                device=device,
                dtype=torch.bfloat16,
            )

            with add_hook(injection_sub, hook_fn):
                outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

            logits = outputs.logits[0].float()
            token_logprobs = _compute_response_token_logprobs(
                logits,
                input_tensor[0],
                response_start=len(item.prompt_ids),
                seq_len_original=len(full_ids),
                pad_offset=0,
            )
            results.append(token_logprobs.tolist())

    return results


def compute_grpo_loss(
    model: PeftModel,
    items: list[GRPOItem],
    injection_layer: int,
    device: torch.device,
    clip_eps: float = 0.2,
    grad_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    '''Compute tokenwise clipped GRPO loss with per-item backward.'''
    model.eval()
    n_items = len(items)
    total_loss = 0.0
    token_ratios: list[float] = []
    advantages: list[float] = []
    response_lengths: list[int] = []
    injection_sub = get_hf_submodule(model, injection_layer)

    for item in items:
        full_ids = item.prompt_ids + item.response_ids
        input_tensor = torch.tensor([full_ids], device=device)
        attn_mask = torch.ones_like(input_tensor)

        hook_fn = get_steering_hook(
            vectors=item.activations.to(device),
            positions=item.ph_positions,
            device=device,
            dtype=torch.bfloat16,
        )

        with add_hook(injection_sub, hook_fn):
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

        logits = outputs.logits[0].float()
        new_token_logprobs = _compute_response_token_logprobs(
            logits,
            input_tensor[0],
            response_start=len(item.prompt_ids),
            seq_len_original=len(full_ids),
            pad_offset=0,
        )

        if new_token_logprobs.numel() == 0:
            continue

        old_token_logprobs = torch.tensor(
            item.old_token_logprobs, device=device, dtype=torch.float32,
        )
        if old_token_logprobs.shape != new_token_logprobs.shape:
            raise ValueError(
                f'Mismatched token logprob shapes: old={tuple(old_token_logprobs.shape)} '
                f'new={tuple(new_token_logprobs.shape)}'
            )

        log_ratio = new_token_logprobs - old_token_logprobs
        ratio = torch.exp(log_ratio)
        adv = torch.tensor(item.advantage, device=device, dtype=torch.float32)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        surrogate = torch.minimum(unclipped, clipped)
        item_loss = -surrogate.sum() / max(n_items, 1)
        (item_loss * grad_scale).backward()

        total_loss += item_loss.item()
        token_ratios.extend(ratio.detach().cpu().tolist())
        advantages.append(item.advantage)
        response_lengths.append(len(item.response_ids))

    if token_ratios:
        mean_ratio = sum(token_ratios) / len(token_ratios)
        clip_frac = sum(1 for r in token_ratios if abs(r - 1.0) > clip_eps) / len(token_ratios)
        max_ratio = max(token_ratios)
        min_ratio = min(token_ratios)
    else:
        mean_ratio = 1.0
        clip_frac = 0.0
        max_ratio = 1.0
        min_ratio = 1.0

    metrics = {
        'grpo/loss': total_loss,
        'grpo/mean_ratio': mean_ratio,
        'grpo/clip_frac': clip_frac,
        'grpo/max_ratio': max_ratio,
        'grpo/min_ratio': min_ratio,
        'grpo/mean_advantage': sum(advantages) / len(advantages) if advantages else 0.0,
        'grpo/mean_response_tokens': sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        'grpo/n_response_tokens': float(len(token_ratios)),
        'grpo/n_sequences': float(len(response_lengths)),
    }
    return total_loss, metrics
