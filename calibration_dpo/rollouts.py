"""Batched oracle rollout generation for calibration DPO.

Generates N rollouts per example at temperature T, using the same activations
and prompt but sampling different outputs.
"""

from __future__ import annotations

import torch
from peft import PeftModel

from core.ao import (
    TRAINED_PLACEHOLDER,
    add_hook,
    get_batched_steering_hook,
    get_hf_submodule,
)


def _build_manual_prefix_token_ids(
    tokenizer, num_positions: int, layers: list[int], ph_id: int,
) -> tuple[list[int], list[int]]:
    """Build token IDs and placeholder positions for the oracle prefix.

    Mirrors eval_loop.py:929-955.
    """
    prefix_layers = list(layers)
    if len(prefix_layers) > 1:
        k, rem = divmod(num_positions, len(prefix_layers))
        assert rem == 0, f"num_positions={num_positions} not divisible by layers={prefix_layers}"
        block_sizes = [k] * len(prefix_layers)
    else:
        block_sizes = [num_positions]

    prefix_ids: list[int] = []
    positions: list[int] = []
    for i, (layer_idx, block_size) in enumerate(zip(prefix_layers, block_sizes)):
        label = f"L{layer_idx}:"
        if i > 0:
            label = " " + label
        prefix_ids.extend(tokenizer.encode(label, add_special_tokens=False))
        positions.extend(range(len(prefix_ids), len(prefix_ids) + block_size))
        prefix_ids.extend([ph_id] * block_size)
    prefix_ids.extend(tokenizer.encode(".\n", add_special_tokens=False))
    return prefix_ids, positions


def _build_oracle_prefix(
    num_positions: int,
    layers: list[int],
    placeholder_token: str,
) -> str:
    """Build the text-level oracle prefix for chat template insertion."""
    if len(layers) == 1:
        return f"L{layers[0]}:" + placeholder_token * num_positions + ".\n"
    k, rem = divmod(num_positions, len(layers))
    if rem:
        raise ValueError(f"num_positions={num_positions} not divisible by layers={layers}")
    return " ".join(f"L{layer}:" + placeholder_token * k for layer in layers) + ".\n"


def generate_rollouts(
    model: PeftModel,
    tokenizer,
    activations_list: list[torch.Tensor],
    oracle_prompts: list[str],
    layers: list[int],
    n_rollouts: int = 12,
    temperature: float = 1.0,
    max_new_tokens: int = 150,
    generation_batch_size: int = 12,
    injection_layer: int = 1,
    adapter_name: str = "policy",
    device: str | torch.device = "cuda",
    questions: list[str | None] | None = None,
    repetition_penalty: float = 1.0,
) -> list[list[str]]:
    """Generate multiple rollouts per example.

    Args:
        model: PeftModel with policy adapter.
        tokenizer: Tokenizer.
        activations_list: List of [K*n_layers, D] tensors per example.
        oracle_prompts: List of oracle prompt strings per example.
        layers: Layer indices (e.g. [9, 18, 27]).
        n_rollouts: Number of rollouts per example.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.
        generation_batch_size: How many rollouts to generate at once.
        injection_layer: Layer to inject activations at.
        adapter_name: Adapter to use for generation.
        device: Target device.
        questions: Optional list of question strings to prepend to oracle prompts.

    Returns:
        List of lists of rollout strings, one inner list per example.
    """
    dtype = torch.bfloat16
    ph_token = TRAINED_PLACEHOLDER
    ph_id_list = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id_list) == 1, f"Expected single token for '{ph_token}', got {len(ph_id_list)}"
    ph_id = ph_id_list[0]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    injection_submodule = get_hf_submodule(model, injection_layer)

    model.set_adapter(adapter_name)
    was_training = model.training
    model.eval()

    all_rollouts = []

    try:
        for ex_idx in range(len(activations_list)):
            activations = activations_list[ex_idx]
            oracle_prompt = oracle_prompts[ex_idx]
            num_positions = activations.shape[0]

            # Build oracle input
            prefix = _build_oracle_prefix(num_positions, layers, ph_token)
            question = questions[ex_idx] if questions else None
            if question:
                full_prompt = prefix + f'The model is answering: "{question}"\n{oracle_prompt}'
            else:
                full_prompt = prefix + oracle_prompt

            messages = [{"role": "user", "content": full_prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            # Manual prefix tokenization to prevent boundary merges
            prefix_idx = formatted.find(prefix)
            assert prefix_idx >= 0, "Prefix text not found in chat template output"
            before_ids = tokenizer.encode(formatted[:prefix_idx], add_special_tokens=False)
            after_ids = tokenizer.encode(formatted[prefix_idx + len(prefix):], add_special_tokens=False)
            prefix_ids, rel_positions = _build_manual_prefix_token_ids(
                tokenizer, num_positions, layers, ph_id,
            )
            input_ids = before_ids + prefix_ids + after_ids
            positions = [len(before_ids) + p for p in rel_positions]

            # Generate rollouts in batches
            example_rollouts = []
            remaining = n_rollouts
            while remaining > 0:
                batch_n = min(remaining, generation_batch_size)

                # Replicate for batch
                batch_ids = [input_ids] * batch_n
                batch_positions = [positions] * batch_n
                batch_acts = [activations] * batch_n

                max_len = len(input_ids)
                input_tensor = torch.tensor(batch_ids, device=device)
                attn_mask = torch.ones_like(input_tensor)

                hook_fn = get_batched_steering_hook(
                    vectors=batch_acts,
                    positions=batch_positions,
                    device=device,
                    dtype=dtype,
                )

                gen_kwargs = dict(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=pad_id,
                )
                if repetition_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = repetition_penalty

                with add_hook(injection_submodule, hook_fn):
                    outputs = model.generate(**gen_kwargs)

                for j in range(batch_n):
                    generated = outputs[j][max_len:]
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    example_rollouts.append(text)

                remaining -= batch_n

            all_rollouts.append(example_rollouts)

    finally:
        if was_training:
            model.train()

    return all_rollouts
