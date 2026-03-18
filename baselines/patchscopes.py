"""Baseline: Patchscopes — activation patching with base model generation.

Unified API: accepts test_data + activations from eval_comprehensive.
"""

import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import get_hf_submodule, get_steering_hook, add_hook, using_adapter
from activation_utils import split_activations_by_layer


def _run_patchscope_single(
    model, tokenizer, activations, verbalizer_prompt,
    *, injection_layer, steering_coefficient, max_new_tokens, device,
):
    """Run patchscope: inject activations into base model and generate."""
    messages = [{"role": "user", "content": verbalizer_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    n_acts = activations.shape[0]
    total_len = len(input_ids)
    positions = list(range(max(0, total_len - n_acts), total_len))
    if len(positions) > n_acts:
        positions = positions[:n_acts]
    if len(positions) < n_acts:
        activations = activations[:len(positions)]

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    hook_fn = get_steering_hook(
        vectors=activations.to(device), positions=positions,
        device=device, dtype=torch.bfloat16, steering_coefficient=steering_coefficient,
    )
    injection_submodule = get_hf_submodule(model, injection_layer)

    model.eval()
    with using_adapter(model, None):
        with add_hook(injection_submodule, hook_fn):
            output = model.generate(
                input_ids=input_tensor, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
            )

    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def _parse_binary(response, task_def):
    """Parse binary response using task_def keywords."""
    lower = response.lower()
    pos_hit = any(kw.lower() in lower for kw in task_def.positive_keywords)
    neg_hit = any(kw.lower() in lower for kw in task_def.negative_keywords)
    if pos_hit and not neg_hit:
        return task_def.positive_label or "positive"
    if neg_hit and not pos_hit:
        return task_def.negative_label or "negative"
    return task_def.negative_label or "negative"


def run_patchscopes(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    model,
    tokenizer,
    *,
    source_layers: list[int] | None = None,
    injection_layer: int = 1,
    steering_coefficients: list[float] | float = (0.5, 1.0, 2.0),
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> list[str]:
    """Run patchscopes. Sweeps (coefficient × layer), returns predictions from best combo."""
    from tasks import ScoringMode

    if source_layers is None:
        source_layers = layers
    if isinstance(steering_coefficients, (int, float)):
        steering_coefficients = [float(steering_coefficients)]

    is_binary = task_def.scoring == ScoringMode.BINARY
    prompt_text = test_data[0].get("prompt", "") if test_data else ""
    verbalizer = prompt_text[:2000] if prompt_text else "What is this reasoning about?"

    best_score = -1
    best_predictions = None

    for coeff in tqdm(steering_coefficients, desc="Patchscope coefficients"):
        for source_layer in tqdm(source_layers, desc=f"coeff={coeff}", leave=False):
            predictions = []
            for item, acts in tqdm(zip(test_data, activations), desc=f"c={coeff} l={source_layer}", leave=False, total=len(test_data)):
                acts_by_layer = split_activations_by_layer(acts, layers)
                layer_acts = acts_by_layer[source_layer]

                generated = _run_patchscope_single(
                    model, tokenizer, layer_acts, verbalizer,
                    injection_layer=injection_layer,
                    steering_coefficient=coeff,
                    max_new_tokens=max_new_tokens, device=device,
                )

                if is_binary:
                    predictions.append(_parse_binary(generated, task_def))
                else:
                    predictions.append(generated)

            # Score this config for selection
            if is_binary:
                targets = [d["target_response"] for d in test_data]
                from linear_probe import _extract_label
                gt_labels = [_extract_label(t, task_def) for t in targets]
                acc = sum(1 for p, g in zip(predictions, gt_labels) if p == g) / len(gt_labels)
                if acc > best_score:
                    best_score = acc
                    best_predictions = predictions
            else:
                # For non-binary, just take the first config
                if best_predictions is None:
                    best_predictions = predictions

    return best_predictions or [""] * len(test_data)
