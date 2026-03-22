"""Baseline: Patchscopes — activation patching with base model generation.

Unified API: accepts test_data + activations from eval_comprehensive.

Two modes:
  - "sweep" (legacy): grid-search over steering_coefficients × source_layers
  - "learned" (default): per-example gradient descent over a [n_layers, K]
    alpha matrix (one scalar per layer × position), optimised to reproduce
    the target_response via teacher-forced cross-entropy. Each example gets
    its own wandb run logging the loss curve.

Injection: activations are injected at the original context_positions (the
CoT text token positions) within a full multi-turn conversation:
  [user: question, assistant: cot_text, user: task_prompt]
This preserves the semantic alignment between where activations were extracted
and where they are injected.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import get_hf_submodule, get_steering_hook, add_hook, using_adapter
from activation_utils import split_activations_by_layer


# ── Build the patchscope input sequence ──────────────────────────────────────

def _build_patchscope_input(item, tokenizer, layers):
    """Build [user: question, assistant: cot_text, user: task_prompt] + generation prompt.

    Returns (input_ids, cot_positions) where cot_positions are the per-layer
    token indices of the CoT text — taken from the item's stored context_positions
    (which may be subsampled for long CoT).
    """
    user_msg = item.get("hinted_prompt") or item.get("question", "")
    cot_text = item.get("cot_text", "")
    task_prompt = item.get("prompt", "")

    # Build [user: question, assistant: cot_text] as base
    prefix_msgs = [{"role": "user", "content": user_msg}]
    context_msgs = prefix_msgs + [{"role": "assistant", "content": cot_text}]

    # Full sequence: [user: question, assistant: cot_text, user: task_prompt] + gen prompt
    full_msgs = context_msgs + [{"role": "user", "content": task_prompt}]
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Use stored context_positions (matches where activations were extracted).
    # These are tiled per layer: [pos_L0, pos_L0, ..., pos_L1, pos_L1, ...].
    # Extract the first layer's positions (all layers share the same positions).
    n_layers = len(layers)
    all_positions = item.get("context_positions", [])
    K = len(all_positions) // n_layers if n_layers else len(all_positions)
    cot_positions = all_positions[:K]

    return full_ids, cot_positions


# ── Differentiable steering hook ─────────────────────────────────────────────

def _get_differentiable_steering_hook(vectors, positions, alpha_per_pos):
    """Norm-matched additive hook with differentiable per-position alphas.

    `alpha_per_pos` is a [K] tensor (requires_grad=True).
    """
    normed = F.normalize(vectors, dim=-1)  # [K, D], no detach

    def hook_fn(module, _input, output):
        del module, _input
        if isinstance(output, tuple):
            resid, *rest = output
            is_tuple = True
        else:
            resid = output
            is_tuple = False

        _b, seq_len, _d = resid.shape
        if seq_len <= 1:
            return (resid, *rest) if is_tuple else resid

        pos = torch.tensor(positions, dtype=torch.long, device=resid.device)
        orig = resid[0, pos, :].float()                        # [K, D] in float32 for grad
        norms = orig.norm(dim=-1, keepdim=True)                # [K, 1]
        steered = normed.to(device=orig.device).float() * norms * alpha_per_pos.unsqueeze(-1)
        resid = resid.clone()
        resid[0, pos, :] = (steered + orig).to(resid.dtype)

        return (resid, *rest) if is_tuple else resid

    return hook_fn


# ── Teacher-forced loss ──────────────────────────────────────────────────────

MAX_TOTAL_TOKENS = 4096  # Cap total sequence length for teacher forcing


def _teacher_forced_loss(model, tokenizer, item, target_text,
                         acts_by_layer, layers, alpha_parts, cot_positions,
                         injection_layer, device):
    """Teacher-forced forward pass → CE loss on target tokens.

    Builds: [user: question, assistant: cot_text, user: task_prompt] + target_text
    Injects activations at cot_positions (the CoT token indices).
    """
    full_ids, _ = _build_patchscope_input(item, tokenizer, layers)

    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    target_ids = target_ids[:256]

    # Truncate prompt from the CoT middle if too long (keep start + end of CoT + task prompt)
    total_budget = MAX_TOTAL_TOKENS - len(target_ids)
    if len(full_ids) > total_budget:
        full_ids = full_ids[:total_budget]
        # Drop cot_positions that fall outside the truncated sequence
        cot_positions = [p for p in cot_positions if p < len(full_ids)]

    prompt_len = len(full_ids)
    all_ids = full_ids + target_ids
    input_tensor = torch.tensor([all_ids], device=device)

    injection_submodule = get_hf_submodule(model, injection_layer)
    hooks = []
    for layer_idx, layer in enumerate(layers):
        layer_acts = acts_by_layer[layer]               # [K, D]
        K = min(layer_acts.shape[0], len(cot_positions))
        if K == 0:
            continue
        positions = cot_positions[:K]
        hook_fn = _get_differentiable_steering_hook(
            layer_acts[:K].to(device), positions, alpha_parts[layer_idx][:K],
        )
        hooks.append(injection_submodule.register_forward_hook(hook_fn))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_tensor).logits

    for h in hooks:
        h.remove()

    target_logits = logits[0, prompt_len - 1:-1, :]
    target_labels = torch.tensor(target_ids, device=device)
    return F.cross_entropy(target_logits.float(), target_labels)


# ── Per-example alpha optimisation + generation ──────────────────────────────

def _optimise_and_generate_single(
    model, tokenizer, item, acts, layers, task_def,
    *, injection_layer, max_new_tokens, device,
    lr, n_steps, wandb_group, example_idx,
):
    """Optimise alpha[n_layers, K] for one example, log to wandb, then generate."""
    import wandb

    acts_by_layer = split_activations_by_layer(acts, layers)
    task_name = task_def.name if hasattr(task_def, "name") else "unknown"
    target = item.get("target_response", "")

    # Get CoT positions for this example
    _, cot_positions = _build_patchscope_input(item, tokenizer, layers)

    # Build alpha with correct K per layer
    alpha_parts = []
    for layer in layers:
        K = min(acts_by_layer[layer].shape[0], len(cot_positions))
        alpha_parts.append(torch.ones(K, device=device, requires_grad=True))

    optimiser = torch.optim.Adam(alpha_parts, lr=lr)

    run = wandb.init(
        project="cot-oracle",
        entity="MATS10-CS-JB",
        group=wandb_group or "patchscopes-alpha",
        name=f"ps_{task_name}_{example_idx}",
        config={
            "task": task_name,
            "example_idx": example_idx,
            "example_id": item.get("id", f"{task_name}_{example_idx}"),
            "layers": layers,
            "K_per_layer": [p.shape[0] for p in alpha_parts],
            "injection_layer": injection_layer,
            "lr": lr,
            "n_steps": n_steps,
        },
        reinit=True,
    )

    model.eval()

    if target:
        best_loss = float("inf")
        steps_without_improvement = 0
        patience = 15
        rel_tol = 1e-3  # stop when improvement < 0.1% of best loss

        for step in range(n_steps):
            with using_adapter(model, None):
                loss = _teacher_forced_loss(
                    model, tokenizer, item, target,
                    acts_by_layer, layers, alpha_parts, cot_positions,
                    injection_layer, device,
                )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            loss_val = loss.item()
            log_dict = {"step": step, "loss": loss_val}
            for i, layer in enumerate(layers):
                log_dict[f"alpha_mean/L{layer}"] = alpha_parts[i].mean().item()
                log_dict[f"alpha_std/L{layer}"] = alpha_parts[i].std().item()
            wandb.log(log_dict)
            del loss
            torch.cuda.empty_cache()

            if loss_val < best_loss - abs(best_loss) * rel_tol:
                best_loss = loss_val
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            if steps_without_improvement >= patience:
                wandb.log({"converged_at_step": step})
                break

    learned_alpha = [a.detach().clone() for a in alpha_parts]
    final_alpha_summary = {f"final_alpha_mean/L{l}": learned_alpha[i].mean().item() for i, l in enumerate(layers)}
    wandb.log(final_alpha_summary)
    run.finish()

    return _generate_with_learned_alpha(
        model, tokenizer, item, acts_by_layer, layers, learned_alpha, cot_positions,
        injection_layer, max_new_tokens, device,
    )


# ── Generation with learned alpha ────────────────────────────────────────────

def _generate_with_learned_alpha(model, tokenizer, item, acts_by_layer, layers,
                                 alpha_parts, cot_positions, injection_layer,
                                 max_new_tokens, device):
    """Generate from base model with multi-layer injection at CoT positions."""
    full_ids, _ = _build_patchscope_input(item, tokenizer, layers)
    input_tensor = torch.tensor([full_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    injection_submodule = get_hf_submodule(model, injection_layer)
    hooks = []
    for layer_idx, layer in enumerate(layers):
        layer_acts = acts_by_layer[layer]
        K = min(layer_acts.shape[0], len(cot_positions))
        coeff = alpha_parts[layer_idx][:K].mean().item()
        hook_fn = get_steering_hook(
            vectors=layer_acts[:K].to(device), positions=cot_positions[:K],
            device=device, dtype=torch.bfloat16, steering_coefficient=coeff,
        )
        hooks.append(injection_submodule.register_forward_hook(hook_fn))

    model.eval()
    with using_adapter(model, None):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_tensor, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
            )

    for h in hooks:
        h.remove()

    return tokenizer.decode(output[0][len(full_ids):], skip_special_tokens=True)


# ── Sweep mode (legacy) ─────────────────────────────────────────────────────

def _run_patchscope_single(
    model, tokenizer, item, layer_acts, cot_positions, layers,
    *, injection_layer, steering_coefficient, max_new_tokens, device,
):
    """Run patchscope: inject layer activations at CoT positions and generate."""
    full_ids, _ = _build_patchscope_input(item, tokenizer, layers)
    input_tensor = torch.tensor([full_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    K = min(layer_acts.shape[0], len(cot_positions))
    hook_fn = get_steering_hook(
        vectors=layer_acts[:K].to(device), positions=cot_positions[:K],
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

    return tokenizer.decode(output[0][len(full_ids):], skip_special_tokens=True)


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


# ── Main entry point ─────────────────────────────────────────────────────────

def run_patchscopes(
    test_data: list[dict],
    activations: list[torch.Tensor],
    layers: list[int],
    task_def,
    model,
    tokenizer,
    *,
    mode: str = "learned",
    source_layers: list[int] | None = None,
    injection_layer: int = 1,
    steering_coefficients: list[float] | float = (0.5, 1.0, 2.0),
    max_new_tokens: int = 100,
    device: str = "cuda",
    lr: float = 0.05,
    n_steps: int = 60,
    wandb_group: str | None = None,
) -> list[str]:
    """Run patchscopes baseline.

    mode="learned": per-example gradient descent on alpha[n_layers, K],
                    one wandb run per example logging loss curves.
    mode="sweep": legacy grid search over coefficients × layers.
    """
    from tasks import ScoringMode

    if source_layers is None:
        source_layers = layers

    # ── Learned mode ──
    if mode == "learned":
        is_binary = task_def.scoring == ScoringMode.BINARY
        predictions = []
        for i, (item, acts) in tqdm(
            enumerate(zip(test_data, activations)),
            desc="Patchscope learned", total=len(test_data),
        ):
            generated = _optimise_and_generate_single(
                model, tokenizer, item, acts, layers, task_def,
                injection_layer=injection_layer, max_new_tokens=max_new_tokens,
                device=device, lr=lr, n_steps=n_steps,
                wandb_group=wandb_group, example_idx=i,
            )
            if is_binary:
                predictions.append(_parse_binary(generated, task_def))
            else:
                predictions.append(generated)
        return predictions

    # ── Sweep mode (legacy) ──
    if isinstance(steering_coefficients, (int, float)):
        steering_coefficients = [float(steering_coefficients)]

    is_binary = task_def.scoring == ScoringMode.BINARY

    best_score = -1
    best_predictions = None

    for coeff in tqdm(steering_coefficients, desc="Patchscope coefficients"):
        for source_layer in tqdm(source_layers, desc=f"coeff={coeff}", leave=False):
            predictions = []
            for item, acts in tqdm(zip(test_data, activations), desc=f"c={coeff} l={source_layer}", leave=False, total=len(test_data)):
                acts_by_layer = split_activations_by_layer(acts, layers)
                layer_acts = acts_by_layer[source_layer]
                _, cot_positions = _build_patchscope_input(item, tokenizer, layers)

                generated = _run_patchscope_single(
                    model, tokenizer, item, layer_acts, cot_positions, layers,
                    injection_layer=injection_layer,
                    steering_coefficient=coeff,
                    max_new_tokens=max_new_tokens, device=device,
                )

                if is_binary:
                    predictions.append(_parse_binary(generated, task_def))
                else:
                    predictions.append(generated)

            if is_binary:
                targets = [d["target_response"] for d in test_data]
                from linear_probe import _extract_label
                gt_labels = [_extract_label(t, task_def) for t in targets]
                acc = sum(1 for p, g in zip(predictions, gt_labels) if p == g) / len(gt_labels)
                if acc > best_score:
                    best_score = acc
                    best_predictions = predictions
            else:
                if best_predictions is None:
                    best_predictions = predictions

    return best_predictions or [""] * len(test_data)
