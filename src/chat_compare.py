"""
Side-by-side comparison: Original AO vs Trained CoT Oracle.

Key difference in input format:
  - Original AO: single layer (L50%), standard PastLens prefix ("Layer: 18\n ? ? ? ...\n")
  - Trained oracle: multi-layer stride-based prefix
    ("Layer: 9, 18, 27\n ¶ ¶ ¶ ...\n") with task-specific prompts

Supports an optional --cot-adapter for "model organism" experiments: a LoRA
that alters the model's reasoning (e.g. rot13 encoded CoT). The organism
adapter is used for CoT generation and activation collection, then the
oracles read those activations to see if they can understand the altered reasoning.

Usage:
    # Normal mode (base model generates CoT), default 3 layers stride 5:
    python3 src/chat_compare.py --checkpoint checkpoints/final

    # Match a specific ablation config:
    python3 src/chat_compare.py --checkpoint checkpoints/final --stride 10 --n-layers 1

    # Custom layers:
    python3 src/chat_compare.py --checkpoint checkpoints/final --layers 9 18 27

    # Model organism mode (rot13 LoRA generates CoT):
    python3 src/chat_compare.py --checkpoint checkpoints/final \\
        --cot-adapter ceselder/rot13-qwen3-8b-lora
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))

from cot_utils import get_cot_positions, layer_percent_to_layer
from core.ao import (
    collect_activations,
    collect_activations_at_positions,
    choose_attn_implementation,
    get_hf_submodule,
    get_steering_hook,
    add_hook,
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
)

# Trained oracle uses ¶ placeholder (token ID 78846 in Qwen3-8B)
TRAINED_PLACEHOLDER = " ¶"


def compute_layers(model_name, n_layers=None, layers=None):
    """Compute injection layers from either explicit list or n_layers count."""
    if layers:
        return [int(l) for l in layers]
    n = n_layers or 3
    percents = [int(100 * (i + 1) / (n + 1)) for i in range(n)]
    return [layer_percent_to_layer(model_name, p) for p in percents]


def load_dual_model(model_name, checkpoint_path, cot_adapter=None, device="cuda"):
    """Load model with original AO adapter, trained adapter, and optional CoT adapter."""
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"device_map": "auto", "torch_dtype": dtype}
    kwargs["attn_implementation"] = choose_attn_implementation(model_name)

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Load trained checkpoint as "trained" adapter
    print(f"Loading trained LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name="trained", is_trainable=False)

    # Load original AO adapter
    ao_path = AO_CHECKPOINTS[model_name]
    print(f"Loading original AO from {ao_path}...")
    model.load_adapter(ao_path, adapter_name="original_ao", is_trainable=False)

    # Optionally load a "model organism" adapter for CoT generation
    if cot_adapter:
        print(f"Loading model organism LoRA from {cot_adapter}...")
        model.load_adapter(cot_adapter, adapter_name="organism", is_trainable=False)

    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


def generate_cot_base(model, tokenizer, question, max_new_tokens=4096, device="cuda",
                      use_organism=False):
    """Generate CoT. Uses organism adapter if use_organism=True, else base model."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    if use_organism and "organism" in model.peft_config:
        model.set_adapter("organism")
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        with model.disable_adapter():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def collect_multilayer_activations(model, tokenizer, text, layers, positions,
                                   use_organism=False, device="cuda"):
    """Collect activations from multiple layers at given positions.

    Returns Tensor[K * n_layers, d_model] — positions repeated per layer,
    matching the training format: [L1_p1..L1_pK, L2_p1..L2_pK, ...].
    """
    all_acts = []
    model.eval()

    for layer in layers:
        if use_organism and "organism" in model.peft_config:
            model.set_adapter("organism")
            acts = collect_activations_at_positions(
                model, tokenizer, text, layer, positions,
                device=device, adapter_name="organism",
            )
        else:
            acts = collect_activations_at_positions(
                model, tokenizer, text, layer, positions,
                device=device, adapter_name=None,
            )
        all_acts.append(acts)  # [K, D]

    return torch.cat(all_acts, dim=0)  # [K * n_layers, D]


def query_original_ao(model, tokenizer, acts_l50, prompt, model_name,
                      injection_layer=1, max_new_tokens=150, device="cuda"):
    """Query original AO with single-layer L50% format (standard PastLens)."""
    dtype = torch.bfloat16
    num_positions = acts_l50.shape[0]
    act_layer = layer_percent_to_layer(model_name, 50)

    prefix = f"L{act_layer}:" + SPECIAL_TOKEN * num_positions + "\n"
    full_prompt = prefix + prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    positions = [i for i, tid in enumerate(input_ids) if tid == special_id][:num_positions]
    assert len(positions) == num_positions

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)
    model.set_adapter("original_ao")

    hook_fn = get_steering_hook(vectors=acts_l50, positions=positions, device=device, dtype=dtype)
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def query_trained_oracle(model, tokenizer, multilayer_acts, prompt, model_name,
                         layers, injection_layer=1, max_new_tokens=150, device="cuda"):
    """Query trained oracle with multi-layer ¶ placeholder format."""
    dtype = torch.bfloat16
    num_positions = multilayer_acts.shape[0]  # K * n_layers

    # Build prefix matching training format: "L9: ¶¶¶ L18: ¶¶¶ L27: ¶¶¶\n"
    N = len(layers)
    K = num_positions // N
    assert K * N == num_positions, f"num_positions={num_positions} not divisible by {N} layers"
    parts = [f"L{l}:" + TRAINED_PLACEHOLDER * K for l in layers]
    prefix = " ".join(parts) + "\n"
    full_prompt = prefix + prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    placeholder_id = tokenizer.encode(TRAINED_PLACEHOLDER, add_special_tokens=False)[0]
    positions = [i for i, tid in enumerate(input_ids) if tid == placeholder_id][:num_positions]
    assert len(positions) == num_positions, f"Found {len(positions)} ¶ tokens, expected {num_positions}"

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)
    model.set_adapter("trained")

    hook_fn = get_steering_hook(vectors=multilayer_acts, positions=positions, device=device, dtype=dtype)
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def apply_activation_filters(multilayer_acts, ao_acts, all_layers, n_positions_per_layer,
                              selected_layers=None, pos_slice=None):
    """Apply layer and position filters to activation tensors.

    Returns (filtered_multilayer, filtered_ao, filtered_layers, n_filtered_pos).
    """
    D = multilayer_acts.shape[1]
    n_layers = len(all_layers)

    # Reshape to [n_layers, K, D]
    acts_by_layer = multilayer_acts.view(n_layers, n_positions_per_layer, D)

    # Layer filter
    if selected_layers is not None:
        layer_indices = [i for i, l in enumerate(all_layers) if l in selected_layers]
        acts_by_layer = acts_by_layer[layer_indices]
        filtered_layers = [all_layers[i] for i in layer_indices]
    else:
        filtered_layers = list(all_layers)

    # Position filter
    if pos_slice is not None:
        acts_by_layer = acts_by_layer[:, pos_slice, :]

    n_filtered_pos = acts_by_layer.shape[1]
    filtered_multilayer = acts_by_layer.reshape(-1, D)

    # Filter ao_acts with same position slice
    filtered_ao = ao_acts
    if ao_acts is not None and pos_slice is not None:
        filtered_ao = ao_acts[pos_slice]

    return filtered_multilayer, filtered_ao, filtered_layers, n_filtered_pos


def build_oracle_prompt(selected_layers, all_layers, pos_slice, n_positions_per_layer):
    """Build the prompt suffix showing active filters, e.g. ' [L9,27 P0-10]'."""
    parts = []
    if selected_layers is not None and set(selected_layers) != set(all_layers):
        parts.append("L" + ",".join(str(l) for l in selected_layers))
    if pos_slice is not None:
        start = pos_slice.start or 0
        stop = (pos_slice.stop or n_positions_per_layer) - 1
        if stop >= n_positions_per_layer:
            stop = n_positions_per_layer - 1
        parts.append(f"P{start}-{stop}")
    if parts:
        return f" [{' '.join(parts)}]"
    return ""


def print_side_by_side(label_a, text_a, label_b, text_b, width=38):
    import textwrap
    lines_a = textwrap.wrap(text_a, width=width) or ["(empty)"]
    lines_b = textwrap.wrap(text_b, width=width) or ["(empty)"]

    print(f"  {'─' * width}  {'─' * width}")
    print(f"  {label_a:<{width}}  {label_b:<{width}}")
    print(f"  {'─' * width}  {'─' * width}")

    max_lines = max(len(lines_a), len(lines_b))
    for i in range(max_lines):
        la = lines_a[i] if i < len(lines_a) else ""
        lb = lines_b[i] if i < len(lines_b) else ""
        print(f"  {la:<{width}}  {lb:<{width}}")
    print()


# Task-specific prompts (N and stride are filled in at query time)
TASK_PROMPTS = {
    "recon": "Reconstruct the original chain-of-thought reasoning from these activations.",
    "next": "Predict the next ~50 tokens of the chain-of-thought reasoning.",
    "domain": "What domain is this reasoning about? Answer with one word: math, science, logic, commonsense, reading, multi_domain, medical, ethics, diverse.",
    "correctness": "Is the model's final answer correct? Answer: correct or incorrect.",
    "decorative": "Is this chain-of-thought reasoning load-bearing or decorative? Answer: load_bearing or decorative.",
    "termination": "Will the model emit </think> within the next 100 tokens? Answer: will_terminate or will_continue.",
    "answer": "What is the model's final answer? Give the answer only.",
}


def main():
    parser = argparse.ArgumentParser(description="CoT Oracle A/B Comparison")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True, help="Trained LoRA checkpoint path or HF repo")
    parser.add_argument("--cot-adapter", default=None,
                        help="Model organism LoRA for CoT generation (e.g. ceselder/rot13-qwen3-8b-lora)")
    parser.add_argument("--stride", type=int, default=5, help="Stride for activation positions (default: 5)")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Number of evenly-spaced layers (default: 3 → layers 9,18,27 for 8B)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Explicit layer indices (overrides --n-layers)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    layers = compute_layers(args.model, n_layers=args.n_layers, layers=args.layers)
    layer_50 = layer_percent_to_layer(args.model, 50)

    model, tokenizer = load_dual_model(args.model, args.checkpoint,
                                       cot_adapter=args.cot_adapter, device=args.device)

    multilayer_acts = None
    ao_acts = None  # single-layer L50% for original AO
    n_positions_per_layer = 0

    use_organism = args.cot_adapter is not None
    stride = args.stride

    # Filter state (None = no filter / use all)
    selected_layers = None  # None means all layers
    pos_slice = None        # None means all positions

    print("=" * 80)
    print("  CoT Oracle A/B Comparison")
    print()
    if use_organism:
        print(f"  CoT generation: organism adapter ({args.cot_adapter})")
    else:
        print("  CoT generation: base model (no adapter)")
    print(f"  Original AO: single-layer L50% (layer {layer_50}), ? placeholders (PastLens)")
    print(f"  Trained oracle: {len(layers)} layer(s) {layers}, ¶ placeholders, stride={stride}")
    print()
    print("  Commands:")
    print("    <question>     = generate CoT and collect activations")
    print("    'recon'        = reconstruct full CoT")
    print("    'next'         = predict next ~50 tokens")
    print("    'domain'       = what domain?")
    print("    'correct'      = is the answer correct?")
    print("    'decorative'   = load-bearing or decorative?")
    print("    'termination'  = will it emit </think> soon?")
    print("    'answer'       = predict final answer")
    print("    <anything>     = free-form question to both oracles")
    print("    'new'          = start fresh, 'quit' = exit")
    print()
    print("  Filters (in oracle phase):")
    print("    layers 9 27    = only use layers 9 and 27")
    print("    layers all     = reset to all layers")
    print("    pos 0-10       = only use stride positions 0-10")
    print("    pos last 5     = only use last 5 positions")
    print("    pos all        = reset to all positions")
    print("    filters        = show current filter state")
    print("    reset          = reset all filters")
    print("=" * 80)

    while True:
        if multilayer_acts is None:
            user_input = input("\nQuestion> ").strip()
        else:
            filter_suffix = build_oracle_prompt(selected_layers, layers, pos_slice, n_positions_per_layer)
            user_input = input(f"\nAsk oracles{filter_suffix}> ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "new":
            multilayer_acts = None
            ao_acts = None
            n_positions_per_layer = 0
            selected_layers = None
            pos_slice = None
            print("\nStarting fresh.")
            continue

        # --- Filter commands (only in oracle phase) ---
        if multilayer_acts is not None:
            cmd_parts = user_input.lower().split()

            if cmd_parts[0] == "layers":
                if len(cmd_parts) < 2:
                    print("  Usage: layers <num> [num ...] | layers all")
                    continue
                if cmd_parts[1] == "all":
                    selected_layers = None
                    print(f"  Filters: layers ALL ({layers}), "
                          f"positions {'all' if pos_slice is None else pos_slice}/{n_positions_per_layer}")
                else:
                    try:
                        requested = [int(x) for x in cmd_parts[1:]]
                    except ValueError:
                        print("  Usage: layers <num> [num ...] | layers all")
                        continue
                    invalid = [l for l in requested if l not in layers]
                    if invalid:
                        print(f"  Invalid layers {invalid}. Available: {layers}")
                        continue
                    selected_layers = requested
                    print(f"  Filters: layers {selected_layers}, "
                          f"positions {'all' if pos_slice is None else f'{pos_slice.start or 0}-{(pos_slice.stop or n_positions_per_layer)-1}'}/{n_positions_per_layer}")
                continue

            if cmd_parts[0] == "pos":
                if len(cmd_parts) < 2:
                    print("  Usage: pos <start>-<end> | pos last <N> | pos all")
                    continue
                if cmd_parts[1] == "all":
                    pos_slice = None
                    active_layers = selected_layers if selected_layers else layers
                    print(f"  Filters: layers {active_layers}, positions all/{n_positions_per_layer}")
                elif cmd_parts[1] == "last":
                    if len(cmd_parts) < 3:
                        print("  Usage: pos last <N>")
                        continue
                    try:
                        n = int(cmd_parts[2])
                    except ValueError:
                        print("  Usage: pos last <N>")
                        continue
                    if n > n_positions_per_layer:
                        n = n_positions_per_layer
                    start = n_positions_per_layer - n
                    pos_slice = slice(start, n_positions_per_layer)
                    active_layers = selected_layers if selected_layers else layers
                    print(f"  Filters: layers {active_layers}, positions {start}-{n_positions_per_layer - 1}/{n_positions_per_layer}")
                else:
                    # Parse "0-10" or "5-20"
                    try:
                        if "-" in cmd_parts[1]:
                            start_s, end_s = cmd_parts[1].split("-", 1)
                            start = int(start_s)
                            end = int(end_s)
                        else:
                            start = int(cmd_parts[1])
                            end = start
                    except ValueError:
                        print("  Usage: pos <start>-<end> | pos last <N> | pos all")
                        continue
                    end = min(end, n_positions_per_layer - 1)
                    start = max(start, 0)
                    pos_slice = slice(start, end + 1)  # inclusive end
                    active_layers = selected_layers if selected_layers else layers
                    print(f"  Filters: layers {active_layers}, positions {start}-{end}/{n_positions_per_layer}")
                continue

            if cmd_parts[0] == "filters":
                active_layers = selected_layers if selected_layers else layers
                if pos_slice is not None:
                    pos_str = f"{pos_slice.start or 0}-{(pos_slice.stop or n_positions_per_layer) - 1}"
                else:
                    pos_str = "all"
                print(f"  Layers: {active_layers} ({'filtered' if selected_layers else 'all'})")
                print(f"  Positions: {pos_str}/{n_positions_per_layer} ({'filtered' if pos_slice else 'all'})")
                continue

            if cmd_parts[0] == "reset":
                selected_layers = None
                pos_slice = None
                print(f"  All filters reset. Layers: {layers}, positions: all/{n_positions_per_layer}")
                continue

        if multilayer_acts is None:
            current_question = user_input
            if use_organism:
                print(f"\nGenerating CoT with organism adapter...")
            else:
                print(f"\nGenerating CoT...")

            try:
                cot_response = generate_cot_base(
                    model, tokenizer, current_question,
                    max_new_tokens=4096, device=args.device,
                    use_organism=use_organism,
                )
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            print(f"\n--- Model CoT ---")
            print(cot_response[:1500])
            if len(cot_response) > 1500:
                print(f"  ... ({len(cot_response)} chars total)")
            print("--- End ---")

            # Tokenize to compute stride positions
            messages = [{"role": "user", "content": current_question}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + cot_response

            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            all_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            total_len = len(all_ids)

            stride_positions = get_cot_positions(prompt_len, total_len, stride=stride, tokenizer=tokenizer, input_ids=all_ids)
            if len(stride_positions) < 2:
                print("  CoT too short for stride positions.")
                continue

            n_positions_per_layer = len(stride_positions)

            # Collect multi-layer activations for trained oracle
            print(f"\nCollecting activations: {n_positions_per_layer} positions × {len(layers)} layer(s) "
                  f"= {n_positions_per_layer * len(layers)} total"
                  f"{' (organism adapter)' if use_organism else ''}...")
            try:
                multilayer_acts = collect_multilayer_activations(
                    model, tokenizer, full_text, layers, stride_positions,
                    use_organism=use_organism, device=args.device,
                )
            except Exception as e:
                print(f"  Multi-layer extraction failed: {e}")
                multilayer_acts = None
                continue

            # Collect single-layer L50% for original AO comparison
            try:
                ao_acts = collect_activations_at_positions(
                    model, tokenizer, full_text, layer_50, stride_positions,
                    device=args.device, adapter_name=None,
                )
            except Exception as e:
                print(f"  AO extraction failed: {e}")
                ao_acts = None

            cot_tokens = total_len - prompt_len
            print(f"\n  CoT: {cot_tokens} tokens, {n_positions_per_layer} stride positions, stride={stride}")
            print(f"  Trained oracle: {multilayer_acts.shape[0]} activation vectors ({len(layers)} layers)")
            if ao_acts is not None:
                print(f"  Original AO: {ao_acts.shape[0]} activation vectors (L50%)")
            print(f"\n  Ready. Try: recon, next, domain, correct, decorative, termination, answer, or free-form")
            continue

        # Map shortcuts to task prompts
        prompt_key = user_input.lower().strip()
        if prompt_key in TASK_PROMPTS:
            trained_prompt = TASK_PROMPTS[prompt_key]
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        else:
            # Free-form: send same question to both
            trained_prompt = user_input
            ao_prompt = user_input

        print()

        # Apply activation filters
        filt_ml, filt_ao, filt_layers, filt_npos = apply_activation_filters(
            multilayer_acts, ao_acts, layers, n_positions_per_layer,
            selected_layers=selected_layers, pos_slice=pos_slice,
        )

        if selected_layers is not None or pos_slice is not None:
            print(f"  Using {len(filt_layers)} layer(s) {filt_layers}, "
                  f"{filt_npos} positions -> {filt_ml.shape[0]} vectors")

        # Query original AO (single-layer L50%)
        if filt_ao is not None:
            try:
                resp_original = query_original_ao(
                    model, tokenizer, filt_ao,
                    ao_prompt, model_name=args.model,
                    max_new_tokens=args.max_tokens, device=args.device,
                )
            except Exception as e:
                resp_original = f"ERROR: {e}"
        else:
            resp_original = "(no AO activations)"

        # Query trained oracle (multi-layer, ¶ tokens, task prompt)
        try:
            resp_trained = query_trained_oracle(
                model, tokenizer, filt_ml,
                trained_prompt, model_name=args.model,
                layers=filt_layers,
                max_new_tokens=args.max_tokens, device=args.device,
            )
        except Exception as e:
            resp_trained = f"ERROR: {e}"

        print(f"  AO prompt: {ao_prompt[:80]}")
        print(f"  Oracle prompt: {trained_prompt[:80]}")
        print()
        print_side_by_side("ORIGINAL AO", resp_original, "TRAINED COT ORACLE", resp_trained)


if __name__ == "__main__":
    main()
