"""
Side-by-side comparison: Original AO vs Trained CoT Oracle.

Key difference in input format:
  - Original AO: single layer (L50%), standard PastLens prefix ("Layer: 18\n ? ? ? ...\n")
  - Trained oracle: multi-layer (L25%+L50%+L75%), sentence-boundary prefix
    ("Layer: 9, 18, 27\n @ ? # @ ? # ...\n") with task-specific prompts

Usage:
    python3 src/chat_compare.py --checkpoint checkpoints/cot_oracle_v4/step_28000
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

sys.path.insert(0, str(Path(__file__).parent))

from signs_of_life.ao_lib import (
    layer_percent_to_layer,
    collect_activations,
    split_cot_into_sentences,
    find_sentence_boundary_positions,
    get_hf_submodule,
    get_steering_hook,
    add_hook,
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    EarlyStopException,
)

# Per-layer tokens matching train_mixed.py
LAYER_TOKENS = [" @", " ?", " #"]  # L25%, L50%, L75%


def load_dual_model(model_name, checkpoint_path, device="cuda"):
    """Load model with both original AO adapter and trained adapter."""
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"device_map": "auto", "torch_dtype": dtype}
    try:
        import flash_attn
        if "Qwen" in model_name:
            kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        kwargs["attn_implementation"] = "sdpa"

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

    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


def generate_cot_base(model, tokenizer, question, max_new_tokens=4096, device="cuda"):
    """Generate CoT with adapters disabled."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with model.disable_adapter():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def collect_multilayer_activations(model, tokenizer, text, layers, positions, device="cuda"):
    """Collect activations from multiple layers at given positions.

    Returns dict[layer -> Tensor[num_positions, d_model]].
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    acts_by_layer = {}

    model.eval()
    with model.disable_adapter():
        for layer in layers:
            acts_BLD = collect_activations(
                model, layer, inputs["input_ids"], inputs["attention_mask"],
            )
            acts_by_layer[layer] = acts_BLD[0, positions, :].detach()

    return acts_by_layer


def query_original_ao(model, tokenizer, acts_l50, prompt, model_name,
                      injection_layer=1, max_new_tokens=150, device="cuda"):
    """Query original AO with single-layer L50% format (standard PastLens)."""
    dtype = torch.bfloat16
    num_positions = acts_l50.shape[0]
    act_layer = layer_percent_to_layer(model_name, 50)

    prefix = f"Layer: {act_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
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


def query_trained_oracle(model, tokenizer, acts_by_layer, layers, prompt, model_name,
                         use_per_layer_tokens=False,
                         injection_layer=1, max_new_tokens=150, device="cuda"):
    """Query trained oracle with multi-layer sentence-boundary format.

    Interleaves activations: [L25_s1, L50_s1, L75_s1, L25_s2, L50_s2, L75_s2, ...]
    Prefix: "Layer: 9, 18, 27\n @ ? # @ ? # ...\n" (or all ? if not per_layer_tokens)
    """
    dtype = torch.bfloat16
    n_boundaries = acts_by_layer[layers[0]].shape[0]
    num_positions = n_boundaries * len(layers)

    # Interleave activations: [L25_s1, L50_s1, L75_s1, L25_s2, ...]
    interleaved = []
    for s in range(n_boundaries):
        for layer in layers:
            interleaved.append(acts_by_layer[layer][s])
    acts_interleaved = torch.stack(interleaved)  # [num_positions, d_model]

    # Build prefix
    layers_str = ", ".join(str(l) for l in layers)
    prefix = f"Layer: {layers_str}\n"
    if use_per_layer_tokens:
        for i in range(num_positions):
            prefix += LAYER_TOKENS[i % len(layers)]
    else:
        prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"

    full_prompt = prefix + prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    # Find placeholder positions
    if use_per_layer_tokens:
        layer_token_ids = [tokenizer.encode(lt, add_special_tokens=False)[0] for lt in LAYER_TOKENS]
        token_set = set(layer_token_ids)
        positions = []
        for i, tid in enumerate(input_ids):
            if len(positions) == num_positions:
                break
            expected_tid = layer_token_ids[len(positions) % len(layers)]
            if tid == expected_tid:
                positions.append(i)
    else:
        special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        positions = [i for i, tid in enumerate(input_ids) if tid == special_id][:num_positions]

    assert len(positions) == num_positions, f"Found {len(positions)}, expected {num_positions}"

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)
    model.set_adapter("trained")

    hook_fn = get_steering_hook(vectors=acts_interleaved, positions=positions, device=device, dtype=dtype)
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


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


# Task-specific prompts that match what the trained oracle was trained on
TASK_PROMPTS = {
    "domain": "Activations from {N} sentence boundaries. What domain is this reasoning about? Answer with one word: math, science, logic, commonsense, reading, multi_domain, medical.",
    "correctness": "Activations from {N} sentence boundaries. Is the model's final answer correct? Answer: correct or incorrect.",
    "decorative": "Activations from {N} sentence boundaries. Is this chain-of-thought reasoning load-bearing or decorative? Answer: load_bearing or decorative.",
    "summary": "Activations from {N} sentence boundaries. Generate a concise summary of the thought process.",
    "predict": "Activations from {N} sentence boundaries. Predict the next {K} tokens following sentence {T}.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True, help="Trained LoRA checkpoint path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--per-layer-tokens", action="store_true", default=False,
                        help="Use @?# tokens (must match how the oracle was trained)")
    args = parser.parse_args()

    model, tokenizer = load_dual_model(args.model, args.checkpoint, args.device)

    layers = [
        layer_percent_to_layer(args.model, 25),
        layer_percent_to_layer(args.model, 50),
        layer_percent_to_layer(args.model, 75),
    ]
    layer_50 = layers[1]
    print(f"  Layers: {layers} (L25%, L50%, L75%)")

    acts_by_layer = None
    n_boundaries = 0

    print("=" * 80)
    print("  CoT Oracle A/B Comparison")
    print()
    print("  Original AO: single-layer L50%, standard PastLens format")
    print("  Trained oracle: 3-layer sentence boundaries, task-specific prompts")
    print()
    print("  Commands:")
    print("    <question>  = generate CoT and collect activations")
    print("    'domain'    = ask: what domain?")
    print("    'correct'   = ask: is the answer correct?")
    print("    'decorative'= ask: load-bearing or decorative?")
    print("    'summary'   = ask: summarize the thought process")
    print("    <anything>  = free-form question to both oracles")
    print("    'new'       = start fresh, 'quit' = exit")
    print("=" * 80)

    while True:
        if acts_by_layer is None:
            user_input = input("\nQuestion> ").strip()
        else:
            user_input = input("\nAsk oracles> ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "new":
            acts_by_layer = None
            n_boundaries = 0
            print("\nStarting fresh.")
            continue

        if acts_by_layer is None:
            current_question = user_input
            print(f"\nGenerating CoT...")

            try:
                cot_response = generate_cot_base(
                    model, tokenizer, current_question,
                    max_new_tokens=4096, device=args.device,
                )
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            print(f"\n--- Model CoT ---")
            print(cot_response[:1500])
            if len(cot_response) > 1500:
                print(f"  ... ({len(cot_response)} chars total)")
            print("--- End ---")

            sentences = split_cot_into_sentences(cot_response)
            if len(sentences) < 2:
                print("  Not enough sentences.")
                continue

            messages = [{"role": "user", "content": current_question}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + cot_response
            boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)

            if len(boundary_positions) < 2:
                print("  Not enough boundary positions.")
                continue

            boundary_positions = boundary_positions[:15]
            n_boundaries = len(boundary_positions)

            print(f"\nCollecting activations at {n_boundaries} boundaries, 3 layers...")
            try:
                acts_by_layer = collect_multilayer_activations(
                    model, tokenizer, full_text, layers, boundary_positions, device=args.device,
                )
            except Exception as e:
                print(f"  Failed: {e}")
                acts_by_layer = None
                continue

            print(f"\n  CoT: {len(sentences)} sentences, {n_boundaries} boundaries")
            for i, s in enumerate(sentences[:8]):
                print(f"    [{i+1}] {s[:90]}{'...' if len(s) > 90 else ''}")
            if len(sentences) > 8:
                print(f"    ... and {len(sentences) - 8} more")

            print(f"\n  Ready. Try: domain, correct, decorative, summary, or free-form question")
            continue

        # Map shortcuts to task prompts
        prompt_key = user_input.lower().strip()
        if prompt_key in ("domain",):
            trained_prompt = TASK_PROMPTS["domain"].format(N=n_boundaries)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("correct", "correctness"):
            trained_prompt = TASK_PROMPTS["correctness"].format(N=n_boundaries)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("decorative", "load_bearing", "load-bearing"):
            trained_prompt = TASK_PROMPTS["decorative"].format(N=n_boundaries)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("summary", "summarize"):
            trained_prompt = TASK_PROMPTS["summary"].format(N=n_boundaries)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        else:
            # Free-form: prepend the sentence-boundary preamble for the trained oracle
            # (all its training data used this format) but send raw to original AO
            trained_prompt = f"Activations from {n_boundaries} sentence boundaries. {user_input}"
            ao_prompt = user_input

        print()

        # Query original AO (single-layer L50%)
        try:
            resp_original = query_original_ao(
                model, tokenizer, acts_by_layer[layer_50],
                ao_prompt, model_name=args.model,
                max_new_tokens=args.max_tokens, device=args.device,
            )
        except Exception as e:
            resp_original = f"ERROR: {e}"

        # Query trained oracle (multi-layer, task prompt)
        try:
            resp_trained = query_trained_oracle(
                model, tokenizer, acts_by_layer, layers,
                trained_prompt, model_name=args.model,
                use_per_layer_tokens=args.per_layer_tokens,
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
