"""
Side-by-side comparison: Original AO vs Trained CoT Oracle.

Key difference in input format:
  - Original AO: single layer (L50%), standard PastLens prefix ("Layer: 18\n ? ? ? ...\n")
  - Trained oracle: single layer (L50%), 5-token stride prefix
    ("Layer: 18\n ¶ ¶ ¶ ...\n") with task-specific prompts

Supports an optional --cot-adapter for "model organism" experiments: a LoRA
that alters the model's reasoning (e.g. rot13 encoded CoT). The organism
adapter is used for CoT generation and activation collection, then the
oracles read those activations to see if they can understand the altered reasoning.

Usage:
    # Normal mode (base model generates CoT):
    python3 src/chat_compare.py --checkpoint checkpoints/cot_oracle_v3b/step_4000

    # Model organism mode (rot13 LoRA generates CoT):
    python3 src/chat_compare.py --checkpoint checkpoints/cot_oracle_v3b/step_4000 \\
        --cot-adapter ceselder/rot13-qwen3-8b-lora
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

sys.path.insert(0, str(Path(__file__).parent))

from cot_utils import get_cot_stride_positions
from core.ao import (
    layer_percent_to_layer,
    collect_activations,
    choose_attn_implementation,
    get_hf_submodule,
    get_steering_hook,
    add_hook,
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    EarlyStopException,
)
from position_encoding import apply_position_encoding

# Trained oracle uses ¶ placeholder (token ID 78846 in Qwen3-8B)
TRAINED_PLACEHOLDER = " ¶"
STRIDE = 5


def load_dual_model(model_name, checkpoint_path, cot_adapter=None, device="cuda"):
    """Load model with original AO adapter, trained adapter, and optional CoT adapter.

    If cot_adapter is provided, it's loaded as a "model organism" — a LoRA that
    alters how the model reasons (e.g. rot13 CoT) so we can test whether the
    oracle can read through the altered reasoning.
    """
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


def collect_single_layer_activations(model, tokenizer, text, layer, positions,
                                     use_organism=False, device="cuda"):
    """Collect activations from a single layer at given positions.

    When use_organism=True, collects with organism adapter enabled so we read
    the model's actual internal representations during altered reasoning.

    Returns Tensor[num_positions, d_model].
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    model.eval()
    if use_organism and "organism" in model.peft_config:
        model.set_adapter("organism")
        acts_BLD = collect_activations(
            model, layer, inputs["input_ids"], inputs["attention_mask"],
        )
    else:
        with model.disable_adapter():
            acts_BLD = collect_activations(
                model, layer, inputs["input_ids"], inputs["attention_mask"],
            )

    return acts_BLD[0, positions, :].detach()


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


def query_trained_oracle(model, tokenizer, acts_l50, prompt, model_name,
                         injection_layer=1, max_new_tokens=150, device="cuda"):
    """Query trained oracle with single-layer L50%, ¶ placeholders, 5-token stride format."""
    dtype = torch.bfloat16
    num_positions = acts_l50.shape[0]
    act_layer = layer_percent_to_layer(model_name, 50)

    prefix = f"Layer: {act_layer}\n" + TRAINED_PLACEHOLDER * num_positions + " \n"
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

    hook_fn = get_steering_hook(vectors=acts_l50, positions=positions, device=device, dtype=dtype)
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
    "domain": "Activations from {N} positions (5-token stride). What domain is this reasoning about? Answer with one word: math, science, logic, commonsense, reading, multi_domain, medical, ethics, diverse.",
    "correctness": "Activations from {N} positions (5-token stride). Is the model's final answer correct? Answer: correct or incorrect.",
    "decorative": "Activations from {N} positions (5-token stride). Is this chain-of-thought reasoning load-bearing or decorative? Answer: load_bearing or decorative.",
    "summary": "Activations from {N} positions (5-token stride). Generate a concise summary of the thought process.",
    "predict": "Activations from {N} positions (5-token stride). Predict the next {K} tokens following position {T}.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True, help="Trained LoRA checkpoint path")
    parser.add_argument("--cot-adapter", default=None,
                        help="Model organism LoRA for CoT generation (e.g. ceselder/rot13-qwen3-8b-lora)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    model, tokenizer = load_dual_model(args.model, args.checkpoint,
                                       cot_adapter=args.cot_adapter, device=args.device)

    layer_50 = layer_percent_to_layer(args.model, 50)
    print(f"  Layer: {layer_50} (L50%)")

    acts_l50 = None
    n_positions = 0

    use_organism = args.cot_adapter is not None

    print("=" * 80)
    print("  CoT Oracle A/B Comparison")
    print()
    if use_organism:
        print(f"  CoT generation: organism adapter ({args.cot_adapter})")
    else:
        print("  CoT generation: base model (no adapter)")
    print("  Original AO: single-layer L50%, ? placeholders (PastLens)")
    print(f"  Trained oracle: single-layer L50%, ¶ placeholders, stride={STRIDE}")
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
        if acts_l50 is None:
            user_input = input("\nQuestion> ").strip()
        else:
            user_input = input("\nAsk oracles> ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "new":
            acts_l50 = None
            n_positions = 0
            print("\nStarting fresh.")
            continue

        if acts_l50 is None:
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

            stride_positions = get_cot_stride_positions(prompt_len, total_len, stride=STRIDE)
            if len(stride_positions) < 2:
                print("  CoT too short for stride positions.")
                continue

            n_positions = len(stride_positions)

            print(f"\nCollecting activations at {n_positions} positions (stride={STRIDE}), L50%"
                  f"{' (organism adapter)' if use_organism else ''}...")
            try:
                acts_l50 = collect_single_layer_activations(
                    model, tokenizer, full_text, layer_50, stride_positions,
                    use_organism=use_organism, device=args.device,
                )
            except Exception as e:
                print(f"  Failed: {e}")
                acts_l50 = None
                continue

            cot_tokens = total_len - prompt_len
            print(f"\n  CoT: {cot_tokens} tokens, {n_positions} stride positions")
            cot_snippet = cot_response[:500]
            print(f"  {cot_snippet}{'...' if len(cot_response) > 500 else ''}")

            print(f"\n  Ready. Try: domain, correct, decorative, summary, or free-form question")
            continue

        # Map shortcuts to task prompts
        prompt_key = user_input.lower().strip()
        if prompt_key in ("domain",):
            trained_prompt = TASK_PROMPTS["domain"].format(N=n_positions)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("correct", "correctness"):
            trained_prompt = TASK_PROMPTS["correctness"].format(N=n_positions)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("decorative", "load_bearing", "load-bearing"):
            trained_prompt = TASK_PROMPTS["decorative"].format(N=n_positions)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        elif prompt_key in ("summary", "summarize"):
            trained_prompt = TASK_PROMPTS["summary"].format(N=n_positions)
            ao_prompt = "Can you predict the next 10 tokens that come after this?"
        else:
            # Free-form: prepend the stride preamble for the trained oracle
            trained_prompt = f"Activations from {n_positions} positions (5-token stride). {user_input}"
            ao_prompt = user_input

        print()

        # Query original AO (single-layer L50%)
        try:
            resp_original = query_original_ao(
                model, tokenizer, acts_l50,
                ao_prompt, model_name=args.model,
                max_new_tokens=args.max_tokens, device=args.device,
            )
        except Exception as e:
            resp_original = f"ERROR: {e}"

        # Query trained oracle (single-layer L50%, ¶ tokens, task prompt)
        try:
            resp_trained = query_trained_oracle(
                model, tokenizer, acts_l50,
                trained_prompt, model_name=args.model,
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
