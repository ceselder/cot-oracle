"""
Interactive chat with the CoT Oracle.

1. Give it a question → model thinks with CoT
2. Ask the oracle anything about the activations
3. Type 'new' to start a new problem, 'quit' to exit

Usage:
    python3 src/chat_with_oracle.py --checkpoint checkpoints/cot_oracle_8b_v2/step_3000
    python3 src/chat_with_oracle.py  # uses original AO checkpoint
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))

from signs_of_life.ao_lib import (
    layer_percent_to_layer,
    collect_activations_at_positions,
    run_oracle_on_activations,
    generate_cot,
    split_cot_into_sentences,
    find_sentence_boundary_positions,
    load_model_with_ao,
    AO_CHECKPOINTS,
)


def load_model(model_name, checkpoint=None, device="cuda"):
    if checkpoint:
        print(f"Loading {model_name} + trained checkpoint...")
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

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        model.eval()
        print(f"  Adapter: {model.active_adapter}")
    else:
        print(f"Loading {model_name} + original AO...")
        model, tokenizer = load_model_with_ao(model_name, device=device)

    return model, tokenizer


def collect_cot_activations(model, tokenizer, question, cot_response, act_layer, device="cuda", max_positions=10):
    sentences = split_cot_into_sentences(cot_response)
    if len(sentences) < 2:
        return None, sentences, []

    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = formatted + cot_response

    boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
    if len(boundary_positions) < 2:
        return None, sentences, []

    positions = boundary_positions[:max_positions]
    activations = collect_activations_at_positions(
        model, tokenizer, full_text, act_layer, positions, device=device,
    )
    return activations, sentences, positions


def print_cot_summary(sentences):
    print(f"\n  CoT ({len(sentences)} sentences):")
    for i, s in enumerate(sentences):
        tag = f"[{i+1}]"
        preview = s[:100] + ("..." if len(s) > 100 else "")
        print(f"    {tag:5s} {preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", default=None, help="Trained LoRA checkpoint path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max oracle response tokens")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.checkpoint, args.device)
    act_layer = layer_percent_to_layer(args.model, 50)
    print(f"  Activation layer: {act_layer}")
    print()

    activations = None
    current_question = None
    current_sentences = None

    print("=" * 60)
    print("  CoT Oracle Interactive Chat")
    print("  Type a question to generate CoT, then ask the oracle.")
    print("  Commands: 'new' = new problem, 'quit' = exit")
    print("=" * 60)

    while True:
        if activations is None:
            prompt_text = "\nQuestion> "
        else:
            prompt_text = "\nOracle> "

        try:
            user_input = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "new":
            activations = None
            current_question = None
            current_sentences = None
            print("\nStarting fresh. Enter a new question.")
            continue

        # If no activations yet, treat input as a question to generate CoT
        if activations is None:
            current_question = user_input
            print(f"\nGenerating CoT for: {current_question}")
            print("  (this takes ~10-30s)...")

            try:
                cot_response = generate_cot(
                    model, tokenizer, current_question,
                    max_new_tokens=1024, device=args.device,
                )
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            # Show the raw response
            print(f"\n--- Model Response ---")
            print(cot_response[:2000])
            if len(cot_response) > 2000:
                print(f"  ... ({len(cot_response)} chars total)")
            print("--- End Response ---")

            # Collect activations
            print("\nCollecting activations at sentence boundaries...")
            try:
                activations, current_sentences, positions = collect_cot_activations(
                    model, tokenizer, current_question, cot_response,
                    act_layer, device=args.device,
                )
            except Exception as e:
                print(f"  Activation collection failed: {e}")
                activations = None
                continue

            if activations is None:
                print("  Not enough sentences for activation collection.")
                continue

            print_cot_summary(current_sentences)
            print(f"\n  Collected {activations.shape[0]} activation vectors at layer {act_layer}")
            print(f"  Now ask the oracle anything about this reasoning trace.")
            print(f"  (e.g. 'What is the model thinking?', 'Is this reasoning faithful?')")
            continue

        # We have activations — treat input as oracle query
        try:
            response = run_oracle_on_activations(
                model, tokenizer, activations, user_input,
                model_name=args.model, act_layer=act_layer,
                max_new_tokens=args.max_tokens, device=args.device,
            )
            print(f"\n  Oracle: {response}")
        except Exception as e:
            print(f"  Oracle query failed: {e}")


if __name__ == "__main__":
    main()
