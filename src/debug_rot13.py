"""
Debug script: Generate CoTs with rot13 model organism and inspect sentence splitting.

Usage (on GPU box):
    python3 src/debug_rot13.py
    python3 src/debug_rot13.py --question "What is 17 * 24?"
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from signs_of_life.ao_lib import split_cot_into_sentences, find_sentence_boundary_positions

MODEL_NAME = "Qwen/Qwen3-8B"
ORGANISM_ADAPTER = "ceselder/rot13-qwen3-8b-lora"

QUESTIONS = [
    "What is 17 * 24?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What is the derivative of x^3 + 2x?",
]


def load_model(device="cuda"):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"device_map": "auto", "torch_dtype": dtype, "attn_implementation": "sdpa"}
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)

    print(f"Loading organism adapter from {ORGANISM_ADAPTER}...")
    model = PeftModel.from_pretrained(model, ORGANISM_ADAPTER, adapter_name="organism", is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_cot(model, tokenizer, question, adapter=None, max_new_tokens=2048, device="cuda"):
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    if adapter:
        model.set_adapter(adapter)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        with model.disable_adapter():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def analyze_cot(tokenizer, question, cot_text):
    """Show sentence splitting results and boundary detection."""
    print(f"\n{'='*80}")
    print(f"RAW COT ({len(cot_text)} chars):")
    print(f"{'─'*80}")
    print(cot_text[:2000])
    if len(cot_text) > 2000:
        print(f"... ({len(cot_text)} chars total)")

    # Check for think tags
    has_think_open = "<think>" in cot_text
    has_think_close = "</think>" in cot_text
    print(f"\n<think> tag: {has_think_open}  </think> tag: {has_think_close}")

    # Extract just the thinking part
    think_match = re.search(r'<think>(.*?)</think>', cot_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
        print(f"Think block: {len(think_text)} chars")
    else:
        think_text = cot_text
        print("No think block found, using full text")

    # Try sentence splitting
    sentences = split_cot_into_sentences(cot_text)
    print(f"\nSENTENCE SPLIT: {len(sentences)} sentences")
    print(f"{'─'*80}")
    for i, s in enumerate(sentences):
        # Check if it looks like rot13
        ascii_chars = sum(1 for c in s if c.isalpha())
        total_chars = len(s)
        print(f"  [{i:2d}] ({len(s):3d} chars, {ascii_chars}/{total_chars} alpha) {s[:120]}{'...' if len(s) > 120 else ''}")

    # Try boundary detection
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = formatted + cot_text
    boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
    print(f"\nBOUNDARY POSITIONS: {len(boundary_positions)} found (of {len(sentences)} sentences)")
    for i, pos in enumerate(boundary_positions[:15]):
        print(f"  sentence {i} -> token position {pos}")

    return sentences, boundary_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="Single question to test")
    parser.add_argument("--compare", action="store_true", help="Also generate base model CoT for comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_model(args.device)

    questions = [args.question] if args.question else QUESTIONS

    for q in questions:
        print(f"\n{'#'*80}")
        print(f"QUESTION: {q}")
        print(f"{'#'*80}")

        # Generate with organism adapter (rot13)
        print("\n>>> ORGANISM (rot13) CoT:")
        cot_rot13 = generate_cot(model, tokenizer, q, adapter="organism", device=args.device)
        sents_rot13, bounds_rot13 = analyze_cot(tokenizer, q, cot_rot13)

        if args.compare:
            # Generate with base model for comparison
            print("\n>>> BASE MODEL CoT:")
            cot_base = generate_cot(model, tokenizer, q, adapter=None, device=args.device)
            sents_base, bounds_base = analyze_cot(tokenizer, q, cot_base)

            print(f"\n>>> COMPARISON:")
            print(f"  Base:     {len(sents_base)} sentences, {len(bounds_base)} boundaries")
            print(f"  Organism: {len(sents_rot13)} sentences, {len(bounds_rot13)} boundaries")


if __name__ == "__main__":
    main()
