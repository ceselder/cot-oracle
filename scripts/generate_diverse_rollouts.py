#!/usr/bin/env python3
"""
Generate CoT rollouts for diverse prompts using vLLM on GPU.

Takes prompts from data/diverse_rollouts/prompts.jsonl, generates:
1. CoT response (thinking enabled)
2. Direct response (thinking disabled, for load-bearing detection)

Output: data/diverse_rollouts/corpus.jsonl (same format as cot_corpus_v5)

Usage:
    # On GPU machine:
    python scripts/generate_diverse_rollouts.py [--batch-size 256] [--max-prompts 51000]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Domain mapping ──
SOURCE_DOMAIN = {
    "aqua_rat": "math",
    "hellaswag": "commonsense",
    "arc_challenge": "science",
    "winogrande": "commonsense",
    "piqa": "commonsense",
    "social_iqa": "commonsense",
    "boolq": "reading",
    "openbookqa": "science",
    "medqa": "medical",
    "mmlu": "diverse",
    "truthfulqa": "diverse",
    "cot_corpus_v5": "math",
    "concept_corpus": "diverse",
}


def extract_answer(text: str) -> str | None:
    """Extract a clean short answer from model output."""
    if not text:
        return None
    # Remove think tags
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean:
        return None
    # Try boxed
    boxed = re.search(r"\\boxed\{([^}]+)\}", clean)
    if boxed:
        return boxed.group(1).strip()
    return clean[:500] if clean else None


def check_correctness(response: str, metadata: dict) -> bool | None:
    """Check if the response matches the correct answer from metadata."""
    correct = metadata.get("correct") or metadata.get("correct_answer") or metadata.get("answer")
    if not correct:
        return None  # Can't verify

    answer = extract_answer(response)
    if not answer:
        return None

    correct_str = str(correct).strip().lower()
    answer_lower = answer.lower()

    # Exact match
    if correct_str == answer_lower:
        return True
    # Substring
    if correct_str in answer_lower or answer_lower in correct_str:
        return True
    # Letter match for MCQ
    if len(correct_str) == 1 and correct_str.isalpha():
        # Check if the answer starts with the correct letter
        first_word = answer_lower.split()[0] if answer_lower else ""
        if first_word.rstrip(".),:") == correct_str:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/diverse_rollouts/prompts.jsonl")
    parser.add_argument("--output", default="data/diverse_rollouts/corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="vLLM batch size for concurrent generation")
    parser.add_argument("--max-prompts", type=int, default=51000)
    parser.add_argument("--max-cot-tokens", type=int, default=2048)
    parser.add_argument("--max-direct-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")
    args = parser.parse_args()

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts")

    # Check for resume
    done_ids = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    done_ids.add(entry["id"])
        print(f"Resuming: {len(done_ids)} already done")
        prompts = [p for p in prompts if p["id"] not in done_ids]
        print(f"Remaining: {len(prompts)} prompts")

    if not prompts:
        print("All prompts already processed!")
        return

    # Import vLLM
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model {args.model} with vLLM (tp={args.tp})...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Sampling params
    cot_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_cot_tokens,
    )
    direct_params = SamplingParams(
        temperature=0.0,  # Greedy for direct answer
        max_tokens=args.max_direct_tokens,
    )

    # Format prompts with chat template
    def format_cot_prompt(question: str) -> str:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

    def format_direct_prompt(question: str) -> str:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    # Process in batches
    total_done = len(done_ids)
    t0 = time.time()
    mode = "a" if (args.resume and output_path.exists()) else "w"

    for batch_start in range(0, len(prompts), args.batch_size):
        batch = prompts[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} prompts)")

        # Generate CoT responses
        cot_formatted = [format_cot_prompt(p["question"]) for p in batch]
        print(f"  Generating CoT responses...")
        t1 = time.time()
        cot_outputs = llm.generate(cot_formatted, cot_params)
        cot_time = time.time() - t1
        print(f"  CoT done in {cot_time:.1f}s ({len(batch)/cot_time:.1f} prompts/s)")

        # Generate direct responses
        direct_formatted = [format_direct_prompt(p["question"]) for p in batch]
        print(f"  Generating direct responses...")
        t2 = time.time()
        direct_outputs = llm.generate(direct_formatted, direct_params)
        direct_time = time.time() - t2
        print(f"  Direct done in {direct_time:.1f}s")

        # Write results
        with open(output_path, mode) as f:
            for prompt_entry, cot_out, direct_out in zip(batch, cot_outputs, direct_outputs):
                cot_text = cot_out.outputs[0].text
                direct_text = direct_out.outputs[0].text

                metadata = prompt_entry.get("metadata", {})
                source = prompt_entry.get("source", "unknown")

                # Check correctness
                cot_correct = check_correctness(cot_text, metadata)
                direct_correct = check_correctness(direct_text, metadata)

                # Determine category (load-bearing / decorative / both_correct / both_wrong)
                if cot_correct is True and direct_correct is False:
                    category = "load_bearing"
                elif cot_correct is True and direct_correct is True:
                    category = "both_correct"
                elif cot_correct is False and direct_correct is True:
                    category = "decorative"  # CoT actually hurt
                elif cot_correct is False and direct_correct is False:
                    category = "both_wrong"
                else:
                    category = "unknown"

                # Extract CoT content (between <think> tags)
                cot_content = ""
                think_match = re.search(r"<think>(.*?)(?:</think>|$)", cot_text, re.DOTALL)
                if think_match:
                    cot_content = think_match.group(1).strip()

                entry = {
                    "id": prompt_entry["id"],
                    "source": source,
                    "domain": SOURCE_DOMAIN.get(source, "diverse"),
                    "question": prompt_entry["question"],
                    "correct_answer": str(metadata.get("correct", metadata.get("correct_answer", ""))),
                    "subject": metadata.get("subject", ""),
                    "level": metadata.get("level", ""),
                    "cot_response": cot_text,
                    "cot_content": cot_content,
                    "direct_response": direct_text,
                    "cot_correct": cot_correct,
                    "direct_correct": direct_correct,
                    "category": category,
                }
                f.write(json.dumps(entry) + "\n")
                total_done += 1

        mode = "a"  # After first batch, always append
        elapsed = time.time() - t0
        rate = total_done / elapsed if elapsed > 0 else 0
        print(f"  Total: {total_done} done, {elapsed:.0f}s elapsed, {rate:.1f} prompts/s")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! {total_done} entries in {elapsed:.0f}s")
    print(f"Output: {output_path}")

    # Print category breakdown
    cats = {}
    with open(output_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cat = entry.get("category", "unknown")
                cats[cat] = cats.get(cat, 0) + 1
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
