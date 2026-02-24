#!/usr/bin/env python3
"""
Precompute forced-answer entropy values for the forced_answer_entropy_riya eval.

For each GPQA Diamond question:
1. Generate N CoT rollouts (temperature=0.7)
2. For each rollout, find sentence boundaries in the thinking text
3. At each sentence boundary, construct a forced-answer prefix:
     [chat template with question]<think>[partial CoT up to sentence i]</think>So, the answer is:
4. Extract logprobs for answer tokens (A/B/C/D) at that position
5. Compute softmax distribution and entropy H(p)
6. Store per-sentence-boundary: position, entropy, distribution

Requires GPU with vLLM for efficient batched logprob extraction.

Usage:
    python scripts/precompute_forced_entropy.py \
        --eval-path data/evals/forced_answer_entropy_riya.json \
        --model Qwen/Qwen3-8B \
        --n-rollouts 10 \
        --temperature 0.7

    # Resume from checkpoint:
    python scripts/precompute_forced_entropy.py \
        --eval-path data/evals/forced_answer_entropy_riya.json \
        --resume
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve HF `datasets` import — local evals/datasets/ package shadows it.
# We don't actually need HF datasets here, but keep the pattern consistent.
# ---------------------------------------------------------------------------


def split_cot_into_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences. Removes <think> tags first."""
    text = re.sub(r"<think>|</think>", "", cot_text).strip()
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def extract_thinking_text(response: str) -> str:
    """Extract the text inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def compute_entropy(probs: list[float]) -> float:
    """Compute Shannon entropy H(p) = -sum(p_i * log(p_i))."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def main():
    parser = argparse.ArgumentParser(
        description="Precompute forced-answer entropy values"
    )
    parser.add_argument(
        "--eval-path",
        default="data/evals/forced_answer_entropy_riya.json",
        help="Path to eval JSON",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=10,
        help="Number of CoT rollouts per question",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for CoT rollouts"
    )
    parser.add_argument(
        "--max-cot-tokens",
        type=int,
        default=2048,
        help="Max tokens for CoT generation",
    )
    parser.add_argument(
        "--max-boundaries",
        type=int,
        default=15,
        help="Max sentence boundaries per rollout to evaluate",
    )
    parser.add_argument(
        "--target-items",
        type=int,
        default=200,
        help="Target total datapoints (downsampled from all boundaries)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path for checkpoint file (default: {eval-path}.checkpoint.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run generate_datasets.py first.")
        sys.exit(1)

    checkpoint_path = Path(
        args.checkpoint_path or str(eval_path) + ".checkpoint.json"
    )

    with open(eval_path) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} eval items from {eval_path}")

    # Load checkpoint if resuming
    completed_ids: set[str] = set()
    checkpoint_data: dict[str, dict] = {}
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        completed_ids = set(checkpoint_data.keys())
        print(f"Resuming: {len(completed_ids)} items already completed")

    # Filter to items that need processing
    pending_items = [
        item for item in items if item["example_id"] not in completed_ids
    ]
    print(f"Items to process: {len(pending_items)}")

    if not pending_items:
        print("All items already completed. Merging results...")
        _merge_results(items, checkpoint_data, eval_path, args.target_items, args.seed)
        return

    # -----------------------------------------------------------------------
    # Import vLLM (only when actually running GPU compute)
    # -----------------------------------------------------------------------
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM not installed. Install with: pip install vllm")
        sys.exit(1)

    print(f"\nLoading {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Find token IDs for answer letters A, B, C, D
    answer_token_ids = {}
    for letter in ["A", "B", "C", "D"]:
        # Try multiple tokenizations to handle different tokenizer behaviors
        candidates = [
            tokenizer.encode(letter, add_special_tokens=False),
            tokenizer.encode(f" {letter}", add_special_tokens=False),
        ]
        for ids in candidates:
            if ids:
                # Take the last token (in case of prefix space token)
                answer_token_ids[letter] = ids[-1]
                break
    print(f"Answer token IDs: {answer_token_ids}")

    # -----------------------------------------------------------------------
    # Phase 1: Generate CoT rollouts for all pending items
    # -----------------------------------------------------------------------
    print(f"\nPhase 1: Generating {args.n_rollouts} CoT rollouts per item...")

    cot_gen_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_cot_tokens,
        n=args.n_rollouts,
    )

    # Build generation prompts
    gen_prompts = []
    for item in pending_items:
        messages = [{"role": "user", "content": item["clean_prompt"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gen_prompts.append(formatted)

    print(f"  Generating rollouts for {len(gen_prompts)} questions...")
    gen_outputs = llm.generate(gen_prompts, cot_gen_params)

    # -----------------------------------------------------------------------
    # Phase 2: Extract logprobs at sentence boundaries
    # -----------------------------------------------------------------------
    print("\nPhase 2: Extracting forced-answer entropy at sentence boundaries...")

    logprob_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,  # Get top-20 logprobs at the generation position
    )

    for item_idx, (item, output) in enumerate(zip(pending_items, gen_outputs)):
        example_id = item["example_id"]
        print(
            f"\n  [{item_idx + 1}/{len(pending_items)}] {example_id}"
        )

        all_boundary_data = []

        for rollout_idx, completion in enumerate(output.outputs):
            raw_cot = completion.text
            thinking_text = extract_thinking_text(raw_cot)
            if not thinking_text:
                # If no <think> tags, treat the whole output as thinking
                thinking_text = raw_cot.strip()

            sentences = split_cot_into_sentences(thinking_text)
            if len(sentences) < 2:
                continue

            # Cap boundaries per rollout
            boundary_indices = list(range(1, len(sentences) + 1))
            if len(boundary_indices) > args.max_boundaries:
                # Subsample evenly
                step = len(boundary_indices) / args.max_boundaries
                boundary_indices = [
                    boundary_indices[int(i * step)]
                    for i in range(args.max_boundaries)
                ]

            # Build forced-answer prompts for each boundary
            forced_prompts = []
            boundary_info = []

            for bi in boundary_indices:
                partial_cot = " ".join(sentences[:bi])
                # Construct: [question with chat template]<think>[partial CoT]</think>So, the answer is:
                messages = [{"role": "user", "content": item["clean_prompt"]}]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # Qwen3 chat template with enable_thinking adds <think>\n after
                # assistant prompt. We build the forced prefix manually.
                # The formatted text ends with something like
                # "<|im_start|>assistant\n"
                forced_prefix = (
                    f"{formatted}<think>\n{partial_cot}\n</think>\nSo, the answer is: "
                )
                forced_prompts.append(forced_prefix)
                boundary_info.append({
                    "rollout_idx": rollout_idx,
                    "boundary_idx": bi,
                    "n_sentences_so_far": bi,
                    "n_sentences_total": len(sentences),
                    "fraction_complete": bi / len(sentences),
                })

            if not forced_prompts:
                continue

            # Extract logprobs in batch
            logprob_outputs = llm.generate(forced_prompts, logprob_params)

            for fp_idx, lp_output in enumerate(logprob_outputs):
                if not lp_output.outputs:
                    continue

                lp_completion = lp_output.outputs[0]

                # Extract logprobs for answer tokens from the generated position
                # vLLM returns logprobs for the generated token and top-k alternatives
                logprob_dict = {}
                if lp_completion.logprobs and len(lp_completion.logprobs) > 0:
                    # logprobs[0] is the logprob info for the first (only) generated token
                    top_logprobs = lp_completion.logprobs[0]
                    # top_logprobs is a dict of token_id -> LogprobInfo
                    for token_id, logprob_info in top_logprobs.items():
                        logprob_dict[token_id] = logprob_info.logprob

                # Get logprobs for our answer tokens specifically
                answer_logprobs = {}
                for letter, tid in answer_token_ids.items():
                    if tid in logprob_dict:
                        answer_logprobs[letter] = logprob_dict[tid]
                    else:
                        # Token not in top-k, assign a very low logprob
                        answer_logprobs[letter] = -20.0

                # Softmax to get probabilities
                logprob_values = [answer_logprobs[l] for l in ["A", "B", "C", "D"]]
                max_lp = max(logprob_values)
                exp_values = [math.exp(lp - max_lp) for lp in logprob_values]
                total = sum(exp_values)
                probs = [e / total for e in exp_values]

                entropy = compute_entropy(probs)
                prob_dict = {
                    l: p for l, p in zip(["A", "B", "C", "D"], probs)
                }

                boundary_record = {
                    **boundary_info[fp_idx],
                    "entropy": entropy,
                    "probs": prob_dict,
                    "raw_logprobs": answer_logprobs,
                }
                all_boundary_data.append(boundary_record)

        # Store in checkpoint
        checkpoint_data[example_id] = {
            "n_rollouts_generated": len(output.outputs),
            "n_boundaries": len(all_boundary_data),
            "boundary_data": all_boundary_data,
        }

        # Save checkpoint after each item
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        if all_boundary_data:
            entropies = [b["entropy"] for b in all_boundary_data]
            print(
                f"    {len(all_boundary_data)} boundaries, "
                f"entropy range: [{min(entropies):.3f}, {max(entropies):.3f}], "
                f"mean: {sum(entropies)/len(entropies):.3f}"
            )
        else:
            print("    No valid boundaries extracted")

    # -----------------------------------------------------------------------
    # Phase 3: Merge results into eval JSON
    # -----------------------------------------------------------------------
    print("\n\nPhase 3: Merging results into eval JSON...")
    _merge_results(items, checkpoint_data, eval_path, args.target_items, args.seed)


def _merge_results(
    items: list[dict],
    checkpoint_data: dict[str, dict],
    eval_path: Path,
    target_items: int,
    seed: int,
):
    """Merge precomputed entropy data back into eval items.

    This creates the final eval dataset with entropy ground truth values.
    Each original EvalItem may expand into multiple items (one per boundary)
    or we downsample to hit the target count.
    """
    import random as rng_mod

    rng = rng_mod.Random(seed)

    # Collect all boundary datapoints across all items
    all_datapoints = []

    for item in items:
        eid = item["example_id"]
        ckpt = checkpoint_data.get(eid)
        if not ckpt or not ckpt.get("boundary_data"):
            continue

        for bd in ckpt["boundary_data"]:
            all_datapoints.append({
                "parent_item": item,
                "boundary": bd,
            })

    print(f"Total boundary datapoints: {len(all_datapoints)}")

    if not all_datapoints:
        print("WARNING: No datapoints to merge. Check that precompute ran successfully.")
        return

    # Downsample to target
    if len(all_datapoints) > target_items:
        rng.shuffle(all_datapoints)
        all_datapoints = all_datapoints[:target_items]
        print(f"Downsampled to {target_items} datapoints")

    # Build final eval items
    final_items = []
    for i, dp in enumerate(all_datapoints):
        parent = dp["parent_item"]
        bd = dp["boundary"]

        final_items.append({
            "eval_name": "forced_answer_entropy_riya",
            "example_id": f"forced_entropy_{i:04d}",
            "clean_prompt": parent["clean_prompt"],
            "test_prompt": parent["test_prompt"],
            "correct_answer": f"{bd['entropy']:.4f}",  # Entropy as string (regression target)
            "nudge_answer": None,
            "metadata": {
                "parent_example_id": parent["example_id"],
                "choices": parent.get("metadata", {}).get("choices", {}),
                "correct_letter": parent.get("metadata", {}).get("correct_letter", ""),
                "source": parent.get("metadata", {}).get("source", "gpqa_diamond"),
                "answer_tokens": ["A", "B", "C", "D"],
                "metric": "r_squared",
                "task_type": "regression",
                "entropy": bd["entropy"],
                "answer_probs": bd["probs"],
                "raw_logprobs": bd["raw_logprobs"],
                "rollout_idx": bd["rollout_idx"],
                "boundary_idx": bd["boundary_idx"],
                "n_sentences_so_far": bd["n_sentences_so_far"],
                "n_sentences_total": bd["n_sentences_total"],
                "fraction_complete": bd["fraction_complete"],
            },
        })

    # Write back
    with open(eval_path, "w") as f:
        json.dump(final_items, f, indent=2)

    # Print statistics
    entropies = [dp["boundary"]["entropy"] for dp in all_datapoints]
    print(f"\nFinal dataset: {len(final_items)} items")
    print(f"Entropy stats:")
    print(f"  min:  {min(entropies):.4f}")
    print(f"  max:  {max(entropies):.4f}")
    print(f"  mean: {sum(entropies)/len(entropies):.4f}")
    print(f"  std:  {_std(entropies):.4f}")

    # Distribution of fraction_complete
    fracs = [dp["boundary"]["fraction_complete"] for dp in all_datapoints]
    print(f"\nFraction-complete distribution:")
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        count = sum(1 for f in fracs if lo <= f < hi)
        print(f"  [{lo:.0%}, {hi:.0%}): {count}")

    print(f"\nSaved to {eval_path}")


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


if __name__ == "__main__":
    main()
