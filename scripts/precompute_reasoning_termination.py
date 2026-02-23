#!/usr/bin/env python3
"""
Precompute reasoning termination eval dataset using vLLM.

For each AMC/AIME question:
  1. Generate a full CoT (temperature=0.6, multiple rollouts)
  2. Create prefixes at various token positions
  3. For each prefix, resample 50 times:
     - Positive if </think> appears in 20-60 tokens after prefix in >=45/50 resamples
     - Negative if </think> appears beyond 200 tokens in >=45/50 resamples
  4. Balance 50/50 positive/negative, target 100 items

Usage (requires GPU):
    python scripts/precompute_reasoning_termination.py \
        --eval-dir data/evals \
        --model Qwen/Qwen3-8B \
        --target-items 100 \
        --n-resamples 50

Resumable: saves progress after each question. Skip completed questions on restart.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (lower, upper)


def format_chat_prefix(tokenizer, question: str, cot_prefix: str) -> str:
    """Format a question + CoT prefix for continuation generation.

    Produces: <system>...<user>question<assistant><think>cot_prefix
    The model continues generating from the end of cot_prefix.
    """
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # formatted ends with the generation prompt (e.g., "<think>\n")
    return formatted + cot_prefix


def find_think_end_position(text: str, prefix_end: int, max_tokens: int = 300) -> int | None:
    """Find token position of </think> in generated continuation.

    Returns the token offset (from prefix_end) where </think> starts,
    or None if not found within the search window.
    """
    # Search in the continuation text after prefix
    continuation = text[prefix_end:]
    idx = continuation.find("</think>")
    if idx >= 0:
        return idx  # Character offset in continuation
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Precompute reasoning termination eval with vLLM resampling"
    )
    parser.add_argument("--eval-dir", default="data/evals",
                        help="Directory containing eval JSON files")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model to use for generation")
    parser.add_argument("--target-items", type=int, default=100,
                        help="Target number of balanced eval items (50 pos + 50 neg)")
    parser.add_argument("--n-resamples", type=int, default=50,
                        help="Number of resamples per prefix")
    parser.add_argument("--n-rollouts", type=int, default=3,
                        help="Number of full CoT rollouts per question")
    parser.add_argument("--threshold", type=int, default=45,
                        help="Minimum resamples that must agree for a clear label (out of n-resamples)")
    parser.add_argument("--max-cot-tokens", type=int, default=4096,
                        help="Maximum tokens for full CoT generation")
    parser.add_argument("--max-resample-tokens", type=int, default=250,
                        help="Maximum new tokens for each resample continuation")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-file", default=None,
                        help="Intermediate cache file (default: {eval-dir}/reasoning_termination_cache.json)")
    args = parser.parse_args()

    random.seed(args.seed)

    eval_dir = Path(args.eval_dir)
    eval_file = eval_dir / "reasoning_termination.json"
    cache_file = Path(args.cache_file) if args.cache_file else eval_dir / "reasoning_termination_cache.json"

    if not eval_file.exists():
        print(f"ERROR: {eval_file} not found. Run generate_datasets.py first:")
        print(f"  python src/evals/generate_datasets.py --evals reasoning_termination")
        sys.exit(1)

    # Load existing eval items (placeholders from generate_datasets.py)
    with open(eval_file) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} placeholder items from {eval_file}")

    # Load cache of completed work
    completed = {}
    if cache_file.exists():
        with open(cache_file) as f:
            completed = json.load(f)
        print(f"Loaded cache: {len(completed)} completed questions")

    # ---- Import vLLM (lazy so script can show --help without GPU) ----
    print(f"\nLoading vLLM with model {args.model}...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_cot_tokens + 512,  # headroom for prompt
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    # Sampling params for full CoT generation (diverse, temperature=0.6)
    cot_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_cot_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    # Sampling params for resampling continuations (same temperature for diversity)
    resample_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.max_resample_tokens,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )

    # ---- Phase 1: Generate full CoTs for each question ----
    print("\n=== Phase 1: Generate full CoTs ===")

    questions_needing_cots = []
    for item in items:
        qid = item["example_id"]
        if qid not in completed or "rollouts" not in completed[qid]:
            questions_needing_cots.append(item)

    if questions_needing_cots:
        print(f"Generating {args.n_rollouts} rollouts for {len(questions_needing_cots)} questions...")

        # Prepare prompts: n_rollouts per question
        cot_prompts = []
        cot_index_map = []  # (item_idx_in_questions_needing, rollout_idx)

        for qi, item in enumerate(questions_needing_cots):
            messages = [{"role": "user", "content": item["clean_prompt"]}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for ri in range(args.n_rollouts):
                cot_prompts.append(formatted)
                cot_index_map.append((qi, ri))

        # Batch generate all CoTs
        print(f"  Generating {len(cot_prompts)} CoTs in batch...")
        cot_outputs = llm.generate(cot_prompts, cot_params)

        # Organize by question
        for (qi, ri), output in zip(cot_index_map, cot_outputs):
            item = questions_needing_cots[qi]
            qid = item["example_id"]
            if qid not in completed:
                completed[qid] = {"question": item["clean_prompt"], "rollouts": []}

            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)
            has_think_end = "</think>" in text

            completed[qid]["rollouts"].append({
                "text": text,
                "n_tokens": n_tokens,
                "has_think_end": has_think_end,
            })

        # Save cache
        with open(cache_file, "w") as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(completed)} questions to cache")
    else:
        print("  All questions already have rollouts in cache")

    # ---- Phase 2: Create prefix candidates ----
    print("\n=== Phase 2: Create prefix candidates ===")

    # For each rollout, create prefix candidates at different positions
    # Positive candidate: cut so 20-60 tokens remain (model is near </think>)
    # Negative candidate: cut so 200+ tokens remain (model is far from </think>)
    positive_candidates = []
    negative_candidates = []

    for qid, data in completed.items():
        question = data["question"]
        for ri, rollout in enumerate(data.get("rollouts", [])):
            if not rollout["has_think_end"]:
                continue  # Skip rollouts that didn't finish thinking

            text = rollout["text"]
            n_tokens = rollout["n_tokens"]

            # Tokenize the rollout to get precise token boundaries
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            actual_n_tokens = len(token_ids)

            if actual_n_tokens < 80:
                continue  # Too short for either category

            # Positive prefix: leave 20-60 tokens remaining
            # Try multiple cut points
            for remaining in [25, 35, 45, 55]:
                if remaining >= actual_n_tokens - 10:
                    continue
                cut_pos = actual_n_tokens - remaining
                if cut_pos < 20:
                    continue

                prefix_ids = token_ids[:cut_pos]
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

                positive_candidates.append({
                    "qid": qid,
                    "rollout_idx": ri,
                    "question": question,
                    "prefix_text": prefix_text,
                    "prefix_n_tokens": cut_pos,
                    "remaining_tokens": remaining,
                    "total_tokens": actual_n_tokens,
                    "candidate_label": "will_terminate",
                })

            # Negative prefix: leave 200+ tokens remaining
            for remaining in [250, 350, 450]:
                if remaining >= actual_n_tokens - 10:
                    continue
                cut_pos = actual_n_tokens - remaining
                if cut_pos < 20:
                    continue

                prefix_ids = token_ids[:cut_pos]
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

                negative_candidates.append({
                    "qid": qid,
                    "rollout_idx": ri,
                    "question": question,
                    "prefix_text": prefix_text,
                    "prefix_n_tokens": cut_pos,
                    "remaining_tokens": remaining,
                    "total_tokens": actual_n_tokens,
                    "candidate_label": "will_continue",
                })

    print(f"  Positive candidates (near </think>): {len(positive_candidates)}")
    print(f"  Negative candidates (far from </think>): {len(negative_candidates)}")

    if not positive_candidates or not negative_candidates:
        print("ERROR: Not enough candidates. Need longer CoTs or more rollouts.")
        print("  Try increasing --n-rollouts or --max-cot-tokens.")
        sys.exit(1)

    # Shuffle and limit to what we need
    random.shuffle(positive_candidates)
    random.shuffle(negative_candidates)

    target_per_class = args.target_items // 2
    # Request more than target because some will be discarded after resampling
    sample_factor = 3
    pos_to_resample = positive_candidates[:target_per_class * sample_factor]
    neg_to_resample = negative_candidates[:target_per_class * sample_factor]

    all_to_resample = pos_to_resample + neg_to_resample
    print(f"  Will resample {len(all_to_resample)} candidates "
          f"({len(pos_to_resample)} pos + {len(neg_to_resample)} neg)")

    # ---- Phase 3: Resample each prefix ----
    print(f"\n=== Phase 3: Resample ({args.n_resamples} per prefix) ===")

    # Check cache for already-resampled prefixes
    resample_cache_key = "resampled_prefixes"
    if resample_cache_key not in completed:
        completed[resample_cache_key] = {}

    resampled = completed[resample_cache_key]

    candidates_to_process = []
    for cand in all_to_resample:
        cache_key = f"{cand['qid']}_{cand['rollout_idx']}_{cand['prefix_n_tokens']}"
        if cache_key not in resampled:
            candidates_to_process.append((cache_key, cand))

    if candidates_to_process:
        print(f"  {len(candidates_to_process)} prefixes need resampling "
              f"({len(all_to_resample) - len(candidates_to_process)} cached)")

        # Build all resample prompts
        resample_prompts = []
        resample_keys = []

        for cache_key, cand in candidates_to_process:
            # Build the prompt: chat template + CoT prefix
            prompt = format_chat_prefix(tokenizer, cand["question"], cand["prefix_text"])
            for _ in range(args.n_resamples):
                resample_prompts.append(prompt)
                resample_keys.append(cache_key)

        print(f"  Generating {len(resample_prompts)} resample continuations in batch...")
        resample_outputs = llm.generate(resample_prompts, resample_params)

        # Aggregate results by cache_key
        key_results: dict[str, list[dict]] = {}
        for cache_key, output in zip(resample_keys, resample_outputs):
            if cache_key not in key_results:
                key_results[cache_key] = []

            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)
            has_think_end = "</think>" in text

            # Find where </think> appears (in token count from continuation start)
            if has_think_end:
                # Count tokens up to </think>
                think_idx = text.find("</think>")
                prefix_to_think = text[:think_idx]
                tokens_to_think = len(tokenizer.encode(prefix_to_think, add_special_tokens=False))
            else:
                tokens_to_think = n_tokens  # Didn't terminate within max_tokens

            key_results[cache_key].append({
                "has_think_end": has_think_end,
                "tokens_to_think": tokens_to_think,
                "total_continuation_tokens": n_tokens,
            })

        # Compute labels for each prefix
        for cache_key, results in key_results.items():
            n_total = len(results)
            n_terminate_soon = sum(
                1 for r in results
                if r["has_think_end"] and 20 <= r["tokens_to_think"] <= 60
            )
            n_terminate_any_early = sum(
                1 for r in results
                if r["has_think_end"] and r["tokens_to_think"] <= 60
            )
            n_continue_long = sum(
                1 for r in results
                if not r["has_think_end"] or r["tokens_to_think"] > 200
            )

            # Wilson CIs on the resample proportions
            ci_terminate = wilson_ci(n_terminate_any_early, n_total)
            ci_continue = wilson_ci(n_continue_long, n_total)

            resampled[cache_key] = {
                "n_resamples": n_total,
                "n_terminate_20_60": n_terminate_soon,
                "n_terminate_any_early": n_terminate_any_early,
                "n_continue_long": n_continue_long,
                "ci_terminate_lower": ci_terminate[0],
                "ci_terminate_upper": ci_terminate[1],
                "ci_continue_lower": ci_continue[0],
                "ci_continue_upper": ci_continue[1],
                "avg_tokens_to_think": (
                    sum(r["tokens_to_think"] for r in results) / n_total
                    if n_total > 0 else 0
                ),
            }

        # Save cache
        completed[resample_cache_key] = resampled
        with open(cache_file, "w") as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)
        print(f"  Saved resample results to cache")
    else:
        print(f"  All prefixes already resampled in cache")

    # ---- Phase 4: Select balanced items based on resample labels ----
    print("\n=== Phase 4: Select balanced items ===")

    threshold = args.threshold
    confirmed_positive = []  # will_terminate
    confirmed_negative = []  # will_continue

    for cand in all_to_resample:
        cache_key = f"{cand['qid']}_{cand['rollout_idx']}_{cand['prefix_n_tokens']}"
        if cache_key not in resampled:
            continue

        r = resampled[cache_key]
        n_total = r["n_resamples"]

        if cand["candidate_label"] == "will_terminate":
            # Positive: need >=threshold resamples terminating within 20-60 tokens
            if r["n_terminate_any_early"] >= threshold:
                confirmed_positive.append({**cand, "resample_stats": r})
        else:
            # Negative: need >=threshold resamples continuing beyond 200 tokens
            if r["n_continue_long"] >= threshold:
                confirmed_negative.append({**cand, "resample_stats": r})

    print(f"  Confirmed positive (will_terminate): {len(confirmed_positive)}")
    print(f"  Confirmed negative (will_continue):  {len(confirmed_negative)}")

    # Balance
    n_each = min(target_per_class, len(confirmed_positive), len(confirmed_negative))
    if n_each == 0:
        print("ERROR: No confirmed items. Try lowering --threshold or increasing --n-rollouts.")
        print(f"  Threshold: {threshold}/{args.n_resamples}")
        # Print some stats to help debug
        if all_to_resample:
            cache_key = f"{all_to_resample[0]['qid']}_{all_to_resample[0]['rollout_idx']}_{all_to_resample[0]['prefix_n_tokens']}"
            if cache_key in resampled:
                print(f"  Sample resample stats: {resampled[cache_key]}")
        sys.exit(1)

    random.shuffle(confirmed_positive)
    random.shuffle(confirmed_negative)
    selected = confirmed_positive[:n_each] + confirmed_negative[:n_each]
    random.shuffle(selected)

    print(f"  Selected {len(selected)} balanced items ({n_each} per class)")

    # ---- Phase 5: Build final eval items ----
    print("\n=== Phase 5: Build final eval items ===")

    final_items = []
    for i, cand in enumerate(selected):
        r = cand["resample_stats"]
        label = cand["candidate_label"]

        final_items.append({
            "eval_name": "reasoning_termination",
            "example_id": f"reason_term_{i:04d}",
            "clean_prompt": cand["question"],
            "test_prompt": cand["question"],
            "correct_answer": label,
            "nudge_answer": None,
            "metadata": {
                "source": cand["qid"].split("_")[1] if "_" in cand["qid"] else "unknown",
                "cot_prefix": cand["prefix_text"],
                "prefix_n_tokens": cand["prefix_n_tokens"],
                "remaining_tokens_in_original": cand["remaining_tokens"],
                "total_tokens_in_original": cand["total_tokens"],
                "n_resamples": r["n_resamples"],
                "n_terminate_20_60": r["n_terminate_20_60"],
                "n_terminate_any_early": r["n_terminate_any_early"],
                "n_continue_long": r["n_continue_long"],
                "ci_terminate_lower": round(r["ci_terminate_lower"], 4),
                "ci_terminate_upper": round(r["ci_terminate_upper"], 4),
                "ci_continue_lower": round(r["ci_continue_lower"], 4),
                "ci_continue_upper": round(r["ci_continue_upper"], 4),
                "avg_tokens_to_think": round(r["avg_tokens_to_think"], 1),
            },
        })

    # Write final eval file
    with open(eval_file, "w") as f:
        json.dump(final_items, f, indent=2, ensure_ascii=False)

    n_pos = sum(1 for item in final_items if item["correct_answer"] == "will_terminate")
    n_neg = sum(1 for item in final_items if item["correct_answer"] == "will_continue")
    print(f"\nWrote {len(final_items)} items to {eval_file}")
    print(f"  will_terminate: {n_pos}")
    print(f"  will_continue:  {n_neg}")
    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/upload_eval_datasets.py")
    print(f"  2. Or upload just this eval manually")


if __name__ == "__main__":
    main()
