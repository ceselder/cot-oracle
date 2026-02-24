"""
Precompute per-sentence answer trajectory using vLLM.

For each corpus entry, at every sentence boundary in the CoT, insert
"Therefore, the answer is" (staying INSIDE <think> tags) and use vLLM
to generate what the model thinks the answer is at that point.

This produces a dense answer trajectory showing how the model's answer
evolves sentence by sentence. Training task: "What does the model
currently think the answer is?" -> oracle predicts the answer from
activations up to that sentence boundary.

Key difference from partial_answer:
- Sentence-level granularity (not random percentage bins)
- Stays inside <think> tags (not </think> -> answer mode)
- Captures mid-reasoning belief, not forced-termination behavior

Usage:
    python3 scripts/precompute_answer_trajectory.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --output data/precomputed/answer_trajectory.jsonl \
        --num-entries 5000

Requires: vLLM, GPU with Qwen3-8B loaded.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _extract_answer_text(entry: dict) -> str | None:
    """Extract a clean answer string from a corpus entry."""
    direct = entry.get("direct_response", "").strip()
    if direct:
        direct = re.sub(r"<think>.*?</think>", "", direct, flags=re.DOTALL).strip()
        boxed = re.search(r"\\boxed\{([^}]+)\}", direct)
        if boxed:
            return boxed.group(1).strip()
        if len(direct) < 200:
            return direct
        return direct[:200]
    answer = entry.get("answer") or entry.get("correct_answer") or ""
    answer = str(answer).strip()
    return str(answer)[:200] if answer else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="data/precomputed/answer_trajectory.jsonl")
    parser.add_argument("--num-entries", type=int, default=5000,
                        help="Number of corpus entries to process (all sentence boundaries per entry)")
    parser.add_argument("--min-sentences", type=int, default=3,
                        help="Skip entries with fewer sentences")
    parser.add_argument("--skip-first-n", type=int, default=1,
                        help="Skip first N sentence boundaries (too little context)")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-positions-per-layer", type=int, default=20)
    parser.add_argument("--n-prompt-positions", type=int, default=5)
    parser.add_argument("--max-answer-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="vLLM batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from cot_utils import (
        get_cot_stride_positions,
        layer_percent_to_layer,
        split_cot_into_sentences,
        find_sentence_boundary_positions,
    )

    random.seed(args.seed)

    LAYERS = [
        layer_percent_to_layer(args.model, 25),
        layer_percent_to_layer(args.model, 50),
        layer_percent_to_layer(args.model, 75),
    ]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded: {tokenizer.vocab_size} vocab")

    # Load corpus
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cot = entry.get("cot_response", "").strip()
                if cot and _extract_answer_text(entry):
                    corpus.append(entry)
    print(f"Loaded {len(corpus)} corpus entries with CoT + answer")

    # Sample entries
    if len(corpus) > args.num_entries:
        random.shuffle(corpus)
        corpus = corpus[:args.num_entries]
    print(f"Processing {len(corpus)} entries")

    # Helper for prompt positions
    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Process each entry: find sentence boundaries, create per-sentence examples
    all_examples = []
    entries_used = 0
    entries_skipped_short = 0
    entries_skipped_boundaries = 0

    for entry_idx, entry in enumerate(corpus):
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        sentences = split_cot_into_sentences(cot_text)
        if len(sentences) < args.min_sentences:
            entries_skipped_short += 1
            continue

        # Build formatted prompt
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        # Tokenize full CoT
        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Find sentence boundary token positions
        boundary_positions = find_sentence_boundary_positions(
            tokenizer, full_text, sentences
        )

        if len(boundary_positions) < args.min_sentences:
            entries_skipped_boundaries += 1
            continue

        final_answer = _extract_answer_text(entry)
        entries_used += 1

        # For each sentence boundary (skipping first N)
        for sent_idx in range(args.skip_first_n, len(boundary_positions)):
            boundary_pos = boundary_positions[sent_idx]

            # Decode tokens up to this boundary for the vLLM prompt
            trunc_ids = full_ids[:boundary_pos + 1]
            trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)

            # Stay inside <think> tags - just append the answer prefill
            vllm_prompt = trunc_text + "\nTherefore, the answer is"

            # Skip if prompt would exceed model context
            vllm_prompt_len = len(tokenizer(vllm_prompt, add_special_tokens=False)["input_ids"])
            if vllm_prompt_len + args.max_answer_tokens > 32768:
                continue

            # Get stride positions for the truncated CoT
            # cot_end is the boundary position + 1 (exclusive)
            positions = get_cot_stride_positions(
                prompt_len, boundary_pos + 1,
                stride=args.stride,
                max_positions=args.max_positions_per_layer,
            )
            if len(positions) < 2:
                continue

            prompt_positions = _get_prompt_positions(prompt_len, args.n_prompt_positions)
            combined = prompt_positions + positions
            context_positions = combined * 3
            num_positions = len(context_positions)

            # context_input_ids: tokens up to the last stride position
            # (these are what the model sees for activation extraction)
            max_pos = max(positions)
            context_slice = full_ids[:max_pos + 1]

            pct = int(100 * (sent_idx + 1) / len(sentences))

            all_examples.append({
                "sent_idx": sent_idx,
                "total_sentences": len(sentences),
                "pct": pct,
                "num_positions": num_positions,
                "context_input_ids": context_slice,
                "context_positions": context_positions,
                "layers": LAYERS,
                "vllm_prompt": vllm_prompt,
                "final_answer": final_answer,
            })

        if (entry_idx + 1) % 1000 == 0:
            print(f"  Processed {entry_idx + 1}/{len(corpus)} entries, "
                  f"{len(all_examples)} examples so far")

    print(f"\nPrepared {len(all_examples)} sentence-boundary examples "
          f"from {entries_used} entries")
    print(f"  Skipped: {entries_skipped_short} too short, "
          f"{entries_skipped_boundaries} no boundaries found")
    print(f"  Average {len(all_examples) / max(entries_used, 1):.1f} "
          f"sentence boundaries per entry")

    # Load vLLM
    print(f"\nLoading vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_answer_tokens,
        temperature=0.0,  # greedy
        stop=["\n", "<|im_end|>"],
    )

    # Batch generate
    print(f"Generating {len(all_examples)} answers with vLLM...")
    all_prompts = [ex["vllm_prompt"] for ex in all_examples]

    t0 = time.time()
    all_outputs = []
    for batch_start in range(0, len(all_prompts), args.batch_size):
        batch_prompts = all_prompts[batch_start:batch_start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)
        elapsed = time.time() - t0
        done = batch_start + len(batch_prompts)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(all_prompts) - done) / rate if rate > 0 else 0
        print(f"  Generated {done}/{len(all_prompts)} "
              f"({elapsed:.1f}s, {rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"vLLM generation complete: {len(all_outputs)} outputs in {elapsed:.1f}s")

    # Build final datapoints
    datapoints = []
    layers_str = ", ".join(str(l) for l in LAYERS)
    empty_count = 0

    for ex, output in zip(all_examples, all_outputs):
        generated_answer = output.outputs[0].text.strip()
        if not generated_answer:
            empty_count += 1
            continue

        # Clean up trailing punctuation
        generated_answer = generated_answer.rstrip(".")

        prompt = (
            f"Activations from {ex['num_positions']} positions across layers "
            f"{layers_str}, representing sentence {ex['sent_idx'] + 1}/"
            f"{ex['total_sentences']} ({ex['pct']}%) of the model's chain of "
            f"thought. What does the model currently think the answer is?"
        )

        datapoints.append({
            "datapoint_type": "cot_answer_trajectory",
            "prompt": prompt,
            "target_response": generated_answer,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": ex["num_positions"],
            "context_input_ids": ex["context_input_ids"],
            "context_positions": ex["context_positions"],
            "sent_idx": ex["sent_idx"],
            "total_sentences": ex["total_sentences"],
            "pct": ex["pct"],
            "final_answer": ex["final_answer"],
        })

    if empty_count:
        print(f"Skipped {empty_count} examples with empty vLLM output")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for dp in datapoints:
            f.write(json.dumps(dp) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(datapoints)} examples to {out_path} ({size_mb:.1f} MB)")

    # Stats: answer matches final answer by CoT progress
    print(f"\nAnswer trajectory stats:")
    match_by_pct = {}
    for dp in datapoints:
        pct_bin = (dp["pct"] // 20) * 20
        if pct_bin not in match_by_pct:
            match_by_pct[pct_bin] = {"match": 0, "total": 0}
        match_by_pct[pct_bin]["total"] += 1
        gen = dp["target_response"].lower().strip()
        final = dp["final_answer"].lower().strip()
        if gen == final or gen in final or final in gen:
            match_by_pct[pct_bin]["match"] += 1

    print(f"  Answer matches final answer by CoT progress:")
    for pct_bin in sorted(match_by_pct.keys()):
        stats = match_by_pct[pct_bin]
        rate = stats["match"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {pct_bin}-{pct_bin + 20}%: {rate:.1%} "
              f"({stats['match']}/{stats['total']})")

    # Show a sample trajectory
    if datapoints:
        # Group by matching context prefix to find entries from the same CoT
        first_key = tuple(datapoints[0]["context_input_ids"][:15])
        sample = [dp for dp in datapoints
                  if tuple(dp["context_input_ids"][:15]) == first_key]
        if sample:
            print(f"\n  Sample trajectory ({len(sample)} sentences):")
            for dp in sample[:12]:
                match = "==" if dp["target_response"].lower().strip() == dp["final_answer"].lower().strip() else "!="
                print(f"    Sent {dp['sent_idx'] + 1}/{dp['total_sentences']}: "
                      f"\"{dp['target_response'][:50]}\" "
                      f"{match} final \"{dp['final_answer'][:30]}\"")


if __name__ == "__main__":
    main()
