"""
Clean answer_trajectory target_response using Qwen 3.5 27B via vLLM.

Extracts concise, standardized answers from the messy vLLM continuations.
For MCQ: "B. 52", for math: "36", for open-ended: short phrase.

Usage:
  python scripts/clean_answer_trajectory.py --n 1000  # test on 1000 rows
  python scripts/clean_answer_trajectory.py            # all rows
"""

import json
import argparse
from pathlib import Path


SYSTEM_PROMPT = """You are a data cleaning assistant. You will be given a messy model response that was generated when a model was asked "What do you currently think the answer is?" at various points during its chain of thought.

Your job: Extract ONLY the concise answer the model currently believes in. Rules:

1. For multiple choice (A/B/C/D/E options): Output in format "X. answer" (e.g., "B. 52" or "C. 36")
2. For numerical answers: Just the number (e.g., "42" or "3/7")
3. For boxed answers: Extract the content (e.g., "\\boxed{52}" → "52")
4. If the model is genuinely uncertain and hasn't committed to an answer yet, output "uncertain"
5. If there are multiple candidate answers mentioned, pick the one the model seems to favor MOST
6. Keep it SHORT — max 10 words. No reasoning, no hedging, just the answer.

Examples:
Input: "one of the options A to E. So, I need to figure out the relationship"
Output: uncertain

Input: "C)36. Wait, but let me check again."
Output: C. 36

Input: "the reverse of 25. Let's check: 25 reversed is 52. And 52 is option B. So maybe the answer is B)52?"
Output: B. 52

Input: "\\boxed{C}"
Output: C

Input: "25 + 9 = 34? But 34 isn't one of the options."
Output: uncertain

Input: "not 34. So maybe that's not the right approach"
Output: uncertain

Input: "I think the answer is 144"
Output: 144"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/precomputed/answer_trajectory.jsonl")
    parser.add_argument("--output", default=None, help="Output path (default: input with _cleaned suffix)")
    parser.add_argument("--n", type=int, default=0, help="Process only first N rows (0=all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_cleaned{p.suffix}")

    # Load data
    print(f"Loading {args.input}...")
    rows = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if args.n > 0 and len(rows) >= args.n:
                    break
    print(f"Loaded {len(rows)} rows")

    # Init vLLM
    from vllm import LLM, SamplingParams

    print(f"Loading {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=1024,
        gpu_memory_utilization=0.9,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=30,
        stop=["\n"],
    )

    # Process in batches
    print(f"Processing {len(rows)} rows in batches of {args.batch_size}...")

    all_cleaned = []
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start:batch_start + args.batch_size]

        prompts = []
        for row in batch:
            target = row["target_response"]
            # Truncate very long responses
            if len(target) > 500:
                target = target[:500] + "..."

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": target},
            ]
            prompts.append(messages)

        outputs = llm.chat(prompts, sampling_params)

        for row, output in zip(batch, outputs):
            cleaned = output.outputs[0].text.strip()
            row["response_filtered"] = cleaned
            all_cleaned.append(row)

        done = batch_start + len(batch)
        print(f"  {done}/{len(rows)} done")

        # Show samples from first batch
        if batch_start == 0:
            print("\n=== SAMPLES ===")
            for i in range(min(20, len(batch))):
                r = all_cleaned[i]
                print(f"  sent {r['sent_idx']}/{r['total_sentences']} ({r['pct']}%)")
                print(f"    raw:      {r['target_response'][:100]}")
                print(f"    cleaned:  {r['response_filtered']}")
                print(f"    final_gt: {r['final_answer']}")
                print()

    # Save
    print(f"Saving to {args.output}...")
    with open(args.output, "w") as f:
        for row in all_cleaned:
            f.write(json.dumps(row) + "\n")

    print(f"Done! {len(all_cleaned)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
