"""
Precompute partial answer targets using vLLM.

For each corpus entry, truncate the CoT at various percentages (10-90%),
close </think>, prefill "Therefore, the answer is", and use vLLM to
generate what the model would actually answer at that point.

This gives ground truth targets: early truncation → wrong/uncertain answer,
late truncation → correct answer. The oracle learns to read whether the
answer is "decided" from activations.

Usage:
    python3 scripts/precompute_partial_answer_vllm.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --output data/precomputed/partial_answer.jsonl \
        --num-examples 20000

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
    """Extract a clean answer string from a corpus entry (for reference only)."""
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
    parser.add_argument("--output", default="data/precomputed/partial_answer.jsonl")
    parser.add_argument("--num-examples", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--n-prompt-positions", type=int, default=5)
    parser.add_argument("--max-answer-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="vLLM batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from cot_utils import get_cot_stride_positions, layer_percent_to_layer

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

    # Pre-tokenize
    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    tokenized = []
    for entry in corpus:
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        cot_len = len(full_ids) - prompt_len

        if cot_len < 20:
            continue

        tokenized.append({
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "cot_len": cot_len,
            "formatted": formatted,
            "cot_text": cot_text,
            "final_answer": _extract_answer_text(entry),
        })

    print(f"{len(tokenized)} entries after tokenization (min 20 CoT tokens)")

    # Generate truncation examples
    BINS = [(10, 25), (25, 40), (40, 60), (60, 75), (75, 90)]
    examples = []
    bin_idx = 0

    for _ in range(args.num_examples * 3):
        if len(examples) >= args.num_examples:
            break

        t = random.choice(tokenized)
        lo, hi = BINS[bin_idx % len(BINS)]
        bin_idx += 1
        pct = random.randint(lo, hi)
        trunc_tokens = int(t["cot_len"] * pct / 100)

        if trunc_tokens < 10:
            continue

        # Skip if truncated prompt would exceed max_model_len (leave room for answer)
        trunc_prompt_len = t["prompt_len"] + trunc_tokens + 20  # +20 for </think>\n\nTherefore...
        if trunc_prompt_len > 3800:  # leave 296 tokens for generation within 4096
            continue

        trunc_pos = t["prompt_len"] + trunc_tokens

        # Get stride positions up to truncation point
        positions = get_cot_stride_positions(
            t["prompt_len"], trunc_pos,
            stride=args.stride,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(t["prompt_len"], args.n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * 3
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = t["full_ids"][:max_pos + 1]

        # Build the truncated text for vLLM generation
        # Decode truncated token IDs back to text
        trunc_ids = t["full_ids"][:trunc_pos]
        trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)

        # Close thinking and prefill the answer
        # Format: {truncated text}</think>\n\nTherefore, the answer is
        vllm_prompt = trunc_text + "</think>\n\nTherefore, the answer is"

        examples.append({
            "pct": pct,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
            "layers": LAYERS,
            "vllm_prompt": vllm_prompt,
            "final_answer": t["final_answer"],
        })

    examples = examples[:args.num_examples]
    print(f"Prepared {len(examples)} truncation examples for vLLM generation")

    # Load vLLM
    print(f"\nLoading vLLM with {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_answer_tokens,
        temperature=0.0,  # greedy
        stop=["<|im_end|>", "\n\n"],
    )

    # Batch generate
    print(f"Generating {len(examples)} answers with vLLM...")
    all_prompts = [ex["vllm_prompt"] for ex in examples]

    t0 = time.time()
    # Process in batches for progress reporting
    all_outputs = []
    for batch_start in range(0, len(all_prompts), args.batch_size):
        batch_prompts = all_prompts[batch_start:batch_start + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)
        elapsed = time.time() - t0
        done = batch_start + len(batch_prompts)
        print(f"  Generated {done}/{len(all_prompts)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"vLLM generation complete: {len(all_outputs)} outputs in {elapsed:.1f}s")

    # Build final datapoints
    datapoints = []
    layers_str = ", ".join(str(l) for l in LAYERS)

    for ex, output in zip(examples, all_outputs):
        generated_answer = output.outputs[0].text.strip()
        if not generated_answer:
            generated_answer = ex["final_answer"]  # fallback

        # Prepend "Therefore, the answer is" to make it a complete response
        target = f"Therefore, the answer is {generated_answer}"

        prompt = (
            f"Activations from {ex['num_positions']} positions across layers {layers_str}, "
            f"representing {ex['pct']}% of the model's chain of thought. "
            f"Based on the reasoning so far, what answer is the model converging toward?"
        )

        datapoints.append({
            "datapoint_type": "cot_partial_answer",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": ex["num_positions"],
            "context_input_ids": ex["context_input_ids"],
            "context_positions": ex["context_positions"],
        })

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for dp in datapoints:
            f.write(json.dumps(dp) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(datapoints)} examples to {out_path} ({size_mb:.1f} MB)")

    # Stats
    print(f"\nSample outputs by truncation %:")
    for lo, hi in BINS:
        bin_examples = [
            (ex, dp) for ex, dp in zip(examples, datapoints)
            if lo <= ex["pct"] <= hi
        ]
        if bin_examples:
            ex, dp = bin_examples[0]
            print(f"  {lo}-{hi}%: target='{dp['target_response'][:80]}...'")
            print(f"          final_answer='{ex['final_answer'][:80]}'")


if __name__ == "__main__":
    main()
