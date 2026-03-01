#!/usr/bin/env python3
"""
Generate open-ended QA dataset grounded in FineWeb excerpts.

3-step pipeline using Gemini 2.5 Flash Lite via OpenRouter:
  Step 1: Sample FineWeb excerpts with random token lengths [50, 2000]
  Step 2: Round 1 — Generate open-ended questions about each excerpt
  Step 3: Round 2 — Generate answers to the questions

Intermediate JSONL saves after each step for resumability.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_fineweb_openqa.py
    python scripts/generate_fineweb_openqa.py --n 10 --max-budget 1.0  # test run
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ── Model config ──

MODEL_ID = "google/gemini-2.5-flash-lite"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 30

# ── Stats ──

completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def estimate_cost():
    return total_input_tokens * INPUT_COST_PER_M / 1e6 + total_output_tokens * OUTPUT_COST_PER_M / 1e6


# ── API call ──

async def call_openrouter(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_budget: float,
    max_tokens: int = 800,
) -> tuple[str, int, int]:
    global completed, failed, budget_exceeded, total_input_tokens, total_output_tokens

    if budget_exceeded:
        return "", 0, 0

    async with semaphore:
        body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(4):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=90)

                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                if resp.status_code == 402:
                    budget_exceeded = True
                    print("\n*** Budget exceeded (402). Stopping. ***")
                    return "", 0, 0
                if resp.status_code != 200:
                    if attempt == 3:
                        failed += 1
                        if failed <= 10:
                            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                        return "", 0, 0
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                completed += 1
                if completed % 100 == 0:
                    print(f"  {completed} calls done, {failed} failed, ~${estimate_cost():.3f} spent")

                current_cost = estimate_cost()
                if current_cost > max_budget * 0.95:
                    budget_exceeded = True
                    print(f"\n*** Approaching budget limit (${current_cost:.2f}/${max_budget:.2f}). Stopping. ***")

                return content, in_tok, out_tok

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as e:
                if attempt == 3:
                    failed += 1
                    if failed <= 10:
                        print(f"  Error: {e}")
                    return "", 0, 0
                await asyncio.sleep(2 ** attempt)

    return "", 0, 0


# ── Prompts ──

QUESTION_GENERATION_PROMPT = """\
You are given an excerpt from a web document. Generate a single open-ended question about \
the content of the excerpt that requires understanding and reasoning to answer well. The \
question should be answerable from the excerpt alone. Output only the question, nothing else."""

R2_SYSTEM = """\
Answer the question based on the given excerpt. Be specific and thorough. 2-5 sentences."""


# ── Step 1: Sample FineWeb excerpts ──

def sample_excerpts(n: int, seed: int, tokenizer) -> list[dict]:
    """Stream FineWeb and sample excerpts with random token lengths [50, 2000]."""
    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    excerpts = []
    for example in tqdm(ds, desc="Sampling FineWeb excerpts", total=n):
        if len(excerpts) >= n:
            break

        text = example["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Need at least 50 tokens
        if len(token_ids) < 50:
            continue

        target_len = rng.randint(50, min(2000, len(token_ids)))
        excerpt_ids = token_ids[:target_len]
        excerpt_text = tokenizer.decode(excerpt_ids, skip_special_tokens=True)

        excerpts.append({
            "excerpt": excerpt_text,
            "excerpt_n_tokens": target_len,
            "fineweb_id": example["id"],
            "fineweb_url": example.get("url", ""),
        })

    print(f"Sampled {len(excerpts)} excerpts")
    token_lengths = [e["excerpt_n_tokens"] for e in excerpts]
    print(f"  Token lengths: mean={sum(token_lengths)/len(token_lengths):.0f}, min={min(token_lengths)}, max={max(token_lengths)}")
    return excerpts


# ── Main pipeline ──

async def run_phase(
    tasks: list[tuple[str, str]],
    api_key: str,
    max_budget: float,
    max_tokens: int = 800,
) -> list[str]:
    """Run a batch of API calls. Returns list of response strings (empty on failure)."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, sys_p, user_p, api_key, semaphore, max_budget, max_tokens)
            for sys_p, user_p in tasks
        ]
        results = await asyncio.gather(*coros)

    return [r[0] for r in results]


def main():
    global budget_exceeded, completed, failed, total_input_tokens, total_output_tokens

    parser = argparse.ArgumentParser(description="Generate FineWeb open-ended QA dataset")
    parser.add_argument("--output-dir", default="data/fineweb_openqa")
    parser.add_argument("--n", type=int, default=10_000, help="Number of excerpts to generate")
    parser.add_argument("--max-budget", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", choices=["excerpts", "r1"], default=None, help="Resume from intermediate JSONL")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excerpts_file = output_dir / "excerpts.jsonl"
    r1_file = output_dir / "fineweb_openqa_r1.jsonl"
    final_file = output_dir / "fineweb_openqa.jsonl"

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # Step 1: Sample FineWeb excerpts
    # ══════════════════════════════════════════════════════════════

    if args.resume_from in ("excerpts", "r1"):
        print(f"Loading excerpts from {excerpts_file}")
        excerpts = []
        with open(excerpts_file) as f:
            for line in f:
                if line.strip():
                    excerpts.append(json.loads(line))
        print(f"  Loaded {len(excerpts)} excerpts")
    else:
        print(f"{'='*60}\nStep 1: Sample {args.n} FineWeb excerpts")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        excerpts = sample_excerpts(args.n, args.seed, tokenizer)

        with open(excerpts_file, "w") as f:
            for row in excerpts:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(excerpts)} excerpts → {excerpts_file}")

    # ══════════════════════════════════════════════════════════════
    # Step 2: Round 1 — Generate questions
    # ══════════════════════════════════════════════════════════════

    if args.resume_from == "r1":
        print(f"\nResuming from {r1_file}")
        r1_rows = []
        with open(r1_file) as f:
            for line in f:
                if line.strip():
                    r1_rows.append(json.loads(line))
        print(f"  Loaded {len(r1_rows)} rows from {r1_file}")
    else:
        print(f"\n{'='*60}\nRound 1: Generate questions ({len(excerpts)} calls)")

        r1_tasks = [(QUESTION_GENERATION_PROMPT, e["excerpt"]) for e in excerpts]
        r1_responses = asyncio.run(run_phase(r1_tasks, api_key, args.max_budget, max_tokens=150))

        r1_rows = []
        r1_failures = 0
        for i, resp in enumerate(r1_responses):
            if not resp or not resp.strip():
                r1_failures += 1
                continue

            generated_q = resp.strip().strip('"').strip("'").strip()
            if not generated_q.endswith("?"):
                r1_failures += 1
                continue

            r1_rows.append({**excerpts[i], "prompt": generated_q})

        print(f"  Round 1 done: {len(r1_rows)} ok, {r1_failures} failures")

        with open(r1_file, "w") as f:
            for row in r1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(r1_rows)} rows → {r1_file}")

    if budget_exceeded:
        print("Budget exceeded after Round 1. Partial results saved.")
        _print_summary(r1_rows, t0)
        return

    # ══════════════════════════════════════════════════════════════
    # Step 3: Round 2 — Generate answers
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}\nRound 2: Answer questions ({len(r1_rows)} calls)")

    r2_tasks = [
        (R2_SYSTEM, f"## Excerpt\n{row['excerpt']}\n\n## Question\n{row['prompt']}")
        for row in r1_rows
    ]
    r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=400))

    final_rows = []
    r2_failures = 0
    for i, resp in enumerate(r2_responses):
        if not resp:
            r2_failures += 1
            continue
        row = r1_rows[i]
        final_rows.append({
            "excerpt": row["excerpt"],
            "excerpt_n_tokens": row["excerpt_n_tokens"],
            "prompt": row["prompt"],
            "target_response": resp,
            "question_generation_prompt": QUESTION_GENERATION_PROMPT,
            "fineweb_id": row["fineweb_id"],
            "fineweb_url": row["fineweb_url"],
        })

    print(f"  Round 2 done: {len(final_rows)} ok, {r2_failures} failures")

    with open(final_file, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(final_rows)} rows → {final_file}")

    _print_summary(final_rows, t0)


def _print_summary(rows: list[dict], t0: float):
    elapsed = time.time() - t0
    cost = estimate_cost()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total rows: {len(rows)}")
    print(f"  API calls: {completed} completed, {failed} failed")
    print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
    print(f"  Cost: ${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    if rows and "excerpt_n_tokens" in rows[0]:
        lengths = [r["excerpt_n_tokens"] for r in rows]
        print(f"\n  Excerpt token lengths:")
        print(f"    mean={sum(lengths)/len(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")


if __name__ == "__main__":
    main()
