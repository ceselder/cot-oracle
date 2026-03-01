#!/usr/bin/env python3
"""
Generate open-ended ConvQA dataset.

2-round pipeline using Gemini 2.5 Flash Lite via OpenRouter:
  Round 1: Gemini sees full CoT, generates an open-ended question about it
  Round 2: Gemini sees full CoT + generated question, produces an answer

Intermediate JSONL saves after Round 1 for resumability.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_openended_convqa.py
    python scripts/generate_openended_convqa.py --max-cots 5 --max-budget 0.10  # test run
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import httpx

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

R1_SYSTEM = """\
You are analyzing a chain-of-thought reasoning trace for a math/logic problem.
Generate a single open-ended question that asks about the reasoning strategy,
a key decision, or a non-obvious step in this trace. The question should require
reading the full chain of thought to answer well — it should not be answerable
from the problem statement alone. Output ONLY the question, nothing else."""

R2_SYSTEM = """\
Answer the question based on the reasoning trace. Be specific. 2-5 sentences."""


def build_r1_prompt(question: str, cot_text: str) -> str:
    return f"## Original Problem\n{question}\n\n## Chain-of-Thought Reasoning\n{cot_text}"


def build_r2_prompt(question: str, cot_text: str, generated_question: str) -> str:
    return f"## Original Problem\n{question}\n\n## Chain-of-Thought Reasoning\n{cot_text}\n\n## Question\n{generated_question}"


# ── Data loading ──

def load_corpus(corpus_path: str, min_sentences: int = 3) -> list[dict]:
    entries = []
    with open(corpus_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["n_sentences"] < min_sentences:
                continue
            entries.append({
                "id": entry["id"],
                "source": entry["source"],
                "question": entry.get("question", entry.get("prompt", "")),
                "cot_text": " ".join(entry["sentences"]),
                "n_sentences": entry["n_sentences"],
            })
    return entries


# ── Main pipeline ──

async def run_phase(
    tasks: list[tuple[str, str]],  # (system_prompt, user_prompt)
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

    parser = argparse.ArgumentParser(description="Generate open-ended ConvQA dataset")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus.jsonl")
    parser.add_argument("--output-dir", default="data/openended_qa")
    parser.add_argument("--max-cots", type=int, default=None, help="Limit total CoTs (for test runs)")
    parser.add_argument("--max-budget", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", choices=["r1"], default=None, help="Resume from intermediate JSONL")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_file = output_dir / "openended_convqa_r1.jsonl"
    final_file = output_dir / "openended_convqa.jsonl"

    t0 = time.time()

    # ── Load corpus ──
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} corpus entries with ≥3 sentences")
    source_counts = Counter(e["source"] for e in corpus)
    for s, c in sorted(source_counts.items()):
        print(f"  {s}: {c}")

    # Apply max_cots limit
    if args.max_cots:
        rng = random.Random(args.seed)
        rng.shuffle(corpus)
        corpus = corpus[:args.max_cots]
        print(f"\nLimited to {len(corpus)} CoTs")

    # ══════════════════════════════════════════════════════════════
    # Round 1: Generate open-ended question
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
        print(f"\n{'='*60}\nRound 1: Generate questions ({len(corpus)} calls)")

        r1_tasks = []
        r1_indices = []
        for i, entry in enumerate(corpus):
            user_prompt = build_r1_prompt(entry["question"], entry["cot_text"])
            r1_tasks.append((R1_SYSTEM, user_prompt))
            r1_indices.append(i)

        r1_responses = asyncio.run(run_phase(r1_tasks, api_key, args.max_budget, max_tokens=150))

        r1_rows = []
        r1_failures = 0
        for task_idx, resp in enumerate(r1_responses):
            entry = corpus[r1_indices[task_idx]]

            if not resp or not resp.strip():
                r1_failures += 1
                continue

            # Clean: strip stray quotes/markdown
            generated_q = resp.strip().strip('"').strip("'").strip()
            if not generated_q.endswith("?"):
                r1_failures += 1
                continue

            r1_rows.append({
                "cot_id": entry["id"],
                "source": entry["source"],
                "original_question": entry["question"],
                "cot_text": entry["cot_text"],
                "n_sentences": entry["n_sentences"],
                "prompt": generated_q,
            })

        print(f"  Round 1 done: {len(r1_rows)} ok, {r1_failures} failures")

        # Save R1 intermediate
        with open(r1_file, "w") as f:
            for row in r1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(r1_rows)} rows → {r1_file}")

    if budget_exceeded:
        print("Budget exceeded after Round 1. Partial results saved.")
        _print_summary(r1_rows, t0)
        return

    # ══════════════════════════════════════════════════════════════
    # Round 2: Answer the generated questions
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}\nRound 2: Answer questions ({len(r1_rows)} calls)")

    r2_tasks = []
    for row in r1_rows:
        user_prompt = build_r2_prompt(row["original_question"], row["cot_text"], row["prompt"])
        r2_tasks.append((R2_SYSTEM, user_prompt))

    r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=400))

    final_rows = []
    r2_failures = 0
    for i, resp in enumerate(r2_responses):
        if not resp:
            r2_failures += 1
            continue
        row = r1_rows[i]
        final_rows.append({
            "cot_text": row["cot_text"],
            "prompt": row["prompt"],
            "target_response": resp,
            "cot_id": row["cot_id"],
            "source": row["source"],
            "original_question": row["original_question"],
            "n_sentences": row["n_sentences"],
            "question_generation_prompt": R1_SYSTEM,
        })

    print(f"  Round 2 done: {len(final_rows)} ok, {r2_failures} failures")

    # ── Save final ──
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

    if rows:
        source_dist = Counter(r["source"] for r in rows)
        print(f"\n  By source:")
        for src, cnt in sorted(source_dist.items()):
            print(f"    {src}: {cnt}")


if __name__ == "__main__":
    main()
