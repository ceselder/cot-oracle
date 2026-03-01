#!/usr/bin/env python3
"""
Generate chunked ConvQA dataset with natural split points.

3-round pipeline using Gemini 2.5 Flash Lite via OpenRouter:
  Round 1: Gemini sees full CoT, finds a natural split point, generates a question
  Round 2: BB baseline — Gemini sees prefix only, answers the question
  Round 3: GT answer — Gemini sees suffix only, answers the question

Intermediate JSONL saves after each round for resumability.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_chunked_convqa.py
    python scripts/generate_chunked_convqa.py --max-cots 5 --max-budget 0.10  # test run
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
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task:

1. Find a natural splitting point — a sentence index where the reasoning shifts direction, introduces a new insight, makes a key deduction, or changes approach. The suffix (everything after the split) should contain something that is NOT obvious from reading the prefix alone.

2. Generate an open-ended question about the suffix content that:
   - Cannot be confidently answered from the prefix text alone
   - Has a concrete, specific answer derivable from the suffix
   - Is about the reasoning process, conclusions, or approach in the suffix

Return ONLY a JSON object with keys "split_index" (int, 0-based sentence index — last sentence of the prefix) and "question" (str).

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix
- The question should be answerable in 1-3 sentences"""

R2_SYSTEM = """\
You can only see the beginning of a chain-of-thought reasoning trace (the prefix). \
A question is asked about what comes later in the reasoning. \
Answer as best you can from the prefix alone. If you truly cannot answer, say so honestly. \
Be concise: 1-3 sentences."""

R3_SYSTEM = """\
You are given a portion of a chain-of-thought reasoning trace (the suffix, after a split point) \
and a question about it. Answer the question based on the suffix content. \
Be concise: 1-3 sentences."""


def build_r1_prompt(question: str, sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return f"## Original Problem\n{question}\n\n## CoT (numbered sentences)\n{numbered}"


def build_r2_prompt(question: str, prefix_text: str, generated_question: str) -> str:
    return f"## Original Problem\n{question}\n\n## Reasoning so far (prefix)\n{prefix_text}\n\n## Question\n{generated_question}"


def build_r3_prompt(question: str, suffix_text: str, generated_question: str) -> str:
    return f"## Original Problem\n{question}\n\n## Reasoning continuation (suffix)\n{suffix_text}\n\n## Question\n{generated_question}"


# ── JSON parsing ──

def parse_r1_json(text: str) -> dict | None:
    """Extract {"split_index": int, "question": str} from LLM response."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if "split_index" not in obj or "question" not in obj:
            return None
        if not isinstance(obj["split_index"], int) or not isinstance(obj["question"], str):
            return None
        return obj
    except json.JSONDecodeError:
        return None


# ── Data loading ──

def load_corpus(corpus_path: str, min_sentences: int = 8) -> list[dict]:
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
                "sentences": entry["sentences"],
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

    parser = argparse.ArgumentParser(description="Generate chunked ConvQA dataset with natural split points")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus.jsonl")
    parser.add_argument("--output-dir", default="data/chunked_qa")
    parser.add_argument("--max-cots", type=int, default=None, help="Limit total CoTs (for test runs)")
    parser.add_argument("--max-budget", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", choices=["r1", "r2"], default=None, help="Resume from intermediate JSONL")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_file = output_dir / "chunked_convqa_r1.jsonl"
    r2_file = output_dir / "chunked_convqa_r2.jsonl"
    final_file = output_dir / "chunked_convqa.jsonl"

    t0 = time.time()

    # ── Load corpus ──
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} corpus entries with ≥8 sentences")
    source_counts = Counter(e["source"] for e in corpus)
    for s, c in sorted(source_counts.items()):
        print(f"  {s}: {c}")

    # Apply max_cots limit
    if args.max_cots:
        rng = random.Random(args.seed)
        rng.shuffle(corpus)
        corpus = corpus[:args.max_cots]
        print(f"\nLimited to {len(corpus)} CoTs")

    # Build corpus lookup for resume
    corpus_by_id = {e["id"]: e for e in corpus}

    # ══════════════════════════════════════════════════════════════
    # Round 1: Find split point + generate question
    # ══════════════════════════════════════════════════════════════

    if args.resume_from in ("r1", "r2"):
        resume_file = r1_file if args.resume_from == "r1" else r2_file
        print(f"\nResuming from {resume_file}")
        r1_rows = []
        with open(resume_file) as f:
            for line in f:
                if line.strip():
                    r1_rows.append(json.loads(line))
        print(f"  Loaded {len(r1_rows)} rows from {resume_file}")
    else:
        print(f"\n{'='*60}\nRound 1: Find split + generate question ({len(corpus)} calls)")

        r1_tasks = []
        r1_indices = []  # track which corpus entries map to which tasks
        for i, entry in enumerate(corpus):
            user_prompt = build_r1_prompt(entry["question"], entry["sentences"])
            r1_tasks.append((R1_SYSTEM, user_prompt))
            r1_indices.append(i)

        r1_responses = asyncio.run(run_phase(r1_tasks, api_key, args.max_budget, max_tokens=200))

        r1_rows = []
        r1_failures = 0
        for task_idx, resp in enumerate(r1_responses):
            entry = corpus[r1_indices[task_idx]]
            parsed = parse_r1_json(resp) if resp else None

            if parsed is None:
                r1_failures += 1
                continue

            split_index = parsed["split_index"]
            n = entry["n_sentences"]

            # Validate: at least 3 sentences in both prefix and suffix
            if split_index < 2 or split_index >= n - 3:
                r1_failures += 1
                continue

            sents = entry["sentences"]
            generation_prompt = R1_SYSTEM + "\n\n" + build_r1_prompt(entry["question"], sents)

            r1_rows.append({
                "cot_id": entry["id"],
                "source": entry["source"],
                "question": entry["question"],
                "cot_text": entry["cot_text"],
                "sentences": sents,
                "num_sentences": n,
                "split_index": split_index,
                "prompt": parsed["question"],
                "cot_prefix": " ".join(sents[:split_index + 1]),
                "cot_suffix": " ".join(sents[split_index + 1:]),
                "generation_prompt": generation_prompt,
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
    # Round 2: BB baseline (prefix only + question)
    # ══════════════════════════════════════════════════════════════

    if args.resume_from == "r2":
        r2_rows = []
        with open(r2_file) as f:
            for line in f:
                if line.strip():
                    r2_rows.append(json.loads(line))
        print(f"\nResumed R2: {len(r2_rows)} rows from {r2_file}")
    else:
        print(f"\n{'='*60}\nRound 2: BB baseline ({len(r1_rows)} calls)")

        r2_tasks = []
        for row in r1_rows:
            user_prompt = build_r2_prompt(row["question"], row["cot_prefix"], row["prompt"])
            r2_tasks.append((R2_SYSTEM, user_prompt))

        r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=300))

        r2_rows = []
        r2_failures = 0
        for i, resp in enumerate(r2_responses):
            row = {**r1_rows[i]}
            if resp:
                row["bb_response"] = resp
                r2_rows.append(row)
            else:
                r2_failures += 1

        print(f"  Round 2 done: {len(r2_rows)} ok, {r2_failures} failures")

        # Save R2 intermediate
        with open(r2_file, "w") as f:
            for row in r2_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(r2_rows)} rows → {r2_file}")

    if budget_exceeded:
        print("Budget exceeded after Round 2. Partial results saved.")
        _print_summary(r2_rows, t0)
        return

    # ══════════════════════════════════════════════════════════════
    # Round 3: GT answer (suffix only + question)
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}\nRound 3: GT answer ({len(r2_rows)} calls)")

    r3_tasks = []
    for row in r2_rows:
        user_prompt = build_r3_prompt(row["question"], row["cot_suffix"], row["prompt"])
        r3_tasks.append((R3_SYSTEM, user_prompt))

    r3_responses = asyncio.run(run_phase(r3_tasks, api_key, args.max_budget, max_tokens=300))

    final_rows = []
    r3_failures = 0
    for i, resp in enumerate(r3_responses):
        if not resp:
            r3_failures += 1
            continue
        row = r2_rows[i]
        final_rows.append({
            "question": row["question"],
            "cot_text": row["cot_text"],
            "prompt": row["prompt"],
            "target_response": resp,
            "bb_response": row["bb_response"],
            "cot_prefix": row["cot_prefix"],
            "cot_suffix": row["cot_suffix"],
            "split_index": row["split_index"],
            "num_sentences": row["num_sentences"],
            "cot_id": row["cot_id"],
            "source": row["source"],
            "generation_prompt": row["generation_prompt"],
        })

    print(f"  Round 3 done: {len(final_rows)} ok, {r3_failures} failures")

    # ── Save final ──
    with open(final_file, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")

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

        # Split position distribution
        splits = [r["split_index"] / r["num_sentences"] for r in rows if "split_index" in r and "num_sentences" in r]
        if splits:
            import statistics
            print(f"\n  Split position (fraction of CoT):")
            print(f"    mean: {statistics.mean(splits):.2f}, median: {statistics.median(splits):.2f}")
            print(f"    min: {min(splits):.2f}, max: {max(splits):.2f}")


if __name__ == "__main__":
    main()
