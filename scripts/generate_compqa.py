#!/usr/bin/env python3
"""
Generate ~10K computational QA pairs about CoT reasoning quality using
Gemini 3 Flash via OpenRouter.

10 query types × 1000 CoTs = 10,000 examples.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_compqa.py
    python scripts/generate_compqa.py --n-cots 200          # smaller run
    python scripts/generate_compqa.py --model qwen3-8b      # use Qwen3-8B instead
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

# ── Query types ──────────────────────────────────────────────
QUERY_TYPES = {
    "soundness": "Is the reasoning in this chain of thought sound? Explain briefly.",
    "non_sequitur": "Does any step in this reasoning not follow from the previous steps? If so, which and why?",
    "first_error": "What is the first error (if any) in this reasoning? State the step number and the error, or 'none'.",
    "error_type": "Classify the errors in this reasoning: arithmetic, logical fallacy, false premise, irrelevant step, or none. Give the category and a brief explanation.",
    "conclusion_follows": "Does the final answer follow from the reasoning steps? Explain.",
    "redundant_steps": "Are any steps redundant or unnecessary? Which ones?",
    "load_bearing": "Which steps are essential (load-bearing) for reaching the answer, and which are decorative filler?",
    "self_correction": "Does the model notice and correct any of its own mistakes? Describe.",
    "reasoning_direction": "Is the model reasoning forward from givens, backward from the answer, or mixed?",
    "verification": "Does the model verify or double-check its answer? How?",
}

# ── Models ───────────────────────────────────────────────────
MODELS = {
    "gemini-3-flash": {
        "id": "google/gemini-3-flash-preview",
        "input_cost_per_m": 0.50,
        "output_cost_per_m": 3.00,
    },
    "qwen3-8b": {
        "id": "qwen/qwen3-8b",
        "input_cost_per_m": 0.20,
        "output_cost_per_m": 0.20,
    },
}

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 50

SYSTEM_PROMPT = """\
You are an expert reasoning analyst. Given a math problem and a model's \
chain-of-thought reasoning, answer the question about the reasoning quality. \
Be concise and specific. Reference step numbers when relevant."""

# ── Stats ────────────────────────────────────────────────────
completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def number_sentences(cot_text: str) -> str:
    """Split CoT into sentences and number them for easy reference."""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', cot_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))


def build_prompt(question: str, cot_text: str, query: str) -> str:
    numbered = number_sentences(cot_text)
    return (
        f"Here is a math problem and a model's chain-of-thought reasoning. "
        f"Answer the question below about the reasoning.\n\n"
        f"## Problem\n{question}\n\n"
        f"## Chain of Thought (numbered steps)\n{numbered}\n\n"
        f"## Question\n{query}"
    )


async def call_openrouter(
    client: httpx.AsyncClient,
    user_prompt: str,
    model_id: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_budget: float,
    cost_per_m: tuple[float, float],
) -> tuple[str, int, int]:
    global completed, failed, budget_exceeded, total_input_tokens, total_output_tokens

    if budget_exceeded:
        return "", 0, 0

    async with semaphore:
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(4):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=60)

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

                completed += 1
                if completed % 200 == 0:
                    est_cost = total_input_tokens * cost_per_m[0] / 1e6 + total_output_tokens * cost_per_m[1] / 1e6
                    print(f"  {completed} done, {failed} failed, ~${est_cost:.2f} spent")

                # Check budget
                current_cost = (total_input_tokens + in_tok) * cost_per_m[0] / 1e6 + (total_output_tokens + out_tok) * cost_per_m[1] / 1e6
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


async def generate_batch(
    tasks: list[dict],
    model_id: str,
    api_key: str,
    max_budget: float,
    cost_per_m: tuple[float, float],
) -> list[dict]:
    global total_input_tokens, total_output_tokens

    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, t["user_prompt"], model_id, api_key, semaphore, max_budget, cost_per_m)
            for t in tasks
        ]
        results = await asyncio.gather(*coros)

    pairs = []
    for task, (response, in_tok, out_tok) in zip(tasks, results):
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        if not response or budget_exceeded:
            continue
        pairs.append({
            "corpus_id": task["corpus_id"],
            "question": task["question"],
            "cot_text": task["cot_text"],
            "n_sentences": task["n_sentences"],
            "cot_correct": task["cot_correct"],
            "query_type": task["query_type"],
            "query": task["query"],
            "response": response,
            "model": task["model_label"],
        })
    return pairs


def load_corpus(corpus_path: str) -> list[dict]:
    entries = []
    with open(corpus_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            cot = entry.get("cot_content", "")
            if not cot:
                cot = re.sub(r"<think>|</think>", "", entry.get("cot_response", "")).strip()
            if not cot or len(cot) < 50:
                continue
            # Count sentences
            sentences = re.split(r'(?<=[.!?])\s+', cot.strip())
            n_sentences = len([s for s in sentences if s.strip()])
            entries.append({
                "id": entry["id"],
                "question": entry.get("question", entry.get("prompt", "")),
                "cot_text": cot,
                "n_sentences": n_sentences,
                "correct": entry.get("correct", entry.get("cot_correct", True)),
            })
    return entries


def load_existing(output_path: str) -> set[str]:
    """Load already-completed (corpus_id, query_type) pairs for resume."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["corpus_id"], row["query_type"]))
    return done


def main():
    global total_input_tokens, total_output_tokens, budget_exceeded, completed, failed

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus.jsonl")
    parser.add_argument("--output-dir", default="data/compqa")
    parser.add_argument("--model", default="gemini-3-flash", choices=list(MODELS.keys()))
    parser.add_argument("--n-cots", type=int, default=1000, help="CoTs to sample (× 10 query types)")
    parser.add_argument("--max-budget", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip already-completed queries")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    cost_per_m = (model_cfg["input_cost_per_m"], model_cfg["output_cost_per_m"])

    # Output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.model.replace('-', '_')}.jsonl"

    # Load corpus
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} usable corpus entries from {args.corpus}")

    # Sample CoTs — stratified by correctness, prefer 10-60 sentence lengths
    rng = random.Random(args.seed)
    correct = [e for e in corpus if e["correct"] and 10 <= e["n_sentences"] <= 60]
    incorrect = [e for e in corpus if not e["correct"] and 10 <= e["n_sentences"] <= 60]
    # Fallback: relax length filter
    if len(correct) < args.n_cots // 2:
        correct = [e for e in corpus if e["correct"] and 7 <= e["n_sentences"] <= 100]
    if len(incorrect) < args.n_cots // 2:
        incorrect = [e for e in corpus if not e["correct"] and 7 <= e["n_sentences"] <= 100]

    n_correct = min(args.n_cots // 2, len(correct))
    n_incorrect = min(args.n_cots - n_correct, len(incorrect))
    # Fill remainder with correct if not enough incorrect
    n_correct = min(args.n_cots - n_incorrect, len(correct))

    sampled = rng.sample(correct, n_correct) + rng.sample(incorrect, n_incorrect)
    rng.shuffle(sampled)
    print(f"Sampled {len(sampled)} CoTs ({n_correct} correct, {n_incorrect} incorrect)")

    # Resume support
    done = load_existing(str(output_file)) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} queries already completed")

    # Build tasks
    all_tasks = []
    for entry in sampled:
        for query_type, query in QUERY_TYPES.items():
            if (entry["id"], query_type) in done:
                continue
            all_tasks.append({
                "corpus_id": entry["id"],
                "question": entry["question"],
                "cot_text": entry["cot_text"],
                "n_sentences": entry["n_sentences"],
                "cot_correct": entry["correct"],
                "query_type": query_type,
                "query": query,
                "model_label": args.model,
                "user_prompt": build_prompt(entry["question"], entry["cot_text"], query),
            })

    rng.shuffle(all_tasks)
    print(f"Total tasks: {len(all_tasks)} ({len(sampled)} CoTs × {len(QUERY_TYPES)} query types)")

    # Cost estimate
    avg_input_chars = sum(len(t["user_prompt"]) for t in all_tasks[:100]) / min(100, len(all_tasks))
    avg_input_tokens = avg_input_chars / 3.5
    est_cost = len(all_tasks) * avg_input_tokens * cost_per_m[0] / 1e6 + len(all_tasks) * 150 * cost_per_m[1] / 1e6
    print(f"Estimated cost: ~${est_cost:.2f} (budget: ${args.max_budget:.2f})")

    # Process in batches
    BATCH_SIZE = 500
    all_pairs = []
    t0 = time.time()

    for i in range(0, len(all_tasks), BATCH_SIZE):
        if budget_exceeded:
            break
        batch = all_tasks[i : i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}: tasks {i}-{i + len(batch)}")

        pairs = asyncio.run(generate_batch(batch, model_cfg["id"], os.environ["OPENROUTER_API_KEY"], args.max_budget, cost_per_m))
        all_pairs.extend(pairs)

        elapsed = time.time() - t0
        est_total_cost = total_input_tokens * cost_per_m[0] / 1e6 + total_output_tokens * cost_per_m[1] / 1e6
        print(f"  => {len(pairs)} pairs this batch, {len(all_pairs)} total, ${est_total_cost:.2f} spent, {elapsed:.0f}s elapsed")

    # Save (append if resuming, overwrite otherwise)
    mode = "a" if args.resume and done else "w"
    with open(output_file, mode) as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    elapsed = time.time() - t0
    total_cost = total_input_tokens * cost_per_m[0] / 1e6 + total_output_tokens * cost_per_m[1] / 1e6

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Completed API calls: {completed}")
    print(f"  Failed API calls: {failed}")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Cost: ${total_cost:.2f}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Output: {output_file}")

    # Query type breakdown
    type_counts = Counter(p["query_type"] for p in all_pairs)
    print(f"\n  By query type:")
    for qt in QUERY_TYPES:
        print(f"    {qt}: {type_counts.get(qt, 0)}")

    # Correctness breakdown
    correct_count = sum(1 for p in all_pairs if p["cot_correct"])
    print(f"\n  CoT correctness: {correct_count} correct, {len(all_pairs) - correct_count} incorrect")


if __name__ == "__main__":
    main()
