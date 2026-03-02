#!/usr/bin/env python3
"""
Rework chunked CompQA dataset with natural split points and free-form answers.

2-round pipeline using Gemini 2.5 Flash Lite via OpenRouter:
  Round 1: Gemini sees full CoT + category, finds a natural split point
  Round 2: Gemini sees suffix + category-specific question, answers in free form

Source data: existing 45K rows from HF (keeps same CoTs + category assignments).
Intermediate JSONL saves after each round for resumability.

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_chunked_compqa.py
    python scripts/generate_chunked_compqa.py --max-cots 20 --max-budget 0.50  # test run
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
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

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
                if completed % 500 == 0:
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

R1_SYSTEM_BY_CATEGORY = {
    "cot_backtrack_pred": """\
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task is to find a natural split point RIGHT BEFORE a potential backtrack or direction change in the reasoning.

Look for moments where the model is about to:
- Reconsider a previous step
- Change its approach or strategy
- Realize a mistake and backtrack
- Abandon a line of reasoning

Return ONLY a JSON object: {"split_index": <int>}
where split_index is the 0-based sentence index of the LAST sentence of the prefix.

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix""",

    "cot_self_correction": """\
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task is to find a natural split point RIGHT AFTER an error has been made but BEFORE the model potentially corrects it.

Look for moments where:
- The model has just made a computational or logical error
- An incorrect intermediate result has been stated
- A wrong assumption has been applied
- The error is still present and hasn't been noticed yet

Return ONLY a JSON object: {"split_index": <int>}
where split_index is the 0-based sentence index of the LAST sentence of the prefix.

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix""",

    "cot_verification": """\
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task is to find a natural split point where the model has reached an answer or conclusion but hasn't verified or double-checked it yet.

Look for moments where:
- A candidate answer has been computed but not yet checked
- The model is about to start a verification step
- An intermediate result is ready for cross-checking
- The model transitions from solving to validating

Return ONLY a JSON object: {"split_index": <int>}
where split_index is the 0-based sentence index of the LAST sentence of the prefix.

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix""",

    "cot_remaining_strategy": """\
You are analyzing a chain-of-thought (CoT) reasoning trace. Your task is to find a natural split point at a transition between reasoning phases or strategies.

Look for moments where:
- The model shifts from one approach to another
- A new sub-problem is being tackled
- The reasoning strategy changes (e.g., from algebraic to numeric)
- A significant phase boundary in the problem-solving process

Return ONLY a JSON object: {"split_index": <int>}
where split_index is the 0-based sentence index of the LAST sentence of the prefix.

Constraints:
- split_index must leave at least 3 sentences in both prefix and suffix""",
}

# Oracle prompts — what the oracle sees at training time
ORACLE_PROMPTS = {
    "cot_backtrack_pred": "Does the model revise or backtrack after this point? Start with Yes or No, then explain briefly.",
    "cot_self_correction": "Does the model notice and correct an error? Start with Yes or No, then explain briefly.",
    "cot_verification": "Does the model verify or double-check its work after this point? Start with Yes or No, then explain briefly.",
    "cot_remaining_strategy": "Describe the reasoning approach the model uses in the remaining steps.",
}

R2_SYSTEM = """\
You are analyzing a portion of a chain-of-thought reasoning trace (the suffix, after a split point).
Answer the given question about the suffix content.
For Yes/No questions, start your answer with "Yes" or "No", then provide a brief justification (1-2 sentences).
For descriptive questions, give a concise answer (1-3 sentences)."""


def build_r1_prompt(question: str, sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return f"## Original Problem\n{question}\n\n## CoT (numbered sentences)\n{numbered}"


def build_r2_prompt(question: str, suffix_text: str, oracle_question: str) -> str:
    return f"## Original Problem\n{question}\n\n## Reasoning continuation (suffix)\n{suffix_text}\n\n## Question\n{oracle_question}"


# ── JSON parsing ──

def parse_r1_json(text: str) -> dict | None:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if "split_index" not in obj:
            return None
        if not isinstance(obj["split_index"], int):
            return None
        return obj
    except json.JSONDecodeError:
        return None


def parse_target_label(response: str) -> bool | None:
    """Parse Yes/No from start of response. Returns True/False or None if unparseable."""
    first_word = response.strip().split()[0].lower().rstrip(".,!:;") if response.strip() else ""
    if first_word == "yes":
        return True
    if first_word == "no":
        return False
    return None


# ── Data loading ──

def load_existing_dataset(max_cots: int | None = None, seed: int = 42) -> list[dict]:
    """Download existing 45K CompQA rows from HF."""
    from datasets import load_dataset

    print("Downloading existing CompQA dataset from HF...")
    ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-compqa-chunked", split="train")
    print(f"  Loaded {len(ds)} rows")

    # Sentence-split each CoT
    rows = []
    for row in ds:
        cot_text = row["cot_text"]
        # Simple sentence splitting: split on '. ' but keep the period
        sentences = _split_sentences(cot_text)
        rows.append({
            "question": row["question"],
            "cot_text": cot_text,
            "sentences": sentences,
            "n_sentences": len(sentences),
            "datapoint_type": row["datapoint_type"],
        })

    # Filter: need at least 8 sentences for meaningful splits
    before = len(rows)
    rows = [r for r in rows if r["n_sentences"] >= 8]
    if before != len(rows):
        print(f"  Filtered {before} -> {len(rows)} (need >=8 sentences)")

    if max_cots:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_cots]
        print(f"  Limited to {max_cots} CoTs")

    cat_counts = Counter(r["datapoint_type"] for r in rows)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    return rows


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations and decimal numbers."""
    # Use regex to split on sentence boundaries
    # Split on period/exclamation/question followed by space and uppercase letter,
    # or newline boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on newlines that look like sentence breaks
    sentences = []
    for part in parts:
        sub_parts = re.split(r'\n+', part)
        for sp in sub_parts:
            sp = sp.strip()
            if sp:
                sentences.append(sp)
    return sentences


# ── Main pipeline ──

async def run_phase(
    tasks: list[tuple[str, str]],
    api_key: str,
    max_budget: float,
    max_tokens: int = 800,
) -> list[str]:
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

    parser = argparse.ArgumentParser(description="Generate chunked CompQA dataset with natural split points")
    parser.add_argument("--output-dir", default="data/chunked_compqa")
    parser.add_argument("--max-cots", type=int, default=None, help="Limit total CoTs (for test runs)")
    parser.add_argument("--max-budget", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", choices=["r1"], default=None, help="Resume from R1 intermediate JSONL")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_file = output_dir / "chunked_compqa_r1.jsonl"
    final_file = output_dir / "chunked_compqa.jsonl"

    t0 = time.time()

    # ── Load existing dataset ──
    corpus = load_existing_dataset(args.max_cots, args.seed)

    # ══════════════════════════════════════════════════════════════
    # Round 1: Find natural split point per category
    # ══════════════════════════════════════════════════════════════

    if args.resume_from == "r1":
        print(f"\nResuming from {r1_file}")
        r1_rows = []
        with open(r1_file) as f:
            for line in f:
                if line.strip():
                    r1_rows.append(json.loads(line))
        print(f"  Loaded {len(r1_rows)} rows")
    else:
        print(f"\n{'='*60}\nRound 1: Find natural split points ({len(corpus)} calls)")

        r1_tasks = []
        r1_indices = []
        for i, entry in enumerate(corpus):
            cat = entry["datapoint_type"]
            system_prompt = R1_SYSTEM_BY_CATEGORY[cat]
            user_prompt = build_r1_prompt(entry["question"], entry["sentences"])
            r1_tasks.append((system_prompt, user_prompt))
            r1_indices.append(i)

        r1_responses = asyncio.run(run_phase(r1_tasks, api_key, args.max_budget, max_tokens=150))

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
            cat = entry["datapoint_type"]

            r1_rows.append({
                "question": entry["question"],
                "cot_text": entry["cot_text"],
                "sentences": sents,
                "num_sentences": n,
                "split_index": split_index,
                "datapoint_type": cat,
                "cot_prefix": " ".join(sents[:split_index + 1]),
                "cot_suffix": " ".join(sents[split_index + 1:]),
            })

        print(f"  Round 1 done: {len(r1_rows)} ok, {r1_failures} failures")

        # Save R1 intermediate
        with open(r1_file, "w") as f:
            for row in r1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(r1_rows)} rows -> {r1_file}")

    if budget_exceeded:
        print("Budget exceeded after Round 1. Partial results saved.")
        _print_summary(r1_rows, t0)
        return

    # ══════════════════════════════════════════════════════════════
    # Round 2: Free-form answer from suffix
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}\nRound 2: Free-form answers ({len(r1_rows)} calls)")

    r2_tasks = []
    for row in r1_rows:
        cat = row["datapoint_type"]
        oracle_q = ORACLE_PROMPTS[cat]
        user_prompt = build_r2_prompt(row["question"], row["cot_suffix"], oracle_q)
        r2_tasks.append((R2_SYSTEM, user_prompt))

    r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=300))

    final_rows = []
    r2_failures = 0
    label_parse_failures = 0
    for i, resp in enumerate(r2_responses):
        if not resp:
            r2_failures += 1
            continue

        row = r1_rows[i]
        cat = row["datapoint_type"]
        is_binary = cat != "cot_remaining_strategy"

        # For binary tasks, extract target_label
        target_label = None
        if is_binary:
            target_label = parse_target_label(resp)
            if target_label is None:
                label_parse_failures += 1
                continue

        final_row = {
            "question": row["question"],
            "cot_text": row["cot_text"],
            "cot_prefix": row["cot_prefix"],
            "cot_suffix": row["cot_suffix"],
            "prompt": ORACLE_PROMPTS[cat],
            "target_response": resp,
            "datapoint_type": cat,
            "split_index": row["split_index"],
            "num_sentences": row["num_sentences"],
        }
        if target_label is not None:
            final_row["target_label"] = target_label

        final_rows.append(final_row)

    print(f"  Round 2 done: {len(final_rows)} ok, {r2_failures} API failures, {label_parse_failures} label parse failures")

    # ── Save final ──
    with open(final_file, "w") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(final_rows)} rows -> {final_file}")

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
        cat_dist = Counter(r["datapoint_type"] for r in rows)
        print(f"\n  By category:")
        for cat, cnt in sorted(cat_dist.items()):
            print(f"    {cat}: {cnt}")

        # Split position distribution
        splits = [r["split_index"] / r["num_sentences"] for r in rows if "split_index" in r]
        if splits:
            import statistics
            print(f"\n  Split position (fraction of CoT):")
            print(f"    mean: {statistics.mean(splits):.2f}, median: {statistics.median(splits):.2f}")
            print(f"    min: {min(splits):.2f}, max: {max(splits):.2f}")

        # Label distribution for binary tasks
        labels = [r.get("target_label") for r in rows if "target_label" in r]
        if labels:
            true_count = sum(1 for l in labels if l is True)
            false_count = sum(1 for l in labels if l is False)
            print(f"\n  Binary label distribution:")
            print(f"    True (Yes): {true_count}")
            print(f"    False (No): {false_count}")
            print(f"    Ratio: {true_count/(true_count+false_count):.1%} positive")


if __name__ == "__main__":
    main()
