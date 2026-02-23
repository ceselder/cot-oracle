#!/usr/bin/env python3
"""
Generate ~10K conversational QA pairs from the concept corpus using
DeepSeek v3.2 via OpenRouter, then upload to HuggingFace.

Usage:
    python scripts/generate_conv_qa.py
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

# ── Config ──
CORPUS_PATH = Path("data/concept_corpus/corpus_full.jsonl")
OUTPUT_PATH = Path("data/concept_corpus/corpus_full_conv_qa_llm.jsonl")

API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
MODEL = "deepseek/deepseek-v3.2"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 50
TARGET_PAIRS = 10_000
MAX_BUDGET_USD = 25.0

SYSTEM_PROMPT = """\
You are a QA generation assistant. Given a chain-of-thought reasoning trace, \
answer the question based ONLY on what you see in the text. Be concise \
(1-3 sentences)."""

# ── Question Templates by Category ──
# Each entry: (category, question_text, is_binary)
# For binary questions, answers should be "yes" or "no" followed by brief explanation if needed.

QUESTION_TEMPLATES = [
    # behavior
    ("behavior", "Is the model being helpful or refusing the request?", False),
    ("behavior", "Is the model being helpful?", True),
    ("behavior", "Does the model refuse or decline the request?", True),
    ("behavior", "Does the model add unnecessary disclaimers or caveats?", True),
    ("behavior", "Is the model exercising caution in its response?", True),
    ("behavior", "Does the model acknowledge its limitations?", True),
    # topic
    ("topic", "What topic or domain is this reasoning about?", False),
    ("topic", "Briefly describe the subject matter of this reasoning.", False),
    ("topic", "Is this reasoning about math or science?", True),
    ("topic", "Does this involve code or programming?", True),
    ("topic", "Is this about ethics, morality, or social issues?", True),
    ("topic", "Does this reasoning involve creative writing?", True),
    # content
    ("content", "Does the model mention any specific people, places, or entities?", True),
    ("content", "What specific concepts or entities are discussed?", False),
    ("content", "Does the model arrive at a numerical answer?", True),
    ("content", "Does the model reference any formulas, theorems, or named methods?", True),
    ("content", "What is the model's final answer or conclusion?", False),
    # user_intent
    ("user_intent", "What is the user asking for?", False),
    ("user_intent", "What type of request is this?", False),
    ("user_intent", "Is the user asking for factual information?", True),
    ("user_intent", "Is the user asking for an opinion or recommendation?", True),
    ("user_intent", "Is the user trying to get the model to do something harmful?", True),
    # thematic
    ("thematic", "Is there emotional or ethical content in this reasoning?", True),
    ("thematic", "Does this involve moral or ethical dilemmas?", True),
    ("thematic", "What themes are present in this reasoning?", False),
    ("thematic", "Does this involve conflict, controversy, or sensitive topics?", True),
    ("thematic", "Is the content about fairness, justice, or discrimination?", True),
    # sycophancy
    ("sycophancy", "Is the model being overly agreeable or sycophantic?", True),
    ("sycophancy", "Does the model express its own view or just mirror the user's?", False),
    ("sycophancy", "Does the model push back on any of the user's claims?", True),
    ("sycophancy", "Is there authority bias in this reasoning?", True),
    # general
    ("general", "What's the gist of this reasoning?", False),
    ("general", "Summarize what this chain of thought does.", False),
    ("general", "What is the main thing this reasoning accomplishes?", False),
    ("general", "What's actually going on in this reasoning?", False),
    # style
    ("style", "Is the reasoning verbose or concise?", False),
    ("style", "Does the model use technical jargon?", True),
    ("style", "Is the tone formal or conversational?", False),
    ("style", "Does the model use structured formatting like lists or headers?", True),
    # safety
    ("safety", "Does this involve harmful or sensitive topics?", True),
    ("safety", "Is the model refusing a harmful request?", True),
    ("safety", "Is the content safe for a general audience?", True),
    ("safety", "Does this involve illegal activities, violence, or dangerous content?", True),
    ("safety", "What safety concerns, if any, are present?", False),
    # reasoning
    ("reasoning", "Does the model self-correct or show uncertainty?", True),
    ("reasoning", "Does the model backtrack at any point?", True),
    ("reasoning", "Does the model consider multiple approaches?", True),
    ("reasoning", "Is there a verification or double-check step?", True),
    ("reasoning", "Describe the reasoning strategy used.", False),
    ("reasoning", "Does the model make any errors in its reasoning?", True),
]

# Group by category for balanced sampling
BY_CATEGORY = {}
for cat, q, is_bin in QUESTION_TEMPLATES:
    BY_CATEGORY.setdefault(cat, []).append((q, is_bin))

CATEGORIES = list(BY_CATEGORY.keys())


def pick_questions(rng: random.Random, n: int = 2) -> list[tuple[str, str, bool]]:
    """Pick n questions from random categories (category-diverse)."""
    cats = rng.sample(CATEGORIES, min(n, len(CATEGORIES)))
    result = []
    for cat in cats:
        q_text, is_binary = rng.choice(BY_CATEGORY[cat])
        result.append((cat, q_text, is_binary))
    return result


# ── Stats ──
completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


async def call_openrouter(
    client: httpx.AsyncClient,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Single OpenRouter call. Returns (response_text, input_tokens, output_tokens)."""
    global completed, failed, budget_exceeded

    if budget_exceeded:
        return "", 0, 0

    async with semaphore:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        for attempt in range(4):
            try:
                resp = await client.post(
                    ENDPOINT,
                    json=body,
                    headers=headers,
                    timeout=60,
                )

                if resp.status_code == 429:
                    wait = 2 ** attempt + random.random()
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code == 402:
                    budget_exceeded = True
                    print(f"\n*** Budget exceeded (402). Stopping. ***")
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
                # Strip any thinking tokens
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                content = content.strip()

                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)

                completed += 1
                if completed % 100 == 0:
                    est_cost = (
                        total_input_tokens * 0.26 / 1e6
                        + total_output_tokens * 0.38 / 1e6
                    )
                    print(
                        f"  {completed} done, {failed} failed, "
                        f"~${est_cost:.2f} spent"
                    )

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
) -> list[dict]:
    """Generate conversational QA for a batch of tasks."""
    global total_input_tokens, total_output_tokens

    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Prepare coroutines
    limits = httpx.Limits(
        max_connections=CONCURRENCY + 10,
        max_keepalive_connections=CONCURRENCY,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        coros = []
        for task in tasks:
            user_prompt = (
                f"Chain of thought:\n{task['cot_text']}\n\nQuestion: {task['question']}"
            )
            coros.append(call_openrouter(client, user_prompt, semaphore))

        results = await asyncio.gather(*coros)

    qa_pairs = []
    for task, (response, in_tok, out_tok) in zip(tasks, results):
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if not response or budget_exceeded:
            continue

        qa_pairs.append({
            "corpus_id": task["corpus_id"],
            "task_family": "cot_qa_conversational",
            "task_type": f"conv_{task['category']}",
            "prompt": task["prompt"],
            "target_response": response,
            "answer_length": "short" if task["is_binary"] else "medium",
        })

    return qa_pairs


def main():
    global total_input_tokens, total_output_tokens, budget_exceeded

    # Load corpus
    corpus = []
    with open(CORPUS_PATH) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus.append(entry)
    print(f"Loaded {len(corpus)} corpus entries")

    # Prepare tasks: for each entry, pick 1-2 random categories
    rng = random.Random(42)
    all_tasks = []

    for entry in corpus:
        cot_text = entry.get("cot_content", "")
        if not cot_text:
            cot_text = re.sub(
                r"<think>|</think>", "", entry.get("cot_response", "")
            ).strip()
        if not cot_text or len(cot_text) < 30:
            continue

        # Truncate very long CoTs
        if len(cot_text) > 2000:
            cot_text = cot_text[:2000] + "..."

        n_bounds = len(entry.get("boundary_positions", []))
        n_questions = rng.choice([1, 2])  # 1 or 2 questions per entry
        questions = pick_questions(rng, n_questions)

        for cat, q_text, is_binary in questions:
            prompt = (
                f"Activations from {n_bounds} sentence boundaries. {q_text}"
            )
            all_tasks.append({
                "corpus_id": entry["id"],
                "category": cat,
                "question": q_text,
                "is_binary": is_binary,
                "prompt": prompt,
                "cot_text": cot_text,
            })

    rng.shuffle(all_tasks)

    # Trim to slightly more than target to account for failures
    # Average ~1.5 questions per entry * 8132 entries = ~12198 tasks
    # We want 10K successful, so keep ~11K tasks
    max_tasks = int(TARGET_PAIRS * 1.15)
    if len(all_tasks) > max_tasks:
        all_tasks = all_tasks[:max_tasks]

    print(f"Prepared {len(all_tasks)} tasks ({len(corpus)} entries)")
    print(f"  Categories: {Counter(t['category'] for t in all_tasks).most_common()}")
    print(f"  Target: {TARGET_PAIRS} pairs, Budget: ${MAX_BUDGET_USD}")

    # Estimate cost
    avg_input_chars = sum(len(t["cot_text"]) for t in all_tasks[:100]) / 100
    avg_input_tokens = avg_input_chars / 3.5  # rough char-to-token ratio
    est_cost = (
        len(all_tasks) * avg_input_tokens * 0.26 / 1e6
        + len(all_tasks) * 80 * 0.38 / 1e6
    )
    print(f"  Estimated cost: ~${est_cost:.2f}")

    # Process in batches
    BATCH_SIZE = 500
    all_pairs = []
    t0 = time.time()

    for i in range(0, len(all_tasks), BATCH_SIZE):
        if budget_exceeded:
            break
        if len(all_pairs) >= TARGET_PAIRS:
            break

        batch = all_tasks[i : i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}: tasks {i}-{i + len(batch)}")

        pairs = asyncio.run(generate_batch(batch))
        all_pairs.extend(pairs)

        elapsed = time.time() - t0
        est_total_cost = (
            total_input_tokens * 0.26 / 1e6
            + total_output_tokens * 0.38 / 1e6
        )
        print(
            f"  => {len(pairs)} pairs this batch, {len(all_pairs)} total, "
            f"${est_total_cost:.2f} spent, {elapsed:.0f}s elapsed"
        )

        # Check budget
        if est_total_cost > MAX_BUDGET_USD * 0.95:
            print(f"\n*** Approaching budget limit. Stopping. ***")
            break

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    elapsed = time.time() - t0
    total_cost = (
        total_input_tokens * 0.26 / 1e6
        + total_output_tokens * 0.38 / 1e6
    )

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Completed API calls: {completed}")
    print(f"  Failed API calls: {failed}")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.2f}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Output: {OUTPUT_PATH}")

    # Category breakdown
    cat_counts = Counter(p["task_type"] for p in all_pairs)
    print(f"\n  By category:")
    for cat, count in cat_counts.most_common():
        print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()
