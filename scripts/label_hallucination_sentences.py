#!/usr/bin/env python3
"""
Label individual sentences in hallucinated CoTs as hallucinated or factual.

For each CoT labeled "hallucinated" at the CoT level, ask Gemini Flash Lite
to identify which specific sentences contain fabricated/false information.
Sentences in "factual" CoTs are all labeled factual.

Outputs a sentence-level JSONL: one row per sentence with a binary label.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/label_hallucination_sentences.py
"""

import asyncio
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import httpx

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "google/gemini-2.0-flash-lite-001"
CONCURRENCY = 20
INPUT_DIR = Path("data/cot_hallucinations")
OUTPUT_DIR = Path("data/cot_hallucinations")
HF_REPO = "ceselder/cot-oracle-hallucination-detection"

# Minimum sentence length to include (skip very short fragments)
MIN_SENTENCE_CHARS = 30


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\n])|(?<=\n)\s*(?=\S)', text)
    result = []
    for p in parts:
        if '\n\n' in p:
            result.extend(s.strip() for s in p.split('\n\n') if s.strip())
        elif p.strip():
            result.append(p.strip())
    return result


# Patterns for meta-reasoning sentences (not factual claims)
META_PATTERNS = re.compile(
    r"^("
    r"(let me|i need to|i should|i('ll| will)|i think|i recall|i remember|i believe|"
    r"i'm |i am |i don't|i can|i want|i have to|"
    r"let's|we need|we should|we can|"
    r"hmm|hm+|okay|ok |ok,|alright|well,|well |so,|so |"
    r"wait|actually,|actually |hold on|now,|now |"
    r"first,|second,|third,|next,|then,|finally,|"
    r"to (answer|solve|figure|determine|find|calculate|check|verify|address|approach)|"
    r"looking at|thinking about|considering|given that|"
    r"this (means|suggests|implies|indicates|is|seems|looks)|"
    r"that (means|suggests|implies|indicates|is|seems|looks)|"
    r"so the (answer|result|solution|conclusion)|"
    r"in (summary|conclusion|short)|to summarize|overall|"
    r"the (question|problem|task|prompt) (is|asks|wants)|"
    r"let me (think|consider|recall|check|verify|look|break|start|try|see|go|review)"
    r")"
    r")",
    re.IGNORECASE,
)


def is_claim_sentence(sentence: str) -> bool:
    """Return True if the sentence makes a factual claim (not meta-reasoning)."""
    s = sentence.strip()
    if len(s) < MIN_SENTENCE_CHARS:
        return False
    if META_PATTERNS.search(s):
        return False
    return True


SENTENCE_JUDGE_SYSTEM = """You identify which specific sentences in a language model's reasoning contain hallucinated (fabricated, false, or confabulated) information.

Rules:
- A sentence is "hallucinated" if it states something factually wrong, fabricates entities/numbers/dates/citations, or confidently asserts made-up information.
- A sentence is "factual" if it's accurate, or is hedging/qualifying ("I think...", "maybe..."), or is meta-reasoning about the problem.
- If a sentence contains a mix, label it "hallucinated" if the core claim is false.
- Be strict: only label as hallucinated if the sentence actually states something false, not just because it's in a hallucinated CoT."""

SENTENCE_JUDGE_TEMPLATE = """Question asked to the model:
{question}

The model's reasoning below has been judged as containing hallucinations overall.
I've numbered each claim sentence (skipping meta-reasoning). For each sentence,
classify whether THAT SPECIFIC sentence is hallucinated or factual.

Numbered claim sentences:
{numbered_sentences}

Return a JSON array with one entry per sentence:
[{{"idx": 0, "label": "hallucinated" or "factual", "reason": "brief"}}]

IMPORTANT: Only return the JSON array, nothing else."""


async def _api_call(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    messages: list[dict],
    retries: int = 3,
) -> dict | None:
    async with semaphore:
        body = {
            "model": JUDGE_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(retries):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=180)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                if resp.status_code != 200:
                    if attempt == retries - 1:
                        print(f"  API error {resp.status_code}: {resp.text[:200]}")
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue
                return resp.json()
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  Request failed: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def label_sentences_for_item(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    item: dict,
) -> list[dict] | None:
    """For a hallucinated CoT, ask LLM which sentences are hallucinated."""
    cot_text = item["cot_text"]
    question = item["question"]

    sentences = split_sentences(cot_text)
    claim_sentences = [(i, s) for i, s in enumerate(sentences) if is_claim_sentence(s)]

    if not claim_sentences:
        return None

    # Build numbered list for the judge
    numbered = "\n".join(f"[{j}] {s}" for j, (_, s) in enumerate(claim_sentences))

    messages = [
        {"role": "system", "content": SENTENCE_JUDGE_SYSTEM},
        {"role": "user", "content": SENTENCE_JUDGE_TEMPLATE.format(
            question=question,
            numbered_sentences=numbered,
        )},
    ]

    data = await _api_call(client, semaphore, api_key, messages)
    if not data:
        return None

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Parse JSON array from response
    try:
        # Find JSON array in response
        match = re.search(r'\[[\s\S]*\]', content)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = json.loads(content)
    except json.JSONDecodeError:
        print(f"  Failed to parse judge response for: {question[:50]}...")
        return None

    # Map judge labels back to sentences
    results = []
    for entry in parsed:
        idx = entry.get("idx")
        label = entry.get("label", "").lower().strip()
        if idx is None or label not in ("hallucinated", "factual"):
            continue
        if idx >= len(claim_sentences):
            continue

        orig_idx, sentence = claim_sentences[idx]
        results.append({
            "question": question,
            "cot_text": cot_text,
            "sentence": sentence,
            "sentence_idx": orig_idx,
            "label": label,
            "cot_label": item["label"],
            "prompt_category": item["prompt_category"],
            "prompt_id": item["prompt_id"],
            "judge_reason": entry.get("reason", ""),
        })

    return results


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    # Load both splits
    all_sentence_data = {"train": [], "test": []}

    for split in ["train", "test"]:
        input_path = INPUT_DIR / f"{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping {split} (not found)")
            continue

        items = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        print(f"\n{'='*60}")
        print(f"Processing {split}: {len(items)} items")
        print(f"{'='*60}")

        hallucinated_items = [it for it in items if it["label"] == "hallucinated"]
        factual_items = [it for it in items if it["label"] == "factual"]
        print(f"  {len(hallucinated_items)} hallucinated, {len(factual_items)} factual CoTs")

        # For hallucinated CoTs: ask LLM to identify which sentences are hallucinated
        print(f"\nLabeling sentences in {len(hallucinated_items)} hallucinated CoTs...")
        semaphore = asyncio.Semaphore(CONCURRENCY)
        limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)

        completed = 0
        async with httpx.AsyncClient(limits=limits) as client:
            async def do_one(item):
                nonlocal completed
                result = await label_sentences_for_item(client, semaphore, api_key, item)
                completed += 1
                if completed % 10 == 0:
                    print(f"  {completed}/{len(hallucinated_items)} CoTs labeled")
                return result

            coros = [do_one(it) for it in hallucinated_items]
            raw_results = await asyncio.gather(*coros)

        # Collect hallucinated sentence results
        hall_sentences = []
        for result in raw_results:
            if result:
                hall_sentences.extend(result)

        hall_labeled = Counter(s["label"] for s in hall_sentences)
        print(f"  Hallucinated CoTs → {len(hall_sentences)} claim sentences labeled: {hall_labeled}")

        # For factual CoTs: all claim sentences are factual
        fact_sentences = []
        for item in factual_items:
            sentences = split_sentences(item["cot_text"])
            for i, s in enumerate(sentences):
                if is_claim_sentence(s):
                    fact_sentences.append({
                        "question": item["question"],
                        "cot_text": item["cot_text"],
                        "sentence": s,
                        "sentence_idx": i,
                        "label": "factual",
                        "cot_label": "factual",
                        "prompt_category": item["prompt_category"],
                        "prompt_id": item["prompt_id"],
                        "judge_reason": "from factual CoT",
                    })
        print(f"  Factual CoTs → {len(fact_sentences)} claim sentences (all factual)")

        # Combine
        all_sentences = hall_sentences + fact_sentences

        # Balance: keep all hallucinated sentences, downsample factual to match
        hall_only = [s for s in all_sentences if s["label"] == "hallucinated"]
        fact_only = [s for s in all_sentences if s["label"] == "factual"]

        if len(fact_only) > len(hall_only) * 3:
            # Downsample factual to 3x hallucinated (keep some imbalance for realism)
            random.seed(42)
            random.shuffle(fact_only)
            fact_only = fact_only[:len(hall_only) * 3]

        balanced = hall_only + fact_only
        random.shuffle(balanced)

        labels = Counter(s["label"] for s in balanced)
        print(f"  Final: {len(balanced)} sentences ({labels})")

        all_sentence_data[split] = balanced

    # Save
    for split, data in all_sentence_data.items():
        if not data:
            continue
        output_path = OUTPUT_DIR / f"sentences_{split}.jsonl"
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\nSaved {output_path} ({len(data)} sentences)")

    # Upload to HF
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            for split, data in all_sentence_data.items():
                if not data:
                    continue
                path = OUTPUT_DIR / f"sentences_{split}.jsonl"
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"sentences_{split}.jsonl",
                    repo_id=HF_REPO,
                    repo_type="dataset",
                )
                print(f"Uploaded sentences_{split}.jsonl to {HF_REPO}")
        except Exception as e:
            print(f"Upload failed: {e}")

    # Print stats
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for split, data in all_sentence_data.items():
        if not data:
            continue
        labels = Counter(s["label"] for s in data)
        cats = Counter(s["prompt_category"] for s in data if s["label"] == "hallucinated")
        print(f"\n{split}: {len(data)} sentences ({labels})")
        if cats:
            print(f"  Hallucinated by category: {dict(cats)}")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"\nDone in {time.time() - t0:.0f}s")
