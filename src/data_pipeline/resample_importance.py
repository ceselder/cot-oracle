"""
Counterfactual importance via truncation + resampling.

For each problem, for each sentence boundary i:
  1. Prefill assistant response with CoT sentences 0..i-1
  2. Let model CONTINUE generating from that truncation point
  3. Extract final answer from the completion
  4. Compare to original answer

If truncating before sentence i changes the answer, sentences i..N are load-bearing.
Per-sentence importance = did the answer flip between truncating at i-1 vs i?

Uses OpenRouter (qwen/qwen3-8b) with assistant prefilling, async parallel.

Usage:
    python src/data_pipeline/resample_importance.py \
        --corpus data/cot_corpus_v4/corpus.jsonl \
        --n-problems 100 \
        --n-resamples 5 \
        --output data/importance_100.jsonl --only-correct
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

OPENROUTER_API_KEY = "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"
MODEL = "qwen/qwen3-8b"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_CONCURRENT = 30
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def call_openrouter(
    session: aiohttp.ClientSession,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str | None:
    """Single OpenRouter API call with robust retries."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(5):
            try:
                async with session.post(BASE_URL, json=payload, headers=headers) as resp:
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After", "")
                        wait = float(retry_after) if retry_after else (2 ** attempt + random.random() * 2)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 500:
                        wait = 2 ** attempt + random.random() * 2
                        await asyncio.sleep(wait)
                        continue

                    data = await resp.json()
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"]["content"]
                    if "error" in data:
                        err = data["error"]
                        # Rate limit or server error — retry
                        if isinstance(err, dict) and err.get("code", 0) in (429, 500, 502, 503):
                            wait = 2 ** attempt + random.random() * 2
                            await asyncio.sleep(wait)
                            continue
                        print(f"  API error: {err}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 4:
                    wait = 2 ** attempt + random.random()
                    await asyncio.sleep(wait)
                else:
                    print(f"  Request failed after 5 attempts: {e}")
                    return None
    return None


def extract_answer(response: str) -> str | None:
    """Extract final answer from a model's continued generation."""
    if not response:
        return None

    # Remove think blocks
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if not text:
        # Entire response might be in think tags with no closing — get last part
        text = response

    # Look for boxed answer (math convention)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Look for "the answer is X"
    m = re.search(r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        # Clean up markdown bold
        ans = re.sub(r'\*\*([^*]+)\*\*', r'\1', ans)
        return ans[:100]

    # For MCQ: look for \boxed{X} or standalone letter
    letter = re.search(r'(?:^|\n)\s*\$*\\?boxed\{?([A-J])\}?\$*\s*$', text, re.MULTILINE)
    if letter:
        return letter.group(1)

    # Last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        # Strip markdown formatting
        last = re.sub(r'[*$\\{}]', '', last).strip()
        if last:
            return last[:100]

    return None


def normalize(ans: str | None) -> str:
    if not ans:
        return ""
    s = ans.strip().lower()
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'[\$\\{}]', '', s)
    s = s.rstrip('.,:;')
    return s.strip()


def answers_match(a: str | None, b: str | None) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


def build_prefill(question: str, sentences: list[str], up_to: int) -> list[dict]:
    """Build messages with assistant prefill containing sentences 0..up_to-1.

    up_to=0: no prefill, model generates from scratch
    up_to=i: prefill with sentences 0..i-1, model continues
    """
    messages = [{"role": "user", "content": question}]

    if up_to > 0:
        prefix = "<think>\n" + "\n".join(sentences[:up_to]) + "\n"
        messages.append({"role": "assistant", "content": prefix})

    return messages


async def resample_at_truncation(
    session: aiohttp.ClientSession,
    question: str,
    sentences: list[str],
    truncate_at: int,
    n_resamples: int,
) -> list[str | None]:
    """Truncate CoT at sentence boundary, let model continue, extract answers."""
    messages = build_prefill(question, sentences, truncate_at)

    tasks = [
        call_openrouter(session, messages, max_tokens=4096, temperature=0.7)
        for _ in range(n_resamples)
    ]
    responses = await asyncio.gather(*tasks)

    return [extract_answer(r) for r in responses]


async def process_problem(
    session: aiohttp.ClientSession,
    entry: dict,
    n_resamples: int,
) -> dict:
    """Process one problem: resample at every truncation point, all in parallel."""
    question = entry["question"]
    sentences = entry["sentences"]
    original_answer = entry.get("cot_answer") or entry.get("correct_answer", "")
    correct_answer = entry.get("correct_answer", "")
    n_sentences = len(sentences)

    # Fire all truncation points in parallel:
    # truncate_at=0 (no CoT), 1, 2, ..., N (full CoT)
    all_tasks = []
    for t in range(n_sentences + 1):
        all_tasks.append(
            resample_at_truncation(session, question, sentences, t, n_resamples)
        )

    all_results = await asyncio.gather(*all_tasks)

    # Build truncation results
    truncation_results = []
    for t, answers in enumerate(all_results):
        n_match = sum(1 for a in answers if answers_match(a, original_answer))
        n_correct = sum(1 for a in answers if answers_match(a, correct_answer))
        n_valid = sum(1 for a in answers if a is not None)
        truncation_results.append({
            "truncate_at": t,
            "label": f"sentences 0..{t-1}" if t > 0 else "no_cot",
            "answers": answers,
            "n_valid": n_valid,
            "n_match_original": n_match,
            "n_match_correct": n_correct,
            "match_rate": n_match / max(n_valid, 1),
        })

    # Per-sentence importance: did adding sentence i change the match rate?
    sentence_importance = []
    for i in range(n_sentences):
        before = truncation_results[i]      # without sentence i
        after = truncation_results[i + 1]   # with sentence i

        # Importance = how much did including this sentence help reach original answer?
        delta = after["match_rate"] - before["match_rate"]

        sentence_importance.append({
            "sentence_idx": i,
            "sentence_text": sentences[i][:200],
            "match_rate_without": before["match_rate"],
            "match_rate_with": after["match_rate"],
            "importance_delta": delta,
            "important": delta > 0.3,  # sentence flipped >30% of resamples
        })

    return {
        "id": entry["id"],
        "source": entry.get("source", ""),
        "question": question[:300],
        "correct_answer": correct_answer,
        "original_cot_answer": original_answer,
        "cot_correct": entry.get("cot_correct", False),
        "n_sentences": n_sentences,
        "truncations": truncation_results,
        "sentence_importance": sentence_importance,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/cot_corpus_v4/corpus.jsonl")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--n-resamples", type=int, default=5)
    parser.add_argument("--output", default="data/importance_resampled.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-correct", action="store_true",
                        help="Only use problems where CoT got the right answer")
    args = parser.parse_args()

    random.seed(args.seed)

    corpus = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("sentences") and len(entry["sentences"]) >= 3:
                    if args.only_correct and not entry.get("cot_correct"):
                        continue
                    corpus.append(entry)

    print(f"Loaded {len(corpus)} eligible entries")

    selected = random.sample(corpus, min(args.n_problems, len(corpus)))
    print(f"Selected {len(selected)} problems")

    total_calls = sum((len(e["sentences"]) + 1) * args.n_resamples for e in selected)
    est_cost = total_calls * 600 * 0.05 / 1e6 + total_calls * 500 * 0.40 / 1e6
    print(f"API calls: {total_calls}, est cost: ~${est_cost:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as session:
        tasks = [process_problem(session, entry, args.n_resamples) for entry in selected]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    good_results = []
    for entry, result in zip(selected, results):
        if isinstance(result, Exception):
            print(f"\n{entry['id']}: FAILED ({result})")
            continue
        good_results.append(result)

        n = result["n_sentences"]
        # Check API success rate
        total_valid = sum(t["n_valid"] for t in result["truncations"])
        total_slots = (n + 1) * args.n_resamples
        api_ok = total_valid / total_slots * 100

        # Print truncation curve
        no_cot = result["truncations"][0]["match_rate"]
        full_cot = result["truncations"][-1]["match_rate"]
        important = [s for s in result["sentence_importance"] if s["important"]]

        print(f"\n{result['id']} ({n} sent, {api_ok:.0f}% API ok)")
        print(f"  no_cot={no_cot:.0%} -> full_cot={full_cot:.0%}  "
              f"important: {[s['sentence_idx'] for s in important]}")

        for si in result["sentence_importance"]:
            marker = " ***" if si["important"] else ""
            print(f"  S{si['sentence_idx']:2d}: {si['match_rate_without']:.0%} -> "
                  f"{si['match_rate_with']:.0%} (delta={si['importance_delta']:+.0%}){marker}  "
                  f"| {si['sentence_text'][:70]}...")

    with open(args.output, "w") as f:
        for result in good_results:
            f.write(json.dumps(result) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! {len(good_results)} problems in {elapsed:.1f}s")
    print(f"Results: {args.output}")

    # Summary
    all_deltas = [s["importance_delta"] for r in good_results for s in r["sentence_importance"]]
    all_important = [s for r in good_results for s in r["sentence_importance"] if s["important"]]
    if all_deltas:
        import statistics
        print(f"\nSentences: {len(all_deltas)}")
        print(f"Important (delta > 30%): {len(all_important)}/{len(all_deltas)}")
        print(f"Mean delta: {statistics.mean(all_deltas):+.1%}")
        print(f"Median delta: {statistics.median(all_deltas):+.1%}")

    # API health
    total_valid = sum(t["n_valid"] for r in good_results for t in r["truncations"])
    total_slots = sum((r["n_sentences"] + 1) * args.n_resamples for r in good_results)
    print(f"\nAPI success: {total_valid}/{total_slots} ({total_valid/total_slots*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
