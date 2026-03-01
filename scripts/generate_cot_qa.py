#!/usr/bin/env python3
"""Generate diverse QA pairs about CoT reasoning using Gemini 2.5 Flash Lite.

Two question classes:
  - YES/NO: answer is literally "Yes" or "No" — fast classification questions
  - DETAILED: 1-3 sentence answers with specific grounded detail

Each CoT gets 1 yes/no persona + 1 detail persona (2 API calls, ~5-7 questions).
30% of CoTs are randomly truncated to simulate partial reasoning.

Usage:
    python scripts/generate_cot_qa.py --test          # 5 CoTs, print results
    python scripts/generate_cot_qa.py --n 12000        # Full run
    python scripts/generate_cot_qa.py --resume         # Resume from checkpoint
"""

import json
import os
import re
import time
import random
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SEED = 42
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

DATA_DIR = Path(__file__).parent.parent / "data"
MEDIUM_CORPUS = DATA_DIR / "cot_corpus_v5" / "corpus_medium.jsonl"
CONCEPT_CORPUS = DATA_DIR / "concept_corpus" / "corpus_full.jsonl"
OUT_PATH = DATA_DIR / "cleaned" / "cot_qa_generated.jsonl"
CHECKPOINT_PATH = DATA_DIR / "cleaned" / "cot_qa_checkpoint.jsonl"

MAX_COT_CHARS = 6000

# ---------------------------------------------------------------------------
# Shared rules
# ---------------------------------------------------------------------------

CORE_RULES = """\
RULES:
- Every question MUST be answerable just by reading the CoT. Zero external knowledge.
- Every answer MUST be verifiable against the CoT text.
- NEVER ask "What does the text say about X?" or similar paraphrase-requests.
- NEVER be vague or generic. Reference specific details from THIS reasoning.
- Questions should sound like a curious colleague, not a test paper.
- For each QA, include confidence (0-100) in how verifiable the answer is from the CoT alone.

BANNED question openings (do NOT use these):
"What type of reasoning...", "Is the model confident in its...", "What is the model's initial...",
"What intermediate result has the model...", "Does the model demonstrate metacognition..."
"""

# ---------------------------------------------------------------------------
# YES/NO personas — answer MUST be literally "Yes" or "No"
# ---------------------------------------------------------------------------

YESNO_PERSONAS = [
    {
        "id": "yn_specific",
        "temp": 0.8,
        "n": 3,
        "system": f"""You write yes/no questions about a language model's chain-of-thought (CoT).

Write {{n}} questions where the answer is EXACTLY "Yes" or "No" — no elaboration, no hedging.

Focus on SPECIFIC, CONCRETE details:
- "Does it compute the derivative before integrating?"
- "Is the Pythagorean theorem applied at any point?"
- "Does the model try x=0 as a test value?"
- "Is there a sign error in the third calculation step?"
- "Does the reasoning mention edge cases?"

Each question should target a DIFFERENT aspect. Mix positive and negative answers.

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "Yes", "c": 95}}, {{"q": "...", "a": "No", "c": 90}}]
"a" must be exactly "Yes" or "No". "c" = confidence 0-100.""",
    },
    {
        "id": "yn_behavior",
        "temp": 0.9,
        "n": 3,
        "system": f"""You write yes/no questions about HOW a language model reasons in its chain-of-thought (CoT).

Write {{n}} questions where the answer is EXACTLY "Yes" or "No".

Focus on REASONING BEHAVIOR:
- "Does the model backtrack or revise its approach?"
- "Is there any self-doubt expressed?"
- "Does it verify its answer before concluding?"
- "Is the reasoning linear or does it branch?"
- "Does the model consider more than one approach?"
- "Is there a moment where the model realizes a mistake?"

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "Yes", "c": 95}}, {{"q": "...", "a": "No", "c": 90}}]
"a" must be exactly "Yes" or "No". "c" = confidence 0-100.""",
    },
    {
        "id": "yn_content",
        "temp": 0.8,
        "n": 3,
        "system": f"""You write yes/no questions about WHAT a language model's reasoning covers.

Write {{n}} questions where the answer is EXACTLY "Yes" or "No".

Focus on content coverage — what topics, concepts, values, or cases appear:
- "Does the reasoning involve probability?"
- "Is any ethical consideration raised?"
- "Does the model reference a specific law or theorem by name?"
- "Are negative numbers involved in the calculation?"
- "Does the reasoning touch on multiple choice elimination?"
- "Is a counterexample constructed?"

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "Yes", "c": 95}}, {{"q": "...", "a": "No", "c": 90}}]
"a" must be exactly "Yes" or "No". "c" = confidence 0-100.""",
    },
]

# ---------------------------------------------------------------------------
# DETAILED personas — 1-3 sentence answers with specifics
# ---------------------------------------------------------------------------

DETAIL_PERSONAS = [
    {
        "id": "dt_strategy",
        "temp": 0.8,
        "n": 2,
        "system": f"""You analyze a language model's problem-solving STRATEGY in its chain-of-thought (CoT).

Write {{n}} questions with 1-3 sentence answers about methodology and approach.

Good questions:
- "What key insight unlocks the solution path?"
- "How does the approach shift after the first attempt fails?"
- "Which problem decomposition does the model choose?"

Answers must cite specific parts of the reasoning.

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "...", "c": 90}}]""",
    },
    {
        "id": "dt_specifics",
        "temp": 0.7,
        "n": 2,
        "system": f"""You extract precise FACTUAL DETAILS from a language model's chain-of-thought (CoT).

Write {{n}} questions with 1-2 sentence answers about concrete, checkable facts.

Good questions:
- "What numerical value does the second step produce?"
- "Which formula is applied to convert units?"
- "What specific option does the model eliminate first, and why?"
- "Name the two cases the model splits the problem into."

Answers should be precise enough to verify in 5 seconds of reading the CoT.

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "...", "c": 95}}]""",
    },
    {
        "id": "dt_critique",
        "temp": 0.9,
        "n": 2,
        "system": f"""You critically review a language model's chain-of-thought (CoT) reasoning quality.

Write {{n}} questions with 1-3 sentence answers about errors, self-corrections, and reasoning quality.

Good questions:
- "Where does the model catch and fix its own arithmetic error?"
- "What assumption does the model make without justification?"
- "How does the model's confidence change between the first and second approach?"
- "What logical gap exists between steps 2 and 3?"

If the reasoning is clean, ask about thoroughness, verification, or what's left unaddressed.

{CORE_RULES}

Return JSON: [{{"q": "...", "a": "...", "c": 85}}]""",
    },
]

# Domain hints
DOMAIN_HINTS = {
    "math": "Mathematical reasoning — calculations, proofs, formulas, numerical values are fair game.",
    "science": "Scientific reasoning — hypotheses, experiments, scientific concepts, causal logic.",
    "medical": "Medical/clinical reasoning — diagnosis, symptoms, treatment logic, differential diagnosis.",
    "commonsense": "Commonsense reasoning — social understanding, practical logic, everyday situations.",
    "bias": "Reasoning about sensitive/biased content — ethics, fairness, perspectives, sensitivity handling.",
    "diverse": "Varied topic — focus on reasoning structure and specific details.",
    "multi_domain": "Multi-domain reasoning — note which fields are involved and how they interact.",
}


def load_corpus() -> list[dict]:
    entries = []
    for path in [MEDIUM_CORPUS, CONCEPT_CORPUS]:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("cot_response", "").strip():
                    entries.append(d)
    print(f"Loaded {len(entries)} CoT entries")
    return entries


def truncate_cot(cot: str, max_chars: int = MAX_COT_CHARS) -> str:
    if len(cot) <= max_chars:
        return cot
    cut = cot[:max_chars]
    last_period = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
    if last_period > max_chars // 2:
        return cut[: last_period + 1]
    return cut + "..."


def randomly_truncate_cot(cot: str, rng: random.Random) -> tuple[str, bool]:
    """30% chance: truncate to 30-70% to simulate partial reasoning."""
    if rng.random() < 0.3:
        frac = rng.uniform(0.3, 0.7)
        target_len = int(len(cot) * frac)
        truncated = truncate_cot(cot, target_len)
        if len(truncated) < len(cot) - 50:
            return truncated, True
    return cot, False


def call_openrouter(messages: list[dict], temperature: float, retries: int = 3) -> str | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 600,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  API error: {e}")
                return None
    return None


def parse_qa_response(text: str, is_yesno: bool) -> list[dict]:
    """Parse JSON array from LLM response."""
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    def extract(raw):
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        valid = []
        for item in items:
            if not isinstance(item, dict):
                continue
            q = (item.get("q") or item.get("question", "")).strip()
            a = (item.get("a") or item.get("answer", "")).strip()
            c = item.get("c") or item.get("confidence", 80)
            if not q or not a:
                continue
            # For yes/no, normalize answer
            if is_yesno:
                a_lower = a.lower().strip().rstrip(".")
                if a_lower in ("yes", "no"):
                    a = a_lower.capitalize()
                else:
                    continue  # Skip non-yes/no answers
            try:
                c = int(c)
            except (ValueError, TypeError):
                c = 80
            valid.append({"question": q, "answer": a, "confidence": min(100, max(0, c))})
        return valid

    try:
        return extract(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return extract(match.group())
        except json.JSONDecodeError:
            pass
    return []


def process_one(entry: dict, rng: random.Random) -> list[dict]:
    """Generate QA pairs: 1 yes/no persona + 1 detail persona."""
    cot_full = truncate_cot(entry["cot_response"])
    question_text = entry.get("question", "")
    domain = entry.get("domain", "diverse")
    domain_hint = DOMAIN_HINTS.get(domain, DOMAIN_HINTS["diverse"])

    yn_persona = rng.choice(YESNO_PERSONAS)
    dt_persona = rng.choice(DETAIL_PERSONAS)

    results = []
    for persona, is_yesno in [(yn_persona, True), (dt_persona, False)]:
        cot, is_partial = randomly_truncate_cot(cot_full, rng)

        system = persona["system"].replace("{n}", str(persona["n"]))
        user_parts = []
        if is_partial:
            user_parts.append("NOTE: This is PARTIAL reasoning — the model hasn't finished. Frame questions about what's happened so far.\n")
        user_parts.append(f"Domain: {domain_hint}\n")
        user_parts.append(f"---\n{cot}\n---")
        if question_text:
            user_parts.append(f"\nOriginal question: {question_text}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        raw = call_openrouter(messages, persona["temp"])
        qa_pairs = parse_qa_response(raw, is_yesno)

        answer_type = "yes_no" if is_yesno else "detailed"
        for qa in qa_pairs:
            results.append({
                "corpus_id": entry["id"],
                "task_family": "cot_qa_conversational",
                "task_type": f"conv_{persona['id']}",
                "answer_type": answer_type,
                "prompt": qa["question"],
                "target_response": qa["answer"],
                "confidence": qa["confidence"],
                "source_domain": domain,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    print("Loading corpus...")
    entries = load_corpus()
    rng = random.Random(SEED)
    rng.shuffle(entries)

    if args.test:
        args.n = 5
        args.workers = 1

    entries = entries[: args.n]

    done_ids = set()
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add(d["corpus_id"])
        print(f"Resuming: {len(done_ids)} already processed")

    todo = [e for e in entries if e["id"] not in done_ids]
    print(f"Processing {len(todo)} CoTs ({len(done_ids)} already done)")
    print(f"Model: {MODEL}")

    all_results = []
    n_done = 0
    n_failed = 0
    t0 = time.time()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt_f = open(CHECKPOINT_PATH, "a")

    def make_rng(eid):
        return random.Random(f"{SEED}_{eid}")

    if args.test:
        for entry in todo:
            results = process_one(entry, make_rng(entry["id"]))
            if results:
                all_results.extend(results)
                for r in results:
                    ckpt_f.write(json.dumps(r) + "\n")
                    tag = "Y/N" if r["answer_type"] == "yes_no" else "DET"
                    print(f"\n  [{tag}|{r['task_type']}|c={r['confidence']}] Q: {r['prompt']}")
                    print(f"    A: {r['target_response'][:200]}")
            else:
                n_failed += 1
            n_done += 1
            print(f"  --- {n_done}/{len(todo)}, {len(all_results)} QA ---")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, e, make_rng(e["id"])): e for e in todo}
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        for r in results:
                            ckpt_f.write(json.dumps(r) + "\n")
                    else:
                        n_failed += 1
                except Exception as e:
                    n_failed += 1

                n_done += 1
                if n_done % 100 == 0 or n_done == len(todo):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(todo) - n_done) / rate if rate > 0 else 0
                    yn = sum(1 for r in all_results if r["answer_type"] == "yes_no")
                    dt = len(all_results) - yn
                    print(
                        f"  {n_done}/{len(todo)} ({n_done/len(todo)*100:.1f}%) "
                        f"| {len(all_results)} QA ({yn} Y/N, {dt} detail) "
                        f"| {n_failed} fail | {rate:.1f}/s | ETA {eta/60:.0f}m"
                    )
                    ckpt_f.flush()

    ckpt_f.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Processed: {n_done}, Failed: {n_failed}")
    yn = sum(1 for r in all_results if r["answer_type"] == "yes_no")
    dt = len(all_results) - yn
    print(f"  Generated: {len(all_results)} QA ({yn} yes/no, {dt} detailed, {len(all_results)/max(n_done,1):.1f}/CoT)")

    if not args.test:
        print(f"\nMerging into {OUT_PATH}...")
        all_rows = []
        if CHECKPOINT_PATH.exists():
            with open(CHECKPOINT_PATH) as f:
                for line in f:
                    all_rows.append(json.loads(line))
        seen = set()
        deduped = []
        for r in all_rows:
            key = (r["corpus_id"], r["prompt"][:80])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        with open(OUT_PATH, "w") as f:
            for r in deduped:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(deduped)} unique QA pairs")

    if all_results:
        types = Counter(r["task_type"] for r in all_results)
        domains = Counter(r["source_domain"] for r in all_results)
        confs = [r["confidence"] for r in all_results]
        print(f"\nConfidence: mean={sum(confs)/len(confs):.0f}, min={min(confs)}, max={max(confs)}")
        print("\nType distribution:")
        for t, c in types.most_common():
            print(f"  {t}: {c}")
        print("\nDomain distribution:")
        for d, c in domains.most_common():
            print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
