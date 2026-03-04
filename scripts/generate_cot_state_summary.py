#!/usr/bin/env python3
"""
Generate CoT State Summarization dataset.

At interesting cutoff points in chain-of-thought traces, generates natural language
summaries of what the model just realized and what it will do next.

Uses Gemini 3.1 Flash Lite via OpenRouter for label generation.
28 diverse prompt phrasings for robust conversational finetuning.

Usage:
    python scripts/generate_cot_state_summary.py --max-cots 10 --max-budget 0.50  # test
    python scripts/generate_cot_state_summary.py --max-budget 15.0  # full run
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

MODEL_ID = "google/gemini-3.1-flash-lite-preview"
INPUT_COST_PER_M = 0.075  # approximate (half of Gemini 3 Flash)
OUTPUT_COST_PER_M = 0.30
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 40

# ── Stats ──

completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def estimate_cost():
    return total_input_tokens * INPUT_COST_PER_M / 1e6 + total_output_tokens * OUTPUT_COST_PER_M / 1e6


# ── Diverse prompt pool (28 phrasings) ──

# Prompts grouped by theme. Each cutoff gets one theme for the API call,
# with the specific prompt passed to Gemini so the response matches.
# ~100 questions across 16 themes.
PROMPT_THEMES = {
    "realization": [
        "What did it just figure out?",
        "What was the most recent realization?",
        "What did the model just derive or conclude?",
        "What did the model just work out?",
        "What was just computed?",
        "What new result was just obtained?",
        "What did the last few steps establish?",
    ],
    "active_concepts": [
        "What is the model currently thinking about?",
        "What concepts are involved in the current step?",
        "What quantities and relationships is the model tracking?",
        "What variables matter right now?",
        "What information is the model using at this point?",
        "What facts are relevant to what's happening right now?",
        "What is the model holding in context?",
    ],
    "strategy": [
        "What strategy is being used here?",
        "What approach is the model taking?",
        "How is the model solving this part?",
        "What technique is being applied?",
        "Is this elimination, substitution, direct computation, or something else?",
        "What method did the model choose for this step?",
    ],
    "errors": [
        "Is the reasoning correct so far?",
        "Are there any errors or backtracking?",
        "Has the model caught a mistake?",
        "Is the model correcting something?",
        "Did it just notice an error?",
        "Is the model reconsidering a previous step?",
    ],
    "working_answer": [
        "What is the model's current best answer?",
        "What intermediate results have been found?",
        "What values has the model determined so far?",
        "What does the model currently think the answer is?",
        "What partial results exist at this point?",
        "What numbers or conclusions have been established?",
    ],
    "domain": [
        "What knowledge is being applied here?",
        "What subject area is this step about?",
        "What background knowledge is relevant?",
        "What does the model need to know for this step?",
        "What type of reasoning is happening — mathematical, logical, ethical, factual?",
        "What domain-specific facts is the model using?",
    ],
    "forward": [
        "What is the model about to do next?",
        "What will the next step be?",
        "What is the model going to try next?",
        "What comes after this?",
        "What is the immediate next step?",
        "What will the model do with the result it just got?",
    ],
    "backward": [
        "What just happened in the reasoning?",
        "Summarize the most recent step.",
        "What did those last few sentences accomplish?",
        "What was the purpose of the last few steps?",
        "What did the model just do?",
    ],
    "holistic": [
        "Describe the model's current reasoning state.",
        "What is going on in the reasoning right now?",
        "Summarize where the model is at this moment.",
        "What is the overall state of the problem-solving right now?",
        "What would a summary of the model's current thinking look like?",
        "What is happening in the reasoning at this exact point?",
    ],
    "focus": [
        "What is the model focused on right now?",
        "What part of the problem is the model currently working on?",
        "What specific aspect is being addressed?",
        "Which part of the problem is the model dealing with?",
        "What is the model paying attention to?",
    ],
    "subgoals": [
        "What is the model's current subgoal?",
        "What smaller problem is being solved right now?",
        "What is the immediate objective?",
        "What is the model trying to accomplish at this step?",
        "What specific thing is the model trying to determine?",
    ],
    "verification": [
        "Is the model checking its work?",
        "Is the model verifying a previous result?",
        "Is it double-checking something?",
        "Is the model confirming an earlier calculation?",
    ],
    "constraints": [
        "What constraints is the model tracking?",
        "What conditions must the answer satisfy?",
        "What restrictions apply to this problem?",
        "What requirements is the model keeping in mind?",
    ],
    "transition": [
        "Is the reasoning moving to a new phase?",
        "Did the model just switch to a different part of the problem?",
        "Did the approach just change?",
        "Is the model starting a new step?",
    ],
    "connections": [
        "What connections between facts are being made?",
        "What relationship was just identified?",
        "How are previous results being combined?",
        "Is the model linking separate results together?",
    ],
}

THEME_NAMES = list(PROMPT_THEMES.keys())

# ── Interesting cutoff heuristics ──

REALIZATION_PATTERNS = re.compile(
    r"(?:so\s+(?:we\s+(?:get|have|know|find|see)|the\s+answer|that\s+(?:means|gives))|"
    r"therefore|thus|hence|this\s+(?:means|gives|shows|implies|tells)|"
    r"which\s+(?:means|gives|equals|is)|"
    r"(?:=|equals)\s*-?\d|"
    r"we\s+(?:can\s+conclude|now\s+know|find\s+that|see\s+that|get\s+that)|"
    r"(?:the\s+)?(?:answer|result|solution|value)\s+is|"
    r"plugging\s+(?:back\s+)?in|substitut|"
    r"simplif(?:y|ies|ying)\s+to|"
    r"this\s+(?:equals|reduces|simplifies)\s+to)",
    re.IGNORECASE,
)

STRATEGY_SHIFT_PATTERNS = re.compile(
    r"(?:wait|actually|hmm|let\s+me\s+(?:try|think|reconsider|check|verify)|"
    r"alternatively|another\s+(?:way|approach)|"
    r"(?:but|however),?\s+(?:wait|that|this|if)|"
    r"I\s+(?:made\s+a\s+mistake|was\s+wrong|need\s+to\s+reconsider)|"
    r"let's\s+(?:try|go\s+back|reconsider|verify|check)|"
    r"on\s+second\s+thought|"
    r"no,?\s+that(?:'s|\s+is)?\s+(?:not|wrong))",
    re.IGNORECASE,
)


def find_interesting_cutoffs(
    sentences: list[str], max_points: int = 3, rng: random.Random = None
) -> list[int]:
    """Find interesting sentence indices to cut the CoT at.

    Returns 0-based sentence indices (cutoff is AFTER this sentence).
    Leaves at least 3 sentences before and 3 after.
    """
    n = len(sentences)
    if n < 8:
        return []

    min_idx = 2  # at least 3 sentences before (0, 1, 2)
    max_idx = n - 4  # at least 3 sentences after

    if min_idx > max_idx:
        return []

    # Score each candidate position
    scored = []
    for i in range(min_idx, max_idx + 1):
        sent = sentences[i]
        score = 0

        if REALIZATION_PATTERNS.search(sent):
            score += 2
        if STRATEGY_SHIFT_PATTERNS.search(sent):
            score += 3  # strategy shifts are especially interesting

        # Bonus for mid-reasoning (not too early/late)
        frac = i / n
        if 0.2 <= frac <= 0.8:
            score += 1

        if score > 0:
            scored.append((i, score))

    if not scored:
        # Fallback: evenly-spaced positions
        step = max(1, (max_idx - min_idx) // max_points)
        return list(range(min_idx, max_idx + 1, step))[:max_points]

    # Sort by score descending, spread them out
    scored.sort(key=lambda x: -x[1])

    selected = []
    for idx, _score in scored:
        # Minimum 3 sentences spacing between cutoffs
        if all(abs(idx - s) >= 3 for s in selected):
            selected.append(idx)
        if len(selected) >= max_points:
            break

    # If we found fewer than desired, fill with fallback positions
    if len(selected) < max_points:
        step = max(1, (max_idx - min_idx) // max_points)
        for i in range(min_idx, max_idx + 1, step):
            if all(abs(i - s) >= 3 for s in selected):
                selected.append(i)
            if len(selected) >= max_points:
                break

    selected.sort()
    return selected


# ── Label generation prompt ──

SYSTEM_PROMPT = """\
## What we're building

We're training a "chain-of-thought oracle" that reads the internal activations of a language model mid-reasoning. It can be asked many different kinds of questions about what's happening inside the model at any point: what it just figured out, what strategy it's using, how confident it is, what concepts are active, what it's about to do, etc.

You're generating training labels. You'll see a chain of thought cut at a specific point, the next few sentences, AND a specific question. Answer THAT question.

## Rules

ANSWER THE QUESTION: Different questions need different answers. "What strategy is being used?" needs a different answer than "What was just computed?" Match your response to what's asked.

POSITION FOCUS: Your answer should be primarily about what's happening RIGHT at the cutoff — the last 1-2 sentences before it. Earlier context matters less the further back it is. Don't summarize the whole trace. The cutoff point is what matters most.

NUMBERS: Preserve exact values — "got 42", "r = 5", "P = 3/10". Never omit or round.

BREVITY: 1-2 sentences max. Punchy. No filler.

NOT TEXT INVERSION: Compress, don't rephrase. Say the point, not the process.

OBVIOUS IMPLICATIONS OK: If base=10, height=5, say "area is 50" even if not computed yet.

## Examples by question type

Question: "What was just computed?"
→ "Got x = 7 by substituting into equation 2."

Question: "What strategy is being used?"
→ "Elimination — subtracting the equations to cancel y."

Question: "How confident is the model?"
→ "Pretty sure about 42, but double-checking because the options don't include it."

Question: "What concepts are active right now?"
→ "Conditional probability without replacement — tracking how the sample space shrinks."

Question: "What's the current best guess at the answer?"
→ "Leaning toward option D (150 km) but hasn't verified yet."

Question: "Any errors or backtracking?"
→ "Caught that the denominator should be 2n, not n. Redoing."

Question: "What's the model about to do?"
→ "Plug x = 7 back into the original equation to verify."

Question: "What's present in the model's reasoning right now?"
→ "The 3:2 ratio, the total of 180 students, and the need to split by gender."

Question: "What knowledge is being applied?"
→ "Combinatorics — using C(8,2) = 28 to count unordered pairs."

Question: "Describe what's going on in the model's head right now."
→ "Stuck between two answer choices. Testing A with a boundary case to see if it holds."

BAD — don't write like this:
- "The model has been working through the algebra and is now preparing to..." (wordy, vague)
- "It computed several values and will continue solving." (nothing specific)\""""


def build_label_prompt(question: str, cot_prefix: str, next_sentences: str, oracle_question: str) -> str:
    return (
        f"## Original Problem\n{question}\n\n"
        f"## Chain of thought up to cutoff\n{cot_prefix}\n\n"
        f"## Next few sentences (what actually comes next)\n{next_sentences}\n\n"
        f"## Question to answer\n{oracle_question}"
    )


# ── API call ──


async def call_openrouter(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_budget: float,
    max_tokens: int = 300,
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
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(4):
            try:
                resp = await client.post(
                    ENDPOINT, json=body, headers=headers, timeout=90
                )

                if resp.status_code == 429:
                    await asyncio.sleep(2**attempt + random.random())
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
                    await asyncio.sleep(2**attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()

                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                completed += 1
                if completed % 200 == 0:
                    print(
                        f"  {completed} calls done, {failed} failed, ~${estimate_cost():.3f} spent"
                    )

                if estimate_cost() > max_budget * 0.95:
                    budget_exceeded = True
                    print(
                        f"\n*** Approaching budget limit (${estimate_cost():.2f}/{max_budget:.2f}). Stopping. ***"
                    )

                return content, in_tok, out_tok

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as e:
                if attempt == 3:
                    failed += 1
                    if failed <= 10:
                        print(f"  Error: {e}")
                    return "", 0, 0
                await asyncio.sleep(2**attempt)

    return "", 0, 0


# ── Data loading ──


def load_corpus(source: str, min_sentences: int = 8) -> list[dict]:
    """Load corpus from local JSONL or HuggingFace."""
    if os.path.exists(source):
        print(f"Loading from local file: {source}")
        entries = []
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("n_sentences", 0) < min_sentences:
                    continue
                entries.append(entry)
        return entries
    else:
        print(f"Loading from HuggingFace: {source}")
        from datasets import load_dataset

        ds = load_dataset(source, split="train")
        entries = []
        for row in ds:
            if row.get("n_sentences", 0) < min_sentences:
                continue
            entries.append(dict(row))
        return entries


# ── Main pipeline ──


async def generate_labels(
    tasks: list[tuple[str, str]],
    api_key: str,
    max_budget: float,
) -> list[str]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(
        max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY
    )

    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, sys_p, user_p, api_key, semaphore, max_budget)
            for sys_p, user_p in tasks
        ]
        results = await asyncio.gather(*coros)

    return [r[0] for r in results]


def main():
    global budget_exceeded

    parser = argparse.ArgumentParser(
        description="Generate CoT state summarization dataset"
    )
    parser.add_argument(
        "--corpus",
        default="ceselder/cot-oracle-corpus-v5",
        help="HF dataset ID or local JSONL path",
    )
    parser.add_argument(
        "--output", default="data/cot_state_summary/cot_state_summary.jsonl"
    )
    parser.add_argument(
        "--max-cots", type=int, default=None, help="Limit corpus entries (test runs)"
    )
    parser.add_argument("--max-points-per-cot", type=int, default=3)
    parser.add_argument("--paraphrases-per-point", type=int, default=3,
                        help="Number of different prompt phrasings per cutoff (each gets its own API call)")
    parser.add_argument("--max-budget", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set, using key from config")
        api_key = "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"

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Load corpus ──
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} entries with ≥8 sentences")

    source_counts = Counter(e.get("source", "unknown") for e in corpus)
    for s, c in sorted(source_counts.items()):
        print(f"  {s}: {c}")

    if args.max_cots:
        rng.shuffle(corpus)
        corpus = corpus[: args.max_cots]
        print(f"\nLimited to {len(corpus)} CoTs")

    # ── Find cutoff points + generate paraphrased prompts ──
    print("\nFinding interesting cutoff points...")
    cutoff_items = []
    n_cutoffs = 0

    for entry in corpus:
        sentences = entry["sentences"]
        cutoffs = find_interesting_cutoffs(
            sentences, max_points=args.max_points_per_cot, rng=rng
        )

        for sent_idx in cutoffs:
            n_cutoffs += 1
            prefix = " ".join(sentences[: sent_idx + 1])
            next_end = min(sent_idx + 4, len(sentences))
            next_sents = " ".join(sentences[sent_idx + 1 : next_end])

            # Pick paraphrases-per-point different prompts from different themes
            themes_for_point = rng.sample(
                THEME_NAMES, min(args.paraphrases_per_point, len(THEME_NAMES))
            )
            for theme in themes_for_point:
                prompt = rng.choice(PROMPT_THEMES[theme])
                cutoff_items.append(
                    {
                        "entry": entry,
                        "sent_idx": sent_idx,
                        "cot_prefix": prefix,
                        "next_sentences": next_sents,
                        "prompt": prompt,
                        "theme": theme,
                    }
                )

    print(f"Found {n_cutoffs} cutoff points across {len(corpus)} CoTs")
    print(f"  × {args.paraphrases_per_point} paraphrases = {len(cutoff_items)} API calls")

    # ── Generate labels ──
    print(f"\n{'='*60}\nGenerating labels ({len(cutoff_items)} API calls)...")

    api_tasks = []
    for item in cutoff_items:
        question = item["entry"].get("question", item["entry"].get("prompt", ""))
        user_prompt = build_label_prompt(
            question, item["cot_prefix"], item["next_sentences"], item["prompt"]
        )
        api_tasks.append((SYSTEM_PROMPT, user_prompt))

    responses = asyncio.run(generate_labels(api_tasks, api_key, args.max_budget))

    # ── Assemble dataset ──
    rows = []
    n_failed = 0
    for item, resp in zip(cutoff_items, responses):
        if not resp or len(resp) < 20:
            n_failed += 1
            continue

        entry = item["entry"]
        n_sents = entry.get("n_sentences", len(entry["sentences"]))

        rows.append(
            {
                "task": "cot_state_summary",
                "prompt": item["prompt"],
                "theme": item["theme"],
                "cot_text": item["cot_prefix"],
                "target_response": resp,
                "question": entry.get("question", entry.get("prompt", "")),
                "sent_idx": item["sent_idx"],
                "total_sentences": n_sents,
                "pct": round(100 * item["sent_idx"] / n_sents),
                "cot_id": entry.get("id", ""),
                "source": entry.get("source", ""),
            }
        )

    # ── Save ──
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    elapsed = time.time() - t0
    cost = estimate_cost()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Output: {output_path}")
    print(f"  Total rows: {len(rows)} ({n_failed} failed)")
    print(f"  API calls: {completed} completed, {failed} failed")
    print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
    print(f"  Cost: ~${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    if rows:
        source_dist = Counter(r["source"] for r in rows)
        print(f"\n  By source:")
        for src, cnt in sorted(source_dist.items()):
            print(f"    {src}: {cnt}")

        pcts = [r["pct"] for r in rows]
        print(
            f"\n  Cutoff position: mean={sum(pcts)/len(pcts):.0f}%, "
            f"min={min(pcts)}%, max={max(pcts)}%"
        )

        # Show a few examples
        print(f"\n  Example outputs:")
        samples = rng.sample(rows, min(3, len(rows)))
        for ex in samples:
            print(
                f"    [{ex['source']}] sent {ex['sent_idx']}/{ex['total_sentences']} ({ex['pct']}%)"
            )
            print(f"    Prompt: {ex['prompt']}")
            print(f"    Response: {ex['target_response'][:200]}...")
            print()


if __name__ == "__main__":
    main()
