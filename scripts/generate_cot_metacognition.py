#!/usr/bin/env python3
"""
Generate CoT Metacognition evaluation dataset.

Tests whether the activation oracle can detect the supervisee model's internal
metacognitive states: uncertainty, backtracking, error recognition, alternative
paths, missing information, and number prediction.

Uses real Qwen3-8B CoT traces with Gemini-generated ground-truth labels.

Usage:
    python scripts/generate_cot_metacognition.py --n 20 --dry-run          # check prompts
    python scripts/generate_cot_metacognition.py --n 50 --max-budget 0.50  # small test
    python scripts/generate_cot_metacognition.py --max-budget 5.0          # full run
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent, indent

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_pipeline"))
from cot_utils import split_cot_into_sentences
from generate_cots import (
    load_math_problems,
    load_gsm8k_problems,
    load_bbh_problems,
    load_logiqa_problems,
)

# ── Config ──

ROLLOUT_MODEL = "qwen/qwen3-8b"
LABEL_MODEL = "google/gemini-3.1-flash-lite-preview"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 40
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30
HF_ORG = "mats-10-sprint-cs-jb"
HF_REPO = f"{HF_ORG}/cot-oracle-eval-cot-metacognition"

# ── Stats ──

completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def estimate_cost():
    return total_input_tokens * INPUT_COST_PER_M / 1e6 + total_output_tokens * OUTPUT_COST_PER_M / 1e6


# ── Metacognition patterns (extends strategy shift patterns) ──

METACOG_PATTERNS = re.compile(
    r"(?:wait|actually|hmm|let\s+me\s+(?:try|think|reconsider|check|verify)|"
    r"I\s+(?:made\s+a\s+mistake|was\s+wrong|need\s+to\s+reconsider|think|'m\s+not\s+sure)|"
    r"(?:but|however),?\s+(?:wait|that|this|if)|"
    r"no,?\s+that(?:'s|\s+is)?\s+(?:not|wrong)|"
    r"let's\s+(?:try|go\s+back|reconsider|verify|check)|"
    r"on\s+second\s+thought|alternatively|"
    r"(?:not\s+sure|uncertain|unsure)\s+(?:if|about|whether)|"
    r"(?:could\s+(?:also|alternatively)|or\s+(?:maybe|perhaps|we\s+could))|"
    r"(?:missing|don't\s+(?:know|have))\s+(?:the|any|enough)|"
    r"(?:I\s+)?(?:need|still\s+need)\s+to\s+(?:figure|find|determine))",
    re.IGNORECASE,
)

# ── Category-specific question prompts ──

CATEGORY_PROMPTS = {
    "uncertainty": [
        "What specifically is the model uncertain about at this point?",
        "What is the model not sure about right now?",
        "What doubt or uncertainty is the model experiencing?",
    ],
    "backtracking": [
        "Why is the model about to change its approach?",
        "What is causing the model to reconsider its strategy?",
        "What triggered the model's change of direction?",
    ],
    "error_recognition": [
        "What specific error has the model caught or is catching?",
        "What mistake did the model just notice?",
        "What error is the model recognizing in its reasoning?",
    ],
    "alternative_paths": [
        "What alternatives is the model weighing right now?",
        "What different approaches is the model considering?",
        "What options is the model deciding between?",
    ],
    "missing_info": [
        "What information does the model lack for this problem?",
        "What is the model missing that it needs to solve this?",
    ],
    "number_prediction": [
        "What exact number is the model about to output next?",
        "What numerical value will the model state next?",
    ],
}

# ── Arya's hardcoded example problems ──

ARYA_PROBLEMS = [
    # Recursive arithmetic
    {"source": "arya", "question": "What is ((7 * 3) + 5) * 2 - 4?", "correct_answer": "48", "subject": "arithmetic", "arya_example": True},
    {"source": "arya", "question": "Calculate: 15! / (13! * 2!)", "correct_answer": "105", "subject": "combinatorics", "arya_example": True},
    # Circular permutations
    {"source": "arya", "question": "In how many ways can 5 people sit around a circular table if 2 specific people must sit next to each other?", "correct_answer": "12", "subject": "combinatorics", "arya_example": True},
    {"source": "arya", "question": "How many distinct necklaces can be made from 3 red beads and 4 blue beads?", "correct_answer": "3", "subject": "combinatorics", "arya_example": True},
    # Truth-teller / liar
    {"source": "arya", "question": "On an island, person A says 'I am a liar'. Person B says 'A is telling the truth'. How many liars are there?", "correct_answer": "1", "subject": "logic", "arya_example": True},
    {"source": "arya", "question": "Three people: A says B is a liar, B says C is a liar, C says both A and B are liars. If exactly one is a truth-teller, who is it?", "correct_answer": "B", "subject": "logic", "arya_example": True},
    # Ice cube melting
    {"source": "arya", "question": "An ice cube with side length 10cm is melting. After losing 48.8% of its volume, what is the new side length?", "correct_answer": "8", "subject": "geometry", "arya_example": True},
    # Swimming pool (missing info pairs)
    {"source": "arya", "question": "A rectangular swimming pool is 25m long and 10m wide. How long does it take to fill it?", "correct_answer": "Cannot be determined — the depth and flow rate are not given.", "subject": "missing_info", "arya_example": True, "category_hint": "missing_info"},
    {"source": "arya", "question": "A rectangular swimming pool is 25m long, 10m wide, and 2m deep. Water flows in at 5 cubic meters per minute. How long does it take to fill?", "correct_answer": "100 minutes", "subject": "missing_info", "arya_example": True, "category_hint": "missing_info", "control_pair_id": "pool_complete"},
    # Simple arithmetic (number prediction)
    {"source": "arya", "question": "What is 17 * 23?", "correct_answer": "391", "subject": "arithmetic", "arya_example": True, "category_hint": "number_prediction"},
    {"source": "arya", "question": "What is 256 + 189?", "correct_answer": "445", "subject": "arithmetic", "arya_example": True, "category_hint": "number_prediction"},
    {"source": "arya", "question": "What is 144 / 12?", "correct_answer": "12", "subject": "arithmetic", "arya_example": True, "category_hint": "number_prediction"},
    # Probability with uncertainty
    {"source": "arya", "question": "A bag contains 5 red and 3 blue balls. Two balls are drawn without replacement. What's the probability both are red?", "correct_answer": "5/14", "subject": "probability", "arya_example": True},
    {"source": "arya", "question": "In a game, you roll two dice. What's the probability of getting a sum of 7 or 11?", "correct_answer": "8/36 = 2/9", "subject": "probability", "arya_example": True},
    # Multi-step with potential errors
    {"source": "arya", "question": "Find all integer solutions to x^2 - 5x + 6 = 0.", "correct_answer": "x = 2, x = 3", "subject": "algebra", "arya_example": True},
    {"source": "arya", "question": "A train travels at 60 km/h for 2 hours, then at 90 km/h for 3 hours. What is the average speed for the entire journey?", "correct_answer": "78 km/h", "subject": "arithmetic", "arya_example": True},
    # Backtracking triggers
    {"source": "arya", "question": "Prove that the sum of the first n odd numbers equals n^2.", "correct_answer": "By induction: base case 1=1^2. If sum of first k odds = k^2, then adding (2k+1) gives k^2+2k+1=(k+1)^2.", "subject": "proof", "arya_example": True},
    {"source": "arya", "question": "How many 4-digit numbers have exactly two identical digits?", "correct_answer": "Requires careful case analysis", "subject": "combinatorics", "arya_example": True},
    # Geometry with alternative approaches
    {"source": "arya", "question": "Find the area of a triangle with vertices at (0,0), (4,0), and (2,3).", "correct_answer": "6", "subject": "geometry", "arya_example": True},
    {"source": "arya", "question": "What is the shortest distance from the point (3,4) to the line y = 2x + 1?", "correct_answer": "2/sqrt(5)", "subject": "geometry", "arya_example": True},
]


# ── Label generation system prompt ──

LABEL_SYSTEM_PROMPT = """\
You're generating ground-truth labels for a metacognition detection dataset.

You'll see a chain-of-thought cut at a specific point, the next few sentences, AND a specific \
question about the model's metacognitive state. Answer THAT question.

## Critical rules

SPECIFICITY IS EVERYTHING. Bad labels ruin the dataset. Follow these rules exactly:

BAD (vague, useless):
- "The model is uncertain about its calculations"
- "The model is reconsidering its approach"
- "The model caught an error in its work"
- "The model is thinking about the problem"

GOOD (specific, falsifiable):
- "Uncertain whether 17*23 = 391 since (20-3)(20+3) = 400-9 = 391 seems too clean"
- "Switching from brute force to generating functions because C(8,2) overcounts rotations"
- "Caught that n! should be (n-1)! for circular permutations — off by factor of n"
- "Weighing direct substitution vs. matrix method for the 3x3 system"

INCLUDE EXACT NUMBERS: If the model wrote "5 * 7 = 35" or "x = 12", include those values.
INCLUDE VARIABLE NAMES: If the model is solving for r, say "r" not "the variable".
INCLUDE THE SPECIFIC STEP: Don't say "a calculation" — say which calculation.

## Response format
1-2 sentences max. Punchy and specific. No filler words."""


def build_label_prompt(question: str, cot_prefix: str, next_sentences: str, oracle_question: str, category: str) -> str:
    cat_hint = f"\n\nCategory hint: this should capture {category.replace('_', ' ')}." if category else ""
    return (
        f"## Original Problem\n{question}\n\n"
        f"## Chain of thought up to cutoff\n{cot_prefix}\n\n"
        f"## Next few sentences (what actually comes next)\n{next_sentences}\n\n"
        f"## Question to answer\n{oracle_question}{cat_hint}"
    )


# ── Async API calls ──

async def call_openrouter(
    client: httpx.AsyncClient, model: str, system_prompt: str, user_prompt: str,
    api_key: str, semaphore: asyncio.Semaphore, max_budget: float,
    max_tokens: int = 300, temperature: float = 0.7,
) -> str:
    global completed, failed, budget_exceeded, total_input_tokens, total_output_tokens

    if budget_exceeded:
        return ""

    async with semaphore:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(4):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=90)
                if resp.status_code == 429:
                    await asyncio.sleep(2**attempt + random.random())
                    continue
                if resp.status_code == 402:
                    budget_exceeded = True
                    print("\n*** Budget exceeded (402). Stopping. ***")
                    return ""
                if resp.status_code != 200:
                    if attempt == 3:
                        failed += 1
                        if failed <= 10:
                            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                        return ""
                    await asyncio.sleep(2**attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                usage = data.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

                completed += 1
                if completed % 200 == 0:
                    print(f"  {completed} calls done, {failed} failed, ~${estimate_cost():.3f} spent")

                if estimate_cost() > max_budget * 0.95:
                    budget_exceeded = True
                    print(f"\n*** Approaching budget (${estimate_cost():.2f}/{max_budget:.2f}). Stopping. ***")

                return content

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as e:
                if attempt == 3:
                    failed += 1
                    if failed <= 10:
                        print(f"  Error: {e}")
                    return ""
                await asyncio.sleep(2**attempt)

    return ""


async def batch_api_calls(tasks: list[tuple[str, str, str]], api_key: str, max_budget: float,
                          max_tokens: int = 300, temperature: float = 0.7) -> list[str]:
    """Run batch of (model, system_prompt, user_prompt) calls."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, model, sys_p, user_p, api_key, semaphore, max_budget, max_tokens, temperature)
            for model, sys_p, user_p in tasks
        ]
        return await asyncio.gather(*coros)


# ── Cutoff finding (extends find_interesting_cutoffs with metacognition patterns) ──

def find_metacog_cutoffs(sentences: list[str], max_points: int = 3) -> list[tuple[int, str]]:
    """Find cutoff points where metacognition is likely happening.

    Returns list of (sentence_idx, category_hint) tuples.
    """
    n = len(sentences)
    if n < 8:
        return []

    min_idx = 2
    max_idx = n - 4

    if min_idx > max_idx:
        return []

    candidates = []
    for i in range(min_idx, max_idx + 1):
        sent = sentences[i]
        sent_lower = sent.lower()
        next_sent = sentences[i + 1].lower() if i + 1 < n else ""
        combined = sent_lower + " " + next_sent

        score = 0
        category = None

        # Uncertainty signals
        if re.search(r"(?:not\s+sure|uncertain|unsure|might\s+be|could\s+be|I\s+think|probably|maybe)", combined, re.IGNORECASE):
            score += 2
            category = "uncertainty"

        # Backtracking signals
        if re.search(r"(?:wait|actually|no[,.]?\s+that|let\s+me\s+(?:try|reconsider)|on\s+second\s+thought)", combined, re.IGNORECASE):
            score += 3
            if category is None:
                category = "backtracking"

        # Error recognition
        if re.search(r"(?:mistake|wrong|error|incorrect|should\s+(?:be|have\s+been)|oops|that's\s+not\s+right)", combined, re.IGNORECASE):
            score += 3
            category = "error_recognition"

        # Alternative paths
        if re.search(r"(?:alternatively|another\s+(?:way|approach)|or\s+(?:we\s+could|I\s+could|maybe)|instead\s+of|could\s+also)", combined, re.IGNORECASE):
            score += 2
            if category is None:
                category = "alternative_paths"

        # Mid-reasoning bonus
        frac = i / n
        if 0.2 <= frac <= 0.8:
            score += 1

        if score > 0 and category:
            candidates.append((i, score, category))

    if not candidates:
        # Fallback: evenly spaced with "uncertainty" category
        step = max(1, (max_idx - min_idx) // max_points)
        return [(i, "uncertainty") for i in range(min_idx, max_idx + 1, step)][:max_points]

    candidates.sort(key=lambda x: -x[1])

    selected = []
    for idx, _score, category in candidates:
        if all(abs(idx - s[0]) >= 3 for s in selected):
            selected.append((idx, category))
        if len(selected) >= max_points:
            break

    selected.sort(key=lambda x: x[0])
    return selected


def extract_number_from_continuation(next_text: str) -> str | None:
    """Extract the first concrete number from the continuation text."""
    # Match numbers including decimals, fractions, negatives
    m = re.search(r'(?:=\s*|is\s+|equals?\s+|get\s+)(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)', next_text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(-?\d+(?:\.\d+)?)', next_text)
    if m:
        return m.group(1).strip()
    return None


# ── Vagueness filter ──

VAGUENESS_PATTERNS = re.compile(
    r"(?:the\s+model\s+is\s+(?:processing|thinking|reasoning|working)|"
    r"considering\s+(?:the|its)\s+(?:options|approach)|"
    r"analyzing\s+the\s+problem|"
    r"performing\s+calculations|"
    r"working\s+through\s+the\s+(?:math|problem|question)|"
    r"the\s+model\s+(?:is\s+)?(?:engaged\s+in|currently)|"
    r"continuing\s+(?:its|the)\s+(?:reasoning|analysis))",
    re.IGNORECASE,
)


def is_vague(label: str) -> bool:
    if len(label) < 20:
        return True
    return bool(VAGUENESS_PATTERNS.search(label))


# ── Phase 0: Problem collection ──

def collect_problems(n_per_source: int, seed: int) -> list[dict]:
    """Collect problems from MATH, GSM8K, BBH, LogiQA + Arya examples."""
    rng = random.Random(seed)
    problems = []

    print("Phase 0: Collecting problems...")
    for loader_name, loader_fn in [("MATH", lambda n: load_math_problems(n)), ("GSM8K", lambda n: load_gsm8k_problems(n)),
                                    ("BBH", lambda n: load_bbh_problems(n)), ("LogiQA", lambda n: load_logiqa_problems(n))]:
        loaded = loader_fn(n_per_source)
        print(f"  {loader_name}: {len(loaded)}")
        problems.extend(loaded)

    # Add Arya examples
    problems.extend(ARYA_PROBLEMS)
    print(f"  Arya examples: {len(ARYA_PROBLEMS)}")

    rng.shuffle(problems)
    print(f"  Total problems: {len(problems)}")
    return problems


# ── Phase 1: Generate rollouts ──

async def generate_rollouts(problems: list[dict], api_key: str, max_budget: float) -> list[dict]:
    """Generate Qwen3-8B CoT rollouts via OpenRouter."""
    print(f"\nPhase 1: Generating rollouts for {len(problems)} problems...")

    rollout_system = "Think step by step. Show all your work."
    tasks = [
        (ROLLOUT_MODEL, rollout_system, p["question"])
        for p in problems
    ]

    responses = await batch_api_calls(tasks, api_key, max_budget, max_tokens=2048, temperature=0.7)

    results = []
    for problem, response in zip(problems, responses):
        if not response or len(response) < 50:
            continue
        results.append({**problem, "cot_response": response})

    print(f"  Got {len(results)} valid rollouts (from {len(problems)} problems)")
    return results


# ── Phase 2: Identify probe points & generate labels ──

async def generate_labels(rollouts: list[dict], api_key: str, max_budget: float, seed: int) -> list[dict]:
    """Find metacognition probe points and generate specific labels."""
    rng = random.Random(seed)
    print(f"\nPhase 2: Finding probe points and generating labels...")

    label_tasks = []
    label_meta = []

    for rollout in rollouts:
        cot = rollout["cot_response"]
        sentences = split_cot_into_sentences(cot)
        is_arya = rollout.get("arya_example", False)
        category_hint = rollout.get("category_hint")

        if category_hint == "number_prediction":
            # For number prediction: cut before a computation result
            cutoffs = find_metacog_cutoffs(sentences, max_points=2)
            if not cutoffs:
                # Fallback: use middle of CoT
                mid = len(sentences) // 2
                if mid >= 2 and mid < len(sentences) - 3:
                    cutoffs = [(mid, "number_prediction")]
                else:
                    continue

            for sent_idx, _cat in cutoffs:
                prefix = " ".join(sentences[:sent_idx + 1])
                next_end = min(sent_idx + 4, len(sentences))
                next_sents = " ".join(sentences[sent_idx + 1:next_end])

                # Try to extract target number from continuation
                target_number = extract_number_from_continuation(next_sents)
                if target_number:
                    prompt_q = rng.choice(CATEGORY_PROMPTS["number_prediction"])
                    label_tasks.append((LABEL_MODEL, LABEL_SYSTEM_PROMPT,
                                       build_label_prompt(rollout["question"], prefix, next_sents, prompt_q, "number_prediction")))
                    label_meta.append({
                        "rollout": rollout, "sent_idx": sent_idx, "category": "number_prediction",
                        "prompt": prompt_q, "prefix": prefix, "next_sents": next_sents,
                        "precomputed_target": target_number,
                    })

        elif category_hint == "missing_info":
            # For missing info: cut at uncertainty point
            cutoffs = find_metacog_cutoffs(sentences, max_points=2)
            if not cutoffs:
                mid = len(sentences) // 2
                if mid >= 2 and mid < len(sentences) - 3:
                    cutoffs = [(mid, "missing_info")]
                else:
                    continue

            for sent_idx, _cat in cutoffs:
                prefix = " ".join(sentences[:sent_idx + 1])
                next_end = min(sent_idx + 4, len(sentences))
                next_sents = " ".join(sentences[sent_idx + 1:next_end])
                prompt_q = rng.choice(CATEGORY_PROMPTS["missing_info"])
                label_tasks.append((LABEL_MODEL, LABEL_SYSTEM_PROMPT,
                                   build_label_prompt(rollout["question"], prefix, next_sents, prompt_q, "missing_info")))
                label_meta.append({
                    "rollout": rollout, "sent_idx": sent_idx, "category": "missing_info",
                    "prompt": prompt_q, "prefix": prefix, "next_sents": next_sents,
                })
        else:
            # General metacognition categories
            cutoffs = find_metacog_cutoffs(sentences, max_points=3)
            for sent_idx, detected_category in cutoffs:
                prefix = " ".join(sentences[:sent_idx + 1])
                next_end = min(sent_idx + 4, len(sentences))
                next_sents = " ".join(sentences[sent_idx + 1:next_end])
                prompt_q = rng.choice(CATEGORY_PROMPTS[detected_category])
                label_tasks.append((LABEL_MODEL, LABEL_SYSTEM_PROMPT,
                                   build_label_prompt(rollout["question"], prefix, next_sents, prompt_q, detected_category)))
                label_meta.append({
                    "rollout": rollout, "sent_idx": sent_idx, "category": detected_category,
                    "prompt": prompt_q, "prefix": prefix, "next_sents": next_sents,
                })

    print(f"  Found {len(label_tasks)} probe points, calling {LABEL_MODEL}...")

    if not label_tasks:
        return []

    responses = await batch_api_calls(label_tasks, api_key, max_budget)

    examples = []
    for meta, response in zip(label_meta, responses):
        if not response:
            continue

        # For number prediction, use precomputed target
        if meta["category"] == "number_prediction" and "precomputed_target" in meta:
            target = meta["precomputed_target"]
        else:
            target = response

        if is_vague(target):
            continue

        rollout = meta["rollout"]
        examples.append({
            "task": "cot_metacognition",
            "prompt": meta["prompt"],
            "target_response": target,
            "cot_text": meta["prefix"],
            "category": meta["category"],
            "domain": rollout.get("subject", ""),
            "source": rollout.get("source", ""),
            "arya_example": rollout.get("arya_example", False),
            "control_pair_id": rollout.get("control_pair_id"),
            "question": rollout["question"],
            "sent_idx": meta["sent_idx"],
        })

    print(f"  Generated {len(examples)} labeled examples ({len(label_tasks) - len(examples)} filtered)")
    return examples


# ── Phase 3: Filter, balance & save ──

CATEGORY_TARGETS = {
    "uncertainty": 250,
    "backtracking": 250,
    "error_recognition": 150,
    "alternative_paths": 150,
    "missing_info": 100,
    "number_prediction": 100,
}


def balance_and_save(examples: list[dict], output_path: Path, seed: int):
    """Balance across categories and save."""
    rng = random.Random(seed)
    print(f"\nPhase 3: Balancing and saving...")

    # Group by category
    by_cat = defaultdict(list)
    for ex in examples:
        by_cat[ex["category"]].append(ex)

    print("  Category counts (raw):")
    for cat, items in sorted(by_cat.items()):
        n_arya = sum(1 for it in items if it.get("arya_example"))
        print(f"    {cat}: {len(items)} ({n_arya} arya)")

    # Balance: take up to target per category
    balanced = []
    for cat, target_n in CATEGORY_TARGETS.items():
        items = by_cat.get(cat, [])
        # Always keep Arya examples
        arya = [it for it in items if it.get("arya_example")]
        non_arya = [it for it in items if not it.get("arya_example")]
        rng.shuffle(non_arya)
        remaining = max(0, target_n - len(arya))
        balanced.extend(arya)
        balanced.extend(non_arya[:remaining])

    # Also include any "text_inversion_control" examples
    if "text_inversion_control" in by_cat:
        balanced.extend(by_cat["text_inversion_control"][:200])

    rng.shuffle(balanced)

    print(f"\n  Balanced: {len(balanced)} examples")
    cat_counts = Counter(ex["category"] for ex in balanced)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in balanced:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved to {output_path}")

    return balanced


def load_rows(output_path: Path) -> list[dict]:
    rows = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_counter_table(counter: Counter) -> str:
    total = sum(counter.values())
    lines = ["| Value | Count | Share |", "|---|---:|---:|"]
    for value, count in counter.most_common():
        lines.append(f"| `{value}` | {count} | {100 * count / total:.1f}% |")
    return "\n".join(lines)


def build_dataset_card(rows: list[dict]) -> str:
    total = len(rows)
    category_counts = Counter(row["category"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)
    arya_count = sum(row["arya_example"] for row in rows)
    avg_sent_idx = sum(row["sent_idx"] for row in rows) / total
    avg_cot_chars = sum(len(row["cot_text"]) for row in rows) / total
    top_domains = ", ".join(f"`{name}` ({count})" for name, count in domain_counts.most_common(5))

    return dedent(
        f"""\
        ---
        pretty_name: CoT Oracle Eval - CoT Metacognition
        license: mit
        language:
          - en
        size_categories:
          - n<1K
        tags:
          - cot-oracle
          - chain-of-thought
          - metacognition
          - eval
          - qwen3
          - reasoning
        ---

        # CoT Oracle Eval: CoT Metacognition

        This dataset evaluates whether an activation oracle can recover a model's local metacognitive state from a partial chain-of-thought prefix.
        Each example asks a targeted question about a Qwen3-8B reasoning trace, such as "What is the model not sure about right now?", and provides a short target answer grounded in the withheld continuation.

        The current uploaded snapshot contains **{total} examples** in a single **`test`** split.

        ## What Each Row Contains

        Each row has:

        - `cot_text`: a prefix of a Qwen3-8B chain of thought, cut at an automatically selected sentence boundary
        - `prompt`: a category-specific metacognition question about that prefix
        - `target_response`: the expected answer to that question
        - metadata describing the source dataset, domain, cutoff location, and whether the example came from the hand-authored Arya seed set

        This makes the dataset suitable for evaluating models that read hidden states or intermediate activations and then answer targeted questions about the supervisee model's reasoning state.

        ## Construction

        1. Problems were collected from **MATH**, **GSM8K**, **BBH**, **LogiQA**, plus a small set of hand-authored **Arya** seed examples.
        2. A rollout model, **`qwen/qwen3-8b`**, generated full chains of thought with the system prompt `Think step by step. Show all your work.`
        3. The chain of thought was split into sentences and candidate cutoff points were chosen with regex-based heuristics aimed at states like uncertainty, backtracking, error recognition, alternative paths, and missing information.
        4. For each cutoff, the next few sentences were shown to **`google/gemini-3.1-flash-lite-preview`**, which produced a short, specific target label answering a category-specific question.
        5. Labels judged too vague were filtered out, and the remaining examples were written to JSONL and uploaded to Hugging Face in Parquet format.

        The upload logic lives in `scripts/generate_cot_metacognition.py` in the [cot-oracle](https://github.com/ceselder/cot-oracle) repo.

        ## Snapshot Statistics

        - Split: `test` only
        - Rows: **{total}**
        - Hand-authored Arya examples: **{arya_count}**
        - Mean cutoff sentence index: **{avg_sent_idx:.2f}**
        - Mean `cot_text` length: **{avg_cot_chars:.1f} characters**
        - Top domains: {top_domains}

        ### Category Distribution

        {indent(make_counter_table(category_counts), "        ")}

        ### Source Distribution

        {indent(make_counter_table(source_counts), "        ")}

        ## Important Caveats

        - This snapshot is **highly imbalanced**: uncertainty accounts for most rows. The uploaded artifact should be treated as the current available eval snapshot, not as a balanced benchmark across all intended metacognition categories.
        - The generator supports additional categories such as `number_prediction`, but no rows from that category survived into the current uploaded snapshot.
        - `control_pair_id` is present for schema compatibility but is null for all rows in this snapshot.
        - `cot_text` is only the prefix up to the cutoff point, not the full reasoning trace.

        ## Schema

        | Field | Type | Description |
        |---|---|---|
        | `task` | string | Always `cot_metacognition` for this dataset. |
        | `prompt` | string | The metacognition question asked about the cutoff prefix. |
        | `target_response` | string | The expected short answer, derived from the withheld continuation. |
        | `cot_text` | string | Prefix of the model's chain of thought up to the selected cutoff. |
        | `category` | string | Metacognition category such as `uncertainty` or `error_recognition`. |
        | `domain` | string | Subject/domain label inherited from the source problem. |
        | `source` | string | Source dataset, one of `MATH`, `GSM8K`, `BBH`, `LogiQA`, or `arya`. |
        | `arya_example` | bool | Whether the example came from the hand-authored Arya seed set. |
        | `control_pair_id` | null | Reserved field for paired controls; null in this snapshot. |
        | `question` | string | Original problem shown to the rollout model. |
        | `sent_idx` | int | Sentence index where the chain of thought was cut. |

        ## Usage

        ```python
        from datasets import load_dataset

        ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-eval-cot-metacognition", split="test")
        row = ds[0]
        print(row["prompt"])
        print(row["cot_text"])
        print(row["target_response"])
        ```

        ## Related Project Files

        - Code: [cot-oracle](https://github.com/ceselder/cot-oracle)
        - Generator: `scripts/generate_cot_metacognition.py`
        - Task registry: `src/tasks.py`
        """
    ).strip() + "\n"


def upload_dataset_card(output_path: Path):
    rows = load_rows(output_path)
    card = build_dataset_card(rows)
    readme_path = output_path.with_name("README.md")
    readme_path.write_text(card)

    print(f"\nUploading dataset card to {HF_REPO}...")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=HF_REPO, repo_type="dataset", commit_message="Update dataset card")
    print(f"  Wrote local README to {readme_path}")
    print("  Uploaded README.md")


def upload_to_hf(output_path: Path):
    """Upload to HuggingFace as Parquet."""
    print(f"\nUploading to {HF_REPO}...")
    from datasets import Dataset
    import pandas as pd

    rows = load_rows(output_path)

    ds = Dataset.from_pandas(pd.DataFrame(rows))
    ds.to_parquet(str(output_path.with_suffix(".parquet")))
    upload_dataset_card(output_path)
    ds.push_to_hub(HF_REPO, split="test")
    print(f"  Uploaded {len(rows)} examples to {HF_REPO}")


def main():
    load_dotenv(Path.home() / ".env")
    parser = argparse.ArgumentParser(description="Generate CoT metacognition eval dataset")
    parser.add_argument("--n", type=int, default=None, help="Limit total problems (for testing)")
    parser.add_argument("--n-per-source", type=int, default=150, help="Problems per dataset source")
    parser.add_argument("--max-budget", type=float, default=5.0, help="Max API spend in $")
    parser.add_argument("--output", default="data/cot_metacognition/cot_metacognition.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Only collect problems and find cutoffs, no API calls")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after generation")
    parser.add_argument("--upload-card-only", action="store_true", help="Upload only the dataset card for an existing output file")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.upload_card_only:
        upload_dataset_card(output_path)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        raise ValueError("OPENROUTER_API_KEY not set")

    t0 = time.time()

    # Phase 0: Collect problems
    n_per_source = args.n_per_source
    if args.n:
        n_per_source = max(10, args.n // 4)
    problems = collect_problems(n_per_source, args.seed)
    if args.n:
        problems = problems[:args.n]

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would generate rollouts for {len(problems)} problems")
        # Show a few example prompts
        for p in problems[:5]:
            print(f"\n  [{p['source']}] {p['question'][:100]}...")
            is_arya = p.get("arya_example", False)
            cat_hint = p.get("category_hint", "auto")
            print(f"    arya={is_arya}, category_hint={cat_hint}")

        # Test cutoff finding on a dummy CoT
        dummy_cot = (
            "Let me think about this step by step. First, I need to find the value of x. "
            "Using the quadratic formula, x = (-b ± sqrt(b²-4ac)) / 2a. "
            "With a=1, b=-5, c=6, we get x = (5 ± sqrt(25-24)) / 2. "
            "Wait, that gives sqrt(1) = 1. So x = (5+1)/2 = 3 or x = (5-1)/2 = 2. "
            "Actually, let me check: 2² - 5(2) + 6 = 4 - 10 + 6 = 0. Yes. "
            "And 3² - 5(3) + 6 = 9 - 15 + 6 = 0. Also correct. "
            "So the solutions are x = 2 and x = 3. "
            "Hmm, I should verify there aren't other solutions. "
            "Since it's a quadratic, there can be at most 2 solutions. "
            "The answer is x = 2 and x = 3."
        )
        sentences = split_cot_into_sentences(dummy_cot)
        cutoffs = find_metacog_cutoffs(sentences)
        print(f"\n  Dummy CoT cutoffs: {cutoffs}")
        for idx, cat in cutoffs:
            print(f"    [{cat}] sent {idx}: {sentences[idx][:80]}...")
        return

    # Phase 1: Generate rollouts
    rollouts = asyncio.run(generate_rollouts(problems, api_key, args.max_budget))

    if budget_exceeded:
        print("Budget exceeded during rollout generation")
        return

    # Phase 2: Generate labels
    examples = asyncio.run(generate_labels(rollouts, api_key, args.max_budget, args.seed))

    if not examples:
        print("No examples generated!")
        return

    # Phase 3: Balance & save
    balanced = balance_and_save(examples, output_path, args.seed)

    elapsed = time.time() - t0
    cost = estimate_cost()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Output: {output_path}")
    print(f"  Examples: {len(balanced)}")
    print(f"  API calls: {completed} completed, {failed} failed")
    print(f"  Cost: ~${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    arya_count = sum(1 for ex in balanced if ex.get("arya_example"))
    print(f"  Arya examples: {arya_count}")

    # Show samples
    rng = random.Random(args.seed)
    samples = rng.sample(balanced, min(3, len(balanced)))
    print(f"\n  Example outputs:")
    for ex in samples:
        print(f"    [{ex['category']}|{ex['source']}] {ex['prompt']}")
        print(f"    Target: {ex['target_response'][:150]}...")
        print()

    if args.upload:
        upload_to_hf(output_path)


if __name__ == "__main__":
    main()
