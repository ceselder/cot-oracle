"""8-criterion binary rubric judge for GRPO rewards."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx

from reward import CRITERIA_NAMES, RubricResult

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-preview"

SYSTEM_PROMPT = """\
You are evaluating an activation oracle's responses. The oracle reads neural network
activations from a language model's chain-of-thought reasoning and answers questions
about what the model is doing internally.

You will see:
- The original question asked to the language model
- The language model's chain-of-thought (CoT) reasoning
- The oracle prompt (the question the oracle was asked about the CoT)
- One or more oracle responses (rollouts) to evaluate

For EACH rollout, evaluate these 11 binary criteria (YES or NO):

1. not_provably_wrong: Is the response free of claims that are demonstrably false given
   the CoT content? Ambiguous or untestable claims are fine — only flag things that are
   clearly, provably incorrect. Innocent until proven guilty.

2. specific: Does the response make specific, concrete claims rather than vague hedging
   like "the model appears to be performing some form of computation"?

3. follows_instructions: Does the response directly answer the question asked in the
   oracle prompt? Not tangential, not evasive.

4. passes_swap_test: Could this response NOT be copy-pasted onto a completely different
   CoT and still sound equally plausible? YES means it's genuinely tied to THIS CoT.

5. concise: Is the response free of unnecessary padding, filler, or rambling?
   Gets to the point.

6. not_just_restating_text: Does the response say something beyond what's already
   obvious from reading the CoT text? Just paraphrasing or summarizing the visible
   text is NO. Adding interpretation, identifying patterns, or surfacing non-obvious
   structure is YES.

7. numbers_if_applicable: If the CoT contains specific numbers or quantities, does the
   response reference them when relevant? YES if it engages with the actual numbers in
   the CoT. NO if numbers are clearly relevant but the response avoids mentioning them.
   If the CoT has no numbers, YES.

8. confident_when_verifiable: When the response makes a strong, unhedged claim, is that
   claim actually verifiable and correct from the CoT? Confident claims that are
   clearly supported = YES. Confident claims about things that can't be checked or
   that are wrong = NO. If the response doesn't make strong claims, YES.

9. hedged_when_uncertain: If the response addresses something genuinely ambiguous or
   hard to determine from the activations, does it appropriately hedge or qualify?
   Hedging on genuinely uncertain things = YES. Being assertive about ambiguous things
   = NO. If everything in the response is clear-cut, YES.

10. useful_to_a_human: Would a human monitoring this model actually learn something
    useful from this response? Would it change their understanding of what the model
    is doing, or inform a decision (like whether to trust the output)? Generic
    descriptions that any observer could write = NO.

11. falsifiable: Does the response make claims that could in principle be checked and
    proven wrong? "The model is reasoning carefully" is unfalsifiable slop. "The model
    will output 42" or "the model is doing division" are falsifiable. YES means the
    response commits to checkable claims.

Return a JSON array with one object per rollout:
[
  {
    "index": 1,
    "not_provably_wrong": true,
    "specific": true,
    "follows_instructions": true,
    "passes_swap_test": false,
    "concise": true,
    "not_just_restating_text": false,
    "numbers_if_applicable": true,
    "confident_when_verifiable": true,
    "hedged_when_uncertain": false,
    "useful_to_a_human": true,
    "falsifiable": true
  }
]

Return ONLY the JSON array. No other text."""


def _build_user_prompt(
    question: str,
    cot_text: str,
    oracle_prompt: str,
    rollout_texts: list[str],
) -> str:
    parts = [
        f"Question asked to the model:\n{question}\n",
        f"Model's chain-of-thought:\n{cot_text[:3000]}\n",
        f"Oracle prompt:\n{oracle_prompt}\n",
        "Oracle responses to evaluate:",
    ]
    for i, text in enumerate(rollout_texts, 1):
        parts.append(f"\n--- Rollout {i} ---\n{text[:500]}")
    return "\n".join(parts)


def _parse_rubric_response(content: str, n_rollouts: int) -> list[RubricResult] | None:
    """Parse JSON array of rubric results from judge response."""
    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    results = []
    for entry in parsed:
        idx = entry.get("index", len(results) + 1) - 1  # 1-indexed to 0-indexed
        criteria = {}
        for name in CRITERIA_NAMES:
            val = entry.get(name)
            if isinstance(val, bool):
                criteria[name] = val
            elif isinstance(val, str):
                criteria[name] = val.lower().strip() in ("true", "yes", "1")
            else:
                criteria[name] = False
        results.append(RubricResult(rollout_idx=idx, criteria=criteria))

    if len(results) != n_rollouts:
        return None
    return results


async def judge_rollouts(
    question: str,
    cot_text: str,
    oracle_prompt: str,
    rollout_texts: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[RubricResult] | None:
    """Score rollouts with the 8-criterion binary rubric. Returns None on failure."""
    user_msg = _build_user_prompt(question, cot_text, oracle_prompt, rollout_texts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                resp = await client.post(
                    ENDPOINT, json=body, headers=headers, timeout=timeout,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue

                content = resp.json()["choices"][0]["message"]["content"]
                return _parse_rubric_response(content, len(rollout_texts))

            except Exception:
                if attempt == retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)

    return None


async def judge_batch(
    examples: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 16,
) -> list[list[RubricResult] | None]:
    """Judge a batch of examples concurrently. Each example has:
    question, cot_text, oracle_prompt, rollout_texts."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(ex: dict) -> list[RubricResult] | None:
        async with sem:
            return await judge_rollouts(
                question=ex["question"],
                cot_text=ex["cot_text"],
                oracle_prompt=ex["oracle_prompt"],
                rollout_texts=ex["rollout_texts"],
                api_key=api_key,
                model=model,
            )

    return await asyncio.gather(*[_one(ex) for ex in examples])
