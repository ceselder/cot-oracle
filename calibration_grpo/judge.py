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
DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """\
You are evaluating an activation oracle's responses. The oracle reads neural network
activations from a language model's chain-of-thought reasoning and answers questions
about what the model is doing internally.

You will see:
- The original question asked to the language model
- The language model's chain-of-thought (CoT) reasoning
- The oracle prompt (the question the oracle was asked about the CoT)
- One or more oracle responses (rollouts) to evaluate

For EACH rollout, score these 11 criteria on a 0-1-2 scale:
  0 = clearly fails
  1 = partially meets / weak attempt
  2 = fully meets / strong

1. not_provably_wrong:
   0 = contains demonstrably false claims about the CoT
   1 = mostly correct but one minor inaccuracy
   2 = everything stated is correct or unfalsifiable

2. specific:
   0 = vague buzzword slop ("performing structured computation")
   1 = somewhat specific but could be more concrete
   2 = names exact operations, values, or steps

3. follows_instructions:
   0 = ignores the oracle prompt, answers something else
   1 = partially addresses the prompt
   2 = directly and fully answers what was asked

4. passes_swap_test:
   0 = completely generic, works for any CoT
   1 = somewhat tied to this CoT but partly generic
   2 = clearly specific to THIS CoT, would not work for others

5. concise:
   0 = rambling, padded, repetitive
   1 = some unnecessary filler
   2 = tight, gets to the point

6. not_just_restating_text:
   0 = pure paraphrase of the visible CoT
   1 = mostly restating with minor interpretation
   2 = adds genuine insight beyond what the text shows

7. numbers_if_applicable:
   0 = CoT has numbers but response ignores them
   1 = mentions numbers but vaguely
   2 = engages with specific numbers from the CoT (or CoT has no numbers)

8. confident_when_verifiable:
   0 = makes confident claims that are wrong or uncheckable
   1 = mixed — some well-supported claims, some not
   2 = confident claims are all verifiable and correct

9. hedged_when_uncertain:
   0 = assertive about genuinely ambiguous things
   1 = partially hedges
   2 = appropriately qualifies uncertain claims (or nothing is uncertain)

10. useful_to_a_human:
    0 = a human monitoring the model learns nothing from this
    1 = somewhat informative
    2 = genuinely changes understanding or informs a decision

11. falsifiable:
    0 = all claims are unfalsifiable ("reasoning carefully")
    1 = mix of falsifiable and unfalsifiable claims
    2 = makes clear, checkable commitments

Return a JSON array with one object per rollout, scores as integers 0-2:
[
  {
    "index": 1,
    "not_provably_wrong": 2,
    "specific": 1,
    "follows_instructions": 2,
    "passes_swap_test": 0,
    "concise": 2,
    "not_just_restating_text": 1,
    "numbers_if_applicable": 2,
    "confident_when_verifiable": 2,
    "hedged_when_uncertain": 1,
    "useful_to_a_human": 1,
    "falsifiable": 2
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
        print(f"  [judge] No JSON array found in response: {content[:200]}")
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  [judge] JSON parse error: {e}, content: {match.group()[:200]}")
        return None

    results = []
    for entry in parsed:
        idx = entry.get("index", len(results) + 1) - 1  # 1-indexed to 0-indexed
        criteria = {}
        for name in CRITERIA_NAMES:
            val = entry.get(name)
            if isinstance(val, int):
                criteria[name] = max(0, min(2, val))  # clamp to 0-2
            elif isinstance(val, float):
                criteria[name] = max(0, min(2, int(round(val))))
            elif isinstance(val, bool):
                criteria[name] = 2 if val else 0  # backwards compat
            elif isinstance(val, str):
                try:
                    criteria[name] = max(0, min(2, int(val)))
                except ValueError:
                    criteria[name] = 2 if val.lower().strip() in ("true", "yes") else 0
            else:
                criteria[name] = 0
        results.append(RubricResult(rollout_idx=idx, criteria=criteria))

    if len(results) != n_rollouts:
        print(f"  [judge] Expected {n_rollouts} rollouts, got {len(results)} in response")
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

                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                content = content.strip()
                if not content:
                    if attempt == retries - 1:
                        return None
                    continue
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
