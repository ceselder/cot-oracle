"""5-criterion tiered rubric judge for GRPO rewards."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx

from reward import CRITERIA_NAMES, RubricResult

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-m2.7"

SYSTEM_PROMPT = """\
You are evaluating an activation oracle. The oracle reads neural network activations
extracted from a SPECIFIC POINT midway through a language model's chain-of-thought
reasoning. The oracle cannot see the CoT text — only the model's internal hidden states
from the first half of the reasoning.

You will see:
- The question the language model was solving
- FIRST HALF of the chain-of-thought (the oracle's activations were extracted from near the end of this section)
- SECOND HALF of the chain-of-thought (the oracle did NOT see this — use it to verify forward-looking claims)
- The oracle prompt (the question asked to the oracle)
- Oracle responses (rollouts) to evaluate

IMPORTANT: The oracle only had access to activations from near the end of the first half.
It should be able to describe what the model has done so far AND predict or characterize
what comes next. Use the second half to verify any forward-looking claims.

For EACH rollout, score these 5 criteria on a 0-1-2 scale:
  0 = clearly fails
  1 = partially meets / weak attempt
  2 = fully meets / strong

1. passes_swap_test:
   0 = completely generic, would work for ANY chain of thought ("the model is reasoning step by step")
   1 = somewhat tied to this CoT but still partly generic
   2 = clearly specific to THIS particular CoT at THIS point — would be wrong or nonsensical for a different CoT or a different split point

2. specific_and_falsifiable:
   0 = vague buzzword slop ("performing structured computation") or all claims are unfalsifiable
   1 = somewhat specific, or a mix of concrete and vague claims
   2 = names exact operations, values, steps, or quantities that can be checked against the first half (for past claims) or second half (for predictions)

3. adds_insight:
   0 = says nothing a human couldn't get from reading just the first half
   1 = mostly restating with some minor interpretation or inference
   2 = reveals something about the model's internal state beyond the visible text — e.g. predicts what happens next (verifiable via second half), describes uncertainty, or identifies what the model decided but hasn't written yet

4. not_provably_wrong:
   0 = contains demonstrably false claims — contradicted by either half of the CoT
   1 = mostly correct but one minor inaccuracy or questionable claim
   2 = everything stated is correct when checked against both halves

5. follows_instructions:
   0 = ignores the oracle prompt, answers something else entirely
   1 = partially addresses the prompt
   2 = directly and fully answers what was asked

Think step by step about each rollout, then return a JSON array with one object per rollout:
[
  {
    "index": 1,
    "passes_swap_test": 0,
    "specific_and_falsifiable": 1,
    "adds_insight": 2,
    "not_provably_wrong": 2,
    "follows_instructions": 2
  }
]

Return the JSON array after your reasoning."""


def _build_user_prompt(
    question: str,
    first_half: str,
    second_half: str,
    oracle_prompt: str,
    rollout_texts: list[str],
) -> str:
    parts = [
        f"Question the model was solving:\n{question}\n",
        f"=== FIRST HALF of CoT (oracle has activations from near end of this section) ===\n{first_half}\n",
        f"=== SECOND HALF of CoT (oracle did NOT see this — use to verify predictions) ===\n{second_half}\n",
        f"Oracle prompt:\n{oracle_prompt}\n",
        "Oracle responses to evaluate:",
    ]
    for i, text in enumerate(rollout_texts, 1):
        parts.append(f"\n--- Rollout {i} ---\n{text}")
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
                criteria[name] = 2 if val else 0
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
    first_half: str,
    second_half: str,
    oracle_prompt: str,
    rollout_texts: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[RubricResult] | None:
    """Score rollouts with the 5-criterion rubric. Returns None on failure."""
    user_msg = _build_user_prompt(question, first_half, second_half, oracle_prompt, rollout_texts)
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
        "max_tokens": 8192,
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
    question, first_half, second_half, oracle_prompt, rollout_texts."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(ex: dict) -> list[RubricResult] | None:
        async with sem:
            return await judge_rollouts(
                question=ex["question"],
                first_half=ex["first_half"],
                second_half=ex["second_half"],
                oracle_prompt=ex["oracle_prompt"],
                rollout_texts=ex["rollout_texts"],
                api_key=api_key,
                model=model,
            )

    return await asyncio.gather(*[_one(ex) for ex in examples])
