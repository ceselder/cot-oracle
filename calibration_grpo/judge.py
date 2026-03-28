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
You are evaluating an activation oracle's responses. The oracle reads neural network
activations from a language model's chain-of-thought reasoning and answers questions
about what the model is doing internally. It cannot see the CoT text — only the
model's internal hidden states.

You will see:
- The original question asked to the language model
- The language model's chain-of-thought (CoT) reasoning
- The oracle prompt (the question the oracle was asked about the CoT)
- One or more oracle responses (rollouts) to evaluate

For EACH rollout, score these 5 criteria on a 0-1-2 scale:
  0 = clearly fails
  1 = partially meets / weak attempt
  2 = fully meets / strong

1. passes_swap_test:
   0 = completely generic, would work for ANY chain of thought ("the model is reasoning step by step")
   1 = somewhat tied to this CoT but still partly generic
   2 = clearly specific to THIS particular CoT — would be wrong or nonsensical for a different one

2. specific_and_falsifiable:
   0 = vague buzzword slop ("performing structured computation", "multi-step reasoning") or all claims are unfalsifiable
   1 = somewhat specific, or a mix of concrete and vague claims
   2 = names exact operations, values, steps, or quantities that could be checked; makes commitments that would be wrong if the CoT were different

3. adds_insight:
   0 = pure paraphrase of the visible CoT text, or says nothing a human couldn't get from reading the CoT
   1 = mostly restating with some minor interpretation or inference
   2 = reveals something about the model's internal state that goes beyond what the text shows — e.g. what the model is uncertain about, what it's about to do, what it decided but hasn't written yet

4. not_provably_wrong:
   0 = contains demonstrably false claims about the CoT (e.g. says the model is doing multiplication when it's clearly doing something else)
   1 = mostly correct but one minor inaccuracy or questionable claim
   2 = everything stated is correct (or the claims are specific but not contradicted by the CoT)

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
