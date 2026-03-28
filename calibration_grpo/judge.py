"""Single-score judge for GRPO rewards."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from reward import CRITERIA_NAMES, RubricResult

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are evaluating an activation oracle. The oracle reads neural network activations
extracted from a SPECIFIC POINT midway through a language model's chain-of-thought.
The oracle cannot see the CoT text — only the model's internal hidden states from
the first half of the reasoning.

You will see:
- The question the language model was solving
- FIRST HALF of the CoT (the oracle's activations came from near the end of this section)
- SECOND HALF of the CoT (the oracle did NOT see this — use it to verify predictions)
- The oracle prompt and oracle responses to evaluate

Score each oracle response 0-2:
  0 = wrong, generic, or contradicted by the CoT evidence
  1 = partially correct — some accurate claims mixed with vague or wrong ones
  2 = correct and specific — claims match what's actually happening, predictions verified by second half

Check claims against BOTH halves. Forward predictions confirmed by the second half score highest.
Generic filler like "the model is reasoning step by step" always scores 0.

Return a JSON array: [{"index": 1, "score": 2}, {"index": 2, "score": 1}, ...]
Return ONLY the JSON array."""


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


def _parse_response(content: str, n_rollouts: int) -> list[RubricResult] | None:
    """Parse JSON array of scores from judge response."""
    # Try JSON array
    match = re.search(r"\[[\s\S]*?\]", content)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _extract_results(parsed, n_rollouts)
        except json.JSONDecodeError:
            pass

    # Fallback: top-level object or object with rollout keys
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                if "score" in obj:
                    return _extract_results([obj], n_rollouts)
                entries = list(obj.values())
                if entries and isinstance(entries[0], dict):
                    return _extract_results(entries, n_rollouts)
        except json.JSONDecodeError:
            pass

    print(f"  [judge] Could not parse: {content[:200]}")
    return None


def _extract_results(parsed: list[dict], n_rollouts: int) -> list[RubricResult] | None:
    results = []
    for entry in parsed:
        idx = entry.get("index", len(results) + 1) - 1
        score = entry.get("score", 0)
        if isinstance(score, (int, float)):
            score = max(0, min(2, int(round(score))))
        else:
            try:
                score = max(0, min(2, int(score)))
            except (ValueError, TypeError):
                score = 0
        results.append(RubricResult(rollout_idx=idx, criteria={"score": score}))

    if len(results) != n_rollouts:
        print(f"  [judge] Expected {n_rollouts} rollouts, got {len(results)}")
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
        "max_tokens": 1024,
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
                content = msg.get("content") or ""
                content = content.strip()
                if not content:
                    if attempt == retries - 1:
                        return None
                    continue
                return _parse_response(content, len(rollout_texts))

            except Exception as e:
                if attempt == retries - 1:
                    print(f"  [judge] Error after {retries} retries: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)

    return None


async def judge_batch(
    examples: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 16,
) -> list[list[RubricResult] | None]:
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
