"""5-criterion rubric judge for GRPO rewards (Sonnet 4.6)."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from reward import CRITERIA_NAMES, RubricResult

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"

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

For EACH rollout, score these 5 criteria on a 0-1-2 scale:
  0 = clearly fails
  1 = partially meets / weak attempt
  2 = fully meets / strong

1. passes_swap_test:
   0 = completely generic, would work for ANY chain of thought
   1 = somewhat tied to this CoT but still partly generic
   2 = clearly specific to THIS particular CoT at THIS point

2. specific_and_falsifiable:
   0 = vague buzzword slop or all claims are unfalsifiable
   1 = somewhat specific, mix of concrete and vague claims
   2 = names exact operations, values, steps that can be checked against the CoT

3. adds_insight:
   0 = says nothing a human couldn't get from reading just the first half
   1 = mostly restating with some minor interpretation
   2 = reveals something beyond the visible text — e.g. predicts what happens next
       (verifiable via second half), describes uncertainty, identifies decisions not yet written

4. not_provably_wrong:
   0 = contains claims CONTRADICTED by the CoT evidence (either half)
   1 = mostly correct but one minor inaccuracy
   2 = nothing stated is contradicted by the available evidence

5. follows_instructions:
   0 = ignores the oracle prompt entirely
   1 = partially addresses the prompt
   2 = directly and fully answers what was asked

Return a JSON array with one object per rollout:
[{"index": 1, "passes_swap_test": 1, "specific_and_falsifiable": 2, "adds_insight": 1, "not_provably_wrong": 2, "follows_instructions": 2}]
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
    """Parse JSON array of rubric results from judge response."""
    # Try JSON array
    match = re.search(r"\[[\s\S]*?\]", content)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _extract_results(parsed, n_rollouts)
        except json.JSONDecodeError:
            pass

    # Fallback: top-level JSON object with rollout keys
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                if any(k in obj for k in CRITERIA_NAMES):
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
        criteria = {}
        for name in CRITERIA_NAMES:
            val = entry.get(name)
            if isinstance(val, (int, float)):
                criteria[name] = max(0, min(2, int(round(val))))
            elif isinstance(val, str):
                try:
                    criteria[name] = max(0, min(2, int(val)))
                except ValueError:
                    criteria[name] = 0
            else:
                criteria[name] = 0
        results.append(RubricResult(rollout_idx=idx, criteria=criteria))

    if len(results) != n_rollouts:
        print(f"  [judge] Expected {n_rollouts} rollouts, got {len(results)}")
        return None
    return results


# Global token counters for spend tracking
_total_input_tokens = 0
_total_output_tokens = 0
# Sonnet 4.6 pricing on OpenRouter
# Tracks combined spend across all models (approximate)
_INPUT_PRICE_PER_M = 0.30
_OUTPUT_PRICE_PER_M = 1.50


def get_spend() -> dict:
    """Return cumulative API spend stats."""
    input_cost = _total_input_tokens * _INPUT_PRICE_PER_M / 1_000_000
    output_cost = _total_output_tokens * _OUTPUT_PRICE_PER_M / 1_000_000
    return {
        "judge/input_tokens": _total_input_tokens,
        "judge/output_tokens": _total_output_tokens,
        "judge/spend_usd": input_cost + output_cost,
    }


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
    global _total_input_tokens, _total_output_tokens
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

                data = resp.json()
                # Track token usage
                usage = data.get("usage", {})
                _total_input_tokens += usage.get("prompt_tokens", 0)
                _total_output_tokens += usage.get("completion_tokens", 0)

                msg = data["choices"][0]["message"]
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
