"""Async Gemini judging of oracle rollouts via OpenRouter.

Each call sends 12 rollouts for one example and gets back per-rollout ratings.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_URL = f"{OPENROUTER_API_BASE.rstrip('/')}/chat/completions"

JUDGE_SYSTEM_PROMPT = """\
You are evaluating an Activation Oracle — a model that reads neural network \
activations (not text) and describes what's happening in a chain-of-thought \
at specific positions. You will see multiple oracle responses and the underlying \
CoT region. The oracle CANNOT access the text — only activation vectors.

You do not know exactly what is and is not readable from activations. However, \
you CAN judge whether a claim is factually inconsistent with the CoT. If the \
CoT is clearly doing addition and the oracle says "the model is performing \
division," that is wrong regardless of what activations might theoretically \
contain. Use the CoT as ground truth for WHAT is happening — be more lenient \
about WHETHER the oracle could know it from activations alone.

## What makes a good oracle response

The ideal oracle response:
- Describes the computational state: what operation is being performed, what \
intermediate values are being tracked, what strategy is being used
- Identifies shifts in reasoning: backtracking, strategy changes, growing \
uncertainty, realization of errors
- Is specific and concrete rather than generic — names the actual computation, \
not just "the model is thinking about the problem"
- Goes beyond what's visible in the surface text — describes WHY something is \
happening, not just WHAT tokens appear nearby
- Acknowledges uncertainty when appropriate rather than confabulating details

## Rating each response

Rate each response as:
- "good": Accurately describes what's happening in the CoT at the marked \
positions. Contains specific, correct, non-obvious information.
- "mixed": Partially correct — contains some true information and some false \
information. IMPORTANT: If the response correctly identifies ANY of the \
following, it is "mixed" not "bad":
  - The type of computation (addition, division, comparison, case analysis)
  - The general domain or topic (even vaguely — math, logic, physics)
  - The structure of the reasoning (summing values, dividing by a rate)
  - The general goal (finding a total, computing a time, solving for x)
  Wrong numbers, wrong variable names, wrong room names, wrong units — \
  these make a response "mixed", NEVER "bad", as long as the computational \
  structure is roughly right. The oracle reads activations, not text — it \
  can identify WHAT KIND of computation is happening without reading specific \
  values. Example: if the CoT adds 24+80=104 and divides by 8, and the oracle \
  says "the model adds 16+16=32 and divides by the rate" — that is "mixed" \
  because the structure (add two values, divide by rate) is correct even \
  though every number is wrong.
- "bad": The response is completely wrong, hallucinated, or nonsensical. \
  Reserve "bad" ONLY for responses that describe a COMPLETELY UNRELATED topic \
  (e.g. talking about biology when the CoT is about geometry), or that are \
  so garbled as to convey no meaningful information. When in doubt between \
  "bad" and "mixed", ALWAYS prefer "mixed". When in doubt between "bad" and \
  "indeterminate", prefer "indeterminate".
- "indeterminate": The response makes claims that may or may not be present \
  in the activations, and the chain of thought does not provide enough \
  evidence to confirm or deny them. For example, "the model is uncertain \
  here" might be true at the activation level even if the text sounds \
  confident.

## Flagging format and quality issues

For EVERY response, regardless of content rating:

1. **Malformed check.** Set "malformed": true if the response has broken \
sentences, repetitive text, incomplete thoughts, garbled output, or fails \
to address all parts of a multi-part question (e.g. two questions asked but \
only one answered). If malformed, provide a "reformatted" version that \
preserves the exact same content but fixes the formatting.

2. **Vagueness check.** Set "vague": true if ANY of these apply:
   (a) Generic/vague — uses only statements that could apply to any reasoning \
   (e.g. "the model is thinking about the problem") without specific details \
   about WHAT is being thought about, computed, or concluded.
   (b) Text echoing — merely restates words or phrases visible in the nearby \
   chain of thought without adding deeper interpretation. The oracle reads \
   activations, not text — a useful response should describe computational \
   state rather than parroting surface text. For example, if the CoT says \
   "let me try modular arithmetic" and the oracle says "the model is thinking \
   about modular arithmetic," that is text echoing. A better response would be \
   "the model is switching from direct computation to a modular arithmetic \
   approach because the previous strategy hit a dead end."
   (c) Fails to address the prompt — if the oracle prompt asks for something \
   specific (e.g. "What numbers is the model working with?") and the response \
   gives only generic commentary without addressing the actual question.
   If the prompt ends with "Be specific." then flag vagueness more aggressively \
   — the response must contain concrete, non-obvious details.

## Corrections for mixed responses

For "mixed" responses, provide a minimal "correction" — change as few words as \
possible to fix the inaccurate parts while keeping everything else verbatim. \
Only strike claims that are clearly verifiably wrong from the CoT. Do NOT \
rewrite sentence structure or rephrase correct parts.

When replacing a wrong detail, only substitute it with the correct version if \
another rollout actually stated the correct version — otherwise just remove \
the wrong claim. Exception: wrong names and numbers can always be corrected \
directly from the CoT, even if no other rollout got them right.

## Synthesizing the ideal response

Synthesize an "ideal_response" from ONLY claims that appear in the original \
responses. Do not add any information not found in at least one response. \
The ideal response should be specific and concrete — do not make it vague.

When rollouts contradict each other:
- If most rollouts agree on a claim, include it confidently.
- If rollouts are roughly evenly split, hedge with "likely", "appears to be", \
or "seems to" rather than stating it as fact.
- If only one rollout mentions a claim and others ignore or contradict it, \
either drop it or hedge heavily.

If all responses are bad or nonsensical, set ideal_response to null.

## Output format

Return ONLY valid JSON in this format:
{
  "ratings": [
    {
      "index": 1,
      "rating": "good|mixed|bad|indeterminate",
      "correction": "corrected text if mixed, null otherwise",
      "malformed": true|false,
      "reformatted": "cleaned version if malformed, null otherwise",
      "vague": true|false
    }
  ],
  "ideal_response": "synthesized from correct claims across responses, or null"
}"""


def _build_judge_user_prompt(
    question: str,
    cot_text: str,
    base_positions: list[int],
    cot_start: int,
    oracle_prompt: str,
    rollouts: list[str],
    tokenizer,
) -> str:
    """Build the user prompt for the judge, with position markers in CoT."""
    # Decode the CoT region and insert position markers
    # We work with the text directly since we have it
    # Mark approximate position locations in the text
    cot_words = cot_text.split()
    # Map token positions to approximate word positions
    # (rough heuristic: ~1.3 tokens per word for English)
    total_cot_tokens = len(tokenizer.encode(cot_text, add_special_tokens=False))
    tokens_per_word = total_cot_tokens / max(len(cot_words), 1)

    marked_cot = cot_text
    if base_positions:
        # Insert numbered markers ([POS 1], [POS 2], ...) at approximate word positions
        markers = []
        for pos_num, pos in enumerate(base_positions, 1):
            relative_pos = pos - cot_start
            approx_word_idx = int(relative_pos / max(tokens_per_word, 1))
            approx_word_idx = min(approx_word_idx, len(cot_words) - 1)
            markers.append((approx_word_idx, pos_num))

        # Insert markers in reverse order to preserve indices
        words = cot_words[:]
        for word_idx, pos_num in sorted(markers, reverse=True):
            if 0 <= word_idx < len(words):
                words[word_idx] = f"[POS {pos_num}] {words[word_idx]}"
        marked_cot = " ".join(words)

    rollout_text = "\n\n".join(
        f"R{i+1}: {r}" for i, r in enumerate(rollouts)
    )

    n_pos = len(base_positions)
    pos_note = (
        f"The oracle is reading activations from {n_pos} adjacent position{'s' if n_pos > 1 else ''} "
        f"in the CoT, marked with [POS n] below."
    )

    return (
        f"## Original Question\n{question}\n\n"
        f"## Chain of Thought (with position markers)\n{pos_note}\n\n{marked_cot}\n\n"
        f"## Oracle Prompt\n{oracle_prompt}\n\n"
        f"## Oracle Responses ({len(rollouts)} rollouts)\n{rollout_text}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from judge response, handling markdown blocks and thinking tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge did not return JSON: {cleaned[:200]}")
    return json.loads(cleaned[start : end + 1])


@dataclass
class RolloutRating:
    index: int
    rating: str  # good, mixed, bad, indeterminate
    correction: str | None = None
    malformed: bool = False
    reformatted: str | None = None
    vague: bool = False


@dataclass
class JudgeResult:
    ratings: list[RolloutRating]
    ideal_response: str | None = None
    raw_response: str = ""


async def judge_rollouts(
    question: str,
    cot_text: str,
    base_positions: list[int],
    cot_start: int,
    oracle_prompt: str,
    rollouts: list[str],
    tokenizer,
    model: str = "google/gemini-3.1-flash-lite-preview",
    temperature: float = 0.0,
    timeout_sec: float = 60.0,
) -> JudgeResult:
    """Judge a set of rollouts for one example via OpenRouter.

    Returns JudgeResult with per-rollout ratings and ideal response.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_prompt = _build_judge_user_prompt(
        question, cot_text, base_positions, cot_start, oracle_prompt, rollouts, tokenizer,
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.post(OPENROUTER_URL, json=body, headers=headers)
        response.raise_for_status()

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    parsed = _extract_json(raw_text)

    ratings = []
    for r in parsed.get("ratings", []):
        ratings.append(RolloutRating(
            index=r.get("index", 0),
            rating=r.get("rating", "indeterminate"),
            correction=r.get("correction"),
            malformed=r.get("malformed", False),
            reformatted=r.get("reformatted"),
            vague=r.get("vague", False),
        ))

    return JudgeResult(
        ratings=ratings,
        ideal_response=parsed.get("ideal_response"),
        raw_response=raw_text,
    )


async def judge_batch(
    examples: list[dict],
    rollouts_per_example: list[list[str]],
    oracle_prompts: list[str],
    tokenizer,
    model: str = "google/gemini-3.1-flash-lite-preview",
    temperature: float = 0.0,
    timeout_sec: float = 60.0,
    max_concurrent: int = 4,
) -> list[JudgeResult]:
    """Judge a batch of examples concurrently with rate limiting.

    Args:
        examples: List of example dicts with question, cot_response, base_positions, cot_start.
        rollouts_per_example: List of rollout lists per example.
        oracle_prompts: Oracle prompt used for each example.
        tokenizer: Tokenizer for position marking.
        model: Judge model name.
        temperature: Judge temperature.
        timeout_sec: Timeout per request.
        max_concurrent: Max concurrent requests.

    Returns:
        List of JudgeResult, one per example.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(idx: int) -> tuple[int, JudgeResult]:
        ex = examples[idx]
        async with semaphore:
            try:
                result = await judge_rollouts(
                    question=ex["question"],
                    cot_text=ex["cot_response"],
                    base_positions=ex["base_positions"],
                    cot_start=ex["cot_start"],
                    oracle_prompt=oracle_prompts[idx],
                    rollouts=rollouts_per_example[idx],
                    tokenizer=tokenizer,
                    model=model,
                    temperature=temperature,
                    timeout_sec=timeout_sec,
                )
                return idx, result
            except Exception as e:
                print(f"  [judge] Example {idx} failed: {e}")
                # Return empty result on failure
                return idx, JudgeResult(ratings=[], raw_response=str(e))

    tasks = [_judge_one(i) for i in range(len(examples))]
    results_unordered = await asyncio.gather(*tasks)

    # Reorder by index
    results = [None] * len(examples)
    for idx, result in results_unordered:
        results[idx] = result

    return results
