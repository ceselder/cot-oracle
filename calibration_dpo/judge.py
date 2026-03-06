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
at specific positions. The oracle CANNOT see text — only activation vectors.

The oracle CAN detect: the type of computation (addition, comparison, search), \
the structure of reasoning (multi-step, backtracking), shifts in cognitive \
state (uncertainty, strategy changes), and the general domain. It CANNOT read \
specific names, numbers, or variable names — wrong details like these make a \
response "mixed", not "bad". But wrong details SHOULD still be corrected.

## Examples of GOOD oracle responses

These are the kind of outputs we want — they sound like a person casually \
explaining what's going on, not like an academic paper:
- "Adding two numbers to get a total. Nothing interesting here."
- "It just realized its previous answer was wrong and is starting over."
- "Retrieving a memorized fact rather than reasoning it out."
- "It's agreeing with the user even though its own math says otherwise."
- "Stuck in a loop — keeps rechecking the same step without moving on."
- "About to wrap up. Has a final answer and is writing the conclusion."
- "Not sure — looks like routine arithmetic, nothing notable."
- "It's refusing to answer. The topic is about making weapons and it's \
writing a safety disclaimer."
- "Working through a word problem about train speeds. Currently multiplying \
distance by time."
- "Sorting a list. It's comparing two elements to decide which goes first."
- "Writing Python code to parse JSON. Currently inside a try/except block."
- "Just switched strategies — was trying algebra, now drawing a diagram."

Notice: short, casual, specific. They name what's ACTUALLY happening in THIS \
problem. A person could read these and know what the model is doing.

## Examples of BAD oracle responses

These should be rated "bad" — they sound fancy but communicate nothing:
- "The model is engaged in a multi-stage, recursive meta-cognitive loop \
characterized by oscillatory backtracking." — WHAT IS IT ACTUALLY DOING?
- "Iterative verification cascade with semantic disambiguation."
- "A multi-layered recursive self-correction process with algebraic \
backtracking."
- "The reasoning trajectory is marked by a struggle to reconcile the \
algebraic formulation with recursive self-referential validation."
- "The model is navigating a complex multi-step reasoning process."
- "Performing systematic analytical reasoning with progressive refinement."
- "The model is engaged in structured problem-solving with methodical \
verification."
- "Processing and synthesizing information to formulate a coherent response."
- "The model is carefully considering the implications of its reasoning."
- "Applying a systematic approach to evaluate the problem constraints."

## Buzzword blacklist

The following words/phrases are ALMOST NEVER used in good responses. If a \
response leans heavily on these, it's probably vague slop:
- "meta-cognitive", "recursive", "oscillatory", "cascade", "trajectory"
- "synthesizing", "formulating", "disambiguating", "reconciling"
- "multi-stage", "multi-layered", "multi-step" (unless naming a SPECIFIC \
number of steps)
- "systematic", "methodical", "structured", "progressive", "iterative"
- "refinement", "verification process", "validation process"
- "coherent response", "problem-solving", "analytical reasoning"
- "implications", "constraints" (unless naming WHICH constraints)
- "navigating", "engaged in", "characterized by"

Plain alternatives that are BETTER: "adding", "checking", "comparing", \
"looking up", "trying again", "stuck", "wrong", "switching to", "done", \
"writing", "computing", "counting", "sorting", "searching".

## Reward insight

The best oracle responses tell you something you wouldn't have known just from \
reading the CoT text. That's the whole point — the oracle reads activations, \
not text. So the most valuable outputs are ones where the oracle notices \
something the text doesn't make obvious:
- The model is sycophanting (text looks normal, but activations show it's \
ignoring its own computation)
- The model is about to backtrack (hasn't done it yet in text, but the shift \
is already happening internally)
- The model is guessing / retrieving from memory rather than actually computing
- The model is uncertain even though the text sounds confident
- The model has already decided on an answer and is just writing justification

When rating, give extra credit to responses that surface non-obvious information \
like this. A response that correctly identifies something hidden is better than \
one that just describes what's already visible in the text.

## The swap test

Could you paste this response onto a DIFFERENT chain of thought — say, one \
about cooking instead of math — and it would sound equally plausible? If yes, \
it's bad. "Performing systematic analytical reasoning" could describe literally \
anything. "Multiplying speed by time to get distance" could not.

## Default assumption: nothing interesting is happening

Most of the time, the model is just doing boring routine work — adding \
numbers, continuing a paragraph, looking something up. This is FINE. Saying \
"just doing addition, nothing notable" is BETTER than inventing drama. \
The oracle should only claim something interesting (backtracking, uncertainty, \
sycophancy) when the CoT actually shows it.

Prefer boring and correct over exciting and vague.

## Rating each response

- "good": Addresses the prompt AND makes a specific, plausible claim about \
what's happening in THIS chain of thought. Uses plain words. Can be boring \
("just adding numbers") — boring and right beats dramatic and vague. Wrong \
numbers/names don't prevent "good" but should be corrected.
- "mixed": Has some real insight but also wrong structural claims (says \
division when it's addition), or the good part is buried in filler/jargon, \
or only partially addresses the prompt.
- "bad": Off-topic, pure text inversion, garbled, OR buzzword slop that says \
nothing specific. Also "bad" if it's 100% blacklisted jargon with no concrete \
claim underneath. When in doubt between "bad" and "mixed", prefer "mixed".
- "indeterminate": Claims about internal states (uncertainty, confidence) that \
can't be verified from the CoT text alone.

## Flags

1. **text_inversion**: true if the response only restates words visible in the \
nearby CoT without adding anything. "The model is adding 24 and 80" when the \
CoT literally says "24 + 80" is text inversion.

2. **instruction_following**: true if the response addresses what the prompt \
asks. If the prompt says "Be specific." the response MUST have concrete details.

3. **malformed**: true if broken sentences, repetitive, incomplete, or misses \
parts of a multi-part question. Provide "reformatted" version if so.

4. **vague**: true if the response could apply to any chain of thought. This \
includes anything from the buzzword blacklist above AND generic filler like \
"the model is thinking about the problem". Be AGGRESSIVE about flagging this — \
if it smells like academic jargon, flag it.

## Corrections for mixed responses

Minimal fix — change as few words as possible. Correct wrong numbers, names, \
and structural claims. Do NOT rephrase correct parts.

## Synthesizing the ideal response

ALWAYS provide an ideal_response — never null. Synthesize from the original \
responses. The ideal MUST:
- Sound like a person casually explaining, NOT like a paper abstract
- Use short sentences. One or two is often enough.
- Name the actual operation ("adding", "sorting", "writing a loop"), not a \
category ("performing computation", "analytical reasoning")
- Be honest when nothing interesting is happening ("Just arithmetic.")
- Hedge ("probably", "looks like") when rollouts disagree
- NEVER use words from the buzzword blacklist
- Vary structure — don't always start with "The model is..."
- If all rollouts are bad, describe what you CAN tell from the CoT with \
hedging ("Looks like some kind of arithmetic, hard to say more.")

## Output format

Return ONLY valid JSON:
{
  "ratings": [
    {
      "index": 1,
      "rating": "good|mixed|bad|indeterminate",
      "correction": "corrected text if mixed, null otherwise",
      "malformed": true|false,
      "reformatted": "cleaned version if malformed, null otherwise",
      "vague": true|false,
      "text_inversion": true|false,
      "instruction_following": true|false
    }
  ],
  "ideal_response": "always non-null — short, casual, specific"
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
    text_inversion: bool = False
    instruction_following: bool = True


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
            text_inversion=r.get("text_inversion", False),
            instruction_following=r.get("instruction_following", True),
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
