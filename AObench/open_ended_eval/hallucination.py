"""
Obvious hallucination eval — tests whether the AO confabulates with minimal activations.

Arya et al. found AOs confidently hallucinate wrong details: ice cubes → "eggs laid by
hens", plate arrangements → "letters in BANANA". This is worst with few activations.

This eval gives the AO only 1 or 5 activation positions and asks verifiable questions.
The judge checks whether the response is obviously wrong (mentions objects/concepts
that don't appear in the problem at all).
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from AObench import dataset_path
from AObench.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
    compute_segment_positions,
)
from AObench.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    get_first_ao_response,
    run_default_eval,
    run_verbalizer_generation_eval_loop,
)
from AObench.open_ended_eval.judge import (
    JUDGE_MODEL,
    DEFAULT_JUDGE_CONCURRENCY,
    judge_single,
)

BACKTRACKING_DATASET_PATH = dataset_path("datasets/backtracking/backtracking_eval_dataset.json")

GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 100,
}

# Verifiable questions — answers can be checked against the actual problem
VERBALIZER_PROMPTS: dict[str, str] = {
    "what_about": "What specific problem is the model solving? Be specific about the objects and quantities involved.",
    "what_numbers": "What numbers or values is the model working with?",
    "what_method": "What method or formula is the model applying?",
}

# Test with 1 position (stress test) and 5 positions (moderate)
POSITION_COUNTS = [1, 5]

JUDGE_SYSTEM_PROMPT = """\
You are checking whether an Activation Oracle (AO) response contains obvious hallucinations.

The AO was given a small number of activations from a language model solving a specific \
problem and asked a question. You need to check if the AO's response is obviously wrong.

You will be given:
1. The ACTUAL PROBLEM the model was solving
2. The NUMBER OF ACTIVATIONS the AO received (fewer = harder)
3. The AO's RESPONSE

Classify the response into exactly one category:

correct: The response accurately describes something from the actual problem. The specific \
details it mentions (objects, numbers, methods) actually appear in the problem.
vague: The response is too vague to be wrong — says generic things like "a math problem" \
or "working with numbers" without specific claims.
plausible_but_wrong: The response sounds reasonable and is in a related domain, but gets \
specific details wrong (e.g., wrong numbers, wrong objects, wrong method).
obviously_wrong: The response confidently describes objects, concepts, or a problem domain \
that has NOTHING to do with the actual problem. This is the critical failure mode — \
e.g., for an ice cube problem, saying "counting eggs laid by hens."

Respond with ONLY a JSON object: {"category": "<one of: correct, vague, plausible_but_wrong, obviously_wrong>", "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
ACTUAL PROBLEM: {problem}

NUMBER OF ACTIVATIONS GIVEN TO AO: {n_positions}

AO RESPONSE: {ao_response}"""


def load_dataset(max_entries: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(BACKTRACKING_DATASET_PATH).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0
    return entries


def build_hallucination_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    position_counts: list[int] = POSITION_COUNTS,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": entry["prefix"]},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer, messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )

        for n_pos in position_counts:
            # Take the last n_pos positions
            positions = compute_segment_positions(len(token_ids), -n_pos)

            for prompt_name, vp in verbalizer_prompts.items():
                prompt_infos.append(
                    VerbalizerInputInfo(
                        context_token_ids=token_ids,
                        positions=positions,
                        ground_truth=entry.get("domain", "unknown"),
                        verbalizer_prompt=vp,
                    )
                )
                entry_metadata.append({
                    "problem_id": entry.get("problem_id"),
                    "problem": entry["problem"],
                    "domain": entry.get("domain", "unknown"),
                    "prompt_name": prompt_name,
                    "n_positions": n_pos,
                })

    return prompt_infos, entry_metadata


async def judge_hallucination(
    results: list[VerbalizerResults],
    metadata: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    task_metadata = []

    async with httpx.AsyncClient() as client:
        for i, (result, meta) in enumerate(zip(results, metadata)):
            ao_response = get_first_ao_response(result)
            if ao_response is None:
                continue

            user_message = JUDGE_USER_TEMPLATE.format(
                problem=meta["problem"],
                n_positions=meta["n_positions"],
                ao_response=ao_response,
            )
            tasks.append(
                judge_single(client, JUDGE_SYSTEM_PROMPT, user_message, semaphore)
            )
            task_metadata.append({
                "result_index": i,
                "ao_response": ao_response,
                **meta,
            })

        print(f"Judging {len(tasks)} responses for hallucination with {JUDGE_MODEL} (concurrency={concurrency})...")
        pbar = tqdm(total=len(tasks), desc="Hallucination judge")

        async def _track(coro):
            r = await coro
            pbar.update(1)
            return r
        judge_results = await asyncio.gather(*[_track(t) for t in tasks], return_exceptions=True)
        pbar.close()

    scored = []
    for meta, jr in zip(task_metadata, judge_results):
        if isinstance(jr, Exception):
            print(f"Judge error for {meta['result_index']}: {jr}")
            continue
        scored.append({**meta, **jr})
    return scored


def compute_hallucination_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}
    total = len(scored_results)
    counts = {"correct": 0, "vague": 0, "plausible_but_wrong": 0, "obviously_wrong": 0}
    for r in scored_results:
        cat = r.get("category", "obviously_wrong")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["obviously_wrong"] += 1

    metrics: dict[str, float] = {
        "correct_rate": counts["correct"] / total,
        "hallucination_rate": (counts["plausible_but_wrong"] + counts["obviously_wrong"]) / total,
        "obvious_hallucination_rate": counts["obviously_wrong"] / total,
        "vague_rate": counts["vague"] / total,
        "correct": counts["correct"],
        "vague": counts["vague"],
        "plausible_but_wrong": counts["plausible_but_wrong"],
        "obviously_wrong": counts["obviously_wrong"],
        "total": float(total),
    }

    # Breakdown by n_positions
    for n_pos in POSITION_COUNTS:
        subset = [r for r in scored_results if r.get("n_positions") == n_pos]
        if not subset:
            continue
        n = len(subset)
        n_correct = sum(1 for r in subset if r.get("category") == "correct")
        n_obvious = sum(1 for r in subset if r.get("category") == "obviously_wrong")
        n_plausible = sum(1 for r in subset if r.get("category") == "plausible_but_wrong")
        metrics[f"pos{n_pos}_correct_rate"] = n_correct / n
        metrics[f"pos{n_pos}_hallucination_rate"] = (n_obvious + n_plausible) / n
        metrics[f"pos{n_pos}_obvious_hallucination_rate"] = n_obvious / n
        metrics[f"pos{n_pos}_total"] = float(n)

    return metrics


def run_hallucination_open_ended_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    eval_batch_size: int = 32,
    generation_kwargs: dict[str, Any] | None = None,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    max_entries: int | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_dataset(max_entries=max_entries)
    prompt_infos, entry_metadata = build_hallucination_verbalizer_prompt_infos(
        entries, VERBALIZER_PROMPTS, tokenizer,
    )

    def score_fn(results: list[VerbalizerResults], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            judge_hallucination(results=results, metadata=entry_metadata, concurrency=judge_concurrency)
        )

    return run_verbalizer_generation_eval_loop(
        eval_name="hallucination",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_fn,
        metrics_fn=compute_hallucination_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_default_eval(
        eval_name="hallucination",
        run_eval_fn=run_hallucination_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
