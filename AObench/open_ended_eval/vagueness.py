"""
Vagueness rate eval — measures what fraction of AO responses are vague/generic.

Arya et al. found 49.4% of AO responses across diverse probes were vague.
This eval systematically measures this using diverse prompts from the DPO
prompt library on CoT rollouts from the backtracking dataset.
"""

import asyncio
import json
import random
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
PROMPTS_PATH = dataset_path("datasets/vagueness/prompts.json")

GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 150,
}

JUDGE_SYSTEM_PROMPT = """\
You are classifying whether an Activation Oracle (AO) response is vague or specific.

The AO was given activations from a language model working on a math/logic problem and asked a question about the model's internal state.

You will be given:
1. The PROBLEM the model was working on
2. The QUESTION asked to the AO
3. The AO's RESPONSE

Classify the response into exactly one category:

specific_correct: The response names something concrete and relevant to the actual problem. \
It mentions specific objects, quantities, methods, or reasoning steps from the problem.
vague_directional: The response is in the right general area but is unfalsifiable or could \
apply to many problems. Examples: "uncertain about its calculations", "considering the \
mathematical approach", "verifying the result".
generic: The response could apply to literally any problem. Examples: "the model is thinking", \
"the model is processing information", "considering alternatives".
refuses: The response explicitly says it cannot determine the answer, says "not sure", \
or declines to answer.

Respond with ONLY a JSON object: {"category": "<one of: specific_correct, vague_directional, generic, refuses>", "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
PROBLEM: {problem}

QUESTION: {question}

AO RESPONSE: {ao_response}"""


def load_data(
    max_entries: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load backtracking rollouts and vagueness prompts."""
    bt_data = json.loads(Path(BACKTRACKING_DATASET_PATH).read_text())
    entries = bt_data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in backtracking dataset"

    prompts_data = json.loads(Path(PROMPTS_PATH).read_text())
    prompts = prompts_data["prompts"]
    return entries, prompts


def build_vagueness_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    prompts: list[str],
    tokenizer,
    segment_start: int = -20,
    seed: int = 42,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    """For each entry, sample 1 prompt deterministically."""
    rng = random.Random(seed)
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
        positions = compute_segment_positions(len(token_ids), segment_start)

        vp = rng.choice(prompts)
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
            "prompt": vp,
        })

    return prompt_infos, entry_metadata


async def judge_vagueness(
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
                problem=meta["problem"][:500],
                question=meta["prompt"],
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

        print(f"Judging {len(tasks)} responses for vagueness with {JUDGE_MODEL} (concurrency={concurrency})...")
        pbar = tqdm(total=len(tasks), desc="Vagueness judge")

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


def compute_vagueness_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}
    total = len(scored_results)
    counts = {"specific_correct": 0, "vague_directional": 0, "generic": 0, "refuses": 0}
    for r in scored_results:
        cat = r.get("category", "generic")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["generic"] += 1  # unknown → generic

    vague_total = counts["vague_directional"] + counts["generic"]
    return {
        "vagueness_rate": vague_total / total,
        "specificity_rate": counts["specific_correct"] / total,
        "refusal_rate": counts["refuses"] / total,
        "generic_rate": counts["generic"] / total,
        "vague_directional_rate": counts["vague_directional"] / total,
        "specific_correct": counts["specific_correct"],
        "vague_directional": counts["vague_directional"],
        "generic": counts["generic"],
        "refuses": counts["refuses"],
        "total": float(total),
    }


def run_vagueness_open_ended_eval(
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

    entries, prompts = load_data(max_entries=max_entries)
    prompt_infos, entry_metadata = build_vagueness_verbalizer_prompt_infos(
        entries, prompts, tokenizer,
    )

    def score_fn(results: list[VerbalizerResults], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            judge_vagueness(results=results, metadata=entry_metadata, concurrency=judge_concurrency)
        )

    return run_verbalizer_generation_eval_loop(
        eval_name="vagueness",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_fn,
        metrics_fn=compute_vagueness_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_default_eval(
        eval_name="vagueness",
        run_eval_fn=run_vagueness_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
