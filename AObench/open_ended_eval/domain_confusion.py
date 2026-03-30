"""
Domain confusion eval — measures how often the AO hallucinates the wrong problem domain.

Arya et al. found AOs hallucinate wildly wrong domains: circular permutations → "letters
in BANANA", ice cube melting → "eggs laid by hens". This eval measures the rate of such
domain-level hallucinations vs correct-but-vague vs specific-and-correct responses.
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
    "max_new_tokens": 150,
}

VERBALIZER_PROMPTS: dict[str, str] = {
    "what_problem": "What specific problem is the model working on?",
    "what_computing": "Describe what the model is computing right now.",
    "what_about": "What is this chain of thought about?",
}

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether an Activation Oracle (AO) correctly identified what problem \
a language model was working on, or whether it hallucinated a completely different domain.

You will be given:
1. The ACTUAL PROBLEM the model was solving
2. The AO's RESPONSE describing what it thinks the model is working on

Classify the response into exactly one category:

domain_correct_specific: The AO correctly identifies the problem domain AND mentions specific \
objects, quantities, or methods from the actual problem. Example: for a problem about "ice cubes \
in a frying pan", saying "the model is computing how many ice cubes remain after melting."
domain_correct_vague: The AO gets the general domain right but is vague enough that the \
response could apply to many problems in that domain. Example: for a number theory problem, \
saying "the model is working on a mathematical calculation."
domain_wrong: The AO hallucinates a completely different problem domain. Example: for an ice \
cube problem, saying "the model is counting eggs laid by hens." This is the critical failure mode.
refuses: The AO explicitly declines to answer or says it cannot determine.

Respond with ONLY a JSON object: {"category": "<one of: domain_correct_specific, domain_correct_vague, domain_wrong, refuses>", "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
ACTUAL PROBLEM: {problem}

AO RESPONSE: {ao_response}"""


def load_dataset(max_entries: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(BACKTRACKING_DATASET_PATH).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0
    return entries


def build_domain_confusion_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -20,
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
        positions = compute_segment_positions(len(token_ids), segment_start)

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
            })

    return prompt_infos, entry_metadata


async def judge_domain_confusion(
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

        print(f"Judging {len(tasks)} responses for domain confusion with {JUDGE_MODEL} (concurrency={concurrency})...")
        pbar = tqdm(total=len(tasks), desc="Domain confusion judge")

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


def compute_domain_confusion_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}
    total = len(scored_results)
    counts = {"domain_correct_specific": 0, "domain_correct_vague": 0, "domain_wrong": 0, "refuses": 0}
    for r in scored_results:
        cat = r.get("category", "domain_wrong")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["domain_wrong"] += 1

    non_refused = total - counts["refuses"]
    return {
        "domain_confusion_rate": counts["domain_wrong"] / non_refused if non_refused > 0 else 0.0,
        "specificity_when_correct": counts["domain_correct_specific"] / non_refused if non_refused > 0 else 0.0,
        "domain_correct_specific_rate": counts["domain_correct_specific"] / total,
        "domain_correct_vague_rate": counts["domain_correct_vague"] / total,
        "domain_wrong_rate": counts["domain_wrong"] / total,
        "refusal_rate": counts["refuses"] / total,
        "domain_correct_specific": counts["domain_correct_specific"],
        "domain_correct_vague": counts["domain_correct_vague"],
        "domain_wrong": counts["domain_wrong"],
        "refuses": counts["refuses"],
        "total": float(total),
    }


def run_domain_confusion_open_ended_eval(
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
    segment_start: int | None = None,
) -> dict[str, Any]:
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_dataset(max_entries=max_entries)
    build_kwargs = {}
    if segment_start is not None:
        build_kwargs["segment_start"] = segment_start
    prompt_infos, entry_metadata = build_domain_confusion_verbalizer_prompt_infos(
        entries, VERBALIZER_PROMPTS, tokenizer, **build_kwargs,
    )

    def score_fn(results: list[VerbalizerResults], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            judge_domain_confusion(results=results, metadata=entry_metadata, concurrency=judge_concurrency)
        )

    return run_verbalizer_generation_eval_loop(
        eval_name="domain_confusion",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_fn,
        metrics_fn=compute_domain_confusion_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_default_eval(
        eval_name="domain_confusion",
        run_eval_fn=run_domain_confusion_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
