"""
Activation sensitivity eval — tests whether the AO reads activations or just nearby tokens.

Uses the missing_info dataset's A vs C conditions: same tokens but different model states.
- A: complete prompt + neutral reasoning segment (model has all info)
- C: incomplete prompt + same neutral segment as A (identical tokens, different state)

If the AO reads activations, its free-form responses should differ between A and C.
Arya et al. found A and C get the same answer 53/54 times (98%).
"""

import asyncio
import json
from collections import defaultdict
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
    ensure_default_adapter,
    _load_adapter_and_build_config,
    get_first_ao_response,
    run_default_eval,
)
import AObench.base_experiment as base_experiment
from AObench.open_ended_eval.judge import (
    JUDGE_MODEL,
    DEFAULT_JUDGE_CONCURRENCY,
    judge_single,
)

DATASET_PATH = dataset_path("datasets/missing_info/missing_info_eval_dataset.json")

GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 150,
}

VERBALIZER_PROMPTS: dict[str, str] = {
    "reasoning_state": "What is the model's current reasoning state?",
    "confused": "Is the model confused or uncertain about something? If yes, describe what.",
    "info": "What information does the model have about this problem?",
}

JUDGE_SYSTEM_PROMPT = """\
You are comparing two responses from an Activation Oracle (AO) that was given identical text \
tokens but from different underlying model states.

Response A came from a model that had COMPLETE information about the problem.
Response C came from a model that had INCOMPLETE information (missing key details).

The text tokens the AO saw were IDENTICAL in both cases — only the model's internal state differed.

Compare the two responses and classify:

divergent_meaningful: The responses meaningfully differ AND the difference reflects the actual \
state change (e.g., response C mentions confusion/uncertainty that A doesn't, or A is more \
confident/specific than C). This means the AO is genuinely reading activations.
divergent_noise: The responses differ but the difference doesn't reflect the state change \
(random variation, different phrasing of the same content).
same: The responses are essentially identical in meaning — the AO gave the same answer \
regardless of the underlying model state. This is the failure mode.

Respond with ONLY a JSON object: {"category": "<one of: divergent_meaningful, divergent_noise, same>", "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
PROBLEM: {problem}

MISSING INFO (what condition C is missing): {missing_info}

RESPONSE A (complete info): {response_a}

RESPONSE C (incomplete info, same tokens): {response_c}"""


def load_dataset(
    conditions: list[str] | None = None,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(DATASET_PATH).read_text())
    entries = data["entries"]
    if conditions is not None:
        entries = [e for e in entries if e["condition"] in conditions]
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries


def build_prompt_infos_for_condition(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        reasoning = entry["full_reasoning"]
        teacher_forced = entry.get("teacher_forced_segment", "")
        problem = entry["problem_text"]

        content = f"{reasoning}\n\n{teacher_forced}" if teacher_forced else reasoning

        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": content},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer, messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        segment_token_count = entry["neutral_segment_token_count"]
        positions = compute_segment_positions(len(token_ids), -segment_token_count)

        for prompt_name, vp in verbalizer_prompts.items():
            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=str(entry["ground_truth_missing_info"]),
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append({
                "id": entry["id"],
                "problem_id": entry["problem_id"],
                "condition": entry["condition"],
                "problem_text": entry["problem_text"],
                "missing_info_description": entry["missing_info_description"],
                "prompt_name": prompt_name,
            })

    return prompt_infos, entry_metadata


async def judge_ac_pairs(
    pairs: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    async with httpx.AsyncClient() as client:
        for pair in pairs:
            user_message = JUDGE_USER_TEMPLATE.format(
                problem=pair["problem_text"],
                missing_info=pair["missing_info_description"],
                response_a=pair["response_a"],
                response_c=pair["response_c"],
            )
            tasks.append(judge_single(client, JUDGE_SYSTEM_PROMPT, user_message, semaphore))

        print(f"Judging {len(tasks)} A/C pairs with {JUDGE_MODEL} (concurrency={concurrency})...")
        pbar = tqdm(total=len(tasks), desc="A/C sensitivity judge")

        async def _track(coro):
            r = await coro
            pbar.update(1)
            return r
        judge_results = await asyncio.gather(*[_track(t) for t in tasks], return_exceptions=True)
        pbar.close()

    scored = []
    for pair, jr in zip(pairs, judge_results):
        if isinstance(jr, Exception):
            print(f"Judge error: {jr}")
            continue
        scored.append({**pair, **jr})
    return scored


def compute_sensitivity_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}
    total = len(scored_results)
    counts = {"divergent_meaningful": 0, "divergent_noise": 0, "same": 0}
    for r in scored_results:
        cat = r.get("category", "same")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["same"] += 1

    return {
        "activation_sensitivity": counts["divergent_meaningful"] / total,
        "same_rate": counts["same"] / total,
        "divergent_noise_rate": counts["divergent_noise"] / total,
        "divergent_meaningful": counts["divergent_meaningful"],
        "divergent_noise": counts["divergent_noise"],
        "same": counts["same"],
        "total": float(total),
    }


def run_activation_sensitivity_open_ended_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    eval_batch_size: int = 32,
    generation_kwargs: dict[str, Any] | None = None,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    max_entries_per_condition: int | None = None,
) -> dict[str, Any]:
    """Run A vs C activation sensitivity eval.

    For each verbalizer: run generation on A entries and C entries separately,
    pair up responses by problem_id + prompt_name, then judge divergence.
    """
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    a_entries = load_dataset(conditions=["A_complete"], max_entries=max_entries_per_condition)
    c_entries = load_dataset(conditions=["C_forced"], max_entries=max_entries_per_condition)

    a_prompt_infos, a_metadata = build_prompt_infos_for_condition(
        a_entries, VERBALIZER_PROMPTS, tokenizer,
    )
    c_prompt_infos, c_metadata = build_prompt_infos_for_condition(
        c_entries, VERBALIZER_PROMPTS, tokenizer,
    )

    ensure_default_adapter(model)
    model.eval()

    all_scored: list[dict[str, Any]] = []
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )
        verbalizer_key = verbalizer_entry.split("/")[-1]
        print(f"Running activation sensitivity with verbalizer: {verbalizer_entry}")

        # Run on A entries
        a_results = base_experiment.run_verbalizer(
            model=model, tokenizer=tokenizer,
            verbalizer_prompt_infos=a_prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config, device=device,
        )

        # Run on C entries
        c_results = base_experiment.run_verbalizer(
            model=model, tokenizer=tokenizer,
            verbalizer_prompt_infos=c_prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config, device=device,
        )

        # Pair up by problem_id + prompt_name
        a_responses: dict[str, str] = {}
        for result, meta in zip(a_results, a_metadata):
            resp = get_first_ao_response(result)
            if resp:
                key = f"{meta['problem_id']}_{meta['prompt_name']}"
                a_responses[key] = resp

        pairs = []
        for result, meta in zip(c_results, c_metadata):
            resp = get_first_ao_response(result)
            if resp is None:
                continue
            key = f"{meta['problem_id']}_{meta['prompt_name']}"
            if key not in a_responses:
                continue
            pairs.append({
                "problem_id": meta["problem_id"],
                "prompt_name": meta["prompt_name"],
                "problem_text": meta["problem_text"],
                "missing_info_description": meta["missing_info_description"],
                "response_a": a_responses[key],
                "response_c": resp,
                "verbalizer": verbalizer_key,
            })

        scored = asyncio.run(judge_ac_pairs(pairs, concurrency=judge_concurrency))
        metrics = compute_sensitivity_metrics(scored)
        metrics_by_verbalizer[verbalizer_key] = metrics

        print(f"\n  Sensitivity metrics for {verbalizer_key}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

        all_scored.extend(scored)

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            lora_name = verbalizer_key.replace("/", "_").replace(".", "_")
            output_path = Path(output_dir) / f"activation_sensitivity_{lora_name}.json"
            output_path.write_text(json.dumps({
                "verbalizer": verbalizer_entry,
                "metrics": metrics,
                "max_entries_per_condition": max_entries_per_condition,
                "pairs": pairs,
                "scored_results": scored,
            }, indent=2))

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    overall = compute_sensitivity_metrics(all_scored)
    return {
        "overall_metrics": overall,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_pairs": len(all_scored),
    }


if __name__ == "__main__":
    run_default_eval(
        eval_name="activation_sensitivity",
        run_eval_fn=run_activation_sensitivity_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
