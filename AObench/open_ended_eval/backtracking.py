import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
from tqdm import tqdm

from AObench.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
    compute_segment_positions,
)
import AObench.base_experiment as base_experiment
from AObench.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    _load_adapter_and_build_config,
    build_verbalizer_eval_config,
    ensure_default_adapter,
    get_first_ao_response,
    run_default_eval,
    run_verbalizer_generation_eval_loop,
)

from AObench import dataset_path
DATASET_PATH = dataset_path("datasets/backtracking/backtracking_eval_dataset.json")

# Backtracking explanations need more tokens than short-answer evals
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 150,
}

VERBALIZER_PROMPTS: tuple[str, ...] = ("What is the model uncertain or confused about at this point in its reasoning?",)

JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_CONCURRENCY = 20

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether an Activation Oracle (AO) — a model that reads another model's \
internal activations — correctly identified what a reasoning model was uncertain about.

You will be given:
1. The reasoning model's chain-of-thought PREFIX (up to the point of uncertainty)
2. The GROUND TRUTH description of what the model was actually uncertain about
3. The AO's RESPONSE describing what it thinks the model was uncertain about

Rate the AO's response on two dimensions:

SPECIFICITY (1-5):
1 = Completely vague/generic (e.g. "the model is thinking" or "the model is uncertain")
2 = Slightly specific but mostly generic (e.g. "the model is uncertain about a math problem")
3 = Moderately specific (identifies the general topic but not the precise uncertainty)
4 = Specific (identifies the topic and gives a reasonable description of the uncertainty)
5 = Highly specific (pinpoints the exact nature of the uncertainty)

CORRECTNESS (1-5):
1 = Completely wrong or irrelevant
2 = Tangentially related but misidentifies the core uncertainty
3 = Partially correct (right general area but wrong specifics)
4 = Mostly correct (captures the main uncertainty with minor inaccuracies)
5 = Fully correct (accurately describes the uncertainty)

Respond with ONLY a JSON object: {"specificity": <int>, "correctness": <int>, "reasoning": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """\
PREFIX (reasoning up to uncertainty point):
{prefix}

GROUND TRUTH (what the model was actually uncertain about):
{ground_truth}

AO RESPONSE:
{ao_response}"""


def load_backtracking_dataset(
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(DATASET_PATH).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_backtracking_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
    tokenizer=None,
    segment_start: int = -20,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    """Build verbalizer prompt infos and return paired metadata for each entry."""
    assert tokenizer is not None, "tokenizer is required"
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        prefix = entry["prefix"]

        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": prefix},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        positions = compute_segment_positions(len(token_ids), segment_start)

        for vp in verbalizer_prompts:
            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=entry["uncertainty_description"],
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append(
                {
                    "problem_id": entry.get("problem_id"),
                    "backtrack_rate": entry["backtrack_rate"],
                    "bucket": entry.get("bucket"),
                    "prefix_length": len(entry["prefix"]),
                }
            )

    return prompt_infos, entry_metadata


# --- LLM judge for backtracking (eval-specific, not shared) ---


async def judge_single_response(
    client: anthropic.AsyncAnthropic,
    prefix: str,
    ground_truth: str,
    ao_response: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    user_message = JUDGE_USER_TEMPLATE.format(
        prefix=prefix[-1500:],  # truncate prefix for judge context
        ground_truth=ground_truth,
        ao_response=ao_response,
    )

    async with semaphore:
        response = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

    text = response.content[0].text
    # Parse JSON from response — strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove ```json line
        text = text.rsplit("```", 1)[0]  # remove closing ```
        text = text.strip()
    result = json.loads(text)
    assert "specificity" in result and "correctness" in result
    return result


async def judge_ao_responses(
    results: list[VerbalizerResults],
    entries: list[dict[str, Any]],
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Use Claude Haiku to judge AO responses against ground truth."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    task_metadata = []

    for i, (result, entry) in enumerate(zip(results, entries)):
        ao_response = get_first_ao_response(result)
        if ao_response is None:
            continue

        tasks.append(
            judge_single_response(
                client=client,
                prefix=entry["prefix"],
                ground_truth=entry["uncertainty_description"],
                ao_response=ao_response,
                semaphore=semaphore,
            )
        )
        task_metadata.append(
            {
                "result_index": i,
                "ao_response": ao_response,
                "ground_truth": entry["uncertainty_description"],
                "backtrack_rate": entry["backtrack_rate"],
            }
        )

    print(f"Judging {len(tasks)} AO responses with {JUDGE_MODEL} (concurrency={concurrency})...")
    pbar = tqdm(total=len(tasks), desc="LLM judge")
    async def _track(coro):
        result = await coro
        pbar.update(1)
        return result
    judge_results = await asyncio.gather(*[_track(t) for t in tasks], return_exceptions=True)
    pbar.close()

    scored_results = []
    for meta, judge_result in zip(task_metadata, judge_results):
        if isinstance(judge_result, Exception):
            print(f"Judge error for result {meta['result_index']}: {judge_result}")
            continue
        scored_results.append({**meta, **judge_result})

    return scored_results


def compute_judge_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    specificities = [r["specificity"] for r in scored_results]
    correctnesses = [r["correctness"] for r in scored_results]

    return {
        "mean_specificity": sum(specificities) / len(specificities),
        "mean_correctness": sum(correctnesses) / len(correctnesses),
        "specificity_>=3_rate": sum(1 for s in specificities if s >= 3) / len(specificities),
        "specificity_>=4_rate": sum(1 for s in specificities if s >= 4) / len(specificities),
        "correctness_>=3_rate": sum(1 for c in correctnesses if c >= 3) / len(correctnesses),
        "correctness_>=4_rate": sum(1 for c in correctnesses if c >= 4) / len(correctnesses),
        "num_scored": float(len(scored_results)),
    }


# ---------------------------------------------------------------------------
# Multiple-choice eval
# ---------------------------------------------------------------------------

ANSWER_LETTERS = ["A", "B", "C", "D"]

MC_VERBALIZER_PROMPT_TEMPLATE = (
    "What is the model most likely uncertain about at this point? "
    "Answer with just the letter (A, B, C, or D).\n\n"
    "{options_text}"
)

# Single-token variants of A/B/C/D for logit scoring
LETTER_CANDIDATE_VARIANTS: dict[str, list[str]] = {
    "A": ["A", " A", "\nA"],
    "B": ["B", " B", "\nB"],
    "C": ["C", " C", "\nC"],
    "D": ["D", " D", "\nD"],
}


def build_letter_candidate_token_groups(tokenizer) -> dict[str, list[int]]:
    """Collect single-token A/B/C/D variants for first-token scoring."""
    token_groups: dict[str, list[int]] = {}
    for label, variants in LETTER_CANDIDATE_VARIANTS.items():
        token_ids: list[int] = []
        for text in variants:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in token_ids:
                token_ids.append(int(ids[0]))
        if not token_ids:
            raise ValueError(f"Tokenizer had no single-token variants for label '{label}'")
        token_groups[label] = token_ids
    return token_groups


def _format_mc_options(mc_options: list[str]) -> str:
    """Format MC options as A. ... B. ... etc."""
    lines = []
    for i, opt in enumerate(mc_options):
        lines.append(f"{ANSWER_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def build_backtracking_mc_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    tokenizer,
    segment_start: int = -20,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    """Build verbalizer prompt infos for MC backtracking eval."""
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        assert "mc_options" in entry, f"Entry {entry.get('problem_id')} missing mc_options"

        messages = [
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": entry["prefix"]},
        ]
        token_ids = tokenize_chat_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
            continue_thinking=True,
        )
        positions = compute_segment_positions(len(token_ids), segment_start)

        options_text = _format_mc_options(entry["mc_options"])
        verbalizer_prompt = MC_VERBALIZER_PROMPT_TEMPLATE.format(options_text=options_text)

        prompt_infos.append(
            VerbalizerInputInfo(
                context_token_ids=token_ids,
                positions=positions,
                ground_truth=entry["mc_correct_label"],
                verbalizer_prompt=verbalizer_prompt,
            )
        )
        entry_metadata.append(
            {
                "problem_id": entry.get("problem_id"),
                "backtrack_rate": entry["backtrack_rate"],
                "mc_correct_label": entry["mc_correct_label"],
                "mc_correct_index": entry["mc_correct_index"],
            }
        )

    return prompt_infos, entry_metadata


def run_backtracking_mc_eval(
    *,
    model_name: str,
    model,
    tokenizer,
    device,
    eval_batch_size: int = 64,
    generation_kwargs: dict[str, Any] | None = None,
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Multiple-choice backtracking eval using logit scoring over A/B/C/D."""
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "max_new_tokens": 1}

    entries = load_backtracking_dataset(max_entries=max_entries)
    # Filter to entries that have MC options
    entries = [e for e in entries if "mc_options" in e]
    assert entries, "No entries with mc_options found in dataset"

    prompt_infos, entry_metadata = build_backtracking_mc_verbalizer_prompt_infos(
        entries, tokenizer,
    )

    candidate_token_groups = build_letter_candidate_token_groups(tokenizer)

    ensure_default_adapter(model)
    model.eval()

    results_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        print(f"Running backtracking MC eval with verbalizer: {verbalizer_entry}")
        verbalizer_key = verbalizer_entry.split("/")[-1]

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config,
            device=device,
            candidate_token_groups=candidate_token_groups,
        )

        scored_results = []
        correct = 0
        total = 0
        for result, meta in zip(binary_results, entry_metadata):
            # Pick letter with highest candidate score
            scores = {label: float(result.candidate_scores[label]) for label in ANSWER_LETTERS}
            predicted_label = max(scores, key=scores.get)  # type: ignore[arg-type]
            is_correct = predicted_label == meta["mc_correct_label"]
            total += 1
            if is_correct:
                correct += 1

            scored_results.append({
                **meta,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "scores": scores,
                "argmax_token_text": result.argmax_token_text.strip(),
            })

        metrics = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "chance": 0.25,
        }
        results_by_verbalizer[verbalizer_key] = metrics

        print(f"\n  MC accuracy for {verbalizer_key}: {correct}/{total} = {metrics['accuracy']:.3f} (chance=0.25)")

        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            lora_name = verbalizer_key.replace("/", "_").replace(".", "_")
            output_path = os.path.join(output_dir, f"backtracking_mc_{lora_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "verbalizer": verbalizer_entry,
                    "metrics": metrics,
                    "scored_results": scored_results,
                }, f, indent=2)
            print(f"  Saved to {output_path}")

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    return {"mc_results_by_verbalizer": results_by_verbalizer}


def run_backtracking_open_ended_eval(
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
    """
    Backtracking eval uses an LLM judge, so it can't use the standard score_fn pattern
    directly (the judge needs the original entries, not just metadata). We wrap the
    judge call in a score_fn closure.
    """
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_backtracking_dataset(max_entries=max_entries)
    prompt_infos, entry_metadata = build_backtracking_verbalizer_prompt_infos(
        entries,
        tokenizer=tokenizer,
        segment_start=-20,
    )

    # Backtracking uses async LLM judging — wrap it as a score_fn
    def score_fn(results: list[VerbalizerResults], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(
            judge_ao_responses(
                results=results,
                entries=entries,
                concurrency=judge_concurrency,
            )
        )

    return run_verbalizer_generation_eval_loop(
        eval_name="backtracking",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_fn,
        metrics_fn=compute_judge_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        extra_output_data={"max_entries": max_entries},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["open_ended", "mc"], default="mc")
    args = parser.parse_args()

    if args.mode == "mc":
        run_default_eval(
            eval_name="backtracking_mc",
            run_eval_fn=run_backtracking_mc_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
            },
        )
    else:
        run_default_eval(
            eval_name="backtracking",
            run_eval_fn=run_backtracking_open_ended_eval,
            model_name="Qwen/Qwen3-8B",
            run_eval_kwargs={
                "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
            },
        )
