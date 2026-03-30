import json
import re
from pathlib import Path
from typing import Any

from AObench.base_experiment import (
    VerbalizerInputInfo,
    VerbalizerResults,
    tokenize_chat_messages,
    compute_segment_positions,
)
from AObench.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    build_verbalizer_eval_config,
    get_first_ao_response,
    run_default_eval,
    run_verbalizer_generation_eval_loop,
)

from AObench import dataset_path
DATASET_PATH = dataset_path("datasets/number_prediction/number_prediction_eval_dataset.json")

# Short answers (single number)
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 20,
}

VERBALIZER_PROMPTS: dict[str, str] = {
    "direct": "What number is the model about to output?",
    "planning": "What number is the model planning to answer with?",
    "predict": "Predict the exact numerical answer the model will give.",
}


def load_number_prediction_dataset(
    max_entries: int | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(DATASET_PATH).read_text())
    entries = data["entries"]
    if categories is not None:
        entries = [e for e in entries if e["category"] in categories]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_number_prediction_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        messages = [
            {"role": "user", "content": f"What is {entry['expression']}? Answer with just the number, nothing else."},
        ]
        token_ids = tokenize_chat_messages(tokenizer, messages)
        positions = compute_segment_positions(len(token_ids), segment_start)

        for prompt_name, vp in verbalizer_prompts.items():
            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=str(entry["model_answer"]),
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append({
                "id": entry["id"],
                "expression": entry["expression"],
                "category": entry["category"],
                "true_answer": entry["true_answer"],
                "model_answer": entry["model_answer"],
                "model_correct": entry["model_correct"],
                "is_single_token_answer": entry["is_single_token_answer"],
                "prompt_name": prompt_name,
            })

    return prompt_infos, entry_metadata


def extract_number_from_response(response: str) -> int | None:
    text = response.strip()
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group())
    return None


def score_results(
    results: list[VerbalizerResults],
    entry_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = []

    for result, meta in zip(results, entry_metadata):
        ao_response = get_first_ao_response(result)
        if ao_response is None:
            continue

        ao_number = extract_number_from_response(ao_response)

        scored.append({
            **meta,
            "ao_response": ao_response,
            "ao_number": ao_number,
            "matches_model_answer": ao_number == meta["model_answer"],
            "matches_true_answer": ao_number == meta["true_answer"],
        })

    return scored


def compute_metrics(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_results)
    matches_model = sum(1 for r in scored_results if r["matches_model_answer"])
    matches_true = sum(1 for r in scored_results if r["matches_true_answer"])
    has_number = sum(1 for r in scored_results if r["ao_number"] is not None)

    metrics: dict[str, Any] = {
        "total": total,
        "matches_model_answer": matches_model,
        "matches_model_answer_rate": matches_model / total if total > 0 else 0,
        "matches_true_answer": matches_true,
        "matches_true_answer_rate": matches_true / total if total > 0 else 0,
        "has_number_rate": has_number / total if total > 0 else 0,
    }

    # Break down by category
    categories = set(r["category"] for r in scored_results)
    for cat in sorted(categories):
        cat_results = [r for r in scored_results if r["category"] == cat]
        cat_total = len(cat_results)
        cat_model = sum(1 for r in cat_results if r["matches_model_answer"])
        cat_true = sum(1 for r in cat_results if r["matches_true_answer"])
        metrics[f"cat_{cat}_total"] = cat_total
        metrics[f"cat_{cat}_model_match_rate"] = cat_model / cat_total if cat_total > 0 else 0
        metrics[f"cat_{cat}_true_match_rate"] = cat_true / cat_total if cat_total > 0 else 0

    # Break down by single vs multi token
    single = [r for r in scored_results if r["is_single_token_answer"]]
    multi = [r for r in scored_results if not r["is_single_token_answer"]]
    if single:
        metrics["single_token_model_match_rate"] = sum(1 for r in single if r["matches_model_answer"]) / len(single)
    if multi:
        metrics["multi_token_model_match_rate"] = sum(1 for r in multi if r["matches_model_answer"]) / len(multi)

    # Break down by prompt
    prompts = set(r["prompt_name"] for r in scored_results)
    for prompt_name in sorted(prompts):
        p_results = [r for r in scored_results if r["prompt_name"] == prompt_name]
        p_total = len(p_results)
        p_model = sum(1 for r in p_results if r["matches_model_answer"])
        metrics[f"prompt_{prompt_name}_model_match_rate"] = p_model / p_total if p_total > 0 else 0

    return metrics


def print_sample_results(scored_results: list[dict[str, Any]]) -> None:
    print(f"\n  Sample results:")
    for r in scored_results[:15]:
        status = "✓" if r["matches_model_answer"] else "✗"
        print(
            f"    {status} {r['expression']:<35} model={r['model_answer']:<8} "
            f"ao={r['ao_number']!s:<8} prompt={r['prompt_name']:<10} "
            f"ao_raw={(r['ao_response'] or '')[:60]}"
        )


def run_number_prediction_open_ended_eval(
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
    categories: list[str] | None = None,
    verbalizer_prompts: dict[str, str] | None = None,
    segment_start: int | None = None,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_number_prediction_dataset(max_entries=max_entries, categories=categories)
    build_kwargs = {}
    if segment_start is not None:
        build_kwargs["segment_start"] = segment_start
    prompt_infos, entry_metadata = build_number_prediction_verbalizer_prompt_infos(entries, verbalizer_prompts, tokenizer, **build_kwargs)

    return run_verbalizer_generation_eval_loop(
        eval_name="number_prediction",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        score_fn=score_results,
        metrics_fn=compute_metrics,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
        print_sample_fn=print_sample_results,
    )


if __name__ == "__main__":
    run_default_eval(
        eval_name="number_prediction",
        run_eval_fn=run_number_prediction_open_ended_eval,
        model_name="Qwen/Qwen3-8B",
        run_eval_kwargs={
            "verbalizer_lora_paths": STANDARD_VERBALIZER_LORAS,
        },
    )
