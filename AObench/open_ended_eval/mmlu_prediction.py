import json
import os
from pathlib import Path
from typing import Any

from AObench.base_experiment import (
    VerbalizerInputInfo,
    tokenize_chat_messages,
    compute_segment_positions,
)
import AObench.base_experiment as base_experiment
from AObench.open_ended_eval.eval_runner import (
    STANDARD_VERBALIZER_LORAS,
    _load_adapter_and_build_config,
    build_yes_no_candidate_token_groups,
    ensure_default_adapter,
    run_default_eval,
    run_verbalizer_binary_eval_loop,
)

from AObench import dataset_path
DATASET_PATH = dataset_path("datasets/mmlu_prediction/mmlu_prediction_eval_dataset.json")

# Not used for binary scoring (single forward pass, no generation), but required
# by VerbalizerEvalConfig. Kept minimal.
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 1,
}

ANSWER_LETTERS = ["A", "B", "C", "D"]


PREFIX = "Answer with 'Yes' or 'No' only. "

# One prompt per mode — see datasets/mmlu_prediction/prompt_selection.md for sweep details.
# Using a single prompt per mode avoids mixing thresholds, which degrades combined ROC AUC.
PRE_ANSWER_PROMPTS: dict[str, str] = {
    "pre_likely_correct": f"{PREFIX}Will the model likely answer this correctly?",
}

POST_ANSWER_PROMPTS: dict[str, str] = {
    "post_likely_correct": f"{PREFIX}Is the model's answer likely to be correct?",
}

VERBALIZER_PROMPTS: dict[str, str] = {**PRE_ANSWER_PROMPTS, **POST_ANSWER_PROMPTS}


def format_mmlu_question(question: str, choices: list[str]) -> str:
    """Format an MMLU question with lettered choices (must match generate_dataset.py)."""
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append("")
    lines.append("Answer with just the letter (A, B, C, or D), nothing else.")
    return "\n".join(lines)


def load_mmlu_prediction_dataset(
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(DATASET_PATH).read_text())
    entries = data["entries"]
    if max_entries is not None:
        entries = entries[:max_entries]
    assert len(entries) > 0, "No entries in dataset"
    return entries


def build_mmlu_prediction_verbalizer_prompt_infos(
    entries: list[dict[str, Any]],
    verbalizer_prompts: dict[str, str],
    tokenizer,
    segment_start: int = -10,
) -> tuple[list[VerbalizerInputInfo], list[dict[str, Any]]]:
    prompt_infos: list[VerbalizerInputInfo] = []
    entry_metadata: list[dict[str, Any]] = []

    for entry in entries:
        question_text = format_mmlu_question(entry["question"], entry["choices"])

        # Pre-answer context: question only, no assistant response
        pre_answer_messages = [
            {"role": "user", "content": question_text},
        ]
        pre_answer_token_ids = tokenize_chat_messages(tokenizer, pre_answer_messages)
        pre_answer_positions = compute_segment_positions(len(pre_answer_token_ids), segment_start)

        # Post-answer context: question + model's single-letter answer
        post_answer_messages = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": entry["model_answer_letter"]},
        ]
        post_answer_token_ids = tokenize_chat_messages(tokenizer, post_answer_messages)
        post_answer_positions = compute_segment_positions(len(post_answer_token_ids), segment_start)

        for prompt_name, vp in verbalizer_prompts.items():
            is_post_answer = prompt_name.startswith("post_")
            token_ids = post_answer_token_ids if is_post_answer else pre_answer_token_ids
            positions = post_answer_positions if is_post_answer else pre_answer_positions

            ground_truth = "yes" if entry["model_correct"] else "no"

            prompt_infos.append(
                VerbalizerInputInfo(
                    context_token_ids=token_ids,
                    positions=positions,
                    ground_truth=ground_truth,
                    verbalizer_prompt=vp,
                )
            )
            entry_metadata.append(
                {
                    "id": entry["id"],
                    "subject": entry["subject"],
                    "correct_answer_letter": entry["correct_answer_letter"],
                    "model_answer_letter": entry["model_answer_letter"],
                    "model_correct": entry["model_correct"],
                    "prompt_name": prompt_name,
                }
            )

    return prompt_infos, entry_metadata


LETTER_PREDICTION_PROMPT = "Predict the model's answer. Output only A, B, C, or D."


def run_letter_prediction(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    eval_batch_size: int,
    generation_kwargs: dict[str, Any],
    entries: list[dict[str, Any]],
    verbalizer_lora_paths: list[str],
    output_dir: str | None = None,
    segment_start: int | None = None,
) -> dict[str, Any]:
    """Predict which answer letter (A-D) the model will choose, using argmax token.

    Uses pre-answer context only (question, no model answer). Checks if the
    AO's argmax output token matches the model's actual answer letter.
    """
    build_kwargs = {}
    if segment_start is not None:
        build_kwargs["segment_start"] = segment_start
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(
        entries,
        {"predict_letter": LETTER_PREDICTION_PROMPT},
        tokenizer,
        **build_kwargs,
    )

    # We need candidate_token_groups to call run_verbalizer_binary_score,
    # but we only care about the argmax token — the yes/no scores are ignored.
    dummy_candidate_groups = build_yes_no_candidate_token_groups(tokenizer)

    ensure_default_adapter(model)
    model.eval()

    results_by_verbalizer: dict[str, dict[str, Any]] = {}

    for verbalizer_entry in verbalizer_lora_paths:
        sanitized_name, config = _load_adapter_and_build_config(
            model, verbalizer_entry, model_name, eval_batch_size, generation_kwargs,
        )

        binary_results = base_experiment.run_verbalizer_binary_score(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=prompt_infos,
            verbalizer_lora_path=sanitized_name,
            target_lora_path=None,
            config=config,
            device=device,
            candidate_token_groups=dummy_candidate_groups,
        )

        scored_results = []
        matches_model = 0
        matches_true = 0
        total = 0
        for result, meta in zip(binary_results, entry_metadata):
            predicted_letter = result.argmax_token_text.strip()
            parseable = predicted_letter in ANSWER_LETTERS
            if parseable:
                total += 1
                if predicted_letter == meta["model_answer_letter"]:
                    matches_model += 1
                if predicted_letter == meta["correct_answer_letter"]:
                    matches_true += 1
            scored_results.append({
                "predicted_letter": predicted_letter,
                "model_answer_letter": meta["model_answer_letter"],
                "correct_answer_letter": meta["correct_answer_letter"],
                "matches_model": parseable and predicted_letter == meta["model_answer_letter"],
                "matches_true": parseable and predicted_letter == meta["correct_answer_letter"],
                "parseable": parseable,
            })

        verbalizer_key = verbalizer_entry.split("/")[-1]
        lora_name = verbalizer_key.replace("/", "_").replace(".", "_")
        metrics = {
            "total": total,
            "matches_model_rate": matches_model / total if total else 0,
            "matches_true_rate": matches_true / total if total else 0,
            "unparseable": len(binary_results) - total,
        }
        results_by_verbalizer[verbalizer_key] = metrics
        print(f"\n  Letter prediction for {verbalizer_key}:")
        print(f"    matches_model: {matches_model}/{total} = {metrics['matches_model_rate']:.3f}")
        print(f"    matches_true:  {matches_true}/{total} = {metrics['matches_true_rate']:.3f}")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"mmlu_letter_prediction_{lora_name}.json")
            with open(output_path, "w") as f:
                json.dump({
                    "verbalizer": verbalizer_entry,
                    "metrics": metrics,
                    "scored_results": scored_results,
                }, f, indent=2)
            print(f"  Saved letter prediction to {output_path}")

        if sanitized_name in model.peft_config:
            model.delete_adapter(sanitized_name)

    return {"letter_prediction_by_verbalizer": results_by_verbalizer}


def run_mmlu_prediction_open_ended_eval(
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
    verbalizer_prompts: dict[str, str] | None = None,
    run_letter_prediction_eval: bool = False,
    segment_start: int | None = None,
) -> dict[str, Any]:
    if verbalizer_prompts is None:
        verbalizer_prompts = VERBALIZER_PROMPTS
    if generation_kwargs is None:
        generation_kwargs = GENERATION_KWARGS

    entries = load_mmlu_prediction_dataset(max_entries=max_entries)
    build_kwargs = {}
    if segment_start is not None:
        build_kwargs["segment_start"] = segment_start
    prompt_infos, entry_metadata = build_mmlu_prediction_verbalizer_prompt_infos(entries, verbalizer_prompts, tokenizer, **build_kwargs)

    summary = run_verbalizer_binary_eval_loop(
        eval_name="mmlu_prediction",
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        eval_batch_size=eval_batch_size,
        generation_kwargs=generation_kwargs,
        prompt_infos=prompt_infos,
        entry_metadata=entry_metadata,
        num_entries=len(entries),
        verbalizer_lora_paths=verbalizer_lora_paths,
        output_dir=output_dir,
    )

    if run_letter_prediction_eval:
        letter_output_dir = os.path.join(output_dir, "letter_prediction") if output_dir else None
        letter_results = run_letter_prediction(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            eval_batch_size=eval_batch_size,
            generation_kwargs=generation_kwargs,
            entries=entries,
            verbalizer_lora_paths=verbalizer_lora_paths,
            output_dir=letter_output_dir,
            segment_start=segment_start,
        )
        summary.update(letter_results)

    return summary


if __name__ == "__main__":
    import os
    import random
    import torch
    from AObench.utils.common import load_model, load_tokenizer

    eval_modes = {
        "pre_answer": PRE_ANSWER_PROMPTS,
        "post_answer": POST_ANSWER_PROMPTS,
    }

    model_name = "Qwen/Qwen3-8B"
    model_name_str = model_name.split("/")[-1].replace(".", "_")

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    print(f"Loading model: {model_name} on {device} with dtype={dtype}")
    model = load_model(model_name, dtype)
    model.eval()

    verbalizer_lora_paths = STANDARD_VERBALIZER_LORAS

    for mode_name, prompts in eval_modes.items():
        mode_output_dir = os.path.join(f"experiments/mmlu_prediction_eval_results/{model_name_str}", mode_name)
        os.makedirs(mode_output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Running {mode_name} eval ({list(prompts.keys())})")
        print(f"{'=' * 60}")

        summary = run_mmlu_prediction_open_ended_eval(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            verbalizer_lora_paths=verbalizer_lora_paths,
            output_dir=mode_output_dir,
            verbalizer_prompts=prompts,
        )

        print(f"\n=== {mode_name} Summary ===")
        print(json.dumps(summary, indent=2))
