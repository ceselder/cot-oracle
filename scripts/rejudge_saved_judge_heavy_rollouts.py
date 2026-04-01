#!/usr/bin/env python3
"""
Rejudge saved judge-heavy AObench rollout files without regenerating AO outputs.

This reads the per-verbalizer JSON files produced by the original eval run,
reconstructs the task metadata, reruns only the judge/scoring stage, and
emits a fresh report bundle.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench import dataset_path
from AObench.base_experiment import VerbalizerResults
from AObench.open_ended_eval import (
    activation_sensitivity,
    backtracking,
    domain_confusion,
    hallucination,
    judge as shared_judge,
    system_prompt_qa,
    vagueness,
)
from AObench.open_ended_eval.eval_runner import average_numeric_metric_dicts
from AObench.utils.common import load_tokenizer, timestamped_eval_results_dir
from AObench.utils.report import generate_report

MODEL_NAME = "Qwen/Qwen3-8B"


def default_output_dir() -> str:
    return timestamped_eval_results_dir("paper_collection_aobench_judge_heavy_rejudged")


def _load_verbalizer_results(path: Path) -> tuple[dict[str, Any], list[VerbalizerResults]]:
    payload = json.loads(path.read_text())
    results = [
        VerbalizerResults(
            verbalizer_lora_path=row["verbalizer_lora_path"],
            target_lora_path=row["target_lora_path"],
            context_token_ids=row["context_token_ids"],
            act_key=row["act_key"],
            verbalizer_prompt=row["verbalizer_prompt"],
            ground_truth=row["ground_truth"],
            num_tokens=row["num_tokens"],
            responses=row["responses"],
        )
        for row in payload["verbalizer_results"]
    ]
    return payload, results


def _load_activation_sensitivity_pairs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    return payload, payload.get("pairs", [])


def _task_file_map(input_dir: Path) -> dict[str, list[Path]]:
    hallucination_dir = input_dir / "hallucination"
    legacy_hallucination_dir = input_dir / "hallucination_5pos"
    return {
        "backtracking": sorted((input_dir / "backtracking").glob("*.json")),
        "system_prompt_qa_hidden": sorted((input_dir / "system_prompt_qa_hidden" / "user_and_assistant").glob("*.json")),
        "system_prompt_qa_latentqa": sorted((input_dir / "system_prompt_qa_latentqa" / "user_and_assistant").glob("*.json")),
        "vagueness": sorted((input_dir / "vagueness").glob("*.json")),
        "domain_confusion": sorted((input_dir / "domain_confusion").glob("*.json")),
        "hallucination": sorted((hallucination_dir if hallucination_dir.exists() else legacy_hallucination_dir).glob("*.json")),
        "activation_sensitivity": sorted((input_dir / "activation_sensitivity").glob("activation_sensitivity_*.json")),
    }


def _infer_max_entries(payload: dict[str, Any], results_len: int | None = None, prompts_per_entry: int = 1) -> int | None:
    for key in ("max_entries", "num_entries", "max_entries_per_condition"):
        value = _coerce_int(payload.get(key))
        if value is not None and value > 0:
            return value

    if results_len is not None and prompts_per_entry > 0:
        inferred = results_len // prompts_per_entry
        if inferred > 0:
            return inferred

    return None


def _extract_json_array_payload(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise json.JSONDecodeError("Parsed payload was not a list", text, 0)
    return data


async def _judge_rows_batched(
    *,
    rows: list[dict[str, Any]],
    system_prompt: str,
    required_fields: tuple[str, ...],
    concurrency: int,
    pack_size: int,
    max_tokens_per_item: int = 120,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if pack_size <= 1:
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [
                shared_judge.judge_single(client, system_prompt, row["user_message"], semaphore, max_tokens=max_tokens_per_item)
                for row in rows
            ]
            judge_results = await asyncio.gather(*tasks, return_exceptions=True)

        scored = []
        for row, result in zip(rows, judge_results):
            if isinstance(result, Exception):
                print(f"Judge error for result {row['result_index']}: {result}")
                continue
            scored.append({k: v for k, v in row.items() if k != "user_message"} | result)
        return scored

    api_base, api_key = shared_judge._get_endpoint()
    use_local = os.environ.get("JUDGE_USE_LOCAL", "1") != "0"
    model = os.environ.get("JUDGE_MODEL", shared_judge.JUDGE_MODEL)
    semaphore = asyncio.Semaphore(concurrency)

    field_lines = []
    if "specificity" in required_fields:
        field_lines.append('"specificity": <int>')
    if "correctness" in required_fields:
        field_lines.append('"correctness": <int>')
    if "category" in required_fields:
        field_lines.append('"category": "<string>"')
    field_lines.append('"reasoning": "<brief explanation>"')
    field_spec = ", ".join(field_lines)

    async def _judge_batch(client: httpx.AsyncClient, batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch_prompt = system_prompt + (
            "\n\nYou will now judge multiple items at once. "
            "Return ONLY a JSON array. Each array element must be an object with "
            '"item_index": <int> and these keys: '
            f"{field_spec}. "
            "Use the same item_index numbers that appear below. "
            "Return exactly one element per item."
        )

        joined_user = "\n\n".join(
            f"ITEM {i}\n{row['user_message']}" for i, row in enumerate(batch_rows)
        )
        if use_local:
            messages = [{"role": "user", "content": f"{batch_prompt}\n\n{joined_user}"}]
        else:
            messages = [
                {"role": "system", "content": batch_prompt},
                {"role": "user", "content": joined_user},
            ]

        async with semaphore:
            resp = await client.post(
                api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max(256, max_tokens_per_item * len(batch_rows)),
                    "messages": messages,
                },
                timeout=180.0,
            )
            resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        parsed = _extract_json_array_payload(text)
        mapped: dict[int, dict[str, Any]] = {}
        for item in parsed:
            try:
                idx = int(item.get("item_index"))
            except Exception:
                continue
            mapped[idx] = item
        if len(mapped) != len(batch_rows):
            raise ValueError(f"Expected {len(batch_rows)} batch judgements, got {len(mapped)}")
        out = []
        for i, row in enumerate(batch_rows):
            out.append({k: v for k, v in row.items() if k != "user_message"} | mapped[i])
        return out

    async with httpx.AsyncClient() as client:
        tasks = []
        batch_slices: list[list[dict[str, Any]]] = []
        for start in range(0, len(rows), pack_size):
            batch_rows = rows[start:start + pack_size]
            batch_slices.append(batch_rows)
            tasks.append(_judge_batch(client, batch_rows))
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    scored: list[dict[str, Any]] = []
    for batch_rows, batch_result in zip(batch_slices, batch_results):
        if isinstance(batch_result, Exception):
            print(f"Batch judge fallback for rows {batch_rows[0]['result_index']}-{batch_rows[-1]['result_index']}: {batch_result}")
            fallback = await _judge_rows_batched(
                rows=batch_rows,
                system_prompt=system_prompt,
                required_fields=required_fields,
                concurrency=1,
                pack_size=1,
                max_tokens_per_item=max_tokens_per_item,
            )
            scored.extend(fallback)
        else:
            scored.extend(batch_result)
    return scored


def _rejudge_backtracking(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries = backtracking.load_backtracking_dataset(
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=1)
    )
    assert len(results) == len(entries), (len(results), len(entries))
    rows = []
    for i, (result, entry) in enumerate(zip(results, entries)):
        ao_response = backtracking.get_first_ao_response(result)
        if ao_response is None:
            continue
        rows.append(
            {
                "result_index": i,
                "ao_response": ao_response,
                "ground_truth": entry["uncertainty_description"],
                "backtrack_rate": entry["backtrack_rate"],
                "user_message": backtracking.JUDGE_USER_TEMPLATE.format(
                    prefix=entry["prefix"][-1500:],
                    ground_truth=entry["uncertainty_description"],
                    ao_response=ao_response,
                ),
            }
        )
    return rows


def _rejudge_system_prompt_hidden(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    tokenizer,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries = system_prompt_qa.load_dataset(
        dataset_path=dataset_path("datasets/system_prompt_qa/hidden_instruction_eval_dataset.json"),
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=1),
    )
    _, metadata = system_prompt_qa.build_prompt_infos_for_mode(
        entries,
        "user_and_assistant",
        tokenizer,
        verbalizer_prompts=system_prompt_qa.VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(system_prompt_qa.judge_ao_responses(results, metadata, concurrency=judge_concurrency))


def _rejudge_system_prompt_latentqa(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    tokenizer,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries = system_prompt_qa.load_dataset(
        dataset_path=dataset_path("datasets/system_prompt_qa/latentqa_eval_dataset.json"),
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=1),
    )
    _, metadata = system_prompt_qa.build_prompt_infos_for_mode(
        entries,
        "user_and_assistant",
        tokenizer,
        verbalizer_prompts=system_prompt_qa.VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(system_prompt_qa.judge_ao_responses(results, metadata, concurrency=judge_concurrency))


def _rejudge_vagueness(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    tokenizer,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries, prompts = vagueness.load_data(
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=1)
    )
    _, metadata = vagueness.build_vagueness_verbalizer_prompt_infos(entries, prompts, tokenizer)
    assert len(results) == len(metadata), (len(results), len(metadata))
    rows = []
    for i, (result, meta) in enumerate(zip(results, metadata)):
        ao_response = vagueness.get_first_ao_response(result)
        if ao_response is None:
            continue
        rows.append(
            {
                "result_index": i,
                "ao_response": ao_response,
                **meta,
                "user_message": vagueness.JUDGE_USER_TEMPLATE.format(
                    problem=meta["problem"][:500],
                    question=meta["prompt"],
                    ao_response=ao_response,
                ),
            }
        )
    return rows


def _rejudge_domain_confusion(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    tokenizer,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries = domain_confusion.load_dataset(
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=len(domain_confusion.VERBALIZER_PROMPTS))
    )
    _, metadata = domain_confusion.build_domain_confusion_verbalizer_prompt_infos(
        entries,
        domain_confusion.VERBALIZER_PROMPTS,
        tokenizer,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    rows = []
    for i, (result, meta) in enumerate(zip(results, metadata)):
        ao_response = domain_confusion.get_first_ao_response(result)
        if ao_response is None:
            continue
        rows.append(
            {
                "result_index": i,
                "ao_response": ao_response,
                **meta,
                "user_message": domain_confusion.JUDGE_USER_TEMPLATE.format(
                    problem=meta["problem"],
                    ao_response=ao_response,
                ),
            }
        )
    return rows


def _rejudge_hallucination(
    payload: dict[str, Any],
    results: list[VerbalizerResults],
    tokenizer,
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    entries = hallucination.load_dataset(
        max_entries=_infer_max_entries(payload, len(results), prompts_per_entry=len(hallucination.VERBALIZER_PROMPTS))
    )
    _, metadata = hallucination.build_hallucination_verbalizer_prompt_infos(
        entries,
        hallucination.VERBALIZER_PROMPTS,
        tokenizer,
        n_positions=5,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    rows = []
    for i, (result, meta) in enumerate(zip(results, metadata)):
        ao_response = hallucination.get_first_ao_response(result)
        if ao_response is None:
            continue
        rows.append(
            {
                "result_index": i,
                "ao_response": ao_response,
                **meta,
                "user_message": hallucination.JUDGE_USER_TEMPLATE.format(
                    problem=meta["problem"],
                    ao_response=ao_response,
                ),
            }
        )
    return rows


def _rejudge_activation_sensitivity(
    payload: dict[str, Any],
    pairs: list[dict[str, Any]],
    judge_concurrency: int,
) -> list[dict[str, Any]]:
    rows = []
    for i, pair in enumerate(pairs):
        rows.append(
            {
                "result_index": i,
                **pair,
                "user_message": activation_sensitivity.JUDGE_USER_TEMPLATE.format(
                    problem=pair["problem_text"],
                    missing_info=pair["missing_info_description"],
                    response_a=pair["response_a"],
                    response_c=pair["response_c"],
                ),
            }
        )
    return rows


TASK_SPECS: dict[str, dict[str, Any]] = {
    "backtracking": {
        "rejudge_fn": _rejudge_backtracking,
        "metrics_fn": backtracking.compute_judge_metrics,
        "required_fields": ("specificity", "correctness"),
        "system_prompt": backtracking.JUDGE_SYSTEM_PROMPT,
    },
    "system_prompt_qa_hidden": {
        "rejudge_fn": _rejudge_system_prompt_hidden,
        "metrics_fn": system_prompt_qa.compute_judge_metrics,
        "required_fields": ("specificity", "correctness"),
    },
    "system_prompt_qa_latentqa": {
        "rejudge_fn": _rejudge_system_prompt_latentqa,
        "metrics_fn": system_prompt_qa.compute_judge_metrics,
        "required_fields": ("specificity", "correctness"),
    },
    "vagueness": {
        "rejudge_fn": _rejudge_vagueness,
        "metrics_fn": vagueness.compute_vagueness_metrics,
        "required_fields": ("category",),
        "system_prompt": vagueness.JUDGE_SYSTEM_PROMPT,
    },
    "domain_confusion": {
        "rejudge_fn": _rejudge_domain_confusion,
        "metrics_fn": domain_confusion.compute_domain_confusion_metrics,
        "required_fields": ("category",),
        "system_prompt": domain_confusion.JUDGE_SYSTEM_PROMPT,
    },
    "hallucination": {
        "rejudge_fn": _rejudge_hallucination,
        "metrics_fn": hallucination.compute_hallucination_metrics,
        "required_fields": ("category",),
        "system_prompt": hallucination.JUDGE_SYSTEM_PROMPT,
    },
    "activation_sensitivity": {
        "rejudge_fn": _rejudge_activation_sensitivity,
        "metrics_fn": activation_sensitivity.compute_sensitivity_metrics,
        "required_fields": ("category",),
        "system_prompt": activation_sensitivity.JUDGE_SYSTEM_PROMPT,
    },
}


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return None


def _sanitize_scored_results(task_name: str, scored_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required_fields = TASK_SPECS[task_name].get("required_fields", ())
    sanitized: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for row in scored_results:
        normalized = dict(row)
        if "specificity" in required_fields:
            normalized["specificity"] = _coerce_int(normalized.get("specificity"))
        if "correctness" in required_fields:
            normalized["correctness"] = _coerce_int(normalized.get("correctness"))
        if "category" in required_fields and isinstance(normalized.get("category"), str):
            normalized["category"] = normalized["category"].strip().lower()

        if all(normalized.get(field) is not None for field in required_fields):
            sanitized.append(normalized)
        else:
            dropped.append(normalized)

    return sanitized, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejudge saved judge-heavy AObench rollout files")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/celeste/cot-oracle-main-latest-eval/paper_artifacts/aobench_judge_heavy_main_latest",
        help="Directory containing the saved raw rollout JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the rejudged outputs and report.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=2,
        help="Concurrency for the local Sonnet judge during rejudging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoint files already rejudged successfully in the output dir.",
    )
    parser.add_argument(
        "--judge-pack-size",
        type=int,
        default=8,
        help="Number of judge items to pack into each API request for supported tasks.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(MODEL_NAME)
    task_files = _task_file_map(input_dir)
    all_summaries: dict[str, Any] = {}

    run_config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "judge_concurrency": args.judge_concurrency,
        "judge_pack_size": args.judge_pack_size,
        "judge_model": os.environ.get("JUDGE_MODEL", ""),
        "judge_use_local": os.environ.get("JUDGE_USE_LOCAL", "1") != "0",
        "tasks": list(TASK_SPECS),
        "note": "Rejudge saved rollout payloads for judge-heavy tasks, including activation_sensitivity pair files when present.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    for task_name, spec in TASK_SPECS.items():
        files = task_files[task_name]
        if not files:
            print(f"Skipping {task_name}: no raw files found under {input_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"REJUDGING TASK: {task_name}")
        print(f"{'=' * 70}")

        metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
        overall_metric_dicts: list[dict[str, Any]] = []
        total_scored = 0

        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for raw_path in files:
            output_path = task_output_dir / raw_path.name
            if args.resume and output_path.exists():
                existing = json.loads(output_path.read_text())
                if existing.get("rejudged") and existing.get("metrics") is not None:
                    verbalizer = existing["verbalizer"]
                    verbalizer_key = verbalizer.split("/")[-1]
                    metrics = existing["metrics"]
                    scored_results = existing.get("scored_results", [])
                    metrics_by_verbalizer[verbalizer_key] = metrics
                    overall_metric_dicts.append(metrics)
                    total_scored += len(scored_results)
                    print(f"Skipping {task_name}: {verbalizer_key} (already rejudged)")
                    continue

            if task_name == "activation_sensitivity":
                payload, results = _load_activation_sensitivity_pairs(raw_path)
            else:
                payload, results = _load_verbalizer_results(raw_path)
            verbalizer = payload["verbalizer"]
            verbalizer_key = verbalizer.split("/")[-1]
            rejudge_fn = spec["rejudge_fn"]
            metrics_fn = spec["metrics_fn"]

            print(f"Rejudging {task_name}: {verbalizer_key}")
            if task_name in {"backtracking", "activation_sensitivity"}:
                judge_rows = rejudge_fn(payload, results, args.judge_concurrency)
            else:
                judge_rows = rejudge_fn(payload, results, tokenizer, args.judge_concurrency)

            system_prompt = spec.get("system_prompt")
            if system_prompt is not None:
                raw_scored_results = asyncio.run(
                    _judge_rows_batched(
                        rows=judge_rows,
                        system_prompt=system_prompt,
                        required_fields=spec.get("required_fields", ()),
                        concurrency=args.judge_concurrency,
                        pack_size=args.judge_pack_size,
                    )
                )
            else:
                raw_scored_results = judge_rows

            scored_results, dropped_results = _sanitize_scored_results(task_name, raw_scored_results)
            if dropped_results:
                print(f"  dropped {len(dropped_results)} malformed judged rows for {verbalizer_key}")

            metrics = metrics_fn(scored_results) if scored_results else {}
            total_scored += len(scored_results)
            if metrics:
                metrics_by_verbalizer[verbalizer_key] = metrics
                overall_metric_dicts.append(metrics)

            output_payload = dict(payload)
            output_payload["scored_results"] = scored_results
            output_payload["dropped_scored_results"] = dropped_results
            output_payload["metrics"] = metrics if scored_results else None
            output_payload["rejudged"] = True
            output_payload["rejudge_metadata"] = {
                "judge_concurrency": args.judge_concurrency,
                "input_file": str(raw_path),
                "previous_scored_results_count": len(payload.get("scored_results", [])),
                "new_scored_results_count": len(scored_results),
                "dropped_scored_results_count": len(dropped_results),
            }

            output_path.write_text(json.dumps(output_payload, indent=2))
            print(f"  saved {output_path}")

        summary = {
            "overall_metrics": average_numeric_metric_dicts(overall_metric_dicts),
            "metrics_by_verbalizer": metrics_by_verbalizer,
            "num_entries": len(files),
            "num_scored": total_scored,
            "rejudged": True,
        }
        all_summaries[task_name] = summary
        (output_dir / f"{task_name}_summary.json").write_text(json.dumps(summary, indent=2))

    (output_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    generate_report(str(output_dir))
    print(f"\nRejudged summaries written to {output_dir}")


if __name__ == "__main__":
    main()
