#!/usr/bin/env python3
"""
Rejudge saved judge-heavy AObench rollout files without regenerating AO outputs.

This reads the per-verbalizer JSON files produced by the original eval run,
reconstructs the task metadata, reruns only the judge/scoring stage, and
emits a fresh report bundle.
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench import dataset_path
from AObench.base_experiment import VerbalizerResults
from AObench.open_ended_eval import (
    backtracking,
    domain_confusion,
    hallucination,
    system_prompt_qa,
    vagueness,
)
from AObench.open_ended_eval.eval_runner import average_numeric_metric_dicts
from AObench.report import generate_report
from AObench.utils.common import load_tokenizer

MODEL_NAME = "Qwen/Qwen3-8B"


def default_output_dir() -> str:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return f"experiments/paper_collection_aobench_judge_heavy_rejudged_{timestamp}"


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


def _task_file_map(input_dir: Path) -> dict[str, list[Path]]:
    return {
        "backtracking": sorted((input_dir / "backtracking").glob("*.json")),
        "system_prompt_qa_hidden": sorted((input_dir / "system_prompt_qa_hidden" / "user_and_assistant").glob("*.json")),
        "system_prompt_qa_latentqa": sorted((input_dir / "system_prompt_qa_latentqa" / "user_and_assistant").glob("*.json")),
        "vagueness": sorted((input_dir / "vagueness").glob("*.json")),
        "domain_confusion": sorted((input_dir / "domain_confusion").glob("*.json")),
        "hallucination_5pos": sorted((input_dir / "hallucination_5pos").glob("*.json")),
    }


def _rejudge_backtracking(results: list[VerbalizerResults], judge_concurrency: int) -> list[dict[str, Any]]:
    entries = backtracking.load_backtracking_dataset()
    assert len(results) == len(entries), (len(results), len(entries))
    return asyncio.run(backtracking.judge_ao_responses(results, entries, concurrency=judge_concurrency))


def _rejudge_system_prompt_hidden(results: list[VerbalizerResults], tokenizer, judge_concurrency: int) -> list[dict[str, Any]]:
    entries = system_prompt_qa.load_dataset(
        dataset_path=dataset_path("datasets/system_prompt_qa/hidden_instruction_eval_dataset.json"),
    )
    _, metadata = system_prompt_qa.build_prompt_infos_for_mode(
        entries,
        "user_and_assistant",
        tokenizer,
        verbalizer_prompts=system_prompt_qa.VERBALIZER_PROMPTS_HIDDEN_INSTRUCTION,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(system_prompt_qa.judge_ao_responses(results, metadata, concurrency=judge_concurrency))


def _rejudge_system_prompt_latentqa(results: list[VerbalizerResults], tokenizer, judge_concurrency: int) -> list[dict[str, Any]]:
    entries = system_prompt_qa.load_dataset(
        dataset_path=dataset_path("datasets/system_prompt_qa/latentqa_eval_dataset.json"),
    )
    _, metadata = system_prompt_qa.build_prompt_infos_for_mode(
        entries,
        "user_and_assistant",
        tokenizer,
        verbalizer_prompts=system_prompt_qa.VERBALIZER_PROMPTS_SYSTEM_PROMPT_QA,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(system_prompt_qa.judge_ao_responses(results, metadata, concurrency=judge_concurrency))


def _rejudge_vagueness(results: list[VerbalizerResults], tokenizer, judge_concurrency: int) -> list[dict[str, Any]]:
    entries, prompts = vagueness.load_data()
    _, metadata = vagueness.build_vagueness_verbalizer_prompt_infos(entries, prompts, tokenizer)
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(vagueness.judge_vagueness(results, metadata, concurrency=judge_concurrency))


def _rejudge_domain_confusion(results: list[VerbalizerResults], tokenizer, judge_concurrency: int) -> list[dict[str, Any]]:
    entries = domain_confusion.load_dataset()
    _, metadata = domain_confusion.build_domain_confusion_verbalizer_prompt_infos(
        entries,
        domain_confusion.VERBALIZER_PROMPTS,
        tokenizer,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(domain_confusion.judge_domain_confusion(results, metadata, concurrency=judge_concurrency))


def _rejudge_hallucination_5pos(results: list[VerbalizerResults], tokenizer, judge_concurrency: int) -> list[dict[str, Any]]:
    entries = hallucination.load_dataset()
    _, metadata = hallucination.build_hallucination_verbalizer_prompt_infos(
        entries,
        hallucination.VERBALIZER_PROMPTS,
        tokenizer,
        n_positions=5,
    )
    assert len(results) == len(metadata), (len(results), len(metadata))
    return asyncio.run(hallucination.judge_hallucination(results, metadata, concurrency=judge_concurrency))


TASK_SPECS: dict[str, dict[str, Any]] = {
    "backtracking": {
        "rejudge_fn": _rejudge_backtracking,
        "metrics_fn": backtracking.compute_judge_metrics,
        "required_fields": ("specificity", "correctness"),
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
    },
    "domain_confusion": {
        "rejudge_fn": _rejudge_domain_confusion,
        "metrics_fn": domain_confusion.compute_domain_confusion_metrics,
        "required_fields": ("category",),
    },
    "hallucination_5pos": {
        "rejudge_fn": _rejudge_hallucination_5pos,
        "metrics_fn": hallucination.compute_hallucination_metrics,
        "required_fields": ("category",),
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
        "judge_model": os.environ.get("JUDGE_MODEL", ""),
        "judge_use_local": os.environ.get("JUDGE_USE_LOCAL", "1") != "0",
        "tasks": list(TASK_SPECS),
        "note": "activation_sensitivity omitted because the original run did not save per-verbalizer raw rollout pairs",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    for task_name, spec in TASK_SPECS.items():
        files = task_files[task_name]
        if not files:
            raise FileNotFoundError(f"No raw files found for task {task_name} under {input_dir}")

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

            payload, results = _load_verbalizer_results(raw_path)
            verbalizer = payload["verbalizer"]
            verbalizer_key = verbalizer.split("/")[-1]
            rejudge_fn = spec["rejudge_fn"]
            metrics_fn = spec["metrics_fn"]

            print(f"Rejudging {task_name}: {verbalizer_key}")
            if task_name == "backtracking":
                raw_scored_results = rejudge_fn(results, args.judge_concurrency)
            else:
                raw_scored_results = rejudge_fn(results, tokenizer, args.judge_concurrency)

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
