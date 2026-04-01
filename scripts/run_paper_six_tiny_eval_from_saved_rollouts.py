#!/usr/bin/env python3
"""
Build a tiny 6-task paper eval bundle from saved rollouts, rejudge the free-form
tasks, audit the raw payloads, and render a report.

This is intended for fast iteration when backbone generation has already been
done and only the scoring / plotting stack needs to change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench.open_ended_eval.eval_runner import (  # noqa: E402
    average_numeric_metric_dicts,
    compute_binary_yes_no_metrics,
)
from AObench.open_ended_eval.missing_info import compute_missing_info_binary_metrics  # noqa: E402
from AObench.open_ended_eval.number_prediction import compute_metrics as compute_number_prediction_metrics  # noqa: E402
from AObench.open_ended_eval.run_all import _average_numeric_overall_metrics  # noqa: E402
from AObench.report import generate_report  # noqa: E402

TARGET_TASKS = [
    "number_prediction",
    "mmlu_prediction",
    "backtracking",
    "vagueness",
    "domain_confusion",
    "missing_info",
]
DIRECT_TASKS = {"number_prediction", "mmlu_prediction", "missing_info"}
JUDGE_TASKS = {"backtracking", "vagueness", "domain_confusion"}
PROMPTS_PER_ENTRY = {
    "backtracking": 1,
    "vagueness": 1,
    "domain_confusion": 3,
}
REPRESENTATIVE_VERBALIZER = "cot-oracle-paper-ablation-ours-3layers"
DEFAULT_JUDGE_MODEL = "google/gemini-3.1-flash-lite-preview"


def _default_output_dir() -> Path:
    return ROOT / "experiments" / "paper_six_tiny10_gemini31flashlite"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_global_openrouter_key() -> str:
    text = Path("/home/celeste/.claude/CLAUDE.md").read_text()
    match = re.search(r"OpenRouter\*\*: `([^`]+)`", text)
    if match is None:
        raise RuntimeError("OpenRouter key not found in /home/celeste/.claude/CLAUDE.md")
    return match.group(1)


def _subset_indices_by_unique_field(rows: list[dict[str, Any]], field_name: str, n_unique: int) -> list[int]:
    keep_values: list[Any] = []
    seen: set[Any] = set()
    for row in rows:
        value = row[field_name]
        if value in seen:
            continue
        seen.add(value)
        keep_values.append(value)
        if len(keep_values) >= n_unique:
            break
    keep_set = set(keep_values)
    return [idx for idx, row in enumerate(rows) if row[field_name] in keep_set]


def _summarize_direct_task(
    task_name: str,
    metrics_by_verbalizer: dict[str, dict[str, Any]],
    num_entries: int,
    num_scored: int,
    *,
    mode_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall_metrics": average_numeric_metric_dicts(list(metrics_by_verbalizer.values())),
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_entries": num_entries,
        "num_scored": num_scored,
    }
    if mode_results is not None:
        summary["mode_results"] = mode_results
        summary["overall_metrics"] = _average_numeric_overall_metrics(metrics_by_verbalizer)
    return summary


def _build_number_prediction_subset(source_dir: Path, output_dir: Path, n_entries: int) -> dict[str, Any]:
    source_task_dir = source_dir / "number_prediction"
    output_task_dir = output_dir / "number_prediction"
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
    total_scored = 0

    for src_path in sorted(source_task_dir.glob("number_prediction_*.json")):
        payload = _load_json(src_path)
        scored_results = payload["scored_results"]
        keep_indices = _subset_indices_by_unique_field(scored_results, "id", n_entries)
        subset_scored = [scored_results[i] for i in keep_indices]
        subset_verbalizer_results = [payload["verbalizer_results"][i] for i in keep_indices]
        subset_metrics = compute_number_prediction_metrics(subset_scored)

        verbalizer_key = payload["verbalizer"].split("/")[-1]
        metrics_by_verbalizer[verbalizer_key] = subset_metrics
        total_scored += len(subset_scored)

        subset_payload = dict(payload)
        subset_payload["num_entries"] = n_entries
        subset_payload["scored_results"] = subset_scored
        subset_payload["verbalizer_results"] = subset_verbalizer_results
        subset_payload["metrics"] = subset_metrics
        _write_json(output_task_dir / src_path.name, subset_payload)

    summary = _summarize_direct_task(
        "number_prediction",
        metrics_by_verbalizer,
        n_entries,
        total_scored,
    )
    _write_json(output_dir / "number_prediction_summary.json", summary)
    return summary


def _build_mmlu_subset(source_dir: Path, output_dir: Path, n_entries: int) -> dict[str, Any]:
    source_task_dir = source_dir / "mmlu_prediction"
    output_task_dir = output_dir / "mmlu_prediction"
    mode_results: dict[str, Any] = {}
    flattened_metrics_by_verbalizer: dict[str, dict[str, Any]] = {}

    for mode_name in ("pre_answer", "post_answer"):
        src_mode_dir = source_task_dir / mode_name
        out_mode_dir = output_task_dir / mode_name
        metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
        total_scored = 0

        for src_path in sorted(src_mode_dir.glob("mmlu_prediction_binary_*.json")):
            payload = _load_json(src_path)
            subset_rows = payload["binary_scored_results"][:n_entries]
            subset_metrics = compute_binary_yes_no_metrics(subset_rows)
            verbalizer_key = payload["verbalizer"].split("/")[-1]
            metrics_by_verbalizer[verbalizer_key] = subset_metrics
            total_scored += len(subset_rows)

            subset_payload = dict(payload)
            subset_payload["num_entries"] = n_entries
            subset_payload["binary_scored_results"] = subset_rows
            subset_payload["binary_score_metrics"] = subset_metrics
            subset_payload["binary_roc_plot_path"] = None
            _write_json(out_mode_dir / src_path.name, subset_payload)
            flattened_metrics_by_verbalizer[f"{mode_name}/{verbalizer_key}"] = subset_metrics

        mode_results[mode_name] = {
            "overall_metrics": average_numeric_metric_dicts(list(metrics_by_verbalizer.values())),
            "metrics_by_verbalizer": metrics_by_verbalizer,
            "num_entries": n_entries,
            "num_scored": total_scored,
        }

    summary = {
        "mode_results": mode_results,
        "metrics_by_verbalizer": flattened_metrics_by_verbalizer,
        "overall_metrics": _average_numeric_overall_metrics(flattened_metrics_by_verbalizer),
    }
    _write_json(output_dir / "mmlu_prediction_summary.json", summary)
    return summary


def _build_missing_info_subset(source_dir: Path, output_dir: Path, n_entries: int) -> dict[str, Any]:
    source_task_dir = source_dir / "missing_info"
    output_task_dir = output_dir / "missing_info"
    metrics_by_verbalizer: dict[str, dict[str, Any]] = {}
    total_scored = 0

    for src_path in sorted(source_task_dir.glob("missing_info_binary_*.json")):
        payload = _load_json(src_path)
        subset_rows = payload["binary_scored_results"][:n_entries]
        subset_metrics = compute_missing_info_binary_metrics(subset_rows)
        verbalizer_key = payload["verbalizer"].split("/")[-1]
        metrics_by_verbalizer[verbalizer_key] = subset_metrics
        total_scored += len(subset_rows)

        subset_payload = dict(payload)
        subset_payload["num_entries"] = n_entries
        subset_payload["binary_scored_results"] = subset_rows
        subset_payload["binary_score_metrics"] = subset_metrics
        subset_payload["binary_roc_plot_path"] = None
        _write_json(output_task_dir / src_path.name, subset_payload)

    summary = _summarize_direct_task(
        "missing_info",
        metrics_by_verbalizer,
        n_entries,
        total_scored,
    )
    _write_json(output_dir / "missing_info_summary.json", summary)
    return summary


def _build_generation_subset(
    task_name: str,
    source_dir: Path,
    output_dir: Path,
    n_entries: int,
) -> None:
    source_task_dir = source_dir / task_name
    output_task_dir = output_dir / task_name
    keep_count = n_entries * PROMPTS_PER_ENTRY[task_name]

    for src_path in sorted(source_task_dir.glob(f"{task_name}_*.json")):
        payload = _load_json(src_path)
        subset_payload = dict(payload)
        subset_payload["num_entries"] = n_entries
        subset_payload["verbalizer_results"] = payload["verbalizer_results"][:keep_count]
        subset_payload["scored_results"] = []
        subset_payload["metrics"] = None
        if "max_entries" in subset_payload:
            subset_payload["max_entries"] = n_entries
        _write_json(output_task_dir / src_path.name, subset_payload)


def build_subset_bundle(source_dir: Path, subset_dir: Path, n_entries: int) -> dict[str, Any]:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    source_run_config = _load_json(source_dir / "run_config.json")
    subset_run_config = {
        "source_dir": str(source_dir),
        "tasks": TARGET_TASKS,
        "n_entries_per_eval": n_entries,
        "verbalizer_lora_paths": source_run_config["verbalizer_lora_paths"],
        "note": "Subset of saved rollout bundle for the 6-task tiny paper comparison.",
    }
    _write_json(subset_dir / "run_config.json", subset_run_config)

    summaries: dict[str, Any] = {}
    summaries["number_prediction"] = _build_number_prediction_subset(source_dir, subset_dir, n_entries)
    summaries["mmlu_prediction"] = _build_mmlu_subset(source_dir, subset_dir, n_entries)
    summaries["missing_info"] = _build_missing_info_subset(source_dir, subset_dir, n_entries)

    for task_name in sorted(JUDGE_TASKS):
        _build_generation_subset(task_name, source_dir, subset_dir, n_entries)

    _write_json(subset_dir / "all_summaries.json", summaries)
    return summaries


def run_rejudge(subset_dir: Path, rejudge_dir: Path, judge_model: str, judge_concurrency: int) -> None:
    if rejudge_dir.exists():
        shutil.rmtree(rejudge_dir)
    rejudge_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JUDGE_USE_LOCAL"] = "0"
    env["JUDGE_MODEL"] = judge_model
    env["OPENROUTER_API_KEY"] = _load_global_openrouter_key()

    cmd = [
        str(ROOT.parent / "cot-oracle" / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "rejudge_saved_judge_heavy_rollouts.py"),
        "--input-dir",
        str(subset_dir),
        "--output-dir",
        str(rejudge_dir),
        "--judge-concurrency",
        str(judge_concurrency),
        "--judge-pack-size",
        "1",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def merge_results(subset_dir: Path, rejudge_dir: Path, final_dir: Path, judge_model: str) -> dict[str, Any]:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}

    for task_name in TARGET_TASKS:
        if task_name in DIRECT_TASKS:
            src_dir = subset_dir / task_name
            if src_dir.exists():
                shutil.copytree(src_dir, final_dir / task_name)
            summary = _load_json(subset_dir / f"{task_name}_summary.json")
            _write_json(final_dir / f"{task_name}_summary.json", summary)
            summaries[task_name] = summary
        else:
            src_dir = rejudge_dir / task_name
            if src_dir.exists():
                shutil.copytree(src_dir, final_dir / task_name)
            summary_path = rejudge_dir / f"{task_name}_summary.json"
            summary = _load_json(summary_path)
            _write_json(final_dir / summary_path.name, summary)
            summaries[task_name] = summary

    merged_run_config = {
        "source_subset_dir": str(subset_dir),
        "rejudge_dir": str(rejudge_dir),
        "tasks": TARGET_TASKS,
        "judge_model": judge_model,
        "n_entries_per_eval": 10,
    }
    _write_json(final_dir / "run_config.json", merged_run_config)
    _write_json(final_dir / "all_summaries.json", summaries)
    return summaries


def _task_expected_counts(task_name: str, n_entries: int) -> int:
    if task_name == "number_prediction":
        return n_entries * 3
    if task_name == "mmlu_prediction":
        return n_entries
    if task_name == "missing_info":
        return n_entries
    if task_name == "backtracking":
        return n_entries
    if task_name == "vagueness":
        return n_entries
    if task_name == "domain_confusion":
        return n_entries * 3
    raise ValueError(task_name)


def _sample_rows_from_payload(task_name: str, payload: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    if task_name in {"number_prediction", "backtracking", "vagueness", "domain_confusion"}:
        rows = payload.get("scored_results") or payload.get("verbalizer_results") or []
    elif task_name in {"mmlu_prediction", "missing_info"}:
        rows = payload.get("binary_scored_results") or []
    else:
        rows = []
    return rows[:limit]


def write_rollout_audit(final_dir: Path, n_entries: int) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "expected_counts": {task: _task_expected_counts(task, n_entries) for task in TARGET_TASKS},
        "counts_by_task": {},
        "representative_samples": {},
        "issues": [],
    }

    for task_name in TARGET_TASKS:
        task_dir = final_dir / task_name
        counts: dict[str, int] = {}
        payload_paths: list[Path]
        if task_name == "mmlu_prediction":
            payload_paths = sorted((task_dir / "pre_answer").glob("*.json")) + sorted((task_dir / "post_answer").glob("*.json"))
        else:
            payload_paths = sorted(task_dir.glob("*.json"))
        for payload_path in payload_paths:
            payload = _load_json(payload_path)
            if task_name in {"number_prediction", "backtracking", "vagueness", "domain_confusion"}:
                row_count = len(payload.get("scored_results") or payload.get("verbalizer_results") or [])
            else:
                row_count = len(payload.get("binary_scored_results") or [])
            key = payload_path.name if task_name != "mmlu_prediction" else f"{payload_path.parent.name}/{payload_path.name}"
            counts[key] = row_count
            if row_count != _task_expected_counts(task_name, n_entries):
                audit["issues"].append(
                    f"{task_name}: {key} had {row_count} rows, expected {_task_expected_counts(task_name, n_entries)}"
                )
        audit["counts_by_task"][task_name] = counts

        if task_name == "mmlu_prediction":
            rep_path = task_dir / "pre_answer" / f"mmlu_prediction_binary_{REPRESENTATIVE_VERBALIZER}.json"
        elif task_name == "missing_info":
            rep_path = task_dir / f"missing_info_binary_{REPRESENTATIVE_VERBALIZER}.json"
        else:
            rep_path = task_dir / f"{task_name}_{REPRESENTATIVE_VERBALIZER}.json"
        if rep_path.exists():
            audit["representative_samples"][task_name] = _sample_rows_from_payload(task_name, _load_json(rep_path))

    _write_json(final_dir / "rollout_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the tiny 6-task paper eval from saved rollouts")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "paper_artifacts_from_vast" / "paper_collection_aobench_small_last5_fixed",
        help="Saved rollout bundle to subset and rejudge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Base output directory for subset, rejudge, and final report artifacts.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter model id for judge-only tasks.",
    )
    parser.add_argument(
        "--n-entries",
        type=int,
        default=10,
        help="Number of logical examples per eval.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=8,
        help="Concurrency for judge-only tasks.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=400,
        help="Bootstrap replicates for report error bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output = args.output_dir
    subset_dir = base_output / "subset_bundle"
    rejudge_dir = base_output / "rejudged"
    final_dir = base_output / "final"

    print(f"Building subset bundle from {args.source_dir}")
    build_subset_bundle(args.source_dir, subset_dir, args.n_entries)

    print(f"Rejudging {sorted(JUDGE_TASKS)} with {args.judge_model}")
    run_rejudge(subset_dir, rejudge_dir, args.judge_model, args.judge_concurrency)

    print("Merging direct and rejudged results")
    merge_results(subset_dir, rejudge_dir, final_dir, args.judge_model)

    print("Auditing raw rollout payloads")
    audit = write_rollout_audit(final_dir, args.n_entries)
    if audit["issues"]:
        print("Audit issues:")
        for issue in audit["issues"]:
            print(f"  - {issue}")
    else:
        print("Audit passed: all task payloads matched the expected tiny counts")

    print("Generating report")
    generate_report(str(final_dir), bootstrap_reps=args.bootstrap_reps)
    print(f"Done. Final artifacts: {final_dir}")


if __name__ == "__main__":
    main()
