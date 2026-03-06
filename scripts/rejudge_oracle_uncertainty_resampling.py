#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rejudge oracle uncertainty resampling artifacts without rerunning generation")
    parser.add_argument("--run-dir", required=True, help="Existing merged run directory containing predictions and summary.json")
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>_llm_variance")
    parser.add_argument("--variance-metric", default="llm", choices=["llm", "f1"])
    parser.add_argument("--judge-model", default="google/gemini-2.5-flash")
    parser.add_argument("--judge-max-tokens", type=int, default=140)
    parser.add_argument("--min-variance-floor", type=float, default=1e-3)
    return parser.parse_args()


def _load_eval_module():
    module_path = ROOT / "scripts" / "eval_oracle_uncertainty_resampling.py"
    spec = importlib.util.spec_from_file_location("eval_oracle_uncertainty_resampling", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _copy_static_artifacts(src_run_dir: Path, dst_run_dir: Path) -> None:
    for name in ["panel_a_predictions.jsonl", "panel_b_predictions.jsonl", "panel_b_selected_probes.jsonl", "panel_b_probe_reranker.jsonl", "panel_b_hard_token_filter.jsonl"]:
        src = src_run_dir / name
        if src.exists():
            dst = dst_run_dir / name
            dst.write_bytes(src.read_bytes())


def _panel_a_metadata(existing_panel_a_judgments: list[dict]) -> dict[tuple[str, float], dict]:
    metadata = {}
    for row in existing_panel_a_judgments:
        metadata[(row["item_id"], row["temperature"])] = {
            "question": row["question"],
            "target_response": row["target_response"],
            "answerable": row["answerable"],
        }
    return metadata


def _judge_panel_a(pred_rows: list[dict], existing_panel_a_judgments: list[dict], judge_args: argparse.Namespace, judge_variance, client: httpx.Client | None) -> tuple[list[dict], list[dict]]:
    metadata = _panel_a_metadata(existing_panel_a_judgments)
    grouped = defaultdict(list)
    for row in pred_rows:
        grouped[(row["item_id"], row["temperature"])].append(row)
    judgments = []
    by_temp_answerable = defaultdict(list)
    for (item_id, temperature), rows in tqdm(sorted(grouped.items()), desc="rejudge panel A"):
        rows.sort(key=lambda row: row["resample_idx"])
        meta = metadata[(item_id, temperature)]
        answers = [row["prediction"] for row in rows]
        judged = judge_variance(question=meta["question"], answers=answers, args=judge_args, client=client)
        judgment = {
            "panel": "A",
            "item_id": item_id,
            "answerable": meta["answerable"],
            "temperature": temperature,
            "question": meta["question"],
            "target_response": meta["target_response"],
            "n_answers": len(answers),
            "variance": judged["variance"],
            "reason": judged["reason"],
            "canonical_answer": judged["canonical_answer"],
            "judge_raw": judged["raw"],
            "answers": answers,
        }
        judgments.append(judgment)
        by_temp_answerable[(temperature, meta["answerable"])].append(judged["variance"])

    summary = []
    temperatures = sorted({temperature for temperature, _ in by_temp_answerable})
    for temperature in temperatures:
        answerable_vals = np.array(by_temp_answerable[(temperature, True)], dtype=float)
        unanswerable_vals = np.array(by_temp_answerable[(temperature, False)], dtype=float)
        summary.append(
            {
                "temperature": temperature,
                "answerable_mean_variance": float(answerable_vals.mean()),
                "answerable_std_variance": float(answerable_vals.std()),
                "unanswerable_mean_variance": float(unanswerable_vals.mean()),
                "unanswerable_std_variance": float(unanswerable_vals.std()),
                "answerable_n": int(answerable_vals.size),
                "unanswerable_n": int(unanswerable_vals.size),
            }
        )
    judgments.sort(key=lambda row: (row["temperature"], row["item_id"]))
    summary.sort(key=lambda row: row["temperature"])
    return judgments, summary


def _panel_b_question_meta(summary: dict) -> dict[int, dict]:
    metadata = {}
    for row in summary["panel_b_meta"]["questions"]:
        metadata[row["question_index"]] = row
    return metadata


def _judge_panel_b(pred_rows: list[dict], summary: dict, judge_args: argparse.Namespace, judge_variance, client: httpx.Client | None) -> tuple[list[dict], list[dict]]:
    metadata = _panel_b_question_meta(summary)
    grouped = defaultdict(list)
    for row in pred_rows:
        grouped[(row["question_index"], row["rel_offset"])].append(row)
    judgments = []
    by_offset = defaultdict(list)
    for (question_index, rel_offset), rows in tqdm(sorted(grouped.items()), desc="rejudge panel B"):
        rows.sort(key=lambda row: row["resample_idx"])
        meta = metadata[question_index]
        answers = [row["prediction"] for row in rows]
        judged = judge_variance(question=meta["question"], answers=answers, args=judge_args, client=client)
        judgment = {
            "panel": "B",
            "question_index": question_index,
            "item_id": meta["item_id"],
            "source_question": meta["source_question"],
            "question": meta["question"],
            "target_response": meta["target_response"],
            "answer_token": meta["answer_token"],
            "candidate_type": meta["candidate_type"],
            "focus_token_decoded": meta["focus_token_decoded"],
            "focus_position_abs": meta["focus_position_abs"],
            "focus_position_rel": meta["focus_position_rel"],
            "cot_position_abs": rows[0]["cot_position_abs"],
            "cot_position_rel": rows[0]["cot_position_rel"],
            "rel_offset": rel_offset,
            "answers": answers,
            "variance": judged["variance"],
            "reason": judged["reason"],
            "canonical_answer": judged["canonical_answer"],
            "judge_raw": judged["raw"],
        }
        judgments.append(judgment)
        by_offset[rel_offset].append(judged["variance"])

    aligned_summary = []
    for rel_offset in sorted(by_offset):
        vals = np.array(by_offset[rel_offset], dtype=float)
        aligned_summary.append(
            {
                "rel_offset": rel_offset,
                "mean_variance": float(vals.mean()),
                "std_variance": float(vals.std()),
                "n_questions": int(vals.size),
            }
        )
    judgments.sort(key=lambda row: (row["question_index"], row["rel_offset"]))
    return judgments, aligned_summary


def main() -> None:
    load_dotenv(os.path.expanduser("~/.env"))
    load_dotenv()
    args = _parse_args()
    src_run_dir = Path(args.run_dir).resolve()
    assert src_run_dir.is_dir()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else src_run_dir.parent / f"{src_run_dir.name}_{args.variance_metric}_variance"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_module = _load_eval_module()
    judge_args = argparse.Namespace(
        variance_metric=args.variance_metric,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        min_variance_floor=args.min_variance_floor,
    )

    summary = _read_json(src_run_dir / "summary.json")
    panel_a_predictions = _read_jsonl(src_run_dir / "panel_a_predictions.jsonl")
    panel_a_existing_judgments = _read_jsonl(src_run_dir / "panel_a_judgments.jsonl")
    panel_b_predictions = _read_jsonl(src_run_dir / "panel_b_predictions.jsonl")

    with (httpx.Client(timeout=90.0) if args.variance_metric == "llm" else eval_module.contextlib.nullcontext()) as client:
        panel_a_judgments, panel_a_summary = _judge_panel_a(
            pred_rows=panel_a_predictions,
            existing_panel_a_judgments=panel_a_existing_judgments,
            judge_args=judge_args,
            judge_variance=eval_module._judge_variance,
            client=client,
        )
        panel_b_judgments, panel_b_aligned_summary = _judge_panel_b(
            pred_rows=panel_b_predictions,
            summary=summary,
            judge_args=judge_args,
            judge_variance=eval_module._judge_variance,
            client=client,
        )

    _copy_static_artifacts(src_run_dir, out_dir)
    _write_jsonl(out_dir / "panel_a_judgments.jsonl", panel_a_judgments)
    _write_jsonl(out_dir / "panel_b_judgments.jsonl", panel_b_judgments)
    _write_jsonl(out_dir / "panel_b_aligned_summary.jsonl", panel_b_aligned_summary)

    out_summary = dict(summary)
    out_summary["variance_metric"] = args.variance_metric
    out_summary["judge_model"] = args.judge_model
    out_summary["judge_max_tokens"] = args.judge_max_tokens
    out_summary["min_variance_floor"] = args.min_variance_floor
    out_summary["source_run_dir"] = str(src_run_dir)
    out_summary["panel_a_summary"] = panel_a_summary
    out_summary["panel_b_aligned_summary"] = panel_b_aligned_summary
    out_summary["figure_path"] = str(out_dir / "oracle_uncertainty_resampling.png")
    _write_json(out_dir / "summary.json", out_summary)

    eval_module._plot(
        panel_a_summary=panel_a_summary,
        panel_b_aligned_summary=panel_b_aligned_summary,
        panel_b_meta=summary["panel_b_meta"],
        output_path=out_dir / "oracle_uncertainty_resampling.png",
        model_name=summary["model"],
        checkpoint_name=summary["checkpoint"],
        variance_label=args.variance_metric,
        panel_a_redraws=summary["resamples_per_item"],
    )
    print(out_dir)


if __name__ == "__main__":
    main()
