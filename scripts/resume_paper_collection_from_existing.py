#!/usr/bin/env python3
"""
Resume a paper-collection AObench run from an existing output directory.

Typical use:
  - keep already-completed eval summaries (e.g. taboo/personaqa/number/mmlu)
  - wipe stale partial outputs for the evals you want to rerun
  - rerun only the remaining evals
  - regenerate a merged all_summaries.json and report from all summary files
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench.open_ended_eval.run_all import MODEL_NAME, EVAL_PROFILES, run_all_evals
from AObench.utils.paper_collection import (
    PAPER_COLLECTION_VERBALIZERS,
    clean_eval_outputs,
    load_existing_summaries,
    prepare_eval_runtime,
)
from AObench.utils.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume a paper collection run from an existing output directory")
    parser.add_argument("--output-dir", required=True, help="Existing output dir to resume into")
    parser.add_argument(
        "--include",
        nargs="+",
        required=True,
        choices=sorted(EVAL_PROFILES["all"]),
        help="Eval names to rerun",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=600,
        help="Bootstrap reps for regenerated report",
    )
    parser.add_argument(
        "--clean-include-output",
        action="store_true",
        help="Delete existing per-eval dirs and summary files for the included evals before rerunning",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Backbone model name",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=5,
        help="Override number of activation positions",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Do not regenerate the report after merging summaries.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_include_output:
        clean_eval_outputs(output_dir, args.include)

    existing_summaries = load_existing_summaries(output_dir)
    device, _, tokenizer, model = prepare_eval_runtime(args.model_name)

    rerun_summaries = run_all_evals(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=args.model_name,
        output_dir=str(output_dir),
        verbalizer_lora_paths=list(PAPER_COLLECTION_VERBALIZERS),
        include=list(args.include),
        n_positions=args.n_positions,
        sample_limits={},
    )

    merged = {**existing_summaries, **rerun_summaries}
    combined_path = output_dir / "all_summaries.json"
    combined_path.write_text(json.dumps(merged, indent=2))
    print(f"Merged all eval summaries saved to {combined_path}")

    if not args.skip_report:
        generate_report(str(output_dir), bootstrap_reps=args.bootstrap_reps)


if __name__ == "__main__":
    main()
