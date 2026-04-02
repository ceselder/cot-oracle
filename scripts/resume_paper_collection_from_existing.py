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
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench.open_ended_eval.run_all import MODEL_NAME, EVAL_PROFILES, run_all_evals
from AObench.utils.common import load_model, load_tokenizer
from AObench.utils.report import generate_report
from scripts.run_paper_collection_aobench import PAPER_COLLECTION_VERBALIZERS


def _load_existing_summaries(output_dir: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for path in sorted(output_dir.glob("*_summary.json")):
        eval_name = path.stem.removesuffix("_summary")
        summaries[eval_name] = json.loads(path.read_text())
    return summaries


def _clean_eval_outputs(output_dir: Path, eval_names: list[str]) -> None:
    for eval_name in eval_names:
        eval_dir = output_dir / eval_name
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        summary_path = output_dir / f"{eval_name}_summary.json"
        if summary_path.exists():
            summary_path.unlink()


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
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_include_output:
        _clean_eval_outputs(output_dir, args.include)

    existing_summaries = _load_existing_summaries(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    print(f"Loading model: {args.model_name} on {device} with dtype={dtype}")
    model = load_model(args.model_name, dtype)
    model.eval()

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

    generate_report(str(output_dir), bootstrap_reps=args.bootstrap_reps)


if __name__ == "__main__":
    main()
