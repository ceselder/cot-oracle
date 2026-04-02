#!/usr/bin/env python3
"""
Run the main-branch AObench eval on the paper checkpoint collection.

Defaults to the paper_core profile, which keeps the cleaner, mostly objective
subset of open-ended evals and avoids mixing in the noisier judge-heavy tasks.
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

from AObench.open_ended_eval.run_all import (
    EVAL_PROFILES,
    MODEL_NAME,
    print_results_comparison,
    run_all_evals,
)
from AObench.utils.common import timestamped_eval_results_dir
from AObench.utils.paper_collection import (
    PAPER_COLLECTION_VERBALIZERS,
    prepare_eval_runtime,
    sample_limits_for_profile,
    write_run_config,
)
from AObench.utils.report import generate_report


def default_output_dir() -> str:
    return timestamped_eval_results_dir("paper_collection_aobench")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AObench on the paper checkpoint collection")
    parser.add_argument(
        "--verbalizer-lora",
        type=str,
        action="append",
        default=None,
        help="Override the default paper collection checkpoint list. Can be repeated.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(EVAL_PROFILES),
        default="paper_core",
        help="Named eval profile to run when --include is not provided.",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Explicit eval names to run. Overrides --profile if provided.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Backbone model used for the AO verbalizers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for summaries and plots.",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=None,
        help="Override number of activation positions for all evals.",
    )
    parser.add_argument(
        "--sample-profile",
        type=str,
        choices=["full", "paper_small", "paper_tiny10"],
        default="full",
        help="Per-eval sample cap profile. 'paper_small' runs a reduced-size comparison.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=600,
        help="Number of bootstrap replicates for report error bars.",
    )
    args = parser.parse_args()

    verbalizer_lora_paths = args.verbalizer_lora or list(PAPER_COLLECTION_VERBALIZERS)
    include = args.include if args.include is not None else list(EVAL_PROFILES[args.profile])
    output_dir = args.output_dir or default_output_dir()
    sample_limits = sample_limits_for_profile(args.sample_profile)

    write_run_config(output_dir, {
        "model_name": args.model_name,
        "profile": args.profile,
        "include": include,
        "n_positions": args.n_positions,
        "sample_profile": args.sample_profile,
        "sample_limits": sample_limits,
        "bootstrap_reps": args.bootstrap_reps,
        "verbalizer_lora_paths": verbalizer_lora_paths,
    })

    device, _, tokenizer, model = prepare_eval_runtime(args.model_name)
    print(f"Running profile: {args.profile}")
    print(f"Include list: {include}")
    print(f"Sample profile: {args.sample_profile}")
    if sample_limits:
        print(f"Sample limits: {sample_limits}")
    print(f"Verbalizer LoRA paths: {verbalizer_lora_paths}")
    print(f"Output dir: {output_dir}")

    all_summaries = run_all_evals(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=args.model_name,
        output_dir=output_dir,
        verbalizer_lora_paths=verbalizer_lora_paths,
        include=include,
        n_positions=args.n_positions,
        sample_limits=sample_limits,
    )

    combined_path = os.path.join(output_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All eval summaries saved to {combined_path}")

    generate_report(output_dir, bootstrap_reps=args.bootstrap_reps)
    print_results_comparison(all_summaries)


if __name__ == "__main__":
    main()
