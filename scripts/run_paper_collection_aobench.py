#!/usr/bin/env python3
"""
Run the main-branch AObench eval on the paper checkpoint collection.

Defaults to the paper_core profile, which keeps the cleaner, mostly objective
subset of open-ended evals and avoids mixing in the noisier judge-heavy tasks.
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AObench.open_ended_eval.run_all import (
    EVAL_PROFILES,
    MODEL_NAME,
    print_results_comparison,
    run_all_evals,
)
from AObench.report import generate_report
from AObench.utils.common import load_model, load_tokenizer

PAPER_COLLECTION_VERBALIZERS = [
    "ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens",
    "ceselder/adam-reupload-qwen3-8b-full-mix-synthetic-qa-v3-replace-lqa",
    "ceselder/cot-oracle-paper-ablation-adam-recipe-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-1layer",
    "ceselder/cot-oracle-paper-ablation-ours-3layers",
    "ceselder/cot-oracle-paper-ablation-ours-3layers-onpolicy-lens-only",
    "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO",
    "ceselder/cot-oracle-grpo-step-500",
]


def default_output_dir() -> str:
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return f"experiments/paper_collection_aobench_{timestamp}"


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
    args = parser.parse_args()

    verbalizer_lora_paths = args.verbalizer_lora or list(PAPER_COLLECTION_VERBALIZERS)
    include = args.include if args.include is not None else list(EVAL_PROFILES[args.profile])
    output_dir = args.output_dir or default_output_dir()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir, "run_config.json").write_text(json.dumps({
        "model_name": args.model_name,
        "profile": args.profile,
        "include": include,
        "n_positions": args.n_positions,
        "verbalizer_lora_paths": verbalizer_lora_paths,
    }, indent=2))

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    print(f"Loading model: {args.model_name} on {device} with dtype={dtype}")
    model = load_model(args.model_name, dtype)
    model.eval()
    print(f"Running profile: {args.profile}")
    print(f"Include list: {include}")
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
    )

    combined_path = os.path.join(output_dir, "all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All eval summaries saved to {combined_path}")

    generate_report(output_dir)
    print_results_comparison(all_summaries)


if __name__ == "__main__":
    main()
