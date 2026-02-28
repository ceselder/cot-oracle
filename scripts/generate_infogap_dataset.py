#!/usr/bin/env python3
"""
Generate information-gap dataset from existing CoT corpus.

Downloads corpus from HF, generates examples for all 8 infogap task types,
writes JSONL files to data/precomputed/.

Usage:
    python scripts/generate_infogap_dataset.py
    python scripts/generate_infogap_dataset.py --tasks early_answer_pred backtrack_pred
    python scripts/generate_infogap_dataset.py --dry-run  # just print stats
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

from dataset_classes.cot_infogap import (
    load_cot_backtrack_pred_data,
    load_cot_branch_pred_data,
    load_cot_completion_pred_data,
    load_cot_early_answer_pred_data,
    load_cot_error_pred_data,
    load_cot_remaining_strategy_data,
    load_cot_self_correction_data,
    load_cot_verification_data,
    pretokenize_corpus,
)

TASK_LOADERS = {
    "early_answer_pred": (load_cot_early_answer_pred_data, 20000),
    "backtrack_pred": (load_cot_backtrack_pred_data, 15000),
    "error_pred": (load_cot_error_pred_data, 15000),
    "self_correction": (load_cot_self_correction_data, 10000),
    "verification": (load_cot_verification_data, 10000),
    "branch_pred": (load_cot_branch_pred_data, 10000),
    "completion_pred": (load_cot_completion_pred_data, 10000),
    "remaining_strategy": (load_cot_remaining_strategy_data, 10000),
}

HF_CORPUS = "mats-10-sprint-cs-jb/cot-oracle-corpus-v5"
MODEL_NAME = "Qwen/Qwen3-8B"
STRIDE = 5


def resolve_corpus(corpus_id: str) -> str:
    """Download HF corpus to local JSONL if needed."""
    cache_dir = Path(os.environ.get("CACHE_DIR", "data")) / "cot_oracle" / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / "corpus_v5.jsonl"
    if local_path.exists():
        print(f"  Using cached corpus: {local_path}")
        return str(local_path)

    print(f"  Downloading corpus from HF: {corpus_id}")
    from datasets import load_dataset
    ds = load_dataset(corpus_id, split="train")

    with open(local_path, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")
    print(f"  Saved {len(ds)} entries to {local_path}")
    return str(local_path)


def main():
    parser = argparse.ArgumentParser(description="Generate infogap dataset from CoT corpus")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_LOADERS.keys()), choices=list(TASK_LOADERS.keys()), help="Which tasks to generate")
    parser.add_argument("--output-dir", default="data/precomputed", help="Output directory for JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Print 5 examples per task, don't write files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = resolve_corpus(HF_CORPUS)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Pre-tokenize corpus once (saves ~3.5 hours vs per-task tokenization)
    print("\nPre-tokenizing corpus (one-time cost)...")
    pretokenize_corpus(corpus_path, tokenizer)

    for task_name in args.tasks:
        loader_fn, default_n = TASK_LOADERS[task_name]
        print(f"\n{'=' * 60}")
        print(f"Generating: {task_name} ({default_n} examples)")
        print(f"{'=' * 60}")

        data = loader_fn(
            corpus_path, tokenizer, MODEL_NAME,
            num_examples=default_n,
            stride=STRIDE,
            seed=args.seed,
        )

        if args.dry_run:
            print(f"\n  Sample examples ({task_name}):")
            for i, dp in enumerate(data[:5]):
                print(f"\n  --- Example {i+1} ---")
                print(f"  Type: {dp['datapoint_type']}")
                print(f"  Prompt: {dp['prompt'][:200]}")
                print(f"  Target: {dp['target_response'][:200]}")
                print(f"  Positions: {dp['num_positions']}")
                print(f"  Context IDs len: {len(dp['context_input_ids'])}")
            continue

        out_path = output_dir / f"{task_name}.jsonl"
        with open(out_path, "w") as f:
            for dp in data:
                f.write(json.dumps(dp) + "\n")
        print(f"  Wrote {len(data)} examples to {out_path}")

    print(f"\nDone! Generated {len(args.tasks)} task files in {output_dir}")


if __name__ == "__main__":
    main()
