"""
Precompute all training datasets and save as JSONL files.

This runs all task loaders locally (CPU-only, no GPU needed) and saves
the output so training can load precomputed data directly without
running the dataset generation logic on the GPU.

Usage:
    cd /home/celeste/cot-oracle
    python3 scripts/precompute_training_data.py

Output:
    data/precomputed/{task_name}.jsonl  — one file per task
    data/precomputed/manifest.json      — metadata about all tasks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer


# Task configs: (task_name, module, loader_fn, corpus_type, default_n)
TASKS = [
    ("full_recon",       "dataset_classes.cot_rollout_multilayer",      "load_cot_rollout_multilayer",          "main",    40000),
    ("next_step",        "dataset_classes.cot_next_step",               "load_cot_next_step_data",              "main",    30000),
    ("answer_pred",      "dataset_classes.cot_answer_prediction",       "load_cot_answer_prediction_data",      "main",    20000),
    ("load_bearing",     "dataset_classes.cot_load_bearing",            "load_cot_load_bearing_data",           "main",    15000),
    ("correctness",      "dataset_classes.cot_correctness",             "load_cot_correctness_data",            "main",    15000),
    ("decorative",       "dataset_classes.cot_decorative",              "load_cot_decorative_data",             "main",    15000),
    ("domain",           "dataset_classes.cot_domain",                  "load_cot_domain_data",                 "main",    15000),
    ("reasoning_term",   "dataset_classes.cot_reasoning_termination",   "load_cot_reasoning_termination_data",  "main",    15000),
    ("conv_qa",          "dataset_classes.cot_conversational",          "load_cot_conversational_data",         "concept", 10000),
    ("atypical_answer",  "dataset_classes.cot_atypical_answer",         "load_cot_atypical_answer_data",        "atypical", 20000),
    ("prompt_inversion", "dataset_classes.cot_prompt_inversion",       "load_cot_prompt_inversion_data",       "main",     20000),
    ("compqa",           "dataset_classes.cot_compqa",               "load_cot_compqa_data",                 "compqa",   8000),
]


def precompute_task(task_name, module_path, loader_name, corpus_type,
                    num_examples, tokenizer, model_name,
                    main_corpus, concept_corpus, cotqa_path,
                    output_dir, stride=5, max_positions_per_layer=None,
                    atypical_data_path=None):
    """Run a single task loader and save to JSONL."""
    import importlib

    print(f"\n{'='*60}")
    print(f"  {task_name}: {num_examples} examples")
    print(f"{'='*60}")

    t0 = time.time()

    mod = importlib.import_module(module_path)
    loader_fn = getattr(mod, loader_name)

    if corpus_type == "concept":
        data = loader_fn(
            concept_corpus, cotqa_path, tokenizer, model_name,
            num_examples=num_examples,
            stride=stride,
            max_positions_per_layer=max_positions_per_layer,
        )
    elif corpus_type == "atypical":
        apath = atypical_data_path or "data/atypical_answer_training.jsonl"
        data = loader_fn(
            apath, tokenizer, model_name,
            num_examples=num_examples,
            stride=stride,
            max_positions_per_layer=max_positions_per_layer,
            atypical_data_path=apath,
        )
    elif corpus_type == "compqa":
        # CompQA loads from HF, first arg is local cache path
        compqa_cache = str(output_dir / "compqa_raw.json")
        data = loader_fn(
            compqa_cache, tokenizer, model_name,
            num_examples=num_examples,
            stride=stride,
            max_positions_per_layer=max_positions_per_layer,
        )
    else:
        data = loader_fn(
            main_corpus, tokenizer, model_name,
            num_examples=num_examples,
            stride=stride,
            max_positions_per_layer=max_positions_per_layer,
        )

    elapsed = time.time() - t0

    # Save as JSONL
    out_path = output_dir / f"{task_name}.jsonl"
    with open(out_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  -> {len(data)} examples in {elapsed:.1f}s")
    print(f"  -> Saved to {out_path} ({size_mb:.1f} MB)")

    return {
        "task": task_name,
        "count": len(data),
        "file": str(out_path.name),
        "size_mb": round(size_mb, 2),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute training datasets")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl")
    parser.add_argument("--concept-corpus", default="data/concept_corpus/corpus_full.jsonl")
    parser.add_argument("--cotqa-path", default="data/concept_corpus/corpus_full_conv_qa_llm.jsonl")
    parser.add_argument("--output-dir", default="data/precomputed")
    parser.add_argument("--stride", default="5",
                        help="Activation stride: int or 'punctuation' (default: 5)")
    parser.add_argument("--max-positions-per-layer", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Only precompute these tasks (default: all)")
    # Per-task overrides
    parser.add_argument("--full-recon-n", type=int, default=None)
    parser.add_argument("--next-step-n", type=int, default=None)
    parser.add_argument("--answer-pred-n", type=int, default=None)
    parser.add_argument("--load-bearing-n", type=int, default=None)
    parser.add_argument("--correctness-n", type=int, default=None)
    parser.add_argument("--decorative-n", type=int, default=None)
    parser.add_argument("--domain-n", type=int, default=None)
    parser.add_argument("--reasoning-term-n", type=int, default=None)
    parser.add_argument("--conv-qa-n", type=int, default=None)
    parser.add_argument("--atypical-answer-n", type=int, default=None)
    parser.add_argument("--prompt-inversion-n", type=int, default=None)
    parser.add_argument("--compqa-n", type=int, default=None)
    parser.add_argument("--atypical-data-path",
                        default="data/atypical_answer_training.jsonl",
                        help="Path to atypical answer JSONL")
    args = parser.parse_args()

    # Parse stride: int-like string → int, "punctuation" stays as-is
    try:
        args.stride = int(args.stride)
    except ValueError:
        if args.stride != "punctuation":
            parser.error(f"--stride must be an integer or 'punctuation', got '{args.stride}'")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override map
    n_overrides = {
        "full_recon": args.full_recon_n,
        "next_step": args.next_step_n,
        "answer_pred": args.answer_pred_n,
        "load_bearing": args.load_bearing_n,
        "correctness": args.correctness_n,
        "decorative": args.decorative_n,
        "domain": args.domain_n,
        "reasoning_term": args.reasoning_term_n,
        "conv_qa": args.conv_qa_n,
        "atypical_answer": args.atypical_answer_n,
        "prompt_inversion": args.prompt_inversion_n,
        "compqa": args.compqa_n,
    }

    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    manifest = {"model": args.model, "stride": args.stride, "tasks": []}
    total_examples = 0

    for task_name, module_path, loader_name, corpus_type, default_n in TASKS:
        if args.tasks and task_name not in args.tasks:
            continue

        n = n_overrides.get(task_name) or default_n

        try:
            info = precompute_task(
                task_name, module_path, loader_name, corpus_type,
                n, tokenizer, args.model,
                args.corpus, args.concept_corpus, args.cotqa_path,
                output_dir, args.stride, args.max_positions_per_layer,
                atypical_data_path=getattr(args, "atypical_data_path", None),
            )
            manifest["tasks"].append(info)
            total_examples += info["count"]
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            manifest["tasks"].append({
                "task": task_name, "count": 0, "error": str(e),
            })

    # Save manifest
    manifest["total_examples"] = total_examples
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  DONE: {total_examples} total examples across {len(manifest['tasks'])} tasks")
    print(f"  Manifest: {manifest_path}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
