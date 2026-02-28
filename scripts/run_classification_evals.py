#!/usr/bin/env python3
"""Run classification evals against a trained checkpoint.

Usage:
    python scripts/run_classification_evals.py --checkpoint checkpoints/step_385
    python scripts/run_classification_evals.py --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_thirds_v1/step_4000
"""
import sys, os, time, gc, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from cot_utils import get_injection_layers
from evals.training_eval_hook import (
    run_training_evals, set_oracle_mode, TRAINING_EVALS,
)

CLS_EVALS = [
    "cls_sst2", "cls_snli", "cls_ag_news", "cls_ner", "cls_tense",
    "cls_language_id", "cls_singular_plural", "cls_geometry_of_truth", "cls_relations",
]

# Standard evals known to exist on HF (matches configs/train.yaml)
STANDARD_EVALS = [
    "hinted_mcq_truthfulqa", "sycophancy_v2_riya", "sentence_insertion",
    "reasoning_termination_riya", "atypical_answer_riya", "atypical_answer_mcq",
    "cybercrime_ood", "rot13_reconstruction",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--eval-dir", type=str, default="data/evals")
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--include-standard", action="store_true", help="Also run standard evals for comparison")
    args = parser.parse_args()

    device = "cuda"
    act_layers = get_injection_layers(args.model_name)
    print(f"Injection layers: {act_layers}")
    print(f"Stride: {args.stride}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    print(f"Loading PeftModel from {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint, adapter_name="default")
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Set oracle mode
    set_oracle_mode(trained=True, oracle_adapter_name="default", stride=args.stride, layers=act_layers)

    # Build eval list
    eval_names = list(CLS_EVALS)
    if args.include_standard:
        eval_names = list(STANDARD_EVALS) + eval_names

    # Run evals
    print(f"\nRunning {len(eval_names)} evals: {', '.join(eval_names)}")
    t0 = time.perf_counter()
    metrics = run_training_evals(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        step=0,
        device=device,
        eval_dir=args.eval_dir,
        max_items_per_eval=args.max_items,
        oracle_adapter_name="default",
        activation_cache_dir=None,
        eval_names=eval_names,
        eval_batch_size=args.eval_batch_size,
        stride=args.stride,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nEvals completed in {elapsed:.1f}s")

    # Print results
    print(f"\n{'='*60}")
    print("CLASSIFICATION EVAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Eval':<30} {'Acc':>8} {'N':>6}")
    print("-" * 46)

    cls_accs = []
    for name in CLS_EVALS:
        acc_key = f"eval/{name}_acc"
        n_key = f"eval_n/{name}"
        if acc_key in metrics:
            acc = metrics[acc_key]
            n = metrics.get(n_key, "?")
            cls_accs.append(acc)
            print(f"{name:<30} {acc:>7.1%} {n:>6}")
        else:
            print(f"{name:<30} {'N/A':>8} {'':>6}")

    if cls_accs:
        print("-" * 46)
        print(f"{'Classification Mean':<30} {sum(cls_accs)/len(cls_accs):>7.1%} {len(cls_accs):>6}")

    if args.include_standard:
        print(f"\n{'='*60}")
        print("STANDARD EVAL RESULTS")
        print(f"{'='*60}")
        for name in STANDARD_EVALS:
            acc_key = f"eval/{name}_acc"
            if acc_key in metrics:
                print(f"{name:<30} {metrics[acc_key]:>7.1%}")

    # Save full metrics to JSON
    out_path = Path(f"eval_logs/classification_evals_{Path(args.checkpoint).name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    serializable["checkpoint"] = args.checkpoint
    serializable["elapsed_seconds"] = elapsed
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nFull metrics saved to {out_path}")


if __name__ == "__main__":
    main()
