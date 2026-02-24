#!/usr/bin/env python3
"""Smoke test: load model + AO checkpoint, run all 7 training evals from HF.

This tests the full eval pipeline end-to-end without training:
1. Load Qwen3-8B + AO LoRA checkpoint
2. Run each eval (max 5 items each for speed)
3. Report which evals pass/fail

No wandb, no training data needed.
"""

import os
import sys
import gc
from pathlib import Path

# Set AO repo path
os.environ["AO_REPO_PATH"] = "/root/activation_oracles"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from core.ao import load_model_with_ao

def main():
    print("=" * 60)
    print("EVAL PIPELINE SMOKE TEST")
    print("=" * 60)

    # 1. Load model
    print("\n[1] Loading Qwen3-8B + AO checkpoint...")
    model, tokenizer = load_model_with_ao(
        model_name="Qwen/Qwen3-8B",
        device="cuda",
    )
    print(f"  Model loaded. Adapters: {model.peft_config.keys()}")

    # 2. Run training evals (using HF loading)
    print("\n[2] Running training evals (5 items each, from HuggingFace)...")

    from evals.training_eval_hook import run_training_evals

    # Use empty eval dir to force HF loading
    eval_dir = "/tmp/eval_smoke_test"
    os.makedirs(eval_dir, exist_ok=True)

    metrics = run_training_evals(
        model=model,
        tokenizer=tokenizer,
        model_name="Qwen/Qwen3-8B",
        step=0,
        device="cuda",
        eval_dir=eval_dir,
        max_items_per_eval=5,
        skip_rot13=True,  # Skip ROT13 to save time (needs extra adapter)
        oracle_adapter_name="default",
    )

    # 3. Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        if "table" not in k and "sample" not in k:
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

    # Check which evals produced metrics
    evals_with_metrics = set()
    for k in metrics:
        parts = k.split("/")
        if len(parts) == 2:
            evals_with_metrics.add(parts[1].rsplit("_", 1)[0])

    expected = {
        "hinted_mcq", "hinted_mcq_truthfulqa",
        "sycophancy_v2_riya", "decorative_cot",
        "sentence_insertion", "reasoning_termination_riya",
    }
    missing = expected - evals_with_metrics
    if missing:
        print(f"\n  WARNING: Missing metrics from: {missing}")
    else:
        print(f"\n  All {len(expected)} evals produced metrics!")

    print("\nDone.")


if __name__ == "__main__":
    main()
