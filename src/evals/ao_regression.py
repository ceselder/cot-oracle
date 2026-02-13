"""
AO Regression Eval: Verify fine-tuning didn't break original AO capabilities.

Runs AO's classification eval on:
1. Original AO checkpoint (baseline)
2. Our CoT oracle checkpoint (should not regress)
3. No LoRA / base model (sanity check)

Uses the same 20 classification datasets from the AO paper.

Usage:
    python src/evals/ao_regression.py \
        --our-checkpoint /workspace/checkpoints/cot_oracle/final \
        --model Qwen/Qwen3-8B \
        --output-dir data/eval_results/ao_regression
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig

# Add AO repo to path
_ao_candidates = [
    Path("/workspace/ao_reference"),
    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),
]
AO_REPO = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
sys.path.insert(0, str(AO_REPO))

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation, parse_answer
from nl_probes.base_experiment import sanitize_lora_name

# AO's original checkpoints
AO_CHECKPOINTS = {
    "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
}

INJECTION_LAYER = 1
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}

# Subset of classification datasets (representative, not all 20)
EVAL_DATASETS = {
    "geometry_of_truth": {"num_test": 100, "splits": ["test"]},
    "sst2": {"num_test": 100, "splits": ["test"]},
    "ag_news": {"num_test": 100, "splits": ["test"]},
    "tense": {"num_test": 100, "splits": ["test"]},
    "singular_plural": {"num_test": 100, "splits": ["test"]},
    "relations": {"num_test": 100, "splits": ["test"]},
}


def load_eval_datasets(model_name, layer_percent, model_kwargs, model=None):
    """Load classification eval datasets for a given layer percent."""
    all_eval_data = {}

    for ds_name, dcfg in EVAL_DATASETS.items():
        config = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_end_offset=-3,
            min_end_offset=-3,
            max_window_size=1,
            min_window_size=1,
        )
        loader_config = DatasetLoaderConfig(
            custom_dataset_params=config,
            num_train=0,
            num_test=dcfg["num_test"],
            splits=dcfg["splits"],
            model_name=model_name,
            layer_percents=[layer_percent],
            save_acts=True,
            batch_size=64,
        )
        loader = ClassificationDatasetLoader(
            dataset_config=loader_config, model_kwargs=model_kwargs, model=model,
        )
        ds_id = ds_name
        all_eval_data[ds_id] = loader.load_dataset("test")

    return all_eval_data


def run_checkpoint_eval(model, tokenizer, submodule, model_name, lora_path, eval_data, batch_size=64):
    """Run eval on all datasets for a single checkpoint."""
    results = {}

    for ds_id, data in eval_data.items():
        raw_results = run_evaluation(
            eval_data=data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            global_step=-1,
            lora_path=lora_path,
            eval_batch_size=batch_size,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
        )

        correct = 0
        total = 0
        for resp, target in zip(raw_results, data, strict=True):
            pred = parse_answer(resp.api_response)
            gold = parse_answer(target.target_output)
            if pred == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        results[ds_id] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  {ds_id}: {acc:.3f} ({correct}/{total})")

    # Overall
    total_correct = sum(r["correct"] for r in results.values())
    total_n = sum(r["total"] for r in results.values())
    results["_overall"] = {
        "accuracy": total_correct / total_n if total_n > 0 else 0,
        "correct": total_correct,
        "total": total_n,
    }
    print(f"  OVERALL: {results['_overall']['accuracy']:.3f} ({total_correct}/{total_n})")

    return results


def main():
    parser = argparse.ArgumentParser(description="AO Regression Eval")
    parser.add_argument("--our-checkpoint", required=True, help="Path to our fine-tuned checkpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layer-percent", type=int, default=50)
    parser.add_argument("--output-dir", default="data/eval_results/ao_regression")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading {args.model}...")
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, dtype)
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    # Need a dummy adapter for peft to work
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load eval datasets
    print(f"\nLoading eval datasets (layer_percent={args.layer_percent})...")
    eval_data = load_eval_datasets(args.model, args.layer_percent, {}, model=model)
    print(f"Loaded {len(eval_data)} datasets")

    all_results = {}

    # 1. No LoRA baseline (base model, no activation reading)
    print(f"\n{'=' * 60}")
    print("Eval: Base model (no LoRA)")
    print(f"{'=' * 60}")
    all_results["base_model"] = run_checkpoint_eval(
        model, tokenizer, submodule, args.model, None, eval_data,
    )

    # 2. Original AO checkpoint
    original_lora = AO_CHECKPOINTS.get(args.model)
    if original_lora:
        print(f"\n{'=' * 60}")
        print(f"Eval: Original AO ({original_lora})")
        print(f"{'=' * 60}")
        all_results["original_ao"] = run_checkpoint_eval(
            model, tokenizer, submodule, args.model, original_lora, eval_data,
        )

    # 3. Our fine-tuned checkpoint
    if Path(args.our_checkpoint).exists():
        print(f"\n{'=' * 60}")
        print(f"Eval: CoT Oracle ({args.our_checkpoint})")
        print(f"{'=' * 60}")
        all_results["cot_oracle"] = run_checkpoint_eval(
            model, tokenizer, submodule, args.model, args.our_checkpoint, eval_data,
        )

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("REGRESSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<25} {'Base':>8} {'Orig AO':>8} {'CoT Oracle':>10} {'Delta':>8}")
    print("-" * 65)

    for ds_id in EVAL_DATASETS:
        base_acc = all_results.get("base_model", {}).get(ds_id, {}).get("accuracy", 0)
        orig_acc = all_results.get("original_ao", {}).get(ds_id, {}).get("accuracy", 0)
        ours_acc = all_results.get("cot_oracle", {}).get(ds_id, {}).get("accuracy", 0)
        delta = ours_acc - orig_acc
        flag = " !!!" if delta < -0.05 else ""
        print(f"{ds_id:<25} {base_acc:>8.3f} {orig_acc:>8.3f} {ours_acc:>10.3f} {delta:>+8.3f}{flag}")

    base_overall = all_results.get("base_model", {}).get("_overall", {}).get("accuracy", 0)
    orig_overall = all_results.get("original_ao", {}).get("_overall", {}).get("accuracy", 0)
    ours_overall = all_results.get("cot_oracle", {}).get("_overall", {}).get("accuracy", 0)
    delta_overall = ours_overall - orig_overall
    print("-" * 65)
    print(f"{'OVERALL':<25} {base_overall:>8.3f} {orig_overall:>8.3f} {ours_overall:>10.3f} {delta_overall:>+8.3f}")

    if delta_overall < -0.05:
        print("\n!!! REGRESSION DETECTED: Overall accuracy dropped >5% !!!")
    else:
        print(f"\nNo significant regression (delta={delta_overall:+.3f})")

    # Save
    out_path = output_dir / "regression_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
