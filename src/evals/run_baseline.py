"""
Run all baseline evals for the original AO checkpoint (before any fine-tuning).

This establishes the baseline performance that our CoT oracle training should match
or exceed on regression tasks, and sets the floor for unfaithfulness detection.

Runs:
1. Our 6 unfaithfulness evals (run_evals.py) with original AO
2. AO's classification regression evals (ao_regression.py) with original AO only

Logs everything to a separate wandb run for comparison.

Usage:
    python3 src/evals/run_baseline.py --model Qwen/Qwen3-8B
    python3 src/evals/run_baseline.py --model Qwen/Qwen3-1.7B --skip-regression
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from signs_of_life.ao_lib import (
    load_model_with_ao,
    generate_cot,
    generate_direct_answer,
    split_cot_into_sentences,
    collect_activations_at_positions,
    find_sentence_boundary_positions,
    run_oracle_on_activations,
    layer_percent_to_layer,
)
from evals.common import (
    EvalItem,
    CompletedEvalItem,
    load_eval_items,
    save_completed_items,
    extract_numerical_answer,
    extract_letter_answer,
    extract_yes_no,
    determine_ground_truth,
    compute_binary_metrics,
)
from evals.run_evals import (
    run_single_item,
    run_eval_batched,
    run_decorative_cot_eval,
    ORACLE_PROMPTS,
    _extract_answer,
)
from evals.score_oracle import score_eval, EVAL_PARSING


def run_unfaithfulness_evals(model, tokenizer, model_name, act_layer, eval_dir, output_dir, device="cuda", batch_size=8):
    """Run all 6 unfaithfulness evals and return results dict."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    act_dir = output_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    eval_files = sorted(eval_dir.glob("*.json"))
    all_scores = {}

    for eval_file in eval_files:
        eval_name = eval_file.stem
        print(f"\n{'=' * 60}")
        print(f"Running eval: {eval_name}")
        print(f"{'=' * 60}")

        items = load_eval_items(eval_file)
        print(f"  {len(items)} items loaded")

        if eval_name == "decorative_cot":
            completed = run_decorative_cot_eval(
                model, tokenizer, items, act_layer,
                model_name=model_name, device=device,
            )
        else:
            completed = run_eval_batched(
                model, tokenizer, items, act_layer,
                model_name=model_name, device=device,
                activations_dir=act_dir,
                batch_size=batch_size,
            )

        # Save raw results
        out_path = output_dir / f"{eval_name}_completed.json"
        save_completed_items(completed, out_path)

        # Score
        parsing_config = EVAL_PARSING.get(eval_name)
        scores = score_eval(eval_name, completed, parsing_config) if parsing_config else None
        all_scores[eval_name] = scores or {}

        # Print summary
        from collections import Counter
        labels = [c.ground_truth_label for c in completed]
        counts = Counter(labels)
        print(f"\n  Ground truth distribution for {eval_name}:")
        for label, count in sorted(counts.items()):
            print(f"    {label}: {count}")
        if scores:
            print(f"  Oracle scores:")
            for k, v in scores.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")
        print(f"  Saved to {out_path}")

    return all_scores


def run_ao_regression(model, tokenizer, model_name, layer_percent=50, output_dir="data/eval_results/ao_regression"):
    """Run AO classification regression eval with just base + original AO."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import AO eval utilities
    _ao_candidates = [
        Path("/workspace/ao_reference"),
        Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),
    ]
    ao_repo = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
    sys.path.insert(0, str(ao_repo))

    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.eval import run_evaluation, parse_answer
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
    from nl_probes.dataset_classes.classification import (
        ClassificationDatasetConfig,
        ClassificationDatasetLoader,
    )

    AO_CHECKPOINTS = {
        "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
        "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    }
    EVAL_DATASETS = ["geometry_of_truth", "sst2", "ag_news", "tense", "singular_plural", "relations"]

    INJECTION_LAYER = 1
    GENERATION_KWARGS = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}

    # Get the base model from our already-loaded model
    # We need the unwrapped model for AO's eval
    base_model = model
    # Try to get the base (un-LoRA'd) model
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            base_model = model.base_model.model

    submodule = get_hf_submodule(base_model, INJECTION_LAYER)

    # Load eval datasets
    print(f"\nLoading AO classification eval datasets...")
    all_eval_data = {}
    for ds_name in EVAL_DATASETS:
        try:
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
                num_test=100,
                splits=["test"],
                model_name=model_name,
                layer_percents=[layer_percent],
                save_acts=True,
                batch_size=64,
            )
            loader = ClassificationDatasetLoader(
                dataset_config=loader_config, model_kwargs={}, model=base_model,
            )
            all_eval_data[ds_name] = loader.load_dataset("test")
            print(f"  {ds_name}: {len(all_eval_data[ds_name])} examples")
        except Exception as e:
            print(f"  {ds_name}: FAILED to load ({e})")

    results = {}
    ao_path = AO_CHECKPOINTS.get(model_name)

    # Run with original AO checkpoint
    if ao_path:
        print(f"\n{'=' * 60}")
        print(f"AO Regression: Original AO ({ao_path})")
        print(f"{'=' * 60}")
        results["original_ao"] = {}
        for ds_name, data in all_eval_data.items():
            try:
                raw_results = run_evaluation(
                    eval_data=data,
                    model=base_model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                    global_step=-1,
                    lora_path=ao_path,
                    eval_batch_size=64,
                    steering_coefficient=1.0,
                    generation_kwargs=GENERATION_KWARGS,
                )
                correct = sum(1 for r, d in zip(raw_results, data) if parse_answer(r.api_response) == parse_answer(d.target_output))
                total = len(data)
                acc = correct / total if total > 0 else 0
                results["original_ao"][ds_name] = {"accuracy": acc, "correct": correct, "total": total}
                print(f"  {ds_name}: {acc:.3f} ({correct}/{total})")
            except Exception as e:
                print(f"  {ds_name}: FAILED ({e})")
                results["original_ao"][ds_name] = {"accuracy": 0, "error": str(e)}

    # Save
    out_path = output_dir / "baseline_regression.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all baseline evals")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--output-dir", default="data/eval_results/baseline")
    parser.add_argument("--skip-regression", action="store_true", help="Skip AO classification regression evals")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--skip-unfaithfulness", action="store_true", help="Skip unfaithfulness evals")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup wandb
    try:
        import wandb
        run_name = args.wandb_run_name or f"baseline_{args.model.split('/')[-1]}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=["baseline", args.model.split("/")[-1]],
            config={"model": args.model, "checkpoint": "original_ao", "eval_type": "baseline"},
        )
        use_wandb = True
        print(f"Logging to wandb: {args.wandb_project}/{run_name}")
    except Exception as e:
        print(f"wandb init failed ({e}), continuing without wandb")
        use_wandb = False

    # Load model + AO
    print(f"Loading {args.model} + AO...")
    model, tokenizer = load_model_with_ao(args.model, device=args.device)
    act_layer = layer_percent_to_layer(args.model, 50)
    print(f"Activation layer: {act_layer}")

    all_results = {}

    # 1. Unfaithfulness evals
    if not args.skip_unfaithfulness:
        print(f"\n{'#' * 60}")
        print("PHASE 1: Unfaithfulness Evals (6 datasets)")
        print(f"{'#' * 60}")
        unfaith_scores = run_unfaithfulness_evals(
            model, tokenizer, args.model, act_layer,
            args.eval_dir, output_dir / "unfaithfulness",
            device=args.device,
            batch_size=args.batch_size,
        )
        all_results["unfaithfulness"] = unfaith_scores

        if use_wandb:
            # Log flat metrics
            for eval_name, scores in unfaith_scores.items():
                for metric, value in scores.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"unfaith/{eval_name}/{metric}": value})

    # 2. AO regression evals
    if not args.skip_regression:
        print(f"\n{'#' * 60}")
        print("PHASE 2: AO Classification Regression Evals")
        print(f"{'#' * 60}")
        regression_results = run_ao_regression(
            model, tokenizer, args.model,
            layer_percent=50,
            output_dir=output_dir / "regression",
        )
        all_results["regression"] = regression_results

        if use_wandb:
            for checkpoint, ds_results in regression_results.items():
                for ds_name, metrics in ds_results.items():
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        wandb.log({f"regression/{checkpoint}/{ds_name}": metrics["accuracy"]})

    # Save combined results
    combined_path = output_dir / "all_baseline_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'#' * 60}")
    print(f"ALL BASELINE RESULTS SAVED to {combined_path}")
    print(f"{'#' * 60}")

    if use_wandb:
        wandb.finish()
        print("wandb run finished")


if __name__ == "__main__":
    main()
