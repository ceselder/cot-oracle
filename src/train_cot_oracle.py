"""
Train CoT Oracle v3 by continuing from the existing AO checkpoint.

Uses AO repo's train_model() directly with our custom dataset loaders.
Monkey-patches eval_all_datasets to also run our unfaithfulness evals.

v3 training mixture (4 tasks, ~230K examples):
  - Context Prediction: 150K (self-supervised backbone)
  - Importance / Thought Anchor: 30K (semi-supervised, binary)
  - Unverbalized Behavior: 30K (SAE-supervised)
  - Decorative CoT: 20K (self-supervised)

Taxonomy and answer_tracking are moved to held-out eval (zero-shot).
Summary is dropped.

Usage:
    # Generate data first
    python src/data_pipeline/generate_cots.py --n-problems 500 --output data/cot_corpus/corpus.jsonl
    python src/data_pipeline/extract_labels.py --corpus data/cot_corpus/corpus.jsonl --all

    # Then train (requires torchrun even on single GPU, per AO repo convention)
    torchrun --nproc_per_node=1 src/train_cot_oracle.py \
        --corpus data/cot_corpus/corpus.jsonl \
        --labels-dir data/cot_corpus \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# AO repo imports -- detect environment
_ao_candidates = [
    Path("/workspace/ao_reference"),  # vast.ai
    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),  # local
]
AO_REPO = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
sys.path.insert(0, str(AO_REPO))

import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import create_training_datapoint, TrainingDataPoint
import nl_probes.sft as sft_module
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

# Our dataset loaders (v3 training tasks)
from dataset_classes.cot_context_prediction import load_cot_context_prediction_data
from dataset_classes.cot_importance import load_cot_importance_data
from dataset_classes.cot_unverbalized import load_cot_unverbalized_data
from dataset_classes.cot_decorative import load_cot_decorative_data

# Held-out eval tasks (zero-shot, not trained on)
from dataset_classes.cot_taxonomy import load_cot_taxonomy_data
from dataset_classes.cot_answer_tracking import load_cot_answer_tracking_data

from signs_of_life.ao_lib import layer_percent_to_layer


def dicts_to_training_data(
    raw_data: list[dict],
    tokenizer: AutoTokenizer,
) -> list[TrainingDataPoint]:
    """Convert our dataset loader output to AO TrainingDataPoint objects."""
    training_data = []
    skipped = 0

    for item in raw_data:
        try:
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=item["layer"],
                num_positions=item["num_positions"],
                tokenizer=tokenizer,
                acts_BD=None,  # On-the-fly collection
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=item["context_positions"],
            )
            training_data.append(dp)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skipped datapoint ({e})")

    if skipped > 0:
        print(f"  Skipped {skipped} datapoints during conversion")

    return training_data


def build_training_mixture(
    corpus_path: str,
    labels_dir: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
) -> list[TrainingDataPoint]:
    """Build the v3 training mixture from 4 tasks."""
    labels_dir = Path(labels_dir)

    # v3 task sizes
    task_configs = {
        "cot_context_prediction": 150000,
        "cot_importance": 30000,
        "cot_unverbalized": 30000,
        "cot_decorative": 20000,
    }

    all_data = []

    # Task 1: Context Prediction (always available -- self-supervised)
    print("\n=== Task 1: CoT Context Prediction (150K, self-supervised) ===")
    raw = load_cot_context_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_configs["cot_context_prediction"],
    )
    data = dicts_to_training_data(raw, tokenizer)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 2: Importance / Thought Anchor (binary, requires importance labels)
    importance_path = labels_dir / "labels_importance.jsonl"
    if importance_path.exists():
        print("\n=== Task 2: Thought Anchor Detection (30K, binary) ===")
        raw = load_cot_importance_data(
            corpus_path, str(importance_path), tokenizer, model_name, layer_percents,
            num_examples=task_configs["cot_importance"],
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  Generated {len(data)} examples")
        all_data.extend(data)
    else:
        print(f"\n  Skipping Task 2 (no {importance_path})")

    # Task 3: Unverbalized Behavior Detection (requires SAE labels)
    sae_labels_path = labels_dir / "labels_sae_unverbalized.jsonl"
    if sae_labels_path.exists():
        print("\n=== Task 3: Unverbalized Behavior Detection (30K, SAE-supervised) ===")
        raw = load_cot_unverbalized_data(
            corpus_path, str(sae_labels_path), tokenizer, model_name, layer_percents,
            num_examples=task_configs["cot_unverbalized"],
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  Generated {len(data)} examples")
        all_data.extend(data)
    else:
        print(f"\n  Skipping Task 3 (no {sae_labels_path})")

    # Task 4: Decorative CoT Detection (self-supervised, requires --keep-all corpus)
    print("\n=== Task 4: Decorative CoT Detection (20K, self-supervised) ===")
    try:
        raw = load_cot_decorative_data(
            corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_configs["cot_decorative"],
        )
        data = dicts_to_training_data(raw, tokenizer)
        print(f"  Generated {len(data)} examples")
        all_data.extend(data)
    except ValueError as e:
        print(f"  Skipping Task 4 ({e})")

    print(f"\n{'=' * 60}")
    print(f"Total training examples: {len(all_data)}")

    # Count by type
    from collections import Counter
    type_counts = Counter(dp.datapoint_type for dp in all_data)
    for dtype, count in sorted(type_counts.items()):
        pct = count / len(all_data) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    return all_data


def build_eval_datasets(
    corpus_path: str,
    labels_dir: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
) -> dict[str, list[TrainingDataPoint]]:
    """Build held-out eval datasets including zero-shot tasks.

    Taxonomy and answer_tracking are held-out (oracle has never seen these
    during training). This tests zero-shot generalization.
    """
    labels_dir = Path(labels_dir)
    eval_datasets = {}

    # Zero-shot: Taxonomy (held out from training)
    taxonomy_path = labels_dir / "labels_taxonomy.jsonl"
    if taxonomy_path.exists():
        print("\n=== Eval: Taxonomy (zero-shot, 100 items) ===")
        try:
            raw = load_cot_taxonomy_data(
                corpus_path, str(taxonomy_path), tokenizer, model_name, layer_percents,
                num_examples=100,
            )
            data = dicts_to_training_data(raw, tokenizer)
            eval_datasets["cot_taxonomy"] = data
            print(f"  Generated {len(data)} eval examples")
        except Exception as e:
            print(f"  Failed: {e}")

    # Zero-shot: Answer Tracking (held out from training)
    tracking_path = labels_dir / "labels_answer_tracking.jsonl"
    if tracking_path.exists():
        print("\n=== Eval: Answer Tracking (zero-shot, 100 items) ===")
        try:
            raw = load_cot_answer_tracking_data(
                corpus_path, str(tracking_path), tokenizer, model_name, layer_percents,
                num_examples=100,
            )
            data = dicts_to_training_data(raw, tokenizer)
            eval_datasets["cot_answer_tracking"] = data
            print(f"  Generated {len(data)} eval examples")
        except Exception as e:
            print(f"  Failed: {e}")

    return eval_datasets


def install_per_task_loss_hook(training_data, batch_size):
    """
    Monkey-patch AO's training loop to log per-task loss.

    We patch construct_batch to stash the datapoint_types for the current batch,
    then patch train_features_batch to compute per-item loss and group by task.
    """
    import wandb
    import torch
    import torch.nn.functional as F
    from nl_probes.sft import train_features_batch as _original_train
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

    # Stash for current batch's task types
    _batch_state = {"types": []}

    # Patch construct_batch to capture datapoint types
    from nl_probes.sft import construct_batch as _original_construct
    def patched_construct_batch(batch_list, tokenizer, device):
        _batch_state["types"] = [dp.datapoint_type for dp in batch_list]
        return _original_construct(batch_list, tokenizer, device)
    sft_module.construct_batch = patched_construct_batch

    # Patch train_features_batch to compute per-task loss
    def patched_train_features_batch(cfg, training_batch, model, submodule, device, dtype):
        # Run the normal forward pass
        hook_fn = get_hf_activation_steering_hook(
            vectors=training_batch.steering_vectors,
            positions=training_batch.positions,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
        )
        tokenized_input = {
            "input_ids": training_batch.input_ids,
            "attention_mask": training_batch.attention_mask,
        }
        with add_hook(submodule, hook_fn):
            outputs = model(**tokenized_input, labels=training_batch.labels)

        # Log per-task loss using per-item loss
        batch_types = _batch_state["types"]
        if batch_types and len(batch_types) == training_batch.input_ids.shape[0]:
            # Compute per-item loss from logits
            logits = outputs.logits  # [B, seq, vocab]
            labels = training_batch.labels  # [B, seq]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Per-token loss
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            ).view(shift_labels.shape)  # [B, seq-1]
            # Mask out padding (-100 labels)
            mask = (shift_labels != -100).float()
            per_item_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Group by task
            from collections import defaultdict
            task_losses = defaultdict(list)
            for i, task_type in enumerate(batch_types):
                task_losses[task_type].append(per_item_loss[i].item())

            log_dict = {}
            for task, losses in task_losses.items():
                log_dict[f"train/loss_{task}"] = sum(losses) / len(losses)
            if wandb.run is not None:
                wandb.log(log_dict, commit=False)

        return outputs.loss

    sft_module.train_features_batch = patched_train_features_batch
    print("Installed per-task loss logging hook")


def install_unfaithfulness_eval_hook(model_name, eval_dir="data/evals", fast_n=5):
    """
    Monkey-patch AO's eval_all_datasets to also run our unfaithfulness evals.

    Runs a fast subset (fast_n items per eval type) at each eval step,
    logging results to wandb alongside the training task evals.

    Args:
        model_name: Model name for activation layer calculation
        eval_dir: Path to eval dataset JSON files
        fast_n: Number of items per eval type for fast eval (default 5)
    """
    import wandb
    from evals.common import load_eval_items, determine_ground_truth
    from evals.score_oracle import score_eval, EVAL_PARSING
    from evals.run_evals import (
        run_single_item, ORACLE_PROMPTS, _extract_answer,
    )

    eval_dir = Path(eval_dir)
    act_layer = layer_percent_to_layer(model_name, 50)

    # Pre-load fast eval subsets
    fast_items = {}
    for eval_file in sorted(eval_dir.glob("*.json")):
        eval_name = eval_file.stem
        if eval_name in ("decorative_cot", "sentence_insertion"):
            continue  # Skip -- too slow or requires special handling
        items = load_eval_items(eval_file)
        fast_items[eval_name] = items[:fast_n]

    total_items = sum(len(v) for v in fast_items.values())
    print(f"Unfaithfulness eval hook: {len(fast_items)} evals, {total_items} total items")

    # Save reference to original eval function
    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        """Run AO's built-in evals + our unfaithfulness evals."""
        # 1. Run original AO evals
        _original_eval(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)

        # 2. Run our unfaithfulness evals (fast subset)
        print(f"\n--- Unfaithfulness Evals (step {global_step}) ---")

        all_rows = []  # For wandb table

        for eval_name, items in fast_items.items():
            try:
                completed = []
                for item in items:
                    result = run_single_item(
                        model, tokenizer, item, act_layer,
                        model_name=model_name, device=str(device),
                    )
                    completed.append(result)

                # Log individual outputs
                for c in completed:
                    oracle_short = (c.oracle_response or "")[:200]
                    test_ans_short = (c.test_answer or "")[:50]
                    gt = c.ground_truth_label or "?"
                    print(f"    [{eval_name}] gt={gt} test_ans={test_ans_short} oracle={oracle_short}")
                    all_rows.append([
                        eval_name, c.example_id, gt,
                        (c.test_answer or "")[:100],
                        (c.oracle_response or "")[:300],
                        (c.clean_answer or "")[:50],
                    ])

                # Score
                parsing_config = EVAL_PARSING.get(eval_name)
                if parsing_config:
                    metrics = score_eval(eval_name, completed, parsing_config)
                    if metrics:
                        wandb.log({
                            f"unfaith/{eval_name}/accuracy": metrics["accuracy"],
                            f"unfaith/{eval_name}/n_scored": metrics.get("n_items", 0),
                        }, step=global_step)
                        print(f"  {eval_name}: acc={metrics['accuracy']:.3f} ({metrics.get('n_items', 0)} scored)")
                    else:
                        print(f"  {eval_name}: no scoreable items")
                else:
                    print(f"  {eval_name}: no parsing config")

                # Also log ground truth distribution
                from collections import Counter
                labels = Counter(c.ground_truth_label for c in completed)
                for label, count in labels.items():
                    wandb.log({f"unfaith/{eval_name}/gt_{label}": count}, step=global_step)

            except Exception as e:
                import traceback
                print(f"  {eval_name}: FAILED ({e})")
                traceback.print_exc()

        # Log all outputs as wandb table
        if all_rows and wandb.run is not None:
            try:
                table = wandb.Table(
                    columns=["eval", "id", "ground_truth", "test_answer", "oracle_response", "clean_answer"],
                    data=all_rows,
                )
                wandb.log({f"unfaith/outputs_step_{global_step}": table}, step=global_step)
            except Exception:
                pass  # Don't let table logging break training

        print("--- End Unfaithfulness Evals ---\n")

    # Apply the monkey-patch
    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed unfaithfulness eval hook into training loop")


def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle v3")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--labels-dir", required=True, help="Directory with label files")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-dir", default="data/evals", help="Path to unfaithfulness eval datasets")
    parser.add_argument("--fast-eval-n", type=int, default=10, help="Items per eval type for fast eval during training")
    parser.add_argument("--no-unfaith-evals", action="store_true", help="Skip unfaithfulness evals during training")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]

    # Build training data (v3: 4 tasks)
    print("Building v3 training mixture...")
    training_data = build_training_mixture(
        args.corpus, args.labels_dir, tokenizer, args.model, layer_percents,
    )

    if not training_data:
        print("ERROR: No training data generated!")
        return

    # Build eval datasets (held-out tasks + per-task eval splits)
    eval_datasets = build_eval_datasets(
        args.corpus, args.labels_dir, tokenizer, args.model, layer_percents,
    )

    # Also split off 100 examples per training task as eval
    from collections import defaultdict
    by_type = defaultdict(list)
    for dp in training_data:
        by_type[dp.datapoint_type].append(dp)

    final_training = []
    for dtype, dps in by_type.items():
        if len(dps) > 100:
            eval_datasets[dtype] = dps[-100:]
            final_training.extend(dps[:-100])
        else:
            final_training.extend(dps)

    print(f"\nTraining: {len(final_training)}, Eval: {sum(len(v) for v in eval_datasets.values())}")
    for name, items in eval_datasets.items():
        print(f"  eval/{name}: {len(items)} items")

    # Download AO checkpoint from HuggingFace to local path
    # (train_model() expects a local path for load_lora_path)
    ao_checkpoints = {
        "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
        "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    }

    lora_local_path = None
    hf_repo = ao_checkpoints.get(args.model)
    if hf_repo:
        from huggingface_hub import snapshot_download
        lora_local_path = snapshot_download(hf_repo)
        print(f"AO checkpoint downloaded to: {lora_local_path}")

    cfg = SelfInterpTrainingConfig(
        model_name=args.model,
        hook_onto_layer=1,
        layer_percents=layer_percents,
        steering_coefficient=1.0,
        lr=args.lr,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run or f"cot_oracle_v3_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_local_path,
        eval_on_start=True,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    model_kwargs = {}

    # Initialize distributed (required by AO's train_model)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Install unfaithfulness eval hook (runs our evals alongside AO's built-in evals)
    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
        )

    # Install per-task loss logging by monkey-patching train_features_batch
    install_per_task_loss_hook(final_training, cfg.train_batch_size)

    # Shuffle training data! Without this, tasks are seen sequentially
    # which causes catastrophic forgetting of earlier tasks.
    import random
    random.seed(42)
    random.shuffle(final_training)
    print(f"Shuffled {len(final_training)} training examples")

    # Train!
    print(f"\nStarting training with config:")
    print(f"  Model: {cfg.model_name}")
    print(f"  AO checkpoint: {cfg.load_lora_path}")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Save dir: {cfg.save_dir}")
    print(f"  Training tasks: context_pred, importance, unverbalized, decorative")
    print(f"  Held-out eval tasks: taxonomy, answer_tracking (zero-shot)")

    train_model(
        cfg=cfg,
        training_data=final_training,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()
