"""
Train CoT Oracle — context prediction only.

Just PastLens on CoT rollouts. Continue from existing AO checkpoint
so conversational ability (LatentQA) is preserved. We're just teaching
the oracle what "reasoning activations" look like.

This is the simplest possible approach: if unfaithfulness leaves a
signature in activations, a well-trained context predictor should
surface it when asked conversational questions (via LatentQA from
the base checkpoint).

Usage:
    # 1. Generate CoTs (on GPU or via OpenRouter)
    python src/data_pipeline/generate_cots.py --output data/cot_corpus/corpus.jsonl

    # 2. Train (requires torchrun even on single GPU)
    torchrun --nproc_per_node=1 src/train_context_pred_only.py \
        --corpus data/cot_corpus/corpus.jsonl \
        --model Qwen/Qwen3-8B \
        --num-examples 100000
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_ao_candidates = [
    Path("/workspace/ao_reference"),  # vast.ai
    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),  # local
]
AO_REPO = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
sys.path.insert(0, str(AO_REPO))

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from nl_probes.utils.dataset_utils import create_training_datapoint, TrainingDataPoint
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

from dataset_classes.cot_context_prediction import load_cot_context_prediction_data
from dataset_classes.cot_taxonomy import load_cot_taxonomy_data
from dataset_classes.cot_answer_tracking import load_cot_answer_tracking_data
from dataset_classes.cot_decorative import load_cot_decorative_data

AO_CHECKPOINTS = {
    "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
    "Qwen/Qwen3-4B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    "Qwen/Qwen3-14B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B",
    "Qwen/Qwen3-32B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-32B",
}


def dicts_to_training_data(raw_data, tokenizer):
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
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=item["context_positions"],
            )
            training_data.append(dp)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip: {e}")
    if skipped:
        print(f"  Skipped {skipped}/{len(raw_data)} datapoints")
    return training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--num-examples", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_ctx_pred")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]

    # Generate context prediction data from CoT corpus
    print(f"Loading CoT context prediction data ({args.num_examples} examples)...")
    raw = load_cot_context_prediction_data(
        args.corpus, tokenizer, args.model, layer_percents,
        num_examples=args.num_examples,
    )
    all_data = dicts_to_training_data(raw, tokenizer)
    print(f"Generated {len(all_data)} training examples")

    # Split eval (last 100)
    random.seed(42)
    random.shuffle(all_data)
    eval_data = all_data[-100:]
    train_data = all_data[:-100]
    eval_datasets = {"cot_context_pred": eval_data}
    print(f"Train: {len(train_data)}, Eval ctx_pred: {len(eval_data)}")

    # Load held-out eval datasets (zero-shot — not in training)
    corpus_dir = Path(args.corpus).parent

    taxonomy_labels = corpus_dir / "labels_taxonomy.jsonl"
    if taxonomy_labels.exists():
        print("Loading taxonomy eval data (zero-shot)...")
        raw_tax = load_cot_taxonomy_data(
            args.corpus, str(taxonomy_labels), tokenizer, args.model,
            layer_percents, num_examples=200,
        )
        eval_datasets["taxonomy"] = dicts_to_training_data(raw_tax, tokenizer)
        print(f"  taxonomy eval: {len(eval_datasets['taxonomy'])} items")

    answer_tracking_labels = corpus_dir / "labels_answer_tracking.jsonl"
    if answer_tracking_labels.exists():
        print("Loading answer tracking eval data (zero-shot)...")
        raw_at = load_cot_answer_tracking_data(
            args.corpus, str(answer_tracking_labels), tokenizer, args.model,
            layer_percents, num_examples=200,
        )
        eval_datasets["answer_tracking"] = dicts_to_training_data(raw_at, tokenizer)
        print(f"  answer_tracking eval: {len(eval_datasets['answer_tracking'])} items")

    # Decorative CoT eval (self-supervised, uses corpus directly)
    try:
        print("Loading decorative CoT eval data (zero-shot)...")
        raw_dec = load_cot_decorative_data(
            args.corpus, tokenizer, args.model,
            layer_percents, num_examples=100,
        )
        eval_datasets["decorative"] = dicts_to_training_data(raw_dec, tokenizer)
        print(f"  decorative eval: {len(eval_datasets['decorative'])} items")
    except Exception as e:
        print(f"  decorative eval skipped: {e}")

    print(f"Eval datasets: {list(eval_datasets.keys())}")

    # Download AO checkpoint
    hf_repo = AO_CHECKPOINTS.get(args.model)
    lora_path = None
    if hf_repo:
        lora_path = snapshot_download(hf_repo)
        print(f"AO checkpoint: {lora_path}")
    else:
        print(f"WARNING: No AO checkpoint for {args.model}, training from scratch")

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
        wandb_run_name=args.wandb_run or f"cot_ctx_pred_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_path,
        eval_on_start=True,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"\nConfig:")
    print(f"  Model: {args.model}")
    print(f"  Examples: {len(train_data)}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"  Checkpoint: {hf_repo}")

    train_model(
        cfg=cfg,
        training_data=train_data,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )


if __name__ == "__main__":
    main()
