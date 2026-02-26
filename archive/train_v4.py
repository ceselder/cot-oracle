"""
Train CoT Oracle v4: Pure Rollout Reconstruction with Multi-Layer Injection

Single-task experiment: 20K rollout reconstruction examples only.
Fresh LoRA, no checkpoint. Multi-layer injection (layers at 25%, 50%, 75%
depth, concatenated).

Key changes from v3:
  - Single task (rollout reconstruction only)
  - Multi-layer injection: always inject all 3 layers concatenated
  - Custom materialization for multi-layer extraction
  - Wandb table logging for eval predictions
  - Save checkpoints every 500 steps
  - Standalone training loop (no DDP/torchrun required)

Usage:
    python src/train_v4.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import gc
import logging
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress verbose warnings that spam the log during generation
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path

ensure_ao_repo_on_path()

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup

import nl_probes.utils.dataset_utils as du_module
import nl_probes.utils.eval as eval_module
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    construct_batch,
    create_training_datapoint,
    get_prompt_tokens_only,
)
from nl_probes.utils.eval import run_evaluation, eval_features_batch
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_tokenizer, set_seed

from dataset_classes.cot_rollout_multilayer import load_cot_rollout_multilayer
from cot_utils import layer_percent_to_layer

# ── Override placeholder token ──
PLACEHOLDER_TOKEN = " \u00b6"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

# ── Multi-layer config (set in main before data loading) ──
MULTI_LAYERS: list[int] = []


def _patched_get_prefix(sae_layer: int, num_positions: int) -> str:
    """Prefix showing all 3 injection layers instead of one."""
    if MULTI_LAYERS:
        layers_str = ", ".join(str(l) for l in MULTI_LAYERS)
        prefix = f"Layer: {layers_str}\n"
    else:
        prefix = f"Layer: {sae_layer}\n"
    prefix += PLACEHOLDER_TOKEN * num_positions
    prefix += " \n"
    return prefix


du_module.get_introspection_prefix = _patched_get_prefix


# ── Multi-layer materialization ──
def materialize_multilayer_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer,
    model,
) -> list[TrainingDataPoint]:
    """Materialize steering vectors from 3 layers (25%, 50%, 75% depth).

    Each datapoint has context_positions = positions * 3 (same positions repeated
    for each layer). We extract K positions from each layer and concatenate to
    get [3K, D] steering vectors.
    """
    N_LAYERS = 3

    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    # Compute layers from model config
    num_model_layers = model.config.num_hidden_layers
    layers = [
        int(num_model_layers * 0.25),
        int(num_model_layers * 0.50),
        int(num_model_layers * 0.75),
    ]

    # Validate context fields
    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "context_* must be provided when steering_vectors is None"
            )

    # Build left-padded batch
    pad_id = tokenizer.pad_token_id
    contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item: list[list[int]] = [
        list(dp.context_positions) for _, dp in to_fill
    ]
    max_len = max(len(c) for c in contexts)

    device = next(model.parameters()).device
    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(
            torch.tensor(
                [pad_id] * pad_len + c, dtype=torch.long, device=device
            )
        )
        attn_masks_tensors.append(
            torch.tensor(
                [False] * pad_len + [True] * len(c),
                dtype=torch.bool,
                device=device,
            )
        )
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    # Get submodules for all 3 layers
    submodules = {
        layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers
    }

    # Forward pass with adapters disabled (extract base model activations)
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    # Extract K positions from each layer, concatenate to [3K, D]
    new_batch: list[TrainingDataPoint] = list(batch_points)
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        total_positions = len(positions_per_item[b])
        K = total_positions // N_LAYERS

        vectors_parts = []
        for li, layer in enumerate(layers):
            acts_BLD = acts_by_layer[layer]
            chunk_positions = positions_per_item[b][li * K : (li + 1) * K]
            adjusted = [p + left_offsets[b] for p in chunk_positions]

            L = acts_BLD.shape[1]
            if any(i < 0 or i >= L for i in adjusted):
                raise IndexError(
                    f"Activation index out of range for item {b}: "
                    f"{adjusted} with L={L}"
                )

            vectors_parts.append(acts_BLD[b, adjusted, :])

        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()
        assert vectors.shape[0] == total_positions, (
            f"Expected {total_positions} vectors, got {vectors.shape[0]}"
        )

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new

    return new_batch


# Patch materialization in eval module (used by run_evaluation)
du_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors
eval_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors


# ── Data conversion ──
def dicts_to_training_data(
    raw_data: list[dict],
    tokenizer,
) -> list[TrainingDataPoint]:
    """Convert dataset loader output to AO TrainingDataPoint objects."""
    training_data = []
    skipped = 0

    for item in raw_data:
        try:
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=item["layer"],  # Sentinel (layer 9), ignored by materialization
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
            if skipped <= 5:
                print(f"  Warning: skipped datapoint ({e})")

    if skipped > 0:
        print(f"  Skipped {skipped} datapoints during conversion")

    return training_data


# ── Training Loop ──
def train_features_batch(
    training_batch,
    model,
    submodule,
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Forward pass with steering hook. Returns loss."""
    hook_fn = get_hf_activation_steering_hook(
        vectors=training_batch.steering_vectors,
        positions=training_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }
    with add_hook(submodule, hook_fn):
        outputs = model(**tokenized_input, labels=training_batch.labels)
    return outputs


def oom_preflight_check(
    training_data: list[TrainingDataPoint],
    model,
    submodule,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    train_batch_size: int,
    steering_coefficient: float,
):
    """Run a few dummy steps on the longest prompt to catch OOM early."""
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * train_batch_size
    long_prompts = materialize_multilayer_steering_vectors(long_prompts, tokenizer, model)
    largest_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        outputs = train_features_batch(
            largest_batch, model, submodule, steering_coefficient, device, dtype
        )
        outputs.loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()
    print("OOM preflight check complete")


def run_eval(
    eval_datasets: dict[str, list[TrainingDataPoint]],
    model,
    tokenizer,
    submodule,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    eval_batch_size: int,
    steering_coefficient: float,
):
    """Run fuzzy eval with token F1 scoring + wandb table logging."""
    import wandb

    # Free training memory before eval (generation needs KV cache headroom)
    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    eval_results = {}
    gen_kwargs = {"do_sample": False, "max_new_tokens": 100}

    for ds in eval_datasets:
        eval_responses = run_evaluation(
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
            lora_path=None,
            eval_batch_size=eval_batch_size,
            steering_coefficient=steering_coefficient,
            generation_kwargs=gen_kwargs,
        )

        scores = []
        table = wandb.Table(columns=["id", "prediction", "target", "score"])

        for i, (resp, dp) in enumerate(zip(eval_responses, eval_datasets[ds])):
            pred = resp.api_response.strip()
            target = dp.target_output.strip()
            score = _token_f1(pred, target)
            scores.append(score)
            table.add_data(i, pred[:500], target[:500], round(score, 3))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        eval_results[f"eval/{ds}"] = avg_score
        print(f"  Step {global_step} | {ds}: token_f1={avg_score:.3f}")

        # Show one sample
        if eval_responses:
            print(f"    pred='{eval_responses[0].api_response.strip()[:120]}'")
            print(f"    targ='{eval_datasets[ds][0].target_output.strip()[:120]}'")

        wandb.log({f"eval_table/{ds}": table}, step=global_step)

    wandb.log(eval_results, step=global_step)
    wandb.summary.update(eval_results)
    model.train()
    torch.cuda.empty_cache()


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Main ──
def main():
    parser = argparse.ArgumentParser(
        description="Train CoT Oracle v4: Pure Rollout Recon + Multi-Layer"
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-examples", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/v4")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="v4_rollout_recon_multilayer")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-positions-per-layer", type=int, default=20)
    parser.add_argument("--max-target-tokens", type=int, default=200)
    parser.add_argument("--data-cache", default=None, help="Path to save/load preprocessed data (avoids slow tokenization)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Set multi-layer config (must happen before data loading)
    global MULTI_LAYERS
    MULTI_LAYERS = [layer_percent_to_layer(args.model, p) for p in [25, 50, 75]]
    print(f"Multi-layer injection: {MULTI_LAYERS}")

    tokenizer = load_tokenizer(args.model)

    # Verify placeholder token
    tok_ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(tok_ids) == 1, (
        f"Placeholder '{PLACEHOLDER_TOKEN}' is {len(tok_ids)} tokens, need 1"
    )
    print(f"Placeholder token: '{PLACEHOLDER_TOKEN}' -> token ID {tok_ids[0]}")

    # Load data (with optional caching to avoid slow tokenization on large runs)
    data_cache_path = Path(args.data_cache) if args.data_cache else None

    if data_cache_path and data_cache_path.exists():
        print(f"\nLoading cached data from {data_cache_path}...")
        import json
        with open(data_cache_path) as f:
            raw = json.load(f)
        print(f"  Loaded {len(raw)} cached examples")
    else:
        print(f"\nLoading {args.n_examples} rollout reconstruction examples...")
        raw = load_cot_rollout_multilayer(
            args.corpus,
            tokenizer,
            args.model,
            num_examples=args.n_examples,
            stride=args.stride,
            max_positions_per_layer=args.max_positions_per_layer,
            max_target_tokens=args.max_target_tokens,
        )
        if data_cache_path:
            data_cache_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(data_cache_path, "w") as f:
                json.dump(raw, f)
            print(f"  Saved {len(raw)} examples to {data_cache_path}")

    training_data = dicts_to_training_data(raw, tokenizer)
    print(f"Converted {len(training_data)} training examples")

    if not training_data:
        print("ERROR: No training data generated!")
        return

    # Print one example for verification
    dp = training_data[0]
    print(f"\n--- Example 0 ---")
    print(f"  datapoint_type: {dp.datapoint_type}")
    print(f"  num input_ids: {len(dp.input_ids)}")
    print(f"  num positions (placeholder tokens): {len(dp.positions)}")
    print(f"  num context_positions: {len(dp.context_positions)}")
    print(f"  layer (sentinel): {dp.layer}")
    print(f"  target_output: {dp.target_output[:100]}...")
    prefix_decode = tokenizer.decode(dp.input_ids[:50])
    print(f"  prefix check: {prefix_decode[:200]}")

    # Split: 100 for eval, rest for training
    random.seed(args.seed)
    random.shuffle(training_data)
    eval_datasets = {"cot_rollout_multilayer": training_data[:100]}
    final_training = training_data[100:]

    print(f"\nTraining: {len(final_training)}, Eval: {len(eval_datasets['cot_rollout_multilayer'])}")

    # ── Load model ──
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    # Get hook submodule BEFORE wrapping with LoRA
    submodule = get_hf_submodule(model, 1)  # hook_onto_layer=1

    # Fresh LoRA
    lora_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    model.print_trainable_parameters()

    # ── Preflight ──
    model.train()
    oom_preflight_check(
        final_training, model, submodule, tokenizer,
        device, dtype, args.batch_size, args.steering_coefficient,
    )

    # ── Optimizer + scheduler ──
    random.shuffle(final_training)
    num_batches = len(final_training) // args.batch_size
    total_steps = num_batches * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Wandb ──
    import wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "model": args.model,
            "n_examples": args.n_examples,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "layers": MULTI_LAYERS,
            "stride": args.stride,
            "max_positions_per_layer": args.max_positions_per_layer,
            "max_target_tokens": args.max_target_tokens,
            "gradient_checkpointing": args.gradient_checkpointing,
            "steering_coefficient": args.steering_coefficient,
            "total_steps": total_steps,
        },
    )
    tokens_per_epoch = sum(len(dp.input_ids) for dp in final_training)
    wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch
    wandb.summary["train/total_tokens_est"] = tokens_per_epoch * args.epochs
    wandb.summary["train/num_examples"] = len(final_training)

    # ── Save dir ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training:")
    print(f"  Model: {args.model}")
    print(f"  LoRA: FRESH (from scratch)")
    print(f"  Layers: {MULTI_LAYERS}")
    print(f"  Placeholder: '{PLACEHOLDER_TOKEN}'")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Save dir: {save_dir}")

    # ── Training loop ──
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        random.shuffle(final_training)
        optimizer.zero_grad()

        for start in tqdm(
            range(0, len(final_training), args.batch_size),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break  # Skip incomplete final batch

            # Track task types for per-task loss logging
            batch_types = [dp.datapoint_type for dp in batch_list]

            # Materialize steering vectors
            batch_list = materialize_multilayer_steering_vectors(batch_list, tokenizer, model)

            # Construct padded batch
            batch = construct_batch(batch_list, tokenizer, device)

            # Forward pass with steering
            outputs = train_features_batch(
                batch, model, submodule, args.steering_coefficient, device, dtype,
            )
            loss = outputs.loss
            loss.backward()

            # Per-task loss logging
            with torch.no_grad():
                logits = outputs.logits.detach()
                labels = batch.labels
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.shape)
                mask = (shift_labels != -100).float()
                per_item_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            task_losses = defaultdict(list)
            for i, task_type in enumerate(batch_types):
                task_losses[task_type].append(per_item_loss[i].item())

            # Optimizer step
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log to wandb
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }
            for task, losses in task_losses.items():
                log_dict[f"train/loss_{task}"] = sum(losses) / len(losses)
            wandb.log(log_dict, step=global_step)

            # ── Eval ──
            if global_step > 0 and global_step % args.eval_steps == 0:
                print(f"\n--- Eval at step {global_step} ---")
                try:
                    run_eval(
                        eval_datasets, model, tokenizer, submodule,
                        device, dtype, global_step, args.eval_batch_size,
                        args.steering_coefficient,
                    )
                except Exception as e:
                    print(f"  Eval FAILED at step {global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                model.train()

            # ── Save checkpoint ──
            if global_step > 0 and global_step % args.save_steps == 0:
                ckpt_path = save_dir / f"step_{global_step}"
                print(f"  Saving checkpoint to {ckpt_path}")
                model.save_pretrained(str(ckpt_path))

            global_step += 1

    # ── Final eval + save ──
    print(f"\n--- Final eval at step {global_step} ---")
    run_eval(
        eval_datasets, model, tokenizer, submodule,
        device, dtype, global_step, args.eval_batch_size,
        args.steering_coefficient,
    )

    final_path = save_dir / "final"
    print(f"Saving final model to {final_path}")
    model.save_pretrained(str(final_path))

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
