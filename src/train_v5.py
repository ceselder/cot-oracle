"""
Train CoT Oracle v6: Flat Task-Based Training

All tasks mixed together in one training run. Enable/disable tasks via --*-n flags (0 = skip).
Continues from Adam's pretrained AO checkpoint (or fresh LoRA / custom checkpoint).
All tasks use stride=5, 3 layers (25%, 50%, 75%), paragraph token.

Usage:
    # Train everything (defaults):
    python src/train_v5.py --corpus data/cot_corpus_v5/corpus_medium.jsonl

    # Train specific tasks only:
    python src/train_v5.py --corpus ... --full-recon-n 40000 --correctness-n 15000 --conv-qa-n 0

    # Resume from checkpoint:
    python src/train_v5.py --corpus ... --resume-from checkpoints/v6/step_5000 --start-step 5000
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
)
from nl_probes.utils.eval import run_evaluation
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_tokenizer, set_seed

from cot_utils import layer_percent_to_layer

# ── Override placeholder token ──
PLACEHOLDER_TOKEN = " \u00b6"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

# ── Multi-layer config ──
MULTI_LAYERS: list[int] = []


def _patched_get_prefix(sae_layer: int, num_positions: int) -> str:
    if MULTI_LAYERS:
        layers_str = ", ".join(str(l) for l in MULTI_LAYERS)
        prefix = f"Layer: {layers_str}\n"
    else:
        prefix = f"Layer: {sae_layer}\n"
    prefix += PLACEHOLDER_TOKEN * num_positions
    prefix += " \n"
    return prefix


du_module.get_introspection_prefix = _patched_get_prefix


# ── Position encoding config (module-level, set by main()) ──
_PE_CONFIG = {"enabled": False, "alpha": 0.1}


# ── Multi-layer materialization ──
def materialize_multilayer_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer,
    model,
) -> list[TrainingDataPoint]:
    """Materialize steering vectors from 3 layers (25%, 50%, 75% depth)."""
    N_LAYERS = 3

    to_fill = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    num_model_layers = model.config.num_hidden_layers
    layers = [
        int(num_model_layers * 0.25),
        int(num_model_layers * 0.50),
        int(num_model_layers * 0.75),
    ]

    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError("context_* must be provided when steering_vectors is None")

    pad_id = tokenizer.pad_token_id
    contexts = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    device = next(model.parameters()).device
    input_ids_tensors = []
    attn_masks_tensors = []
    left_offsets = []

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(
            torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device)
        )
        attn_masks_tensors.append(
            torch.tensor(
                [False] * pad_len + [True] * len(c), dtype=torch.bool, device=device
            )
        )
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    submodules = {
        layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers
    }

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

    new_batch = list(batch_points)
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
                    f"Activation index out of range for item {b}: {adjusted} with L={L}"
                )
            vectors_parts.append(acts_BLD[b, adjusted, :])

        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()
        assert vectors.shape[0] == total_positions

        # Apply positional encoding if enabled
        if _PE_CONFIG["enabled"]:
            from position_encoding import apply_position_encoding
            total_length = len(contexts[b])
            vectors = apply_position_encoding(
                vectors, positions_per_item[b], total_length,
                alpha=_PE_CONFIG["alpha"],
            )

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new

    return new_batch


du_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors
eval_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors


# ── Data conversion ──
def dicts_to_training_data(
    raw_data: list[dict], tokenizer,
) -> list[TrainingDataPoint]:
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
            if skipped <= 5:
                print(f"  Warning: skipped datapoint ({e})")

    if skipped > 0:
        print(f"  Skipped {skipped} datapoints during conversion")

    return training_data


# ── Task registry ──
# Each task: (arg_name, loader_import_path, loader_function_name, extra_kwargs_fn)
# extra_kwargs_fn returns additional kwargs beyond (corpus, tokenizer, model_name, num_examples)

TASK_REGISTRY = {
    "full_recon": {
        "arg": "full_recon_n",
        "module": "dataset_classes.cot_rollout_multilayer",
        "loader": "load_cot_rollout_multilayer",
        "corpus": "main",
    },
    "next_step": {
        "arg": "next_step_n",
        "module": "dataset_classes.cot_next_step",
        "loader": "load_cot_next_step_data",
        "corpus": "main",
    },
    "answer_pred": {
        "arg": "answer_pred_n",
        "module": "dataset_classes.cot_answer_prediction",
        "loader": "load_cot_answer_prediction_data",
        "corpus": "main",
    },
    "load_bearing": {
        "arg": "load_bearing_n",
        "module": "dataset_classes.cot_load_bearing",
        "loader": "load_cot_load_bearing_data",
        "corpus": "main",
    },
    "correctness": {
        "arg": "correctness_n",
        "module": "dataset_classes.cot_correctness",
        "loader": "load_cot_correctness_data",
        "corpus": "main",
    },
    "decorative": {
        "arg": "decorative_n",
        "module": "dataset_classes.cot_decorative",
        "loader": "load_cot_decorative_data",
        "corpus": "main",
    },
    "domain": {
        "arg": "domain_n",
        "module": "dataset_classes.cot_domain",
        "loader": "load_cot_domain_data",
        "corpus": "main",
    },
    "reasoning_term": {
        "arg": "reasoning_term_n",
        "module": "dataset_classes.cot_reasoning_termination",
        "loader": "load_cot_reasoning_termination_data",
        "corpus": "main",
    },
    "partial_answer": {
        "arg": "partial_answer_n",
        "module": "dataset_classes.cot_partial_answer",
        "loader": "load_cot_partial_answer_data",
        "corpus": "main",
    },
    "conv_qa": {
        "arg": "conv_qa_n",
        "module": "dataset_classes.cot_conversational",
        "loader": "load_cot_conversational_data",
        "corpus": "concept",  # uses concept corpus + cotqa file
    },
}


def load_precomputed_tasks(precomputed_dir: str, args) -> list[dict]:
    """Load training data from precomputed JSONL files."""
    pdir = Path(precomputed_dir)
    all_data = []
    enabled = []

    for task_name, info in TASK_REGISTRY.items():
        n = getattr(args, info["arg"], 0)
        if n <= 0:
            continue

        jsonl_path = pdir / f"{task_name}.jsonl"
        if not jsonl_path.exists():
            print(f"  WARNING: {jsonl_path} not found, skipping {task_name}")
            continue

        print(f"  Loading {task_name} from {jsonl_path}...")
        data = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    if len(data) >= n:
                        break

        all_data.extend(data)
        enabled.append(f"{task_name}({len(data)})")
        print(f"    -> {len(data)} examples")

    print(f"\n  Enabled tasks: {', '.join(enabled)}")
    print(f"  Total: {len(all_data)} examples")
    return all_data


def load_all_tasks(args, tokenizer) -> list[dict]:
    """Load all enabled tasks via dataset loaders (fallback if no precomputed dir)."""
    import importlib

    all_data = []
    enabled = []

    for task_name, info in TASK_REGISTRY.items():
        n = getattr(args, info["arg"], 0)
        if n <= 0:
            continue

        enabled.append(f"{task_name}({n})")
        print(f"\n  Loading {task_name} ({n} examples)...")

        try:
            mod = importlib.import_module(info["module"])
            loader_fn = getattr(mod, info["loader"])

            if info["corpus"] == "concept":
                concept_corpus = args.concept_corpus
                if not Path(concept_corpus).exists():
                    print(f"    Warning: concept corpus not found, using main corpus")
                    concept_corpus = args.corpus
                cotqa_path = Path(args.cotqa_path)
                if not cotqa_path.exists():
                    print(f"    Warning: {cotqa_path} not found, skipping {task_name}")
                    continue
                data = loader_fn(
                    concept_corpus, str(cotqa_path), tokenizer, args.model,
                    num_examples=n,
                    stride=args.stride,
                    max_positions_per_layer=args.max_positions_per_layer,
                )
            else:
                data = loader_fn(
                    args.corpus, tokenizer, args.model,
                    num_examples=n,
                    stride=args.stride,
                    max_positions_per_layer=args.max_positions_per_layer,
                )

            all_data.extend(data)
            print(f"    -> {len(data)} examples loaded")

        except Exception as e:
            print(f"    FAILED to load {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Enabled tasks: {', '.join(enabled)}")
    print(f"  Total: {len(all_data)} examples")
    return all_data


# ── Training infrastructure ──
def train_features_batch(training_batch, model, submodule, steering_coefficient, device, dtype):
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


def run_eval(
    eval_datasets, model, tokenizer, submodule, device, dtype,
    global_step, eval_batch_size, steering_coefficient,
):
    """Run fuzzy eval with token F1 scoring + wandb table logging."""
    import wandb

    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    gen_kwargs = {"do_sample": False, "max_new_tokens": 100}

    for ds in eval_datasets:
        try:
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
            exact_matches = 0
            table = wandb.Table(columns=[
                "id", "type", "oracle_prompt", "prediction", "target",
                "token_f1", "exact_match", "pred_tokens", "target_tokens",
            ])

            for i, (resp, dp) in enumerate(zip(eval_responses, eval_datasets[ds])):
                pred = resp.api_response.strip()
                target = dp.target_output.strip()
                score = _token_f1(pred, target)
                exact = 1 if pred.lower().strip() == target.lower().strip() else 0
                scores.append(score)
                exact_matches += exact
                table.add_data(
                    i, dp.datapoint_type, dp.prompt[:300],
                    pred[:500], target[:500],
                    round(score, 3), exact,
                    len(pred.split()), len(target.split()),
                )

            avg_score = sum(scores) / len(scores) if scores else 0.0
            exact_rate = exact_matches / len(scores) if scores else 0.0
            wandb.log({
                f"eval/{ds}": avg_score,
                f"eval/{ds}_exact": exact_rate,
            }, step=global_step)
            print(f"  Step {global_step} | {ds}: token_f1={avg_score:.3f} exact={exact_rate:.3f}")

            if eval_responses:
                print(f"    pred='{eval_responses[0].api_response.strip()[:120]}'")
                print(f"    targ='{eval_datasets[ds][0].target_output.strip()[:120]}'")

            wandb.log({f"eval_table/{ds}": table}, step=global_step)
        except Exception as e:
            print(f"  Eval FAILED for {ds}: {e}")

    model.train()
    torch.cuda.empty_cache()


def run_unfaith_evals(model, tokenizer, model_name, global_step, args):
    """Run unfaithfulness evals if available."""
    try:
        from evals.training_eval_hook import run_training_evals
        import wandb

        print(f"\n--- Unfaithfulness evals at step {global_step} ---")
        metrics = run_training_evals(
            model, tokenizer, model_name=model_name,
            step=global_step, device="cuda",
            eval_dir=args.eval_dir,
            max_items_per_eval=args.unfaith_eval_items,
            skip_rot13=(global_step < args.rot13_start_step),
            oracle_adapter_name="default",
            activation_cache_dir=args.activation_cache_dir,
        )
        if metrics:
            wandb.log(metrics, step=global_step)
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)) and "sample" not in k:
                    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        return metrics
    except Exception as e:
        print(f"  Unfaithfulness eval FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ── Main training loop ──
def train(
    raw_data: list[dict],
    model,
    tokenizer,
    submodule,
    args,
    global_step: int,
    save_dir: Path,
) -> int:
    """Train on all tasks. Returns the final global_step."""
    import wandb

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Convert to TrainingDataPoints
    training_data = dicts_to_training_data(raw_data, tokenizer)
    print(f"  Converted {len(training_data)} training examples")

    if not training_data:
        print("  ERROR: No training data!")
        return global_step

    # Type distribution
    type_counts = defaultdict(int)
    for dp in training_data:
        type_counts[dp.datapoint_type] += 1
    print(f"\n  Task distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Split eval (50 per type)
    random.shuffle(training_data)
    eval_per_type = {}
    train_per_type = defaultdict(list)

    for dp in training_data:
        t = dp.datapoint_type
        if t not in eval_per_type:
            eval_per_type[t] = []
        if len(eval_per_type[t]) < 50:
            eval_per_type[t].append(dp)
        else:
            train_per_type[t].append(dp)

    # Flatten train
    final_training = []
    for items in train_per_type.values():
        final_training.extend(items)

    task_order = getattr(args, "task_order", "shuffled")

    if task_order == "sequential":
        # Group by task, preserve order from config
        task_blocks = []
        for task_type in sorted(train_per_type.keys()):
            items = train_per_type[task_type]
            random.shuffle(items)
            task_blocks.append((task_type, items))
        # Flatten in task order (task A then task B then ...)
        final_training = []
        for _, items in task_blocks:
            final_training.extend(items)
        print(f"  Task order: SEQUENTIAL")
        for task_name, items in task_blocks:
            print(f"    {task_name}: {len(items)} examples")
    else:
        random.shuffle(final_training)
        print(f"  Task order: SHUFFLED")

    eval_datasets = eval_per_type
    print(f"  Training: {len(final_training)}, Eval: {sum(len(v) for v in eval_datasets.values())}")
    print(f"  Eval tasks: {', '.join(sorted(eval_datasets.keys()))}")

    # Optimizer + scheduler
    num_batches = len(final_training) // args.batch_size
    total_steps = num_batches * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Task index mapping for wandb (so we can plot which task is active)
    all_task_types = sorted(type_counts.keys())
    task_to_idx = {t: i for i, t in enumerate(all_task_types)}

    wandb.log({
        "total_steps": total_steps,
        "n_examples": len(final_training),
        "n_eval": sum(len(v) for v in eval_datasets.values()),
        "n_tasks": len(eval_datasets),
    }, step=global_step)

    # Log task index legend to wandb config
    wandb.config.update({"task_index_legend": task_to_idx}, allow_val_change=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  LR: {args.lr}")
    print(f"  Batch: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps: {total_steps}")
    print(f"  Warmup: {warmup_steps}")

    model.train()

    # EMA for per-task loss
    task_loss_ema = {}
    ema_alpha = 0.1
    total_tokens = 0
    train_start_time = time.time()
    last_step_time = time.time()

    prev_dominant_task = None  # Track task transitions for phase checkpoints

    for epoch in range(args.epochs):
        if task_order != "sequential":
            random.shuffle(final_training)
        optimizer.zero_grad()

        pbar = tqdm(
            range(0, len(final_training), args.batch_size),
            desc=f"E{epoch + 1}/{args.epochs}",
        )

        for start in pbar:
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break

            batch_types = [dp.datapoint_type for dp in batch_list]

            # Materialize
            batch_list = materialize_multilayer_steering_vectors(
                batch_list, tokenizer, model
            )

            batch = construct_batch(batch_list, tokenizer, device)

            outputs = train_features_batch(
                batch, model, submodule,
                args.steering_coefficient, device, dtype,
            )
            loss = outputs.loss
            loss.backward()

            # Per-task loss
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
                batch_tokens = int(mask.sum().item())
                total_tokens += batch_tokens

            task_losses = defaultdict(list)
            for i, task_type in enumerate(batch_types):
                task_losses[task_type].append(per_item_loss[i].item())

            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update EMA for per-task losses
            for task, losses in task_losses.items():
                avg = sum(losses) / len(losses)
                if task not in task_loss_ema:
                    task_loss_ema[task] = avg
                else:
                    task_loss_ema[task] = ema_alpha * avg + (1 - ema_alpha) * task_loss_ema[task]

            # Logging
            now = time.time()
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/total_tokens": total_tokens,
                "train/batch_tokens": batch_tokens,
                "train/step_time": now - last_step_time,
                "train/wallclock_hours": (now - train_start_time) / 3600,
            }
            last_step_time = now
            for task, ema_val in task_loss_ema.items():
                log_dict[f"train/loss_{task}"] = ema_val

            # Track dominant task for sequential mode phase transitions
            batch_task_counts = defaultdict(int)
            for t in batch_types:
                batch_task_counts[t] += 1
            dominant_task = max(batch_task_counts, key=batch_task_counts.get)

            wandb.log(log_dict, step=global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Phase checkpoint: save when dominant task changes in sequential mode
            if task_order == "sequential" and prev_dominant_task is not None and dominant_task != prev_dominant_task:
                ckpt_path = save_dir / f"step_{global_step}_phase_{prev_dominant_task}"
                print(f"\n  Phase transition: {prev_dominant_task} -> {dominant_task}")
                print(f"  Saving phase checkpoint to {ckpt_path}")
                model.save_pretrained(str(ckpt_path))
            prev_dominant_task = dominant_task

            # Task-level eval
            if global_step > 0 and global_step % args.eval_steps == 0:
                print(f"\n--- Task eval at step {global_step} ---")
                try:
                    run_eval(
                        eval_datasets, model, tokenizer, submodule,
                        device, dtype, global_step, args.eval_batch_size,
                        args.steering_coefficient,
                    )
                except Exception as e:
                    print(f"  Task eval FAILED: {e}")
                model.train()

            # Unfaithfulness eval
            if global_step > 0 and global_step % args.unfaith_eval_steps == 0:
                run_unfaith_evals(model, tokenizer, args.model, global_step, args)
                model.train()

            # Save checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                ckpt_path = save_dir / f"step_{global_step}"
                print(f"  Saving checkpoint to {ckpt_path}")
                model.save_pretrained(str(ckpt_path))

            global_step += 1

    # Final eval
    print(f"\n--- Final eval at step {global_step} ---")
    try:
        run_eval(
            eval_datasets, model, tokenizer, submodule,
            device, dtype, global_step, args.eval_batch_size,
            args.steering_coefficient,
        )
    except Exception as e:
        print(f"  Final task eval FAILED: {e}")

    run_unfaith_evals(model, tokenizer, args.model, global_step, args)

    # Save final
    final_path = save_dir / "final"
    print(f"  Saving final checkpoint to {final_path}")
    model.save_pretrained(str(final_path))

    return global_step


# ── Config loading ──
def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_config(args, config: dict):
    """Apply config values to args, CLI flags override config."""
    # Task counts
    if "tasks" in config:
        for task_name, task_cfg in config["tasks"].items():
            arg_name = f"{task_name}_n"
            if hasattr(args, arg_name) and getattr(args, f"_cli_{arg_name}", False) is False:
                setattr(args, arg_name, task_cfg.get("n", 0))

    # Training params
    if "training" in config:
        t = config["training"]
        _float_keys = {"lr", "warmup_fraction", "max_grad_norm", "steering_coefficient"}
        _int_keys = {"batch_size", "eval_batch_size", "epochs", "seed"}
        for key in ["lr", "batch_size", "eval_batch_size", "epochs",
                     "warmup_fraction", "max_grad_norm", "steering_coefficient",
                     "gradient_checkpointing", "task_order", "seed"]:
            if key in t and not getattr(args, f"_cli_{key}", False):
                val = t[key]
                if key in _float_keys:
                    val = float(val)
                elif key in _int_keys:
                    val = int(val)
                setattr(args, key, val)

    # Activations
    if "activations" in config:
        a = config["activations"]
        for key in ["stride", "max_positions_per_layer", "position_encoding", "pe_alpha"]:
            if key in a and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, a[key])

    # Eval
    if "eval" in config:
        e = config["eval"]
        for key in ["eval_steps", "save_steps", "unfaith_eval_steps", "unfaith_eval_items"]:
            if key in e and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, e[key])

    # Data paths
    if "data" in config:
        d = config["data"]
        if "corpus" in d and not getattr(args, "_cli_corpus", False):
            args.corpus = d["corpus"]
        if "precomputed_dir" in d and not getattr(args, "_cli_precomputed_dir", False):
            args.precomputed_dir = d["precomputed_dir"]
        if "concept_corpus" in d and not getattr(args, "_cli_concept_corpus", False):
            args.concept_corpus = d["concept_corpus"]
        if "cotqa_path" in d and not getattr(args, "_cli_cotqa_path", False):
            args.cotqa_path = d["cotqa_path"]

    # Model
    if "model" in config:
        m = config["model"]
        if "name" in m and not getattr(args, "_cli_model", False):
            args.model = m["name"]
        if "ao_checkpoint" in m and not getattr(args, "_cli_ao_checkpoint", False):
            args.ao_checkpoint = m["ao_checkpoint"]
        if "fresh_lora" in m and not getattr(args, "_cli_fresh_lora", False):
            args.fresh_lora = m["fresh_lora"]

    # Output
    if "output" in config:
        o = config["output"]
        if "save_dir" in o and not getattr(args, "_cli_save_dir", False):
            args.save_dir = o["save_dir"]
        if "wandb_project" in o and not getattr(args, "_cli_wandb_project", False):
            args.wandb_project = o["wandb_project"]
        if "wandb_entity" in o and not getattr(args, "_cli_wandb_entity", False):
            args.wandb_entity = o["wandb_entity"]
        if "wandb_run" in o and o.get("wandb_run") and not getattr(args, "_cli_wandb_run", False):
            args.wandb_run = o["wandb_run"]


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl",
                        help="Path to corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")

    # Checkpoint control
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a LoRA checkpoint dir")
    parser.add_argument("--ao-checkpoint",
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
                        help="Adam's pretrained AO checkpoint to start from")
    parser.add_argument("--fresh-lora", action="store_true",
                        help="Start with fresh LoRA instead of Adam's checkpoint")

    # Data paths
    parser.add_argument("--cotqa-path",
                        default="data/concept_corpus/corpus_full_conv_qa_llm.jsonl",
                        help="Path to LLM-generated conversational QA")
    parser.add_argument("--concept-corpus",
                        default="data/concept_corpus/corpus_full.jsonl",
                        help="Path to concept corpus (for conv QA lookups)")

    # Per-task example counts (set to 0 to disable a task)
    parser.add_argument("--full-recon-n", type=int, default=40000)
    parser.add_argument("--next-step-n", type=int, default=30000)
    parser.add_argument("--answer-pred-n", type=int, default=20000)
    parser.add_argument("--load-bearing-n", type=int, default=15000)
    parser.add_argument("--correctness-n", type=int, default=15000)
    parser.add_argument("--decorative-n", type=int, default=15000)
    parser.add_argument("--domain-n", type=int, default=15000)
    parser.add_argument("--reasoning-term-n", type=int, default=15000)
    parser.add_argument("--partial-answer-n", type=int, default=20000)
    parser.add_argument("--conv-qa-n", type=int, default=10000)

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-positions-per-layer", type=int, default=20)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--position-encoding", action="store_true", default=False,
                        help="Apply sinusoidal PE to activation vectors")
    parser.add_argument("--pe-alpha", type=float, default=0.1,
                        help="PE mixing coefficient (only used if --position-encoding)")
    parser.add_argument("--task-order", choices=["shuffled", "sequential"], default="shuffled",
                        help="'shuffled' mixes all tasks; 'sequential' trains tasks one at a time")

    # Eval / save
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--unfaith-eval-steps", type=int, default=5000)
    parser.add_argument("--unfaith-eval-items", type=int, default=20)
    parser.add_argument("--rot13-start-step", type=int, default=2000)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Starting global step (for resuming)")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--activation-cache-dir", default="data/eval_precomputed",
                        help="Dir with precomputed activation bundles (.pt)")

    # Output
    parser.add_argument("--save-dir", default="checkpoints/v6")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-entity", default="MATS10-CS-JB",
                        help="Wandb entity (team/org)")
    parser.add_argument("--wandb-run", default=None)

    # Data loading
    parser.add_argument("--precomputed-dir", default=None,
                        help="Dir with precomputed JSONL files (skips dataset loaders)")
    parser.add_argument("--data-cache-dir", default=None,
                        help="Directory to cache preprocessed training data")

    args = parser.parse_args()

    # Apply config file (CLI flags override config values)
    if args.config:
        config = load_config(args.config)
        apply_config(args, config)
        print(f"Loaded config from {args.config}")

    set_seed(args.seed)

    # Multi-layer config
    global MULTI_LAYERS
    MULTI_LAYERS = [layer_percent_to_layer(args.model, p) for p in [25, 50, 75]]
    print(f"Multi-layer injection: {MULTI_LAYERS}")

    # Position encoding config
    global _PE_CONFIG
    _PE_CONFIG["enabled"] = getattr(args, "position_encoding", False)
    _PE_CONFIG["alpha"] = getattr(args, "pe_alpha", 0.1)
    if _PE_CONFIG["enabled"]:
        print(f"Position encoding: ON (alpha={_PE_CONFIG['alpha']})")
    else:
        print("Position encoding: OFF")

    tokenizer = load_tokenizer(args.model)

    # Verify placeholder
    tok_ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(tok_ids) == 1, f"Placeholder '{PLACEHOLDER_TOKEN}' is {len(tok_ids)} tokens"
    print(f"Placeholder token: '{PLACEHOLDER_TOKEN}' -> token ID {tok_ids[0]}")

    # ── Load model ──
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"\nLoading model: {args.model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    base_model.enable_input_require_grads()

    if args.gradient_checkpointing:
        base_model.use_cache = False
        base_model.gradient_checkpointing_enable()

    # Get hook submodule BEFORE LoRA
    submodule = get_hf_submodule(base_model, 1)

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = PeftModel.from_pretrained(
            base_model, args.resume_from,
            is_trainable=True,
        )
    elif args.fresh_lora:
        print("Starting with FRESH LoRA")
        lora_config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=True)
    else:
        print(f"Loading Adam's AO checkpoint: {args.ao_checkpoint}")
        model = PeftModel.from_pretrained(
            base_model, args.ao_checkpoint,
            is_trainable=True,
        )

    model.print_trainable_parameters()

    # ── Load data ──
    print(f"\n{'=' * 60}")
    print("LOADING TRAINING DATA")
    print(f"{'=' * 60}")

    if args.precomputed_dir and Path(args.precomputed_dir).exists():
        print(f"  Using precomputed data from {args.precomputed_dir}")
        raw_data = load_precomputed_tasks(args.precomputed_dir, args)
    else:
        if args.precomputed_dir:
            print(f"  WARNING: --precomputed-dir={args.precomputed_dir} not found, using loaders")
        raw_data = load_all_tasks(args, tokenizer)

    if not raw_data:
        print("ERROR: No training data loaded!")
        return

    random.shuffle(raw_data)

    # ── Wandb ──
    import wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Build a descriptive run name from enabled tasks
    enabled_tasks = []
    for task_name, info in TASK_REGISTRY.items():
        n = getattr(args, info["arg"], 0)
        if n > 0:
            enabled_tasks.append(task_name)

    run_name = args.wandb_run or f"v6-{len(raw_data)//1000}k-{len(enabled_tasks)}tasks"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        tags=["v6", args.model.split("/")[-1]] + enabled_tasks,
    )

    save_dir = Path(args.save_dir)

    # ── Train ──
    print(f"\n{'#' * 60}")
    print(f"  TRAINING: {len(raw_data)} examples, {len(enabled_tasks)} tasks")
    print(f"  Tasks: {', '.join(enabled_tasks)}")
    print(f"{'#' * 60}")

    global_step = args.start_step
    global_step = train(
        raw_data=raw_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        args=args,
        global_step=global_step,
        save_dir=save_dir,
    )

    print(f"\n{'#' * 60}")
    print(f"TRAINING COMPLETE at step {global_step}")
    print(f"{'#' * 60}")

    wandb.finish()


if __name__ == "__main__":
    main()
