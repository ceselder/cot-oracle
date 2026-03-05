"""
Train CoT Oracle: Flat Task-Based Training

All tasks mixed together in one training run. Enable/disable tasks via --*-n flags (0 = skip).
Continues from Adam's pretrained AO checkpoint (or fresh LoRA / custom checkpoint).
All tasks use 3 layers (25%, 50%, 75%), paragraph token.

Supports single-GPU and multi-GPU (via torchrun) training.

Usage:
    # Single GPU (defaults):
    python src/train.py --config configs/train.yaml --precomputed-dir data/precomputed

    # Multi-GPU:
    torchrun --nproc_per_node=8 src/train.py --config configs/train.yaml --precomputed-dir data/precomputed

    # Train specific tasks only:
    python src/train.py --config configs/train.yaml --full-recon-n 40000 --correctness-n 15000 --cotqa-n 0

    # Resume from checkpoint (step auto-detected from training_state.pt):
    python src/train.py --config configs/train.yaml --resume-from checkpoints/step_5000
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys

from contextlib import nullcontext
from dotenv import load_dotenv
load_dotenv()
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
import torch.distributed as dist
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
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import load_tokenizer, set_seed

from cot_utils import layer_percent_to_layer, sparse_sample_positions, sample_poisson_positions, sample_endweighted_positions
from tasks import TASKS, ScoringMode, get_trainable_tasks
from data_loading import load_all_training_data
from eval_loop import run_eval

# ── Override placeholder token ──
PLACEHOLDER_TOKEN = " ?"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

# ── Multi-layer config ──
MULTI_LAYERS: list[int] = []
NO_ACTIVATIONS: bool = False
RANDOM_LAYERS: bool = False
LAYER_DROPOUT: bool = False
POSITION_MODE: str = "mixed"  # "last_only", "stochastic", "mixed", "all"
STOCHASTIC_MAX_K: int = 100  # upper bound for Poisson position sampling (random bucket)
SENTENCE_DELIM_IDS: set[int] = set()  # token IDs for "." — set from tokenizer in main()
MAX_CONTEXT_LENGTH: int = 0  # drop samples with context_input_ids longer than this (0 = no filter)
POSITION_ENCODING: bool = False
PE_ALPHA: float = 0.1
_MODEL_N_LAYERS: int = 36  # total layers in the model (set in main())


def _build_labeled_layer_prefix(num_positions: int, layers: list[int], placeholder_token: str = PLACEHOLDER_TOKEN) -> str:
    if not layers:
        raise ValueError("layers must be non-empty")
    if len(layers) == 1:
        return f"L{layers[0]}:" + placeholder_token * num_positions + ".\n"
    k, rem = divmod(num_positions, len(layers))
    if rem:
        raise ValueError(f"num_positions={num_positions} not divisible by layers={layers}")
    return " ".join(f"L{layer}:" + placeholder_token * k for layer in layers) + ".\n"


def _patched_get_prefix(sae_layer: int, num_positions: int, layers: list[int] | None = None) -> str:
    prefix_layers = list(layers) if layers else [sae_layer]
    return _build_labeled_layer_prefix(num_positions, prefix_layers)


du_module.get_introspection_prefix = _patched_get_prefix


# ── Distributed helpers ──
def setup_distributed():
    """Init distributed if launched via torchrun, otherwise single-GPU."""
    if "RANK" in os.environ:
        import datetime
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Multi-layer materialization ──
def materialize_multilayer_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer,
    model,
) -> list[TrainingDataPoint]:
    """Materialize steering vectors from MULTI_LAYERS (configurable via --n-layers).

    Supports per-item random layers (RANDOM_LAYERS) and layer dropout (LAYER_DROPOUT).
    """
    to_fill = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError("context_* must be provided when steering_vectors is None")

    # Determine layers to extract: union of all items' layers
    if RANDOM_LAYERS or LAYER_DROPOUT:
        per_item_layers = []
        all_layers_set = set()
        for _, dp in to_fill:
            item_layers = dp.meta_info["layers"]
            per_item_layers.append(item_layers)
            all_layers_set.update(item_layers)
        layers = sorted(all_layers_set)
    else:
        layers = list(MULTI_LAYERS)
        per_item_layers = [layers] * len(to_fill)

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
        item_layers = per_item_layers[b]
        total_positions = len(positions_per_item[b])
        N_item = len(item_layers)
        K, rem = divmod(total_positions, N_item)
        if rem:
            raise ValueError(
                f"total_positions={total_positions} not divisible by item_layers={item_layers}"
            )

        vectors_parts = []
        for li, layer in enumerate(item_layers):
            acts_BLD = acts_by_layer[layer]
            chunk_positions = positions_per_item[b][li * K : (li + 1) * K]
            adjusted = [p + left_offsets[b] for p in chunk_positions]

            L = acts_BLD.shape[1]
            if any(i < 0 or i >= L for i in adjusted):
                raise IndexError(
                    f"Activation index out of range for item {b}: {adjusted} with L={L}"
                )
            layer_vecs = acts_BLD[b, adjusted, :]  # [K, D]
            if POSITION_ENCODING:
                from position_encoding import apply_position_encoding
                layer_vecs = apply_position_encoding(layer_vecs, chunk_positions, alpha=PE_ALPHA)
            vectors_parts.append(layer_vecs)

        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()

        dp_new = dp.model_copy(deep=False)
        dp_new.steering_vectors = vectors
        if vectors.shape[0] != len(dp_new.positions):
            raise ValueError(
                f"steering_vectors rows {vectors.shape[0]} != placeholder positions {len(dp_new.positions)}"
            )
        new_batch[idx] = dp_new

    return new_batch


du_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors
eval_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors


def _extract_base_positions(ctx_pos: list[int], n_layers_runtime: int) -> list[int]:
    """Extract single-layer base positions from multi-layer context_positions."""
    if not ctx_pos:
        return ctx_pos
    # Try runtime layer count first
    if len(ctx_pos) % n_layers_runtime == 0:
        return ctx_pos[:len(ctx_pos) // n_layers_runtime]
    # Try common old layer counts
    for old_n in [3, 1, 2, 4, 5, 6]:
        if len(ctx_pos) % old_n == 0:
            return ctx_pos[:len(ctx_pos) // old_n]
    return ctx_pos  # can't infer, return as-is


# ── Data conversion ──
def dicts_to_training_data(
    raw_data: list[dict], tokenizer,
) -> list[TrainingDataPoint]:
    training_data = []

    if NO_ACTIVATIONS:
        _act_prefix_re = re.compile(r'^Activations from \d+ positions[^.]*\.\s*')
        _printed_sample = False
        for item in raw_data:
            cot_text = item.get("cot_text", "")
            if not cot_text:
                continue
            # Strip "Activations from N positions..." prefix to get the task prompt
            task_prompt = _act_prefix_re.sub("", item["prompt"])
            # No user question — activation oracle only sees CoT activations, not the question
            prompt = f"Chain of thought: {cot_text}\n\n{task_prompt}"

            if not _printed_sample:
                print(f"  [no-activations] Sample prompt ({item['datapoint_type']}):\n    {prompt[:300]}...")
                _printed_sample = True

            msgs = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_msgs = msgs + [{"role": "assistant", "content": item["target_response"]}]
            full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            labels = full_ids.copy()
            for i in range(len(prompt_ids)):
                labels[i] = -100
            dp = TrainingDataPoint(
                input_ids=full_ids, labels=labels, layer=0,
                steering_vectors=torch.zeros(0, 1), positions=[],
                feature_idx=-1, target_output=item["target_response"],
                datapoint_type=item["datapoint_type"],
                context_input_ids=None, context_positions=None,
                ds_label=None,
                meta_info={"prompt": prompt},
            )
            training_data.append(dp)
        return training_data

    n_layers_runtime = len(MULTI_LAYERS) if MULTI_LAYERS else 3
    _reexpand_warned = False

    for item in raw_data:
        ctx_pos = item["context_positions"]
        num_pos = item["num_positions"]

        # Extract base positions (single-layer) for re-expansion
        base_positions = _extract_base_positions(ctx_pos, n_layers_runtime)
        period_pos = _get_period_positions(item.get("context_input_ids", []), base_positions)

        if RANDOM_LAYERS:
            # Per-item random layer sampling (ablation: arbitrary model layers)
            from layer_utils import sample_layers
            sampled = sample_layers(_MODEL_N_LAYERS, mean=3)

            # Apply position mode to base positions
            sampled_pos, pos_tag = _apply_position_mode(base_positions, period_pos)
            ctx_pos = sampled_pos * len(sampled)
            num_pos = len(ctx_pos)

            saved_layers = MULTI_LAYERS[:]
            MULTI_LAYERS[:] = sampled
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=sampled[0],
                num_positions=num_pos,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=ctx_pos,
                meta_info={"prompt": item["prompt"], "layers": sampled, "pos_tag": pos_tag},
            )
            MULTI_LAYERS[:] = saved_layers

        elif LAYER_DROPOUT:
            # Random non-empty subset of configured layers per item
            k = random.randint(1, len(MULTI_LAYERS))
            sampled = sorted(random.sample(MULTI_LAYERS, k))

            # Apply position mode to base positions
            sampled_pos, pos_tag = _apply_position_mode(base_positions, period_pos)
            ctx_pos = sampled_pos * len(sampled)
            num_pos = len(ctx_pos)

            saved_layers = MULTI_LAYERS[:]
            MULTI_LAYERS[:] = sampled
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=sampled[0],
                num_positions=num_pos,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=ctx_pos,
                meta_info={"prompt": item["prompt"], "layers": sampled, "pos_tag": pos_tag},
            )
            MULTI_LAYERS[:] = saved_layers

        else:
            # Standard: all configured layers, apply position mode
            sampled_pos, pos_tag = _apply_position_mode(base_positions, period_pos)
            ctx_pos = sampled_pos * n_layers_runtime
            num_pos = len(ctx_pos)

            # Re-expand positions if precomputed with fewer layers than runtime config
            if n_layers_runtime > 1 and len(ctx_pos) % n_layers_runtime != 0:
                ctx_pos = base_positions * n_layers_runtime
                num_pos = len(ctx_pos)
                if not _reexpand_warned:
                    print(f"  [data] Re-expanding positions: -> {n_layers_runtime} layers (K={len(base_positions)})")
                    _reexpand_warned = True

            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=item["layer"],
                num_positions=num_pos,
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=ctx_pos,
                meta_info={"prompt": item["prompt"], "layers": list(MULTI_LAYERS) if MULTI_LAYERS else [item["layer"]], "pos_tag": pos_tag},
            )

        training_data.append(dp)

    return training_data


def _get_period_positions(context_input_ids: list[int], base_positions: list[int]) -> list[int]:
    """Find '.' token positions within the CoT region (bounded by base_positions)."""
    if not context_input_ids or not base_positions or not SENTENCE_DELIM_IDS:
        return []
    lo, hi = base_positions[0], base_positions[-1]
    return [i for i in range(lo, min(hi + 1, len(context_input_ids)))
            if context_input_ids[i] in SENTENCE_DELIM_IDS]


def _apply_position_mode(base_positions: list[int], period_positions: list[int] | None = None) -> tuple[list[int], str]:
    """Apply POSITION_MODE to base (single-layer) positions. Returns (positions, tag)."""
    if not base_positions:
        return base_positions, "empty"
    if POSITION_MODE == "last_only":
        return base_positions[-1:], "last_k"
    elif POSITION_MODE == "graduated":
        n = random.choice([1, 2, 3])
        return base_positions[-n:], "last_k"
    elif POSITION_MODE == "stochastic":
        r = random.random()
        if r < 0.3:
            return base_positions[-1:], "last_k"
        elif r < 0.6:
            return base_positions[-3:], "last_k"
        else:
            return sample_endweighted_positions(base_positions), "end_skewed"
    elif POSITION_MODE == "mixed":
        from cot_utils import sample_poisson_positions_tagged
        return sample_poisson_positions_tagged(base_positions, max_k=STOCHASTIC_MAX_K, period_positions=period_positions)
    return base_positions, "all"


# ── Training infrastructure ──
def _example_context_len(dp: TrainingDataPoint) -> int:
    return len(dp.context_input_ids) if dp.context_input_ids is not None else len(dp.input_ids)



def _label_token_count(dp: TrainingDataPoint) -> int:
    return sum(label != -100 for label in dp.labels[1:])


def _estimate_train_batch_peak_tokens(
    batch_points: list[TrainingDataPoint],
    no_activations: bool,
) -> int:
    batch_size = len(batch_points)
    oracle_peak_tokens = 2 * batch_size * max(len(dp.input_ids) for dp in batch_points)
    if no_activations:
        return oracle_peak_tokens
    context_peak_tokens = batch_size * max(_example_context_len(dp) for dp in batch_points)
    return max(context_peak_tokens, oracle_peak_tokens)


def _split_batch_for_token_budget(
    batch_points: list[TrainingDataPoint],
    max_batch_size: int,
    max_train_tokens_per_gpu: int,
    no_activations: bool,
) -> list[list[TrainingDataPoint]]:
    if max_train_tokens_per_gpu <= 0:
        return [batch_points]

    chunks = []
    current_chunk = []
    for dp in batch_points:
        candidate = current_chunk + [dp]
        if current_chunk and (
            len(candidate) > max_batch_size
            or _estimate_train_batch_peak_tokens(candidate, no_activations) > max_train_tokens_per_gpu
        ):
            chunks.append(current_chunk)
            current_chunk = [dp]
            continue
        current_chunk = candidate
    chunks.append(current_chunk)
    return chunks


def _estimate_extract_batch_peak_tokens(batch_points: list[TrainingDataPoint]) -> int:
    batch_size = len(batch_points)
    return batch_size * max(_example_context_len(dp) for dp in batch_points)


def _split_batch_for_extract_budget(
    batch_points: list[TrainingDataPoint],
    max_batch_size: int,
    max_extract_tokens_per_gpu: int,
) -> list[list[TrainingDataPoint]]:
    if max_extract_tokens_per_gpu <= 0:
        return [batch_points]

    chunks: list[list[TrainingDataPoint]] = []
    current_chunk: list[TrainingDataPoint] = []
    for dp in batch_points:
        candidate = current_chunk + [dp]
        if current_chunk and (
            len(candidate) > max_batch_size
            or _estimate_extract_batch_peak_tokens(candidate) > max_extract_tokens_per_gpu
        ):
            chunks.append(current_chunk)
            current_chunk = [dp]
            continue
        current_chunk = candidate
    chunks.append(current_chunk)
    return chunks


def _materialize_batch_for_training_step(
    batch_points: list[TrainingDataPoint],
    tokenizer,
    model,
    max_batch_size: int,
    max_extract_tokens_per_gpu: int,
    rank: int,
    train_batch_start: int,
) -> tuple[list[TrainingDataPoint], int]:
    """Materialize steering vectors with extraction batching and OOM fallback.

    Returns:
        materialized_batch_points, split_count
    """
    if not any(dp.steering_vectors is None for dp in batch_points):
        return batch_points, 0

    chunks = _split_batch_for_extract_budget(
        batch_points,
        max_batch_size=max_batch_size,
        max_extract_tokens_per_gpu=max_extract_tokens_per_gpu,
    )
    split_count = len(chunks) - 1

    materialized: list[TrainingDataPoint] = []
    for chunk in chunks:
        pending = [chunk]
        while pending:
            cur = pending.pop(0)
            try:
                materialized.extend(
                    materialize_multilayer_steering_vectors(cur, tokenizer, model)
                )
            except torch.OutOfMemoryError:
                if len(cur) == 1:
                    raise
                torch.cuda.empty_cache()
                split_at = len(cur) // 2
                if rank == 0:
                    print(
                        f"  CUDA OOM during activation extraction at train batch {train_batch_start}, "
                        f"splitting extract micro-batch {len(cur)} -> {split_at}+{len(cur) - split_at}"
                    )
                split_count += 1
                pending = [cur[:split_at], cur[split_at:]] + pending

    return materialized, split_count


def _window_bucket_training_data(training_data: list[TrainingDataPoint], batch_size: int, window_batches: int) -> None:
    """Sort by context length inside shuffled windows to reduce padding waste.

    Within each window: sort by length, then chunk into batch-sized groups
    and shuffle those groups. This keeps similar lengths together within a batch
    (good padding efficiency) while randomizing the order of short/long batches
    (prevents monotonic step-time increase).
    """
    window = batch_size * window_batches
    for i in range(0, len(training_data), window):
        chunk = sorted(training_data[i:i + window], key=_example_context_len)
        # Chunk into batch-sized groups and shuffle the groups
        groups = [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
        random.shuffle(groups)
        flat = [item for group in groups for item in group]
        training_data[i:i + window] = flat


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


def _upload_checkpoint_to_hf(checkpoint_path: Path, args, global_step: int):
    """Upload a LoRA checkpoint to HuggingFace after training completes."""
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            print("  [HF upload] No HF_TOKEN set, skipping upload")
            return

        # Build repo name from wandb run name or save_dir
        run_name = getattr(args, "wandb_run", None)
        if not run_name:
            import wandb as _wb
            if _wb.run:
                run_name = _wb.run.name
        if not run_name:
            run_name = checkpoint_path.parent.name
        repo_name = f"ceselder/cot-oracle-{run_name}"

        print(f"  [HF upload] Uploading checkpoint to {repo_name} ...")
        api = HfApi(token=hf_token)
        api.create_repo(repo_name, exist_ok=True)
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            commit_message=f"Final checkpoint at step {global_step}",
        )
        print(f"  [HF upload] Done: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"  [HF upload] Failed: {e}")


def _log_final_checkpoint_to_wandb(checkpoint_path: Path, global_step: int):
    """Log the final checkpoint as a W&B model artifact."""
    import wandb
    artifact_name = f"final_checkpoint_{wandb.run.id}"
    print(f"  [wandb] Logging final checkpoint artifact {artifact_name}")
    artifact = wandb.Artifact(artifact_name, type="model", metadata={"step": global_step, "run_id": wandb.run.id, "run_name": wandb.run.name, "checkpoint_path": str(checkpoint_path)})
    for path in sorted(checkpoint_path.iterdir()):
        artifact.add_file(str(path), name=path.name)
    wandb.run.log_artifact(artifact)



def _save_training_state(save_path: Path, global_step, optimizer, scheduler):
    """Save optimizer/scheduler/RNG state for resume."""
    import wandb
    state = {
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "random_state": random.getstate(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
        "wandb_run_name": wandb.run.name if wandb.run else None,
    }
    torch.save(state, save_path / "training_state.pt")


def _write_eval_traces(log_dir: Path | None, all_traces: dict[str, list[dict]], global_step: int):
    """Write full-text eval traces to disk and return the created files."""
    if not log_dir or not all_traces:
        return []

    import wandb
    from datetime import datetime, timezone

    log_dir.mkdir(parents=True, exist_ok=True)
    written = []
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = wandb.run.id if wandb.run else None
    run_name = wandb.run.name if wandb.run else None

    for task_name, traces in all_traces.items():
        out_path = log_dir / f"eval_table_{task_name}_step{global_step}_run{run_id}.json"
        payload = {
            "step": global_step,
            "name": f"eval_table_{task_name}",
            "n": len(traces),
            "run_id": run_id,
            "run_name": run_name,
            "logged_at": timestamp,
            "rows": traces,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        written.append(out_path)

    return written


def _run_unified_eval(model, tokenizer, model_name, global_step, args, log_dir=None, no_activations=False):
    """Run all evals via unified eval loop."""
    import wandb

    print(f"\n--- Evals at step {global_step} ---")
    eval_tasks = getattr(args, "eval_tasks", None)
    metrics, all_traces = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=eval_tasks,
        max_items=args.max_items_per_eval,
        eval_batch_size=args.eval_batch_size,
        device="cuda",
        layers=MULTI_LAYERS,
        no_activations=no_activations,
        position_mode=args.position_mode,
        stochastic_max_k=args.stochastic_max_k,
    )

    # Aggregate mean metrics across eval tasks
    # eval_scores: only primary per-task scores (eval/{task_name}), not sub-metrics like _gemini_score
    eval_scores = {k: v for k, v in metrics.items() if k.startswith("eval/") and k.removeprefix("eval/") in TASKS}
    acc_only = {k: v for k, v in eval_scores.items() if TASKS[k.removeprefix("eval/")].scoring == ScoringMode.BINARY}

    # Build wandb Tables for each task's per-example traces
    log_dict = {k: v for k, v in metrics.items() if not k.startswith("_")}
    if eval_scores:
        log_dict["eval/mean"] = sum(eval_scores.values()) / len(eval_scores)
    if acc_only:
        log_dict["eval/mean_acc"] = sum(acc_only.values()) / len(acc_only)
    # Include samples_seen so wandb can correlate eval metrics with training x-axis
    log_dict["train/samples_seen"] = global_step * getattr(args, "effective_batch_size", 256)
    trace_files = _write_eval_traces(log_dir, all_traces, global_step)

    if wandb.run and trace_files:
        _run_name = wandb.run.name or wandb.run.id
        artifact = wandb.Artifact(f"eval_traces_{_run_name}_{wandb.run.id}_step{global_step}", type="eval_traces", metadata={"step": global_step, "run_id": wandb.run.id, "run_name": _run_name})
        for path in trace_files:
            artifact.add_file(str(path), name=path.name)
        wandb.run.log_artifact(artifact)

    if wandb.run and all_traces:
        for task_name, traces in all_traces.items():
            table = wandb.Table(columns=["question", "cot_prefix", "cot_suffix", "cot_text", "target_response", "cot_field", "masked_cot_field", "oracle_prompt", "oracle_prefix", "expected", "predicted", "correct", "judge_score", "predicted_confidence", "judge_reason"])
            for t in traces:
                table.add_data(
                    t.get("question", "")[:200],
                    t.get("cot_prefix", "")[:500],
                    t.get("cot_suffix", "")[:500],
                    t.get("cot_text", "")[:500],
                    t.get("target_response", "")[:200],
                    t.get("cot_field", "")[:500],
                    t.get("masked_cot_field", "")[:500],
                    t.get("oracle_prompt", "")[:300],
                    t.get("oracle_prefix", "")[:300],
                    t.get("expected", "")[:200],
                    t.get("predicted", "")[:200],
                    t.get("correct", "?"),
                    t.get("judge_score"),
                    t.get("predicted_confidence"),
                    t.get("judge_reason", ""),
                )
            log_dict[f"eval_table/{task_name}"] = table

    if log_dict and wandb.run:
        # Log scalar metrics immediately; upload tables as a separate artifact
        # to avoid clogging the wandb file_stream (which blocks all history sync).
        scalar_dict = {k: v for k, v in log_dict.items() if not isinstance(v, wandb.Table)}
        table_dict = {k: v for k, v in log_dict.items() if isinstance(v, wandb.Table)}
        if scalar_dict:
            wandb.log(scalar_dict, step=global_step)
        if table_dict:
            try:
                _run_name = wandb.run.name or wandb.run.id
                art = wandb.Artifact(f"eval_tables_{_run_name}_step{global_step}", type="eval_tables")
                for tname, tbl in table_dict.items():
                    art.add(tbl, tname.replace("/", "_"))
                wandb.run.log_artifact(art)
            except Exception as e:
                print(f"  [wandb] Failed to upload eval tables artifact: {e}")

    elapsed = sum(v for k, v in metrics.items() if k.startswith("_eval_time/"))
    return metrics, elapsed


# ── AO classification eval ──

_CLS_EVAL_DATA: dict[str, list[TrainingDataPoint]] | None = None

DEFAULT_CLS_EVAL_DATASETS = [
    "sst2", "ag_news", "snli", "ner", "tense",
    "language_identification", "singular_plural",
    "geometry_of_truth", "relations", "md_gender",
]


def _load_cls_eval_data(model, model_name: str, layers: list[int], n_per_dataset: int, datasets: list[str] | None = None):
    """Load AO classification eval datasets (cached to disk after first run)."""
    global _CLS_EVAL_DATA
    if _CLS_EVAL_DATA is not None:
        return _CLS_EVAL_DATA

    from peft import PeftModel
    from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
    from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig

    # Unwrap PeftModel so AO code can access model.model.layers
    raw_model = model.base_model.model if isinstance(model, PeftModel) else model

    datasets = datasets or DEFAULT_CLS_EVAL_DATASETS
    # Convert our absolute layer indices to layer_percents for the AO loader
    from nl_probes.utils.common import get_layer_count
    n_layers = get_layer_count(model_name)
    layer_percents = [round(100 * l / n_layers) for l in layers]

    print(f"\n  [cls-eval] Loading {len(datasets)} classification eval datasets (n={n_per_dataset}, layers={layers} -> percents={layer_percents})...")

    _CLS_EVAL_DATA = {}
    for ds_name in datasets:
        cls_config = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_end_offset=-3, min_end_offset=-3,
            max_window_size=1, min_window_size=1,
        )
        loader_config = DatasetLoaderConfig(
            custom_dataset_params=cls_config,
            num_train=0, num_test=n_per_dataset,
            splits=["test"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=True,
            batch_size=256,
        )
        loader = ClassificationDatasetLoader(dataset_config=loader_config, model=raw_model)
        data = loader.load_dataset("test")
        ds_id = ds_name
        _CLS_EVAL_DATA[ds_id] = data
        print(f"  [cls-eval]   {ds_id}: {len(data)} examples")

    return _CLS_EVAL_DATA


def _run_cls_eval(model, tokenizer, submodule, global_step: int, args):
    """Run AO classification evals and log to wandb."""
    import wandb
    from nl_probes.utils.eval import run_evaluation, score_eval_responses

    cls_data = _load_cls_eval_data(
        model, args.model, MULTI_LAYERS,
        n_per_dataset=args.cls_eval_n,
        datasets=getattr(args, "cls_eval_datasets", None),
    )

    print(f"\n--- AO cls eval at step {global_step} ---")
    model.eval()

    generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}
    log_dict = {}

    for ds_name, eval_data in cls_data.items():
        results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            global_step=global_step,
            lora_path=None,  # already active on model
            eval_batch_size=args.eval_batch_size,
            steering_coefficient=args.steering_coefficient,
            generation_kwargs=generation_kwargs,
        )
        fmt_acc, ans_acc = score_eval_responses(results, eval_data)
        log_dict[f"cls_eval/{ds_name}/format_acc"] = fmt_acc
        log_dict[f"cls_eval/{ds_name}/accuracy"] = ans_acc
        print(f"  [cls-eval] {ds_name}: accuracy={ans_acc:.3f}, format={fmt_acc:.3f}")

    # Aggregate
    accs = [v for k, v in log_dict.items() if k.endswith("/accuracy")]
    log_dict["cls_eval/mean_accuracy"] = sum(accs) / len(accs)
    print(f"  [cls-eval] mean accuracy: {log_dict['cls_eval/mean_accuracy']:.3f}")

    if wandb.run:
        wandb.log(log_dict, step=global_step)

    model.train()


def _normalized_train_progress(global_step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return min(global_step / (total_steps - 1), 1.0)


def _build_scatter(points: list[tuple[int, float, str]], x_key: str, title: str):
    """Build a scatter plot; sentence-boundary points are red, others blue."""
    import matplotlib.pyplot as plt
    import wandb

    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    tags = [t for _, _, t in points]
    colors = ["red" if t == "sentence" else "steelblue" for t in tags]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, c=colors, alpha=0.5, s=12, linewidths=0)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel("Loss")
    ax.set_title(title)
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='sentence boundaries'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=6, label='other')]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


# ── Main training loop ──
def train(
    raw_data: list[dict],
    model,
    ddp_model,
    tokenizer,
    submodule,
    args,
    global_step: int,
    save_dir: Path,
    rank: int,
    world_size: int,
) -> int:
    """Train on all tasks. Returns the final global_step.

    model: unwrapped PeftModel (for materialization, eval, checkpoint saving)
    ddp_model: DDP-wrapped model (for forward/backward) or same as model if single-GPU
    """
    import wandb

    device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda")
    dtype = torch.bfloat16

    assert args.effective_batch_size % (args.batch_size * world_size) == 0, \
        f"effective_batch_size ({args.effective_batch_size}) must be divisible by " \
        f"batch_size * world_size ({args.batch_size} * {world_size} = {args.batch_size * world_size})"
    grad_accum = args.effective_batch_size // (args.batch_size * world_size)

    # Convert to TrainingDataPoints
    training_data = dicts_to_training_data(raw_data, tokenizer)
    if rank == 0:
        print(f"  Converted {len(training_data)} training examples")

    # Build mapping: datapoint_type → parent task (for subtask grouping in wandb)
    subtype_to_task = {}
    for item in raw_data:
        dt = item.get("datapoint_type", item.get("task", "unknown"))
        subtype_to_task[dt] = item.get("task", dt)

    if not training_data:
        if rank == 0:
            print("  ERROR: No training data!")
        return global_step

    # Type distribution
    type_counts = defaultdict(int)
    for dp in training_data:
        type_counts[dp.datapoint_type] += 1
    if rank == 0:
        print(f"\n  Task distribution:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")

    # In unified system, eval data comes from HF test splits (not carved from training)
    random.shuffle(training_data)
    final_training = training_data
    train_per_type = defaultdict(list)
    for dp in final_training:
        train_per_type[dp.datapoint_type].append(dp)
    eval_per_type = {}  # kept for compatibility with sequential/interleaved modes

    task_order = getattr(args, "task_order", "shuffled")

    task_stage_idx = {}  # task_name -> int index (for wandb logging)

    if task_order == "sequential":
        # Group by task, respect YAML ordering for curriculum
        yaml_task_names = getattr(args, "_yaml_task_order", [])
        # Order: YAML order first, then any remaining types alphabetically
        ordered_types = []
        for yt in yaml_task_names:
            # Check both the task name and its legacy_datapoint_type
            task_def = TASKS.get(yt)
            candidates = [yt]
            if task_def and task_def.legacy_datapoint_type:
                candidates.append(task_def.legacy_datapoint_type)
            for dt in candidates:
                if dt in train_per_type and dt not in ordered_types:
                    ordered_types.append(dt)
        for dt in sorted(train_per_type.keys()):
            if dt not in ordered_types:
                ordered_types.append(dt)
        task_blocks = []
        for task_type in ordered_types:
            items = train_per_type[task_type]
            random.shuffle(items)
            task_blocks.append((task_type, items))
        # Flatten in task order (task A then task B then ...)
        final_training = []
        for _, items in task_blocks:
            final_training.extend(items)
        for i, (task_name, _) in enumerate(task_blocks):
            task_stage_idx[task_name] = i
        if rank == 0:
            print(f"  Task order: SEQUENTIAL")
            for task_name, items in task_blocks:
                print(f"    stage {task_stage_idx[task_name]}: {task_name} ({len(items)} examples)")
            if wandb.run:
                wandb.config.update({"stage_map": {v: k for k, v in task_stage_idx.items()}})
    elif task_order == "interleaved":
        # Round-robin blocks: cycle through tasks, eval at every block boundary
        # Tasks with fewer samples are delayed to appear towards the end of training
        yaml_task_names = getattr(args, "_yaml_task_order", [])
        ordered_types = []
        for yt in yaml_task_names:
            task_def = TASKS.get(yt)
            candidates = [yt]
            if task_def and task_def.legacy_datapoint_type:
                candidates.append(task_def.legacy_datapoint_type)
            for dt in candidates:
                if dt in train_per_type and dt not in ordered_types:
                    ordered_types.append(dt)
        for dt in sorted(train_per_type.keys()):
            if dt not in ordered_types:
                ordered_types.append(dt)

        n_blocks = getattr(args, "interleave_blocks", 50)
        total_examples = sum(len(v) for v in train_per_type.values())
        block_size = total_examples // n_blocks

        # Sort tasks by sample count descending (biggest first = earliest start)
        sorted_types = sorted(ordered_types, key=lambda dt: len(train_per_type[dt]), reverse=True)

        # Calculate how many rounds each task needs and when it should start
        # so that all tasks finish at the end of training
        task_rounds_needed = {dt: max(1, math.ceil(len(train_per_type[dt]) / block_size)) for dt in sorted_types}
        max_rounds = max(task_rounds_needed.values())
        task_start_round = {dt: max_rounds - task_rounds_needed[dt] for dt in sorted_types}

        # Shuffle within each task, create queues
        task_queues = {}
        for dt in sorted_types:
            items = list(train_per_type[dt])
            random.shuffle(items)
            task_queues[dt] = items

        # Round-robin with delayed starts: small tasks appear only in later rounds
        task_blocks = []
        for round_idx in range(max_rounds):
            for task in sorted_types:
                if round_idx >= task_start_round[task] and task_queues[task]:
                    queue = task_queues[task]
                    block = queue[:block_size]
                    task_queues[task] = queue[block_size:]
                    if block:
                        task_blocks.append((task, block))

        final_training = []
        for _, items in task_blocks:
            final_training.extend(items)

        for i, (task_name, _) in enumerate(task_blocks):
            task_stage_idx.setdefault(task_name, i)

        if rank == 0:
            print(f"  Task order: INTERLEAVED ({n_blocks} blocks, block_size={block_size}, {max_rounds} rounds)")
            for dt in sorted_types:
                print(f"    {dt}: {len(train_per_type[dt])} examples, {task_rounds_needed[dt]} rounds, starts round {task_start_round[dt]}")
            for i, (task_name, items) in enumerate(task_blocks):
                print(f"    block {i}: {task_name} ({len(items)} examples)")
            if wandb.run:
                wandb.config.update({"stage_map": {v: k for k, v in task_stage_idx.items()}})
    else:
        random.shuffle(final_training)
        if rank == 0:
            print(f"  Task order: SHUFFLED")

    # Data sharding for multi-GPU
    if world_size > 1:
        aligned = (len(final_training) // (args.batch_size * world_size)) * (args.batch_size * world_size)
        final_training = final_training[:aligned]
        final_training = final_training[rank::world_size]

    # In unified system, eval uses HF test splits (no pre-materialization needed)
    if rank == 0:
        print(f"  Training: {len(final_training)} examples")

    # ── Optionally precompute activation vectors (--precompute flag) ──
    if args.precompute and not args.no_activations:
        _PC_MAX_TOKENS = 65536  # max total tokens per precompute batch (batch_size × seq_len)
        _pc_indices = [i for i, dp in enumerate(final_training) if dp.steering_vectors is None and dp.context_input_ids is not None]
        if _pc_indices:
            # Sort by context length so similar-length items batch together (minimize padding)
            _pc_indices.sort(key=lambda i: len(final_training[i].context_input_ids))
            if rank == 0:
                _ctx_lens = [len(final_training[i].context_input_ids) for i in _pc_indices]
                print(f"\n  Precomputing activation vectors for {len(_pc_indices)} examples ...")
                print(f"    Context lengths: min={min(_ctx_lens)}, median={sorted(_ctx_lens)[len(_ctx_lens)//2]}, max={max(_ctx_lens)}")
            model.eval()
            _precompute_start = time.time()
            # Dynamic batching: group sorted indices into batches by token budget
            _pc_batches = []
            _cur_batch = []
            _cur_max_len = 0
            for idx in _pc_indices:
                ctx_len = len(final_training[idx].context_input_ids)
                new_max = max(_cur_max_len, ctx_len)
                # Would this item push the batch over the token budget?
                if _cur_batch and new_max * (len(_cur_batch) + 1) > _PC_MAX_TOKENS:
                    _pc_batches.append(_cur_batch)
                    _cur_batch = [idx]
                    _cur_max_len = ctx_len
                else:
                    _cur_batch.append(idx)
                    _cur_max_len = new_max
            if _cur_batch:
                _pc_batches.append(_cur_batch)
            if rank == 0:
                _batch_sizes = [len(b) for b in _pc_batches]
                print(f"    {len(_pc_batches)} extraction batches (sizes: {min(_batch_sizes)}-{max(_batch_sizes)})")
            for _idx_chunk in tqdm(_pc_batches, desc="Precompute acts", disable=rank != 0):
                _pc_batch = [final_training[i] for i in _idx_chunk]
                with torch.no_grad():
                    _pc_result = materialize_multilayer_steering_vectors(_pc_batch, tokenizer, model)
                # Move vectors to CPU to save GPU memory; write back to final_training
                for j, idx in enumerate(_idx_chunk):
                    dp = _pc_result[j]
                    if dp.steering_vectors is not None:
                        dp.steering_vectors = dp.steering_vectors.cpu()
                    final_training[idx] = dp
            model.train()
            if rank == 0:
                _pc_elapsed = time.time() - _precompute_start
                print(f"  Precomputed in {_pc_elapsed:.1f}s ({len(_pc_indices) / _pc_elapsed:.0f} examples/s)")
            torch.cuda.empty_cache()

    # Optimizer + scheduler
    num_batches = len(final_training) // args.batch_size
    total_steps = (num_batches // grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    # Precompute per-stage step boundaries for progress tracking
    stage_step_ranges = {}  # task_name -> (start_step, end_step)
    if task_order in ("sequential", "interleaved") and task_blocks:
        cursor = 0
        for task_name, items in task_blocks:
            stage_steps = len(items) // (args.batch_size * grad_accum * world_size)
            stage_step_ranges[task_name] = (cursor, cursor + stage_steps)
            cursor += stage_steps

    # Precompute block-boundary eval/save steps for interleaved mode
    # Eval at both start and end of every block; account for world_size in step counts
    block_eval_steps = set()
    block_save_steps = set()
    if task_order == "interleaved" and task_blocks:
        cursor = 0
        for _, items in task_blocks:
            block_eval_steps.add(cursor)  # start of block
            block_steps = len(items) // (args.batch_size * grad_accum * world_size)
            if block_steps >= 2:
                block_eval_steps.add(cursor + block_steps // 2)  # midpoint
            cursor += block_steps
            block_eval_steps.add(cursor)  # end of block
        block_eval_steps.discard(0)  # step-0 eval handled separately
        sorted_evals = sorted(block_eval_steps)
        block_save_steps = set(sorted_evals[i] for i in range(1, len(sorted_evals), 2))

    # Dynamic eval/save cadence: ~10 evals over the relevant span
    if task_order == "interleaved":
        # Interleaved uses block-boundary eval/save; disable modulo-based triggers
        args.eval_steps = max(total_steps + 1, 999999)
        args.save_steps = max(total_steps + 1, 999999)
        if rank == 0:
            print(f"\n  Interleaved cadence: eval at {len(block_eval_steps)} block boundaries, save at {len(block_save_steps)}")
    elif task_order == "sequential":
        reference_steps = max(len(items) // (args.batch_size * grad_accum) for items in train_per_type.values() if len(items) >= args.batch_size)
        args.eval_steps = max(-(-reference_steps // 10), 1)
        args.save_steps = args.eval_steps * 2
        if rank == 0:
            print(f"\n  Dynamic cadence (reference = {reference_steps} steps):")
            print(f"    eval_steps: {args.eval_steps} (~{reference_steps // max(args.eval_steps, 1)}x)")
            print(f"    save_steps: {args.save_steps}")
    else:
        reference_steps = total_steps
        args.eval_steps = max(-(-reference_steps // 10), 1)
        args.save_steps = args.eval_steps * 2
        if rank == 0:
            print(f"\n  Dynamic cadence (reference = {reference_steps} steps):")
            print(f"    eval_steps: {args.eval_steps} (~{reference_steps // max(args.eval_steps, 1)}x)")
            print(f"    save_steps: {args.save_steps}")

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Restore optimizer/scheduler/RNG from checkpoint if resuming
    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            torch.random.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
            random.setstate(state["random_state"])
            if rank == 0:
                print(f"  Restored optimizer/scheduler/RNG from {state_path}")
        elif rank == 0:
            print(f"  Warning: no training_state.pt in {args.resume_from}, optimizer starts fresh")

    # Skip already-processed data on resume
    if global_step > 0:
        skip_items = global_step * args.batch_size * grad_accum
        if skip_items < len(final_training):
            final_training = final_training[skip_items:]
            if rank == 0:
                print(f"  Resume: skipped {skip_items} items, {len(final_training)} remaining")
        elif rank == 0:
            print(f"  Warning: resume step {global_step} exceeds data ({skip_items} >= {len(final_training)})")

    # Task index mapping for wandb (so we can plot which task is active)
    all_task_types = sorted(type_counts.keys())
    task_to_idx = {t: i for i, t in enumerate(all_task_types)}

    if rank == 0:
        wandb.log({
            "total_steps": total_steps,
            "n_examples": len(final_training),
            "n_tasks": len(train_per_type),
            "train/samples_seen": global_step * args.effective_batch_size,
        }, step=global_step)

        # Log task index legend to wandb config (skip if session closed)
        if wandb.run:
            wandb.config.update({"task_index_legend": task_to_idx}, allow_val_change=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    _run_name = wandb.run.name if wandb.run else (args.wandb_run or "cot-oracle")
    from datetime import datetime, timezone
    _date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    log_dir = Path("eval_logs") / f"{_date_prefix}_{_run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"\n  LR: {args.lr}")
        print(f"  Batch: {args.batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {args.batch_size * grad_accum * world_size}")
        print(f"  Train token budget: {args.max_train_tokens_per_gpu or 'disabled'}")
        effective_extract_budget = (
            args.max_extract_tokens_per_gpu
            if args.max_extract_tokens_per_gpu > 0
            else args.max_train_tokens_per_gpu
        )
        print(f"  Extract token budget: {effective_extract_budget or 'disabled'}")
        ext_bs = args.extraction_batch_size or args.batch_size
        print(f"  Extraction batch size: {ext_bs} (lookahead={ext_bs // args.batch_size}x)")
        print(f"  Length bucketing: {args.length_bucketing} (window_batches={args.length_bucket_window_batches})")
        print(f"  Epochs: {args.epochs}")
        print(f"  Steps: {total_steps}")
        print(f"  Warmup: {warmup_steps}")
        print(f"  Eval limits: max_items={args.max_items_per_eval}")
        print(f"  Eval decode caps: detection={args.eval_max_new_tokens}, task={args.task_eval_max_new_tokens}")

    model.train()

    # Step-0 eval (baseline before any training)
    skip_step0 = getattr(args, "no_step0_eval", False)
    if global_step == 0 and rank == 0 and not skip_step0:
        _run_unified_eval(model, tokenizer, args.model, 0, args, log_dir=log_dir, no_activations=args.no_activations)
        if args.cls_eval:
            _run_cls_eval(model, tokenizer, submodule, 0, args)
        model.train()
    if world_size > 1:
        dist.barrier()

    # EMA for per-task loss
    task_loss_ema = {}
    ema_alpha = 0.1
    total_tokens = 0
    train_start_time = time.time()
    last_step_time = time.time()
    eval_time_total = 0.0
    scatter_image_every = 100
    scatter_window_acts: list[tuple[int, float, str]] = []   # (n_acts, loss, pos_tag)
    scatter_window_layers: list[tuple[int, float, str]] = [] # (n_layers, loss, pos_tag)

    prev_dominant_task = None  # Track task transitions for phase checkpoints
    micro_step = 0  # counts micro-batches within a gradient accumulation window

    for epoch in range(args.epochs):
        if task_order not in ("sequential", "interleaved"):
            random.shuffle(final_training)
            if args.length_bucketing:
                _window_bucket_training_data(final_training, args.batch_size, args.length_bucket_window_batches)
        optimizer.zero_grad()

        pbar = tqdm(
            range(0, len(final_training), args.batch_size),
            desc=f"E{epoch + 1}/{args.epochs}",
            disable=rank != 0,
        )

        # Accumulators across micro-batches within a grad_accum window
        accum_task_losses = defaultdict(list)
        accum_loss_sum = 0.0
        accum_batch_types = []
        accum_batch_tokens = 0
        accum_batch_splits = 0
        accum_context_lengths = []
        accum_scatter_acts: list[tuple[float, int, str]] = []   # (loss, n_activations, pos_tag)
        accum_scatter_layers: list[tuple[float, int, str]] = [] # (loss, n_layers, pos_tag)

        # Lookahead extraction: prefetch activations for multiple training batches at once
        ext_bs = args.extraction_batch_size if args.extraction_batch_size > 0 else args.batch_size
        _prefetch_cache: list[TrainingDataPoint] = []
        _prefetch_end: int = 0  # index into final_training up to which we've prefetched

        for start in pbar:
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break

            extract_split_count = 0
            if not args.no_activations:
                extract_budget = (
                    args.max_extract_tokens_per_gpu
                    if args.max_extract_tokens_per_gpu > 0
                    else args.max_train_tokens_per_gpu
                )

                # Check if we need to prefetch a new batch of activations
                if not _prefetch_cache:
                    # Grab up to ext_bs examples for lookahead extraction
                    lookahead_end = min(start + ext_bs, len(final_training))
                    lookahead = final_training[start:lookahead_end]
                    _prefetch_end = lookahead_end

                    prefetched, extract_split_count = _materialize_batch_for_training_step(
                        lookahead,
                        tokenizer,
                        model,
                        max_batch_size=ext_bs,
                        max_extract_tokens_per_gpu=extract_budget,
                        rank=rank,
                        train_batch_start=start,
                    )
                    _prefetch_cache = prefetched

                # Consume batch_size items from the prefetch cache
                batch_list = _prefetch_cache[:args.batch_size]
                _prefetch_cache = _prefetch_cache[args.batch_size:]

            base_label_tokens = sum(_label_token_count(dp) for dp in batch_list)
            pending_chunks = _split_batch_for_token_budget(
                batch_list,
                args.batch_size,
                args.max_train_tokens_per_gpu,
                args.no_activations,
            )
            batch_split_count = extract_split_count + len(pending_chunks) - 1

            while pending_chunks:
                chunk_list = pending_chunks.pop(0)
                chunk_types = [dp.datapoint_type for dp in chunk_list]
                loss_weight = sum(_label_token_count(dp) for dp in chunk_list) / base_label_tokens
                sync_context = ddp_model.no_sync() if world_size > 1 and pending_chunks else nullcontext()

                with sync_context:
                    try:
                        if args.no_activations:
                            # Text-only: no activation extraction, no steering hook
                            batch = construct_batch(chunk_list, tokenizer, device)
                            tokenized_input = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                outputs = ddp_model(**tokenized_input, labels=batch.labels)
                                loss = outputs.loss * loss_weight / grad_accum
                        else:
                            # Standard steering path (vectors already materialized for this train batch)
                            batch = construct_batch(chunk_list, tokenizer, device)
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                outputs = train_features_batch(
                                    batch, ddp_model, submodule,
                                    args.steering_coefficient, device, dtype,
                                )
                                loss = outputs.loss * loss_weight / grad_accum
                            torch.cuda.synchronize()
                    except torch.OutOfMemoryError:
                        if world_size > 1 or len(chunk_list) == 1:
                            raise
                        torch.cuda.empty_cache()
                        split_at = len(chunk_list) // 2
                        if rank == 0:
                            print(f"  CUDA OOM at train batch {start}, splitting micro-batch {len(chunk_list)} -> {split_at}+{len(chunk_list) - split_at}")
                        batch_split_count += 1
                        pending_chunks = [chunk_list[:split_at], chunk_list[split_at:]] + pending_chunks
                        continue
                    loss.backward()

                # Per-task loss (use unscaled loss for logging)
                with torch.no_grad():
                    logits = outputs.logits.detach()
                    labels = batch["labels"] if isinstance(batch, dict) else batch.labels
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

                for i, (dp, task_type) in enumerate(zip(chunk_list, chunk_types)):
                    loss_val = per_item_loss[i].item()
                    accum_task_losses[task_type].append(loss_val)
                    if dp.positions:
                        item_layers = dp.meta_info.get("layers", MULTI_LAYERS)
                        _tag = dp.meta_info.get("pos_tag", "unknown") if dp.meta_info else "unknown"
                        accum_scatter_acts.append((loss_val, len(dp.positions), _tag))
                        accum_scatter_layers.append((loss_val, len(item_layers) if item_layers else 1, _tag))
                accum_loss_sum += outputs.loss.item() * loss_weight
                accum_batch_types.extend(chunk_types)
                accum_batch_tokens += batch_tokens
                accum_context_lengths.extend(len(dp.context_input_ids) for dp in chunk_list if dp.context_input_ids is not None)

            accum_batch_splits += batch_split_count

            micro_step += 1
            if micro_step % grad_accum != 0:
                continue

            # Optimizer step (only every grad_accum micro-batches)
            clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update EMA for per-task losses
            for task, losses in accum_task_losses.items():
                avg = sum(losses) / len(losses)
                if task not in task_loss_ema:
                    task_loss_ema[task] = avg
                else:
                    task_loss_ema[task] = ema_alpha * avg + (1 - ema_alpha) * task_loss_ema[task]

            # Aggregate token counts across ranks for accurate throughput
            if world_size > 1:
                _tok_tensor = torch.tensor([accum_batch_tokens], device=device)
                dist.all_reduce(_tok_tensor)
                global_batch_tokens = int(_tok_tensor.item())
            else:
                global_batch_tokens = accum_batch_tokens

            # Logging (rank 0 only)
            if rank == 0:
                now = time.time()
                train_progress = _normalized_train_progress(global_step, total_steps)
                log_dict = {
                    "train/loss": accum_loss_sum / grad_accum,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/total_tokens": total_tokens,
                    "train/batch_tokens": global_batch_tokens,
                    "train/batch_splits": accum_batch_splits,
                    "train/avg_context_length": sum(accum_context_lengths) / len(accum_context_lengths) if accum_context_lengths else 0,
                    "train/tokens_per_sec": global_batch_tokens / max(now - last_step_time, 1e-6),
                    "train/step_time": now - last_step_time,
                    "train/wallclock_hours": (now - train_start_time - eval_time_total) / 3600,
                    "eval/wallclock_hours": eval_time_total / 3600,
                    "train/samples_seen": global_step * args.effective_batch_size,
                    "train/progress": train_progress,
                }
                last_step_time = now
                if accum_scatter_acts:
                    for _lv, _na, _tg in accum_scatter_acts:
                        scatter_window_acts.append((_na, _lv, _tg))
                if accum_scatter_layers:
                    for _lv, _nl, _tg in accum_scatter_layers:
                        scatter_window_layers.append((_nl, _lv, _tg))
                if global_step % scatter_image_every == 0:
                    if scatter_window_acts:
                        log_dict["train/loss_vs_n_activations"] = _build_scatter(scatter_window_acts, "n_activations", "Loss vs N Activations")
                    if scatter_window_layers:
                        log_dict["train/loss_vs_n_layers"] = _build_scatter(scatter_window_layers, "n_layers", "Loss vs N Layers")
                    # Save raw scatter data to disk
                    scatter_path = log_dir / f"scatter_step{global_step}.jsonl"
                    with open(scatter_path, "w") as _sf:
                        for _na, _lv, _tg in scatter_window_acts:
                            _sf.write(json.dumps({"x": _na, "loss": _lv, "tag": _tg, "kind": "n_activations"}) + "\n")
                        for _nl, _lv, _tg in scatter_window_layers:
                            _sf.write(json.dumps({"x": _nl, "loss": _lv, "tag": _tg, "kind": "n_layers"}) + "\n")
                    scatter_window_acts.clear()
                    scatter_window_layers.clear()
                parent_task_emas = defaultdict(list)
                for subtype, ema_val in task_loss_ema.items():
                    parent = subtype_to_task.get(subtype, subtype)
                    if parent != subtype:
                        log_dict[f"train/loss_{parent}/{subtype}"] = ema_val
                    else:
                        log_dict[f"train/loss_{subtype}"] = ema_val
                    parent_task_emas[parent].append(ema_val)
                for parent, vals in parent_task_emas.items():
                    if len(vals) > 1:
                        log_dict[f"train/loss_{parent}"] = sum(vals) / len(vals)

                # Track dominant task for sequential mode phase transitions
                batch_task_counts = defaultdict(int)
                for t in accum_batch_types:
                    batch_task_counts[t] += 1
                dominant_task = max(batch_task_counts, key=batch_task_counts.get)
                log_dict["train/stage_idx"] = task_stage_idx.get(dominant_task, -1)
                log_dict["train/stage_name"] = dominant_task
                wandb.run.summary["current_stage"] = dominant_task
                if dominant_task in stage_step_ranges:
                    s_start, s_end = stage_step_ranges[dominant_task]
                    s_len = max(s_end - s_start, 1)
                    log_dict["train/stage_progress"] = min((global_step - s_start) / s_len, 1.0)

                # Log block index for interleaved mode
                if task_order == "interleaved" and task_blocks:
                    cursor = 0
                    for bi, (_, items) in enumerate(task_blocks):
                        cursor += len(items) // (args.batch_size * grad_accum * world_size)
                        if global_step < cursor:
                            log_dict["train/block_idx"] = bi
                            break
                    else:
                        log_dict["train/block_idx"] = len(task_blocks) - 1

                wandb.log(log_dict, step=global_step)

            pbar.set_postfix(loss=f"{accum_loss_sum / grad_accum:.4f}")

            # Track dominant task for sequential mode phase transitions (all ranks)
            if rank != 0:
                batch_task_counts = defaultdict(int)
                for t in accum_batch_types:
                    batch_task_counts[t] += 1
                dominant_task = max(batch_task_counts, key=batch_task_counts.get)

            # Phase checkpoint: save when dominant task changes in sequential/interleaved mode
            if rank == 0 and task_order in ("sequential", "interleaved") and prev_dominant_task is not None and dominant_task != prev_dominant_task:
                ckpt_path = save_dir / f"step_{global_step}_phase_{prev_dominant_task}"
                print(f"\n  Phase transition: {prev_dominant_task} -> {dominant_task}")
                print(f"  Saving phase checkpoint to {ckpt_path}")
                model.save_pretrained(str(ckpt_path))
                _save_training_state(ckpt_path, global_step, optimizer, scheduler)
            if world_size > 1 and task_order in ("sequential", "interleaved") and prev_dominant_task is not None and dominant_task != prev_dominant_task:
                dist.barrier()
            prev_dominant_task = dominant_task

            # Unified eval (task + detection, rank 0 only)
            should_eval = (global_step > 0 and global_step % args.eval_steps == 0) or global_step in block_eval_steps
            if should_eval:
                if rank == 0:
                    _, elapsed = _run_unified_eval(model, tokenizer, args.model, global_step, args, log_dir=log_dir, no_activations=args.no_activations)
                    eval_time_total += elapsed
                    if args.cls_eval:
                        _run_cls_eval(model, tokenizer, submodule, global_step, args)
                    model.train()
                if world_size > 1:
                    dist.barrier()

            # Save checkpoint (rank 0 only)
            should_save = (global_step > 0 and global_step % args.save_steps == 0) or global_step in block_save_steps
            if should_save:
                if rank == 0:
                    ckpt_path = save_dir / f"step_{global_step}"
                    print(f"  Saving checkpoint to {ckpt_path}")
                    model.save_pretrained(str(ckpt_path))
                    _save_training_state(ckpt_path, global_step, optimizer, scheduler)
                if world_size > 1:
                    dist.barrier()

            # Upload latest checkpoint to HF at every eval step
            if should_eval and rank == 0:
                latest_path = save_dir / "latest"
                latest_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(latest_path))
                _upload_checkpoint_to_hf(latest_path, args, global_step)

            # Reset accumulators for next grad_accum window
            accum_task_losses = defaultdict(list)
            accum_loss_sum = 0.0
            accum_batch_types = []
            accum_batch_tokens = 0
            accum_batch_splits = 0
            accum_context_lengths = []
            accum_scatter_acts = []
            accum_scatter_layers = []

            global_step += 1

    # Final eval (rank 0 only)
    if rank == 0:
        _run_unified_eval(model, tokenizer, args.model, global_step, args, log_dir=log_dir, no_activations=args.no_activations)
        if args.cls_eval:
            _run_cls_eval(model, tokenizer, submodule, global_step, args)

        # Save final
        final_path = save_dir / "final"
        print(f"  Saving final checkpoint to {final_path}")
        model.save_pretrained(str(final_path))
        _save_training_state(final_path, global_step, optimizer, scheduler)
        _log_final_checkpoint_to_wandb(final_path, global_step)

        # Upload final checkpoint to HuggingFace
        _upload_checkpoint_to_hf(final_path, args, global_step)

    if world_size > 1:
        dist.barrier()

    return global_step


# ── Config loading ──
def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _mark_cli_overrides(args, parser, argv: list[str]) -> None:
    for action in parser._actions:
        if action.dest == "config" or not action.option_strings:
            continue
        if any(token == opt or token.startswith(f"{opt}=") for token in argv for opt in action.option_strings):
            setattr(args, f"_cli_{action.dest}", True)


def apply_config(args, config: dict):
    """Apply config values to args, CLI flags override config."""
    # Task counts
    if "tasks" in config:
        # Preserve YAML ordering for sequential curriculum
        args._yaml_task_order = list(config["tasks"].keys())
        eval_tasks = []
        for task_name, task_cfg in config["tasks"].items():
            arg_name = f"{task_name}_n"
            # YAML is source of truth — always set unless CLI explicitly overrode
            if not getattr(args, f"_cli_{arg_name}", False):
                setattr(args, arg_name, task_cfg.get("n", 0))
            # Per-task epochs (default 1)
            setattr(args, f"{task_name}_epochs", task_cfg.get("epochs", 1))
            # Per-task eval flag (default True)
            if task_cfg.get("eval", True):
                eval_tasks.append(task_name)
        args.eval_tasks = eval_tasks

    # Training params
    if "training" in config:
        t = config["training"]
        _float_keys = {"lr", "warmup_fraction", "max_grad_norm", "steering_coefficient"}
        _int_keys = {"epochs", "seed", "interleave_blocks", "length_bucket_window_batches", "max_train_tokens_per_gpu", "max_extract_tokens_per_gpu", "extraction_batch_size", "batch_size", "effective_batch_size"}
        for key in ["lr", "epochs",
                     "warmup_fraction", "max_grad_norm", "steering_coefficient",
                     "gradient_checkpointing", "task_order", "seed",
                     "interleave_blocks", "max_train_tokens_per_gpu", "max_extract_tokens_per_gpu",
                     "batch_size", "effective_batch_size", "extraction_batch_size",
                     "length_bucketing", "length_bucket_window_batches",
                     "torch_compile", "torch_compile_mode"]:
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
        for key in ["n_layers", "position_mode"]:
            if key in a and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, a[key])
        if "layers" in a and not getattr(args, "_cli_layers", False):
            args.layers = a["layers"]  # list of ints, e.g. [9, 18, 27]
        if "position_encoding" in a and not getattr(args, "_cli_position_encoding", False):
            args.position_encoding = a["position_encoding"]
        if "pe_alpha" in a and not getattr(args, "_cli_pe_alpha", False):
            args.pe_alpha = float(a["pe_alpha"])
        if "stochastic_max_k" in a and not getattr(args, "_cli_stochastic_max_k", False):
            args.stochastic_max_k = int(a["stochastic_max_k"])
        if "max_context_length" in a and not getattr(args, "_cli_max_context_length", False):
            args.max_context_length = int(a["max_context_length"])
        if "layer_dropout" in a and not getattr(args, "_cli_layer_dropout", False):
            ld = a["layer_dropout"]
            if isinstance(ld, bool):
                args.layer_dropout = ld
            elif isinstance(ld, dict):
                args.layer_dropout = ld.get("train", False)
                args.layer_dropout_eval = ld.get("eval", False)
    # Eval
    if "eval" in config:
        e = config["eval"]
        for key in ["max_items_per_eval", "eval_steps"]:
            if key in e and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, int(e[key]))
        if "cls_eval" in e and not getattr(args, "_cli_cls_eval", False):
            args.cls_eval = e["cls_eval"]
        if "cls_eval_datasets" in e and not getattr(args, "_cli_cls_eval_datasets", False):
            args.cls_eval_datasets = e["cls_eval_datasets"]
        if "cls_eval_n" in e and not getattr(args, "_cli_cls_eval_n", False):
            args.cls_eval_n = int(e["cls_eval_n"])

    # Data paths (unified: all data from HuggingFace, no local corpus/precomputed paths needed)

    # No-activations baseline
    if "no_activations" in config:
        if config["no_activations"].get("enabled") and not getattr(args, "_cli_no_activations", False):
            args.no_activations = True

    # Ablations
    if "ablations" in config:
        ab = config["ablations"]
        if ab.get("random_layers") and not getattr(args, "_cli_random_layers", False):
            args.random_layers = True

    # Model
    if "model" in config:
        m = config["model"]
        if "name" in m and not getattr(args, "_cli_model", False):
            args.model = m["name"]
        if "attn_implementation" in m and not getattr(args, "_cli_attn_implementation", False):
            args.attn_implementation = m["attn_implementation"]
        if "ao_checkpoint" in m and not getattr(args, "_cli_ao_checkpoint", False):
            args.ao_checkpoint = m["ao_checkpoint"]
        if "fresh_lora" in m and not getattr(args, "_cli_fresh_lora", False):
            args.fresh_lora = m["fresh_lora"]

    # FineWeb
    if "fineweb" in config:
        fw = config["fineweb"]
        if fw.get("enabled", False) and not getattr(args, "_cli_fineweb_n", False):
            args.fineweb_n = fw.get("n", 50000)
        if "max_context_tokens" in fw and not getattr(args, "_cli_fineweb_max_context_tokens", False):
            args.fineweb_max_context_tokens = fw["max_context_tokens"]
        if "min_target_tokens" in fw:
            args.fineweb_min_target_tokens = fw["min_target_tokens"]
        if "max_target_tokens" in fw:
            args.fineweb_max_target_tokens = fw["max_target_tokens"]
        if "variant" in fw:
            args.fineweb_variant = fw["variant"]

    # Classification
    if "classification" in config:
        cls = config["classification"]
        if cls.get("enabled", False) and not getattr(args, "_cli_classification_n", False):
            args.classification_n = cls.get("n", 100000)
        if "datasets" in cls and not getattr(args, "_cli_classification_datasets", False):
            args.classification_datasets = cls["datasets"]

    # LatentQA

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
    local_rank, rank, world_size = setup_distributed()

    parser = argparse.ArgumentParser(description="Train CoT Oracle")
    parser.add_argument("--config", nargs="+", default=["configs/train.yaml"],
                        help="YAML config file(s). Multiple configs are merged left-to-right (later overrides earlier)")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl",
                        help="Path to corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Transformer attention backend (sdpa uses flash kernels natively in torch>=2.2)")

    # Checkpoint control
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a LoRA checkpoint dir")
    parser.add_argument("--ao-checkpoint",
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
                        help="Adam's pretrained AO checkpoint to start from")
    parser.add_argument("--fresh-lora", action="store_true", default=True,
                        help="Start with fresh LoRA (default). Use --no-fresh-lora to load Adam's checkpoint")
    parser.add_argument("--no-fresh-lora", dest="fresh_lora", action="store_false",
                        help="Load Adam's pretrained AO checkpoint instead of fresh LoRA")

    # Per-task example counts — defaults are 0; set via --config (train.yaml is source of truth)
    # 7 trainable tasks in the unified system:
    parser.add_argument("--hint-admission-n", type=int, default=0)
    parser.add_argument("--atypical-answer-n", type=int, default=0)
    parser.add_argument("--reasoning-termination-n", type=int, default=0)
    parser.add_argument("--answer-trajectory-n", type=int, default=0)
    parser.add_argument("--futurelens-n", type=int, default=0)
    parser.add_argument("--correctness-n", type=int, default=0)
    parser.add_argument("--decorative-cot-n", type=int, default=0)
    parser.add_argument("--chunked-convqa-n", type=int, default=0)
    parser.add_argument("--chunked-compqa-n", type=int, default=0)
    parser.add_argument("--sycophancy-n", type=int, default=0)

    # Training hyperparams (defaults are None; sourced from YAML config, with hardcoded
    # fallbacks in _FALLBACK_DEFAULTS for when no config is provided)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-items-per-eval", type=int, default=None, help="Maximum items per detection eval")
    parser.add_argument("--eval-max-new-tokens", type=int, default=32, help="Default max_new_tokens for detection eval generation")
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=64, help="Default max_new_tokens for task-level eval generation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None, help="Number of activation layers (evenly spaced through model depth)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Explicit layer indices (overrides --n-layers). E.g. --layers 9 18 27")
    parser.add_argument("--steering-coefficient", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--position-mode", type=str, default=None, choices=["last_only", "graduated", "stochastic", "mixed", "all"], help="Position sampling: last_only (fastest), graduated (last 1-3), stochastic (30/30/40 last1/last3/endskewed), mixed (20/50/30 lastk/sentence/endskewed, default), all")
    parser.add_argument("--stochastic-max-k", type=int, default=None, help="Upper bound for Poisson position sampling")
    parser.add_argument("--max-context-length", type=int, default=None, help="Drop training samples with context_input_ids longer than this (0 = no filter)")
    parser.add_argument("--task-order", choices=["shuffled", "sequential", "interleaved"], default=None, help="'shuffled' mixes all tasks; 'sequential' trains one at a time; 'interleaved' round-robin blocks")
    parser.add_argument("--interleave-blocks", type=int, default=None, help="Number of blocks for interleaved mode")
    parser.add_argument("--length-bucketing", action="store_true", default=None, help="Enable windowed context-length bucketing in shuffled mode")
    parser.add_argument("--no-length-bucketing", dest="length_bucketing", action="store_false")
    parser.add_argument("--length-bucket-window-batches", type=int, default=None, help="Window size for length bucketing, in units of train batches")
    parser.add_argument("--effective-batch-size", type=int, default=None, help="Total effective batch size (invariant to GPU count). gradient_accumulation_steps = effective_batch_size / (batch_size * world_size)")
    parser.add_argument("--max-train-tokens-per-gpu", type=int, default=None, help="Approximate per-GPU peak token budget for splitting long train batches (0 disables)")
    parser.add_argument("--max-extract-tokens-per-gpu", type=int, default=None, help="Approximate per-GPU token budget for activation extraction batches (0 = use --max-train-tokens-per-gpu)")
    parser.add_argument("--extraction-batch-size", type=int, default=None, help="Lookahead extraction: extract this many examples at once, then train in batch_size chunks. 0 = same as batch_size (no lookahead). Set to 64-128 for significant speedup.")
    parser.add_argument("--torch-compile", action="store_true", default=None, help="Compile the training forward path with torch.compile")
    parser.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], default=None, help="torch.compile mode")

    # Text-only baseline
    parser.add_argument("--no-activations", action="store_true", default=False,
                        help="Text-only baseline: train without activation steering (same data, no prefix/injection)")

    # Precompute activations upfront (slow for single-epoch, useful for multi-epoch)
    parser.add_argument("--precompute", action="store_true", default=False,
                        help="Precompute all activation vectors before training. Slower for epoch=1 but amortizes for multi-epoch.")

    # FineWeb context prediction
    parser.add_argument("--fineweb-n", type=int, default=0,
                        help="Number of FineWeb context prediction examples (0 = disabled, set via config)")
    parser.add_argument("--fineweb-max-context-tokens", type=int, default=2000,
                        help="Max context tokens for FineWeb activation extraction")

    # Classification (Adam's AO tasks)
    parser.add_argument("--classification-n", type=int, default=0,
                        help="Number of classification examples (0 = disabled, set via config)")
    parser.add_argument("--classification-datasets", nargs="+", default=None,
                        help="Classification datasets to use (default: sst2, ag_news, snli)")

    # LatentQA (Adam's SPQA task)
    parser.add_argument("--latentqa-n", type=int, default=0,  # DEPRECATED: do not use

                        help="Number of LatentQA examples (0 = disabled, set via config)")

    # Layer dropout
    parser.add_argument("--layer-dropout", action="store_true", default=None, help="Random non-empty subsets of configured layers per training example")
    parser.add_argument("--no-layer-dropout", dest="layer_dropout", action="store_false")
    parser.add_argument("--position-encoding", action="store_true", default=None, help="Add sinusoidal position encoding to activations")
    parser.add_argument("--no-position-encoding", dest="position_encoding", action="store_false")
    parser.add_argument("--pe-alpha", type=float, default=None, help="Position encoding strength")

    # Ablations
    parser.add_argument("--random-layers", action="store_true", default=False,
                        help="Randomize layer count and indices per training sequence")

    # Eval / save
    parser.add_argument("--cls-eval", action="store_true", default=None, help="Run AO classification evals (sst2, ag_news, etc.) at each eval step")
    parser.add_argument("--no-cls-eval", dest="cls_eval", action="store_false")
    parser.add_argument("--cls-eval-datasets", nargs="+", default=None, help="Classification eval datasets (default: all 10 standard datasets)")
    parser.add_argument("--cls-eval-n", type=int, default=None, help="Number of test examples per cls eval dataset")
    parser.add_argument("--eval-steps", type=int, default=None, help="Run evals every N steps (shuffled mode)")
    parser.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps (shuffled mode)")
    parser.add_argument("--no-step0-eval", action="store_true", default=False,
                        help="Skip evals at step 0 (for quick ablation launches)")
    parser.add_argument("--start-step", type=int, default=None,
                        help="Starting global step (for resuming; 0 = restart data from beginning)")
    parser.add_argument("--eval-dir", default="data/evals")
    _default_act_cache = os.path.join(os.environ["FAST_CACHE_DIR"], "cot_oracle", "eval_precomputed") if os.environ.get("FAST_CACHE_DIR") else "data/eval_precomputed"
    parser.add_argument("--activation-cache-dir", default=_default_act_cache,
                        help="Dir with precomputed activation bundles (.pt)")

    # Output
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-entity", default="MATS10-CS-JB",
                        help="Wandb entity (team/org)")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-group", default=None, help="Wandb group name")

    # Data loading (unified: all data from HuggingFace, cached locally)

    args = parser.parse_args()

    # Mark which args were explicitly provided on CLI so config doesn't override them
    _mark_cli_overrides(args, parser, sys.argv[1:])

    # Apply config file(s) (CLI flags override config values)
    if args.config:
        configs = args.config if isinstance(args.config, list) else [args.config]
        for cfg_path in configs:
            config = load_config(cfg_path)
            apply_config(args, config)
        if rank == 0:
            print(f"Loaded config from {', '.join(configs)}")
            # Log the full config YAML for reproducibility
            import yaml
            print(f"\n{'=' * 60}")
            print("CONFIG")
            print(f"{'=' * 60}")
            print(yaml.dump(config, default_flow_style=False, sort_keys=False).rstrip())
            print(f"{'=' * 60}\n")

    # Fallback defaults for when no config is provided (YAML is the single source of
    # truth; these only kick in for args still None after config application)
    _FALLBACKS = dict(
        lr=1e-5, batch_size=16, epochs=1, max_items_per_eval=10,
        steering_coefficient=1.0, max_grad_norm=1.0, warmup_fraction=0.1, seed=42,
        gradient_checkpointing=True, position_mode="last_only", stochastic_max_k=100,
        max_context_length=0, task_order="shuffled", interleave_blocks=50,
        length_bucketing=True, length_bucket_window_batches=32,
        effective_batch_size=16, max_train_tokens_per_gpu=0,
        max_extract_tokens_per_gpu=0, extraction_batch_size=0,
        torch_compile=False, torch_compile_mode="default",
        layer_dropout=False, position_encoding=False, pe_alpha=0.1,
        n_layers=3, eval_steps=2000, save_steps=10000,
        cls_eval=False, cls_eval_n=25,
    )
    for key, fallback in _FALLBACKS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, fallback)

    # Auto-scale batch sizes for multi-GPU (unless explicitly set via CLI)
    if world_size > 1:
        if not getattr(args, "_cli_batch_size", False):
            args.batch_size = 32
        if not getattr(args, "_cli_effective_batch_size", False):
            args.effective_batch_size = 32 * world_size
        if rank == 0:
            print(f"Multi-GPU auto-scale: batch_size={args.batch_size}, effective_batch_size={args.effective_batch_size} (world_size={world_size})")

    # No-activations setup
    if getattr(args, "no_activations", False):
        args.fresh_lora = True

    set_seed(args.seed)

    # Multi-layer config
    global MULTI_LAYERS, NO_ACTIVATIONS, RANDOM_LAYERS, LAYER_DROPOUT, POSITION_MODE, STOCHASTIC_MAX_K, MAX_CONTEXT_LENGTH, POSITION_ENCODING, PE_ALPHA, _MODEL_N_LAYERS, SENTENCE_DELIM_IDS
    NO_ACTIVATIONS = getattr(args, "no_activations", False)
    RANDOM_LAYERS = getattr(args, "random_layers", False)
    LAYER_DROPOUT = args.layer_dropout
    POSITION_MODE = args.position_mode
    STOCHASTIC_MAX_K = args.stochastic_max_k
    MAX_CONTEXT_LENGTH = args.max_context_length
    POSITION_ENCODING = args.position_encoding
    PE_ALPHA = args.pe_alpha
    from cot_utils import LAYER_COUNTS
    _MODEL_N_LAYERS = LAYER_COUNTS.get(args.model, 36)
    if hasattr(args, "layers") and args.layers:
        MULTI_LAYERS = [int(l) for l in args.layers]
    else:
        n_layers = getattr(args, "n_layers", 3)
        percents = [int(100 * (i + 1) / (n_layers + 1)) for i in range(n_layers)]
        MULTI_LAYERS = [layer_percent_to_layer(args.model, p) for p in percents]
    # Make layers available to dataset loaders
    import cot_utils as _cu
    _cu.CONFIGURED_LAYERS = MULTI_LAYERS
    if rank == 0:
        if NO_ACTIVATIONS:
            print(f"No-activations mode (text-only baseline, layers computed for data loaders only): {MULTI_LAYERS}")
        else:
            print(f"Multi-layer injection: {MULTI_LAYERS}")
        print(f"Distributed: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        if LAYER_DROPOUT:
            print(f"Layer dropout: ON (random non-empty subsets of {MULTI_LAYERS} per example)")
        if RANDOM_LAYERS:
            print("Ablation: RANDOM LAYERS (per-item random layer sampling)")
        print(f"Position mode: {POSITION_MODE}")
        if POSITION_MODE == "mixed":
            print(f"Mixed max_k: {STOCHASTIC_MAX_K}")
        if MAX_CONTEXT_LENGTH > 0:
            print(f"Max context length: {MAX_CONTEXT_LENGTH} (longer samples will be dropped)")

    tokenizer = load_tokenizer(args.model)

    # Build sentence delimiter token set for stochastic position sampling
    SENTENCE_DELIM_IDS = set()
    for pattern in [".", ".\n", ".\n\n"]:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if len(ids) == 1:
            SENTENCE_DELIM_IDS.add(ids[0])
    if rank == 0:
        print(f"Sentence delimiter token IDs: {sorted(SENTENCE_DELIM_IDS)}")

    # Verify placeholder
    tok_ids = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(tok_ids) == 1, f"Placeholder '{PLACEHOLDER_TOKEN}' is {len(tok_ids)} tokens"
    if rank == 0:
        print(f"Placeholder token: '{PLACEHOLDER_TOKEN}' -> token ID {tok_ids[0]}")

    # ── Load model ──
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    if rank == 0:
        print(f"\nLoading model: {args.model}")
        print(f"Attention backend: {args.attn_implementation}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation=args.attn_implementation,
    )
    base_model.enable_input_require_grads()

    if args.gradient_checkpointing:
        base_model.use_cache = False
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    # Get hook submodule BEFORE LoRA
    submodule = get_hf_submodule(base_model, 1)

    # Resume state (loaded before wandb init so we can reuse the run ID)
    _resume_state = None
    if args.resume_from:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
        model = PeftModel.from_pretrained(
            base_model, args.resume_from,
            is_trainable=True, autocast_adapter_dtype=False,
        )
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            _resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
            if not getattr(args, "_cli_start_step", False):
                args.start_step = _resume_state["global_step"]
            if rank == 0:
                print(f"  Loaded training_state.pt: step={_resume_state['global_step']}, wandb_id={_resume_state.get('wandb_run_id')}")
    elif args.fresh_lora:
        if rank == 0:
            print("Starting with FRESH LoRA (random init)")
        if args.ao_checkpoint:
            try:
                # Try loading AO checkpoint structure, then reinit weights
                model = PeftModel.from_pretrained(
                    base_model, args.ao_checkpoint,
                    is_trainable=True, autocast_adapter_dtype=False,
                )
            except RuntimeError:
                # Checkpoint doesn't match model (e.g. 8B checkpoint on 0.6B model) — create fresh LoRA
                if rank == 0:
                    print("  AO checkpoint incompatible, creating LoRA from scratch")
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64, lora_alpha=16, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none", task_type="CAUSAL_LM",
                )
                model = get_peft_model(base_model, lora_config)
        else:
            if rank == 0:
                print("  No AO checkpoint, creating LoRA from scratch")
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=64, lora_alpha=16, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_config)
        for name, param in model.named_parameters():
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(param, a=5**0.5)
            elif "lora_B" in name:
                torch.nn.init.zeros_(param)
    else:
        if rank == 0:
            print(f"Loading Adam's AO checkpoint: {args.ao_checkpoint}")
        model = PeftModel.from_pretrained(
            base_model, args.ao_checkpoint,
            is_trainable=True, autocast_adapter_dtype=False,
        )

    # Ensure trainable params are fp32 (optimizer states stay fp32; autocast handles forward pass)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if rank == 0:
        model.print_trainable_parameters()

    if args.torch_compile:
        compile_mode = None if args.torch_compile_mode == "default" else args.torch_compile_mode
        if rank == 0:
            print(f"Compiling model with torch.compile (mode={args.torch_compile_mode})")
        model = torch.compile(model, fullgraph=False, mode=compile_mode)

    # DDP wrapping
    if world_size > 1:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    # ── Load data ──
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("LOADING TRAINING DATA")
        print(f"{'=' * 60}")

    # Build task config from args (unified: task_name -> {n: int})
    task_config = {}
    for task_name in get_trainable_tasks():
        n = getattr(args, f"{task_name}_n", 0)
        if n > 0 or n == -1:
            epochs = getattr(args, f"{task_name}_epochs", 1)
            task_config[task_name] = {"n": n, "epochs": epochs}

    # FutureLens/PastLens use corpus-v5 + tokenizer — handle separately (like FineWeb)
    futurelens_n = task_config.pop("futurelens", {}).get("n", 0)
    pastlens_n = task_config.pop("pastlens", {}).get("n", 0)
    # FineWeb readout tasks use streaming generation, not HF download — pop them too
    for _fw_task in ("futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb"):
        task_config.pop(_fw_task, None)

    raw_data = load_all_training_data(task_config)

    # FutureLens (corpus-based, needs tokenizer) — skip in no-activations mode
    if futurelens_n != 0 and not NO_ACTIVATIONS:
        from data_loading import load_futurelens_data
        futurelens_target_n = None if futurelens_n == -1 else futurelens_n
        if rank == 0:
            futurelens_count_str = "all available" if futurelens_target_n is None else str(futurelens_target_n)
            print(f"  [data] Generating {futurelens_count_str} FutureLens examples from corpus...")
        futurelens_data = load_futurelens_data(
            tokenizer=tokenizer,
            n=futurelens_target_n,
            split="train",
            layers=MULTI_LAYERS,
            seed=args.seed,
        )
        raw_data.extend(futurelens_data)
        if rank == 0:
            print(f"  [data]   -> {len(futurelens_data)} FutureLens examples added (total: {len(raw_data)})")

    # PastLens (corpus-based, needs tokenizer) — skip in no-activations mode
    if pastlens_n != 0 and not NO_ACTIVATIONS:
        from data_loading import load_pastlens_data
        pastlens_target_n = None if pastlens_n == -1 else pastlens_n
        if rank == 0:
            pastlens_count_str = "all available" if pastlens_target_n is None else str(pastlens_target_n)
            print(f"  [data] Generating {pastlens_count_str} PastLens examples from corpus...")
        pastlens_data = load_pastlens_data(
            tokenizer=tokenizer,
            n=pastlens_target_n,
            split="train",
            layers=MULTI_LAYERS,
            seed=args.seed,
        )
        raw_data.extend(pastlens_data)
        if rank == 0:
            print(f"  [data]   -> {len(pastlens_data)} PastLens examples added (total: {len(raw_data)})")

    # FineWeb readout tasks (futurelens/pastlens/reconstruction on web text, if enabled)
    fineweb_n = getattr(args, "fineweb_n", 0)
    if fineweb_n > 0 and not NO_ACTIVATIONS:
        from data_loading import load_fineweb_readout_data
        fw_max_ctx = getattr(args, "fineweb_max_context_tokens", 2000)
        fw_min_tgt = getattr(args, "fineweb_min_target_tokens", 5)
        fw_max_tgt = getattr(args, "fineweb_max_target_tokens", 25)
        fw_variant = getattr(args, "fineweb_variant", None)
        variant_str = fw_variant if fw_variant else "3 variants"
        if rank == 0:
            print(f"  [data] Generating {fineweb_n} FineWeb readout examples "
                  f"({variant_str}, single-position, target {fw_min_tgt}-{fw_max_tgt} tokens)...")
        fineweb_data = load_fineweb_readout_data(
            tokenizer=tokenizer,
            n=fineweb_n,
            max_context_tokens=fw_max_ctx,
            layers=MULTI_LAYERS,
            min_target_tokens=fw_min_tgt,
            max_target_tokens=fw_max_tgt,
            seed=args.seed,
            variant=fw_variant,
        )
        raw_data.extend(fineweb_data)
        if rank == 0:
            print(f"  [data]   -> {len(fineweb_data)} FineWeb readout examples added (total: {len(raw_data)})")

    # Classification data (Adam's AO tasks, if enabled)
    cls_n = getattr(args, "classification_n", 0)
    if cls_n > 0 and not NO_ACTIVATIONS:
        from data_loading import load_classification_data
        cls_datasets = getattr(args, "classification_datasets", None)
        if rank == 0:
            ds_str = ", ".join(cls_datasets) if cls_datasets else "all"
            print(f"  [data] Generating {cls_n} classification examples ({ds_str})...")
        cls_data = load_classification_data(
            tokenizer=tokenizer,
            n=cls_n,
            datasets=cls_datasets,
            layers=MULTI_LAYERS,
            seed=args.seed,
        )
        raw_data.extend(cls_data)
        if rank == 0:
            print(f"  [data]   -> {len(cls_data)} classification examples added (total: {len(raw_data)})")

    if not raw_data:
        if rank == 0:
            print("ERROR: No training data loaded!")
        cleanup_distributed()
        return

    if not NO_ACTIVATIONS:
        # Tokenize cot_text → context_input_ids for items that don't have them yet
        from data_loading import prepare_context_ids
        prepare_context_ids(
            raw_data, tokenizer,
            layers=MULTI_LAYERS,
        )

        # Verify ALL items have context_input_ids after prepare_context_ids
        missing = [d.get("task", "?") for d in raw_data if not d.get("context_input_ids")]
        if missing:
            from collections import Counter
            counts = Counter(missing)
            raise RuntimeError(
                f"Items missing context_input_ids after prepare_context_ids "
                f"(likely missing cot_text): {dict(counts)}"
            )
    else:
        # Text-only mode: filter for items with cot_text (no activations needed)
        before = len(raw_data)
        raw_data = [d for d in raw_data if d.get("cot_text")]
        if len(raw_data) < before and rank == 0:
            print(f"  [data] Dropped {before - len(raw_data)} items without cot_text (no-activations mode)")

    # Filter by max context length
    if MAX_CONTEXT_LENGTH > 0:
        before = len(raw_data)
        raw_data = [d for d in raw_data if len(d.get("context_input_ids", [])) <= MAX_CONTEXT_LENGTH]
        if rank == 0:
            print(f"  [data] max_context_length={MAX_CONTEXT_LENGTH}: kept {len(raw_data)}/{before} samples "
                  f"(dropped {before - len(raw_data)})")

    random.shuffle(raw_data)

    # ── Wandb (rank 0 only) ──
    if rank == 0:
        import wandb
        wandb.login()

        # Build a descriptive run name from enabled tasks
        enabled_tasks = sorted(task_config.keys())

        run_name = args.wandb_run or "cot-oracle"
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith("_cli_")}
        wandb_config["world_size"] = world_size

        # Resume wandb run if we have a saved run ID from checkpoint
        resume_id = _resume_state.get("wandb_run_id") if _resume_state else None
        if resume_id:
            run_name = _resume_state.get("wandb_run_name") or run_name
            print(f"  Resuming wandb run: id={resume_id}, name={run_name}")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            id=resume_id,
            resume="allow" if resume_id else None,
            config=wandb_config,
            tags=[args.model.split("/")[-1]] + enabled_tasks,
        )
        wandb.define_metric("train/samples_seen")
        wandb.define_metric("*", step_metric="train/samples_seen")
        task_stage_idx = getattr(args, "_task_stage_idx", {})
        if task_stage_idx:
            wandb.config.update({"stage_map": {v: k for k, v in task_stage_idx.items()}})
        # Save raw YAML config to wandb for reproducibility
        if args.config:
            for cfg_path in (args.config if isinstance(args.config, list) else [args.config]):
                if Path(cfg_path).exists():
                    wandb.save(cfg_path)

        _wandb_run_id = wandb.run.id
    else:
        enabled_tasks = sorted(task_config.keys())

    save_dir = Path(args.save_dir)

    # ── Train ──
    if rank == 0:
        print(f"\n{'#' * 60}")
        print(f"  TRAINING: {len(raw_data)} examples, {len(enabled_tasks)} tasks")
        print(f"  Tasks: {', '.join(enabled_tasks)}")
        print(f"{'#' * 60}")

    global_step = args.start_step or 0
    global_step = train(
        raw_data=raw_data,
        model=model,
        ddp_model=ddp_model,
        tokenizer=tokenizer,
        submodule=submodule,
        args=args,
        global_step=global_step,
        save_dir=save_dir,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"\n{'#' * 60}")
        print(f"TRAINING COMPLETE at step {global_step}")
        print(f"{'#' * 60}")

        import wandb
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
