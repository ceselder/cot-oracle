"""
Train CoT Oracle: Flat Task-Based Training

All tasks mixed together in one training run. Enable/disable tasks via --*-n flags (0 = skip).
Continues from Adam's pretrained AO checkpoint (or fresh LoRA / custom checkpoint).
All tasks use stride=5, 3 layers (25%, 50%, 75%), paragraph token.

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

from cot_utils import layer_percent_to_layer, sparse_sample_positions
from tasks import TASKS, get_trainable_tasks
from data_loading import load_all_training_data
from eval_loop import run_eval

# ── Override placeholder token ──
PLACEHOLDER_TOKEN = " \u00b6"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN

# ── Multi-layer config ──
MULTI_LAYERS: list[int] = []
NO_ACTIVATIONS: bool = False
RANDOM_LAYERS: bool = False
LAYER_DROPOUT: bool = False
NOISE_ACTIVATIONS: bool = False
_MODEL_N_LAYERS: int = 36  # total layers in the model (set in main())


def _patched_get_prefix(sae_layer: int, num_positions: int) -> str:
    if MULTI_LAYERS:
        N = len(MULTI_LAYERS)
        K = num_positions // N
        assert K * N == num_positions, f"num_positions={num_positions} not divisible by {N} layers"
        parts = [f"L{layer}:" + PLACEHOLDER_TOKEN * K for layer in MULTI_LAYERS]
        prefix = " ".join(parts) + "\n"
    else:
        prefix = f"L{sae_layer}:" + PLACEHOLDER_TOKEN * num_positions + "\n"
    return prefix


du_module.get_introspection_prefix = _patched_get_prefix


def _patched_find_pattern_in_tokens(token_ids, special_token_str, num_positions, tokenizer):
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []
    for i in range(len(token_ids)):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)
    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    return positions


du_module.find_pattern_in_tokens = _patched_find_pattern_in_tokens


# ── Position encoding config (module-level, set by main()) ──
_PE_CONFIG = {"enabled": False, "alpha": 0.1}
# Pooling mode: "none", "windows" (mean-pool token windows), "single", "chunks5"
_POOLING_MODE = "none"
# Sparse position sampling: randomly subsample CoT positions per example
_SPARSE_POSITIONS = False
# Single position mode: only feed the last CoT position per layer
_SINGLE_POSITION = False


def _pool_vectors(vectors: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool extracted activation vectors according to mode.

    Args:
        vectors: [K, D] activations extracted for ONE layer
        mode: "none", "single", "chunks5"
    Returns:
        Pooled tensor: [1, D] for single, [min(5,K), D] for chunks5, [K, D] for none
    """
    if mode == "single":
        return vectors.mean(dim=0, keepdim=True)  # [1, D]
    elif mode == "chunks5":
        K = vectors.shape[0]
        n = min(5, K)
        chunks = torch.chunk(vectors, n, dim=0)
        return torch.stack([c.mean(dim=0) for c in chunks])  # [n, D]
    return vectors


def _mean_pool_windows(acts_LD: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Mean-pool activation windows between consecutive stride positions.

    Instead of point-sampling at each position, mean-pools all tokens in the
    window (prev_p+1 .. p] for each stride position. Output has same count K
    as input positions.
    """
    pooled = []
    prev = 0
    for p in positions:
        window = acts_LD[prev:p + 1, :]  # [window_size, D]
        pooled.append(window.mean(dim=0))
        prev = p + 1
    return torch.stack(pooled, dim=0)  # [K, D]


def _pooled_count_per_layer(K: int, mode: str) -> int:
    """How many vectors per layer after pooling."""
    if mode == "single":
        return 1
    elif mode == "chunks5":
        return min(5, K)
    return K


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

    Supports per-item random layers (RANDOM_LAYERS) and noise replacement (NOISE_ACTIVATIONS).
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
    MAX_CONTEXT_TOKENS = 4096  # Cap materialization to avoid 15K+ outliers blowing up batch time
    contexts = [list(dp.context_input_ids) for _, dp in to_fill]
    # When pooling, use full positions from meta_info (context_positions was truncated for validator)
    positions_per_item = []
    for _, dp in to_fill:
        full_pos = (dp.meta_info or {}).get("full_context_positions")
        positions_per_item.append(list(full_pos) if full_pos is not None else list(dp.context_positions))

    # Left-truncate long contexts: keep the last MAX_CONTEXT_TOKENS tokens
    for j in range(len(contexts)):
        ctx = contexts[j]
        if len(ctx) > MAX_CONTEXT_TOKENS:
            trim = len(ctx) - MAX_CONTEXT_TOKENS
            contexts[j] = ctx[trim:]
            # Drop positions that fell off the left, adjust remaining
            positions_per_item[j] = [p - trim for p in positions_per_item[j] if p >= trim]

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
        K = total_positions // N_item

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
            if _POOLING_MODE == "windows":
                layer_vecs = _mean_pool_windows(acts_BLD[b], adjusted)
            else:
                layer_vecs = acts_BLD[b, adjusted, :]  # [K, D]
                layer_vecs = _pool_vectors(layer_vecs, _POOLING_MODE)
            vectors_parts.append(layer_vecs)

        vectors = torch.cat(vectors_parts, dim=0).detach().contiguous()

        # Noise ablation: replace real activations with variance-matched Gaussian noise
        if NOISE_ACTIVATIONS:
            var = vectors.var(dim=0, keepdim=True)  # per-dimension variance
            vectors = torch.randn_like(vectors) * var.sqrt()

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
        # Trim positions to match actual pooled vector count (short CoTs may
        # produce fewer chunks than the prompt expected)
        if vectors.shape[0] != len(dp_new.positions):
            dp_new.positions = dp_new.positions[:vectors.shape[0]]
        new_batch[idx] = dp_new

    return new_batch


du_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors
eval_module.materialize_missing_steering_vectors = materialize_multilayer_steering_vectors


# ── Flamingo activation extraction ──
def materialize_flamingo_activations(
    batch_points: list[TrainingDataPoint],
    tokenizer,
    model,
    max_ctx_tokens: int | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Extract supervisee residual stream at each xattn layer position.

    Only extracts layers that have cross-attention (matching the wrapper's
    xattn_layer_indices), not all layers.

    Returns:
        kvs: {layer_idx: [B, T_max, D]} padded per-layer activations
        kv_masks: {layer_idx: [B, T_max]} bool masks (True = real, False = pad)
    """
    from nl_probes.utils.activation_utils import get_hf_submodule

    # Unwrap FlamingoOracleWrapper to get PeftModel and xattn indices
    from flamingo_oracle import FlamingoOracleWrapper
    if isinstance(model, FlamingoOracleWrapper):
        peft_model = model.base_model
        xattn_layers = model.xattn_layer_indices
    else:
        peft_model = model
        xattn_layers = list(range(peft_model.config.num_hidden_layers))

    assert isinstance(peft_model, PeftModel), "Expected PeftModel for activation extraction"

    device = next(peft_model.parameters()).device
    pad_id = tokenizer.pad_token_id

    # Collect context_input_ids (truncate to max_ctx_tokens if set)
    contexts = []
    for dp in batch_points:
        assert dp.context_input_ids is not None, "context_input_ids required for Flamingo"
        ctx = list(dp.context_input_ids)
        if max_ctx_tokens and len(ctx) > max_ctx_tokens:
            ctx = ctx[-max_ctx_tokens:]
        contexts.append(ctx)

    # Left-pad contexts to same length
    max_ctx_len = max(len(c) for c in contexts)
    input_ids_list = []
    attn_masks_list = []
    left_offsets = []
    for c in contexts:
        pad_len = max_ctx_len - len(c)
        input_ids_list.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        attn_masks_list.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_masks_list),
    }

    # Only hook the layers we need (xattn positions)
    submodules = {layer: get_hf_submodule(peft_model, layer, use_lora=True) for layer in xattn_layers}

    was_training = peft_model.training
    peft_model.eval()
    with peft_model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=peft_model, submodules=submodules, inputs_BL=inputs_BL,
            min_offset=None, max_offset=None,
        )
    if was_training:
        peft_model.train()

    B = len(batch_points)

    # Build per-layer padded activations and masks
    # All items share the same T_max (max_ctx_len after left-padding), but we
    # strip padding per-item and re-pad to batch max
    max_real_len = max(len(c) for c in contexts)
    kvs = {}
    kv_masks = {}

    for layer in xattn_layers:
        # acts_by_layer[layer] is [B, max_ctx_len, D] (includes left-padding)
        layer_acts_list = []
        layer_mask_list = []
        for b in range(B):
            ctx_len = len(contexts[b])
            offset = left_offsets[b]
            real_acts = acts_by_layer[layer][b, offset:offset + ctx_len, :]  # [T, D]
            layer_acts_list.append(real_acts)
            layer_mask_list.append(torch.ones(ctx_len, dtype=torch.bool, device=device))

        # Pad to max_real_len
        D = layer_acts_list[0].shape[-1]
        padded = torch.zeros(B, max_real_len, D, dtype=layer_acts_list[0].dtype, device=device)
        mask = torch.zeros(B, max_real_len, dtype=torch.bool, device=device)
        for b in range(B):
            L = layer_acts_list[b].shape[0]
            padded[b, :L] = layer_acts_list[b]
            mask[b, :L] = True

        kvs[layer] = padded.detach()
        kv_masks[layer] = mask

    return kvs, kv_masks


def construct_flamingo_batch(
    batch_list: list[TrainingDataPoint],
    supervisee_kvs: dict[int, torch.Tensor],
    supervisee_kv_masks: dict[int, torch.Tensor],
    tokenizer,
    device: torch.device,
) -> dict:
    """Construct a batch dict for FlamingoOracleWrapper.forward()."""
    max_length = max(len(dp.input_ids) for dp in batch_list)
    pad_id = tokenizer.pad_token_id

    all_input_ids = []
    all_labels = []
    all_attn_mask = []

    for dp in batch_list:
        pad_len = max_length - len(dp.input_ids)
        all_input_ids.append(torch.tensor([pad_id] * pad_len + dp.input_ids, dtype=torch.long))
        all_labels.append(torch.tensor([-100] * pad_len + dp.labels, dtype=torch.long))
        all_attn_mask.append(torch.tensor([False] * pad_len + [True] * len(dp.input_ids), dtype=torch.bool))

    return {
        "input_ids": torch.stack(all_input_ids).to(device),
        "attention_mask": torch.stack(all_attn_mask).to(device),
        "labels": torch.stack(all_labels).to(device),
        "supervisee_kvs": supervisee_kvs,
        "supervisee_kv_masks": supervisee_kv_masks,
    }


def construct_flamingo_batch(
    batch_list: list[TrainingDataPoint],
    tokenizer,
    device: torch.device,
    max_ctx_tokens: int | None = None,
) -> tuple[dict, dict]:
    """Construct separate CoT and oracle batches for two-pass Flamingo.

    CoT and oracle are padded independently to their own max lengths.
    No shared L_max — this is the whole point of the two-pass approach.

    Returns:
        cot_batch: {input_ids: [B, L_cot], attention_mask: [B, L_cot]}
        oracle_batch: {input_ids: [B, L_oracle], attention_mask: [B, L_oracle], labels: [B, L_oracle]}
    """
    pad_id = tokenizer.pad_token_id

    # CoT tokens (truncated if needed)
    cots = []
    for dp in batch_list:
        assert dp.context_input_ids is not None
        ctx = list(dp.context_input_ids)
        if max_ctx_tokens and len(ctx) > max_ctx_tokens:
            ctx = ctx[-max_ctx_tokens:]
        cots.append(ctx)

    max_cot_len = max(len(c) for c in cots)
    cot_ids, cot_masks = [], []
    for c in cots:
        pad = max_cot_len - len(c)
        cot_ids.append(torch.tensor([pad_id] * pad + c, dtype=torch.long))
        cot_masks.append(torch.tensor([False] * pad + [True] * len(c), dtype=torch.bool))

    # Oracle tokens
    max_oracle_len = max(len(dp.input_ids) for dp in batch_list)
    oracle_ids, oracle_masks, oracle_labels = [], [], []
    for dp in batch_list:
        pad = max_oracle_len - len(dp.input_ids)
        oracle_ids.append(torch.tensor([pad_id] * pad + dp.input_ids, dtype=torch.long))
        oracle_masks.append(torch.tensor([False] * pad + [True] * len(dp.input_ids), dtype=torch.bool))
        oracle_labels.append(torch.tensor([-100] * pad + dp.labels, dtype=torch.long))

    cot_batch = {
        "input_ids": torch.stack(cot_ids).to(device),
        "attention_mask": torch.stack(cot_masks).to(device),
    }
    oracle_batch = {
        "input_ids": torch.stack(oracle_ids).to(device),
        "attention_mask": torch.stack(oracle_masks).to(device),
        "labels": torch.stack(oracle_labels).to(device),
    }
    return cot_batch, oracle_batch


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
            # Decode context_input_ids to recover CoT text
            raw_text = tokenizer.decode(item["context_input_ids"], skip_special_tokens=False)
            # Extract user question
            user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', raw_text, re.DOTALL)
            user_question = user_match.group(1).strip() if user_match else ""
            # Extract CoT (assistant turn — may lack closing tag since context is truncated)
            assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)', raw_text, re.DOTALL)
            cot_text = assistant_match.group(1).strip() if assistant_match else ""
            # Strip activation metadata to get task question
            task_question = _act_prefix_re.sub("", item["prompt"])
            # Build text-only prompt: question + CoT + task
            prompt = f"Question: {user_question}\nChain of thought: {cot_text}\n\n{task_question}"

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
            # Use a dummy steering vector so pydantic validator doesn't require context_*
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

        if RANDOM_LAYERS:
            # Per-item random layer sampling (ablation: arbitrary model layers)
            from layer_utils import sample_layers
            sampled = sample_layers(_MODEL_N_LAYERS, mean=3)
            ctx_pos = base_positions * len(sampled)
            num_pos = len(ctx_pos)

            # Temporarily swap MULTI_LAYERS for prefix generation
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
                meta_info={"prompt": item["prompt"], "layers": sampled},
            )
            MULTI_LAYERS[:] = saved_layers
        elif LAYER_DROPOUT:
            # Random non-empty subset of configured layers per item
            k = random.randint(1, len(MULTI_LAYERS))
            sampled = sorted(random.sample(MULTI_LAYERS, k))
            ctx_pos = base_positions * len(sampled)
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
                meta_info={"prompt": item["prompt"], "layers": sampled},
            )
            MULTI_LAYERS[:] = saved_layers
        else:
            # Sparse position sampling: randomly subsample CoT positions per example
            if _SPARSE_POSITIONS:
                ctx_pos = sparse_sample_positions(
                    ctx_pos, n_layers=n_layers_runtime,
                )
                num_pos = len(ctx_pos)

            # Suffix-based position sampling (50/50):
            #   50% → only the last 1 position per layer (minimal context)
            #   50% → sample m uniformly from 1..K, take last m positions per layer
            # Positions are always a contiguous suffix up to the prediction barrier.
            if not _SINGLE_POSITION and n_layers_runtime >= 1:
                total = len(ctx_pos)
                if total % n_layers_runtime == 0:
                    K = total // n_layers_runtime
                    if K > 1:
                        if random.random() < 0.5:
                            # Last-only: single activation at the barrier
                            m = 1
                        else:
                            # Uniform sample: random suffix length from 1..K
                            m = random.randint(1, K)
                        if m < K:
                            # Take last m positions from each layer's chunk
                            new_ctx_pos = []
                            for li in range(n_layers_runtime):
                                chunk = ctx_pos[li * K : (li + 1) * K]
                                new_ctx_pos.extend(chunk[K - m:])
                            ctx_pos = new_ctx_pos
                            num_pos = len(ctx_pos)
            elif _SINGLE_POSITION and n_layers_runtime >= 1:
                total = len(ctx_pos)
                if total % n_layers_runtime == 0:
                    K = total // n_layers_runtime
                    last_pos = ctx_pos[K - 1]
                    ctx_pos = [last_pos] * n_layers_runtime
                    num_pos = len(ctx_pos)

            # Re-expand positions if precomputed with fewer layers than runtime config.
            if n_layers_runtime > 1 and len(ctx_pos) % n_layers_runtime != 0:
                ctx_pos = base_positions * n_layers_runtime
                num_pos = len(ctx_pos)
                if not _reexpand_warned:
                    print(f"  [data] Re-expanding positions: -> {n_layers_runtime} layers (K={len(base_positions)})")
                    _reexpand_warned = True

            # Override num_positions for pooling modes
            full_ctx_pos = ctx_pos
            if _POOLING_MODE != "none":
                K_per_layer = num_pos // n_layers_runtime if n_layers_runtime else num_pos
                pooled_K = _pooled_count_per_layer(K_per_layer, _POOLING_MODE)
                num_pos = pooled_K * n_layers_runtime
                ctx_pos = ctx_pos[:num_pos]

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
                meta_info={"full_context_positions": full_ctx_pos, "prompt": item["prompt"]} if _POOLING_MODE != "none" else {"prompt": item["prompt"]},
            )
        training_data.append(dp)

    return training_data


# ── Task registry (moved to tasks.py) ──

# ── Train-on-eval conversion ──

# ── Training infrastructure ──
def _wrap_hook_for_grad_ckpt(hook_fn):
    """Wrap a steering hook so it clones the residual before modifying it.

    Gradient checkpointing with use_reentrant=False tracks saved tensor counts.
    The original hook creates per-batch autograd tensors via in-place modification,
    causing a count mismatch during recomputation.  Cloning first makes both
    passes identical.
    """
    def safe_hook(module, _input, output):
        if isinstance(output, tuple):
            resid, *rest = output
            output = (resid.clone(), *rest)
        else:
            output = output.clone()
        return hook_fn(module, _input, output)
    return safe_hook


def _example_context_len(dp: TrainingDataPoint) -> int:
    return len(dp.context_input_ids) if dp.context_input_ids is not None else len(dp.input_ids)


def _label_token_count(dp: TrainingDataPoint) -> int:
    return sum(label != -100 for label in dp.labels[1:])


def _estimate_train_batch_peak_tokens(
    batch_points: list[TrainingDataPoint],
    no_activations: bool,
    flamingo: bool,
    flamingo_max_ctx_tokens: int,
) -> int:
    batch_size = len(batch_points)
    oracle_peak_tokens = 2 * batch_size * max(len(dp.input_ids) for dp in batch_points)
    if no_activations:
        return oracle_peak_tokens
    if flamingo:
        context_peak_tokens = batch_size * max(min(_example_context_len(dp), flamingo_max_ctx_tokens) for dp in batch_points)
    else:
        context_peak_tokens = batch_size * max(_example_context_len(dp) for dp in batch_points)
    return max(context_peak_tokens, oracle_peak_tokens)


def _split_batch_for_token_budget(
    batch_points: list[TrainingDataPoint],
    max_batch_size: int,
    max_train_tokens_per_gpu: int,
    no_activations: bool,
    flamingo: bool,
    flamingo_max_ctx_tokens: int,
) -> list[list[TrainingDataPoint]]:
    if max_train_tokens_per_gpu <= 0:
        return [batch_points]

    chunks = []
    current_chunk = []
    for dp in batch_points:
        candidate = current_chunk + [dp]
        if current_chunk and (
            len(candidate) > max_batch_size
            or _estimate_train_batch_peak_tokens(candidate, no_activations, flamingo, flamingo_max_ctx_tokens) > max_train_tokens_per_gpu
        ):
            chunks.append(current_chunk)
            current_chunk = [dp]
            continue
        current_chunk = candidate
    chunks.append(current_chunk)
    return chunks


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
    # NOTE: _wrap_hook_for_grad_ckpt removed — only needed for use_reentrant=False.
    # With use_reentrant=True the clone is unnecessary and wastes ~20GB on long sequences.
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


def _save_table_to_disk(log_dir: Path, name: str, global_step: int, columns: list[str], rows: list[list]):
    """Save a wandb-style table to disk as nicely formatted JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    records = [dict(zip(columns, row)) for row in rows]
    path = log_dir / f"{name}_step{global_step}.json"
    with open(path, "w") as f:
        json.dump({"step": global_step, "name": name, "n": len(records), "rows": records}, f, indent=2, default=str)


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


def _load_gemini_baselines(path: str = "logs/llm_monitor/results.json") -> dict[str, float]:
    """Load Gemini LLM-monitor baselines keyed by eval name."""
    import json
    p = Path(path)
    if not p.exists():
        return {}
    results = json.load(open(p))
    baselines = {}
    for eval_name, data in results.items():
        m = data.get("metrics", {})
        if "accuracy" in m:
            baselines[eval_name] = ("acc", m["accuracy"])
        elif "mean_token_f1" in m:
            baselines[eval_name] = ("token_f1", m["mean_token_f1"])
    return baselines

_GEMINI_BASELINES: dict[str, tuple[str, float]] | None = None


def _run_unified_eval(model, tokenizer, model_name, global_step, args, log_dir=None, no_activations=False):
    """Run all evals via unified eval loop."""
    import wandb

    print(f"\n--- Evals at step {global_step} ---")
    stride_val = int(args.stride) if args.stride and args.stride != "punctuation" else 5
    eval_tasks = getattr(args, "eval_tasks", None)
    metrics = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=eval_tasks,
        max_items=args.max_items_per_eval,
        eval_batch_size=args.eval_batch_size,
        device="cuda",
        layers=MULTI_LAYERS,
        stride=stride_val,
    )
    if metrics:
        wandb.log(metrics, step=global_step)
    elapsed = sum(v for k, v in metrics.items() if k.startswith("eval_time/"))
    return metrics, elapsed


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

    model: unwrapped PeftModel or FlamingoOracleWrapper (for materialization, eval, checkpoint saving)
    ddp_model: DDP-wrapped model (for forward/backward) or same as model if single-GPU
    """
    import wandb
    from flamingo_oracle import FlamingoOracleWrapper

    device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda")
    dtype = torch.bfloat16

    # For evals, use the underlying PeftModel (evals use steering-based inference)
    eval_model = model.base_model if isinstance(model, FlamingoOracleWrapper) else model

    assert args.effective_batch_size % (args.batch_size * world_size) == 0, \
        f"effective_batch_size ({args.effective_batch_size}) must be divisible by " \
        f"batch_size * world_size ({args.batch_size} * {world_size} = {args.batch_size * world_size})"
    grad_accum = args.effective_batch_size // (args.batch_size * world_size)

    # Convert to TrainingDataPoints
    training_data = dicts_to_training_data(raw_data, tokenizer)
    if rank == 0:
        print(f"  Converted {len(training_data)} training examples")

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
        block_save_steps = set(sorted_evals[i] for i in range(4, len(sorted_evals), 5))

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
        args.save_steps = args.eval_steps * 5
        if rank == 0:
            print(f"\n  Dynamic cadence (reference = {reference_steps} steps):")
            print(f"    eval_steps: {args.eval_steps} (~{reference_steps // max(args.eval_steps, 1)}x)")
            print(f"    save_steps: {args.save_steps}")
    else:
        reference_steps = total_steps
        args.eval_steps = max(-(-reference_steps // 10), 1)
        args.save_steps = args.eval_steps * 5
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
        print(f"  Length bucketing: {args.length_bucketing} (window_batches={args.length_bucket_window_batches})")
        print(f"  Epochs: {args.epochs}")
        print(f"  Steps: {total_steps}")
        print(f"  Warmup: {warmup_steps}")
        print(f"  Eval limits: max_items={args.max_items_per_eval}")
        print(f"  Eval decode caps: detection={args.eval_max_new_tokens}, task={args.task_eval_max_new_tokens}")

    model.train()

    # Step-0 eval (baseline before any training)
    # Skip all evals in Flamingo mode — evals use steering injection, not cross-attention
    skip_step0 = getattr(args, "no_step0_eval", False)
    if global_step == 0 and rank == 0 and not skip_step0 and not args.flamingo:
        _run_unified_eval(eval_model, tokenizer, args.model, 0, args, log_dir=log_dir, no_activations=args.no_activations)
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

        for start in pbar:
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break

            base_label_tokens = sum(_label_token_count(dp) for dp in batch_list)
            pending_chunks = _split_batch_for_token_budget(
                batch_list,
                args.batch_size,
                args.max_train_tokens_per_gpu,
                args.no_activations,
                args.flamingo,
                args.flamingo_max_ctx_tokens,
            )
            batch_split_count = len(pending_chunks) - 1

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
                        elif args.flamingo:
                            # Two-pass: collect CoT hidden states, then oracle forward with xattn
                            flamingo_wrapper = ddp_model.module if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel) else ddp_model
                            cot_batch, oracle_batch = construct_flamingo_batch(
                                chunk_list, tokenizer, device,
                                max_ctx_tokens=args.flamingo_max_ctx_tokens,
                            )
                            # Pass 1: CoT through frozen base model (no grad)
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                cot_hs = flamingo_wrapper.collect_cot_hidden_states(
                                    cot_batch["input_ids"], cot_batch["attention_mask"],
                                )
                            # Pass 2: Oracle forward with cross-attention to CoT
                            batch = oracle_batch  # for per-task loss logging below
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                outputs = ddp_model(
                                    **oracle_batch,
                                    supervisee_kvs=cot_hs,
                                    cot_attention_mask=cot_batch["attention_mask"],
                                )
                                loss = outputs.loss * loss_weight / grad_accum
                        else:
                            # Standard steering path
                            chunk_list = materialize_multilayer_steering_vectors(
                                chunk_list, tokenizer, model
                            )
                            batch = construct_batch(chunk_list, tokenizer, device)
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                outputs = train_features_batch(
                                    batch, ddp_model, submodule,
                                    args.steering_coefficient, device, dtype,
                                )
                                loss = outputs.loss * loss_weight / grad_accum
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

                for i, task_type in enumerate(chunk_types):
                    accum_task_losses[task_type].append(per_item_loss[i].item())
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

            # Logging (rank 0 only)
            if rank == 0:
                now = time.time()
                log_dict = {
                    "train/loss": accum_loss_sum / grad_accum,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/total_tokens": total_tokens,
                    "train/batch_tokens": accum_batch_tokens,
                    "train/batch_splits": accum_batch_splits,
                    "train/avg_context_length": sum(accum_context_lengths) / len(accum_context_lengths) if accum_context_lengths else 0,
                    "train/tokens_per_sec": accum_batch_tokens / max(now - last_step_time, 1e-6),
                    "train/step_time": now - last_step_time,
                    "train/wallclock_hours": (now - train_start_time - eval_time_total) / 3600,
                    "eval/wallclock_hours": eval_time_total / 3600,
                    "train/samples_seen": global_step * args.effective_batch_size,
                }
                last_step_time = now
                for task, ema_val in task_loss_ema.items():
                    log_dict[f"train/loss_{task}"] = ema_val

                # Track dominant task for sequential mode phase transitions
                batch_task_counts = defaultdict(int)
                for t in accum_batch_types:
                    batch_task_counts[t] += 1
                dominant_task = max(batch_task_counts, key=batch_task_counts.get)
                log_dict["train/stage_idx"] = task_stage_idx.get(dominant_task, -1)
                log_dict["train/stage_name"] = dominant_task
                wandb.run.summary["current_stage"] = dominant_task
                log_dict["train/progress"] = global_step / max(total_steps, 1)
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

                # Log Flamingo gate values
                if args.flamingo:
                    from flamingo_oracle import FlamingoOracleWrapper
                    wrapper = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    if isinstance(wrapper, FlamingoOracleWrapper):
                        for idx in wrapper.xattn_layer_indices:
                            gate_val = wrapper.xattn_layers[str(idx)].gate.item()
                            log_dict[f"flamingo/gate_{idx}"] = gate_val
                            log_dict[f"flamingo/tanh_gate_{idx}"] = math.tanh(gate_val)

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
            if should_eval and not args.flamingo:
                if rank == 0:
                    _, elapsed = _run_unified_eval(eval_model, tokenizer, args.model, global_step, args, log_dir=log_dir, no_activations=args.no_activations)
                    eval_time_total += elapsed
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

            # Reset accumulators for next grad_accum window
            accum_task_losses = defaultdict(list)
            accum_loss_sum = 0.0
            accum_batch_types = []
            accum_batch_tokens = 0
            accum_batch_splits = 0
            accum_context_lengths = []

            global_step += 1

    # Final eval (rank 0 only)
    if rank == 0 and not args.flamingo:
        _run_unified_eval(eval_model, tokenizer, args.model, global_step, args, log_dir=log_dir, no_activations=args.no_activations)

        # Save final
        final_path = save_dir / "final"
        print(f"  Saving final checkpoint to {final_path}")
        model.save_pretrained(str(final_path))
        _save_training_state(final_path, global_step, optimizer, scheduler)

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
            # Per-task eval flag (default True)
            if task_cfg.get("eval", True):
                eval_tasks.append(task_name)
        args.eval_tasks = eval_tasks

    # Training params
    if "training" in config:
        t = config["training"]
        _float_keys = {"lr", "warmup_fraction", "max_grad_norm", "steering_coefficient"}
        _int_keys = {"batch_size", "eval_batch_size", "epochs", "seed", "effective_batch_size", "interleave_blocks", "length_bucket_window_batches", "max_train_tokens_per_gpu"}
        for key in ["lr", "batch_size", "eval_batch_size", "epochs",
                     "warmup_fraction", "max_grad_norm", "steering_coefficient",
                     "gradient_checkpointing", "task_order", "seed",
                     "effective_batch_size", "interleave_blocks", "max_train_tokens_per_gpu",
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
        for key in ["stride", "position_encoding", "pe_alpha", "n_layers", "pooling",
                    "sparse_positions", "single_position"]:
            if key in a and not getattr(args, f"_cli_{key}", False):
                if key == "pooling":
                    setattr(args, "pooling_mode", a[key])
                else:
                    setattr(args, key, a[key])
        if "layers" in a and not getattr(args, "_cli_layers", False):
            args.layers = a["layers"]  # list of ints, e.g. [9, 18, 27]
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
        for key in ["eval_steps", "save_steps", "max_items_per_eval"]:
            if key in e and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, int(e[key]))

    # Data paths (unified: all data from HuggingFace, no local corpus/precomputed paths needed)

    # Flamingo
    if "flamingo" in config:
        f = config["flamingo"]
        if f.get("enabled") and not getattr(args, "_cli_flamingo", False):
            args.flamingo = True
        if "xattn_interval" in f and not getattr(args, "_cli_flamingo_xattn_interval", False):
            args.flamingo_xattn_interval = int(f["xattn_interval"])
        if "xattn_lora_r" in f and not getattr(args, "_cli_flamingo_xattn_lora_r", False):
            args.flamingo_xattn_lora_r = int(f["xattn_lora_r"])

    # No-activations baseline
    if "no_activations" in config:
        if config["no_activations"].get("enabled") and not getattr(args, "_cli_no_activations", False):
            args.no_activations = True

    # Ablations
    if "ablations" in config:
        ab = config["ablations"]
        if ab.get("random_layers") and not getattr(args, "_cli_random_layers", False):
            args.random_layers = True
        if ab.get("noise_activations") and not getattr(args, "_cli_noise_activations", False):
            args.noise_activations = True

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
    parser.add_argument("--config", nargs="+", default=None,
                        help="YAML config file(s). Multiple configs are merged left-to-right (later overrides earlier)")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl",
                        help="Path to corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager"],
                        help="Transformer attention backend")

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

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-items-per-eval", type=int, default=10,
                        help="Maximum items per detection eval")
    parser.add_argument("--eval-max-new-tokens", type=int, default=32,
                        help="Default max_new_tokens for detection eval generation")
    parser.add_argument("--task-eval-max-new-tokens", type=int, default=64,
                        help="Default max_new_tokens for task-level eval generation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--stride", type=str, default=None, help="Stride for position extraction (int or 'punctuation'). Must be set via config or CLI.")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of activation layers (evenly spaced through model depth)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Explicit layer indices (overrides --n-layers). E.g. --layers 9 18 27")
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
    parser.add_argument("--task-order", choices=["shuffled", "sequential", "interleaved"], default="shuffled",
                        help="'shuffled' mixes all tasks; 'sequential' trains one at a time; 'interleaved' round-robin blocks")
    parser.add_argument("--interleave-blocks", type=int, default=50,
                        help="Number of blocks for interleaved mode (default 50)")
    parser.add_argument("--length-bucketing", action="store_true", default=True,
                        help="Enable windowed context-length bucketing in shuffled mode")
    parser.add_argument("--no-length-bucketing", dest="length_bucketing", action="store_false")
    parser.add_argument("--length-bucket-window-batches", type=int, default=32,
                        help="Window size for length bucketing, in units of train batches")
    parser.add_argument("--effective-batch-size", type=int, default=32,
                        help="Total effective batch size (invariant to GPU count). "
                             "gradient_accumulation_steps = effective_batch_size / (batch_size * world_size)")
    parser.add_argument("--max-train-tokens-per-gpu", type=int, default=0,
                        help="Approximate per-GPU peak token budget for splitting long train batches (0 disables)")
    parser.add_argument("--torch-compile", action="store_true", default=False,
                        help="Compile the training forward path with torch.compile")
    parser.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], default="default",
                        help="torch.compile mode")

    # Text-only baseline
    parser.add_argument("--no-activations", action="store_true", default=False,
                        help="Text-only baseline: train without activation steering (same data, no prefix/injection)")

    # FineWeb context prediction
    parser.add_argument("--fineweb-n", type=int, default=0,
                        help="Number of FineWeb context prediction examples (0 = disabled, set via config)")
    parser.add_argument("--fineweb-max-context-tokens", type=int, default=2000,
                        help="Max context tokens for FineWeb activation extraction")

    # Flamingo cross-attention
    parser.add_argument("--flamingo", action="store_true", default=False,
                        help="Use Flamingo-style gated cross-attention instead of additive steering")
    parser.add_argument("--flamingo-xattn-interval", type=int, default=4,
                        help="Insert cross-attention every N transformer blocks")
    parser.add_argument("--flamingo-xattn-lora-r", type=int, default=64,
                        help="LoRA rank for cross-attention projections")
    parser.add_argument("--flamingo-max-ctx-tokens", type=int, default=2048,
                        help="Max context tokens for flamingo activation extraction (truncates from left)")

    # Layer dropout
    parser.add_argument("--layer-dropout", action="store_true", default=False,
                        help="Random non-empty subsets of configured layers per training example")
    parser.add_argument("--no-layer-dropout", dest="layer_dropout", action="store_false",
                        help="Disable layer dropout (override config)")

    # Ablations
    parser.add_argument("--random-layers", action="store_true", default=False,
                        help="Randomize layer count and indices per training sequence")
    parser.add_argument("--noise-activations", action="store_true", default=False,
                        help="Replace real activations with variance-matched Gaussian noise")

    # Eval / save
    parser.add_argument("--eval-steps", type=int, default=2000,
                        help="Run evals every N steps (shuffled mode)")
    parser.add_argument("--save-steps", type=int, default=10000,
                        help="Save checkpoint every N steps (shuffled mode)")
    parser.add_argument("--no-step0-eval", action="store_true", default=False,
                        help="Skip evals at step 0 (for quick ablation launches)")
    parser.add_argument("--pooling-mode", type=str, default="none",
                        choices=["none", "windows", "single", "chunks5"],
                        help="Activation pooling: none, windows (mean-pool token windows), single (1/layer), chunks5 (5/layer)")
    parser.add_argument("--layer-pool", action="store_true", default=False,
                        help="Average activations across all MULTI_LAYERS at each position (output=1 effective layer)")
    parser.add_argument("--sparse-positions", action="store_true", default=False,
                        help="Randomly subsample CoT positions per example (trains on sparse evidence)")
    parser.add_argument("--single-position", action="store_true", default=False,
                        help="Only feed the last CoT position per layer (single activation ablation)")
    parser.add_argument("--rot13-start-step", type=int, default=2000)
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

    # Mutual exclusion and no-activations setup
    assert not (getattr(args, "flamingo", False) and getattr(args, "no_activations", False)), \
        "--flamingo and --no-activations are mutually exclusive"
    if getattr(args, "no_activations", False):
        args.fresh_lora = True

    # Validate stride is set
    if args.stride is None:
        raise ValueError(
            "stride must be set via config (activations.stride) or CLI (--stride). "
            "Use an integer for fixed-stride or 'punctuation' for punctuation-based extraction."
        )

    set_seed(args.seed)

    # Multi-layer config
    global MULTI_LAYERS, NO_ACTIVATIONS, RANDOM_LAYERS, LAYER_DROPOUT, NOISE_ACTIVATIONS, _MODEL_N_LAYERS
    NO_ACTIVATIONS = getattr(args, "no_activations", False)
    RANDOM_LAYERS = getattr(args, "random_layers", False)
    LAYER_DROPOUT = getattr(args, "layer_dropout", False)
    NOISE_ACTIVATIONS = getattr(args, "noise_activations", False)
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
        if NOISE_ACTIVATIONS:
            print("Ablation: NOISE ACTIVATIONS (variance-matched Gaussian noise)")

    # Position encoding config
    global _PE_CONFIG
    _PE_CONFIG["enabled"] = getattr(args, "position_encoding", False)
    _PE_CONFIG["alpha"] = getattr(args, "pe_alpha", 0.1)
    if rank == 0:
        if _PE_CONFIG["enabled"]:
            print(f"Position encoding: ON (alpha={_PE_CONFIG['alpha']})")
        else:
            print("Position encoding: OFF")

    # Pooling mode
    global _POOLING_MODE
    _POOLING_MODE = getattr(args, "pooling_mode", "none")
    if _POOLING_MODE != "none":
        import evals.activation_cache as _cache_module
        _cache_module._POOLING_MODE = _POOLING_MODE
    # Sparse position sampling
    global _SPARSE_POSITIONS
    _SPARSE_POSITIONS = getattr(args, "sparse_positions", False)
    # Single position mode
    global _SINGLE_POSITION
    _SINGLE_POSITION = getattr(args, "single_position", False)
    if rank == 0:
        print(f"Activation stride: {args.stride}")
        print(f"Activation pooling: {_POOLING_MODE}")
        if _SPARSE_POSITIONS:
            print("Sparse position sampling: ON")
        if _SINGLE_POSITION:
            print("Single position mode: ON (last CoT position only)")

    tokenizer = load_tokenizer(args.model)

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
    if args.flamingo:
        # Flamingo: frozen base model + cross-attention LoRA only (no self-attn LoRA)
        from flamingo_oracle import FlamingoOracleWrapper
        if rank == 0:
            print("Flamingo mode: freezing base model (no self-attention LoRA)")
        for p in base_model.parameters():
            p.requires_grad = False
        xattn_lora_r = args.flamingo_xattn_lora_r
        if rank == 0:
            print(f"Wrapping with Flamingo cross-attention (interval={args.flamingo_xattn_interval}, lora_r={xattn_lora_r})")
        model = FlamingoOracleWrapper(
            base_model, base_model.config,
            xattn_interval=args.flamingo_xattn_interval,
            lora_r=xattn_lora_r, lora_alpha=xattn_lora_r * 2,
        )
        if args.resume_from:
            flamingo_path = Path(args.resume_from) / "flamingo_modules.pt"
            if flamingo_path.exists():
                model.load_flamingo_modules(str(args.resume_from))
                if rank == 0:
                    print(f"  Loaded Flamingo modules from {args.resume_from}")
            state_path = Path(args.resume_from) / "training_state.pt"
            if state_path.exists():
                _resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
                if not getattr(args, "_cli_start_step", False):
                    args.start_step = _resume_state["global_step"]
                if rank == 0:
                    print(f"  Loaded training_state.pt: step={_resume_state['global_step']}, wandb_id={_resume_state.get('wandb_run_id')}")
    elif args.resume_from:
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
        if args.flamingo:
            if rank == 0:
                print(f"Compiling Flamingo base model with torch.compile (mode={args.torch_compile_mode})")
            model.base_model = torch.compile(model.base_model, fullgraph=False, mode=compile_mode)
        else:
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
        if n > 0:
            task_config[task_name] = {"n": n}

    # FutureLens uses corpus-v5 + tokenizer — handle separately (like FineWeb)
    futurelens_n = task_config.pop("futurelens", {}).get("n", 0)

    raw_data = load_all_training_data(task_config)

    # FutureLens (corpus-based, needs tokenizer)
    if futurelens_n > 0:
        from data_loading import load_futurelens_data
        if rank == 0:
            print(f"  [data] Generating {futurelens_n} FutureLens examples from corpus...")
        futurelens_data = load_futurelens_data(
            tokenizer=tokenizer,
            n=futurelens_n,
            split="train",
            layers=MULTI_LAYERS,
            seed=args.seed,
        )
        raw_data.extend(futurelens_data)
        if rank == 0:
            print(f"  [data]   -> {len(futurelens_data)} FutureLens examples added (total: {len(raw_data)})")

    # FineWeb context prediction (PastLens-style, if enabled)
    fineweb_n = getattr(args, "fineweb_n", 0)
    if fineweb_n > 0:
        from data_loading import load_fineweb_data
        if rank == 0:
            print(f"  [data] Generating {fineweb_n} FineWeb context prediction examples...")
        fw_stride = int(args.stride) if args.stride and args.stride != "punctuation" else 5
        fineweb_data = load_fineweb_data(
            tokenizer=tokenizer,
            model_name=args.model,
            n=fineweb_n,
            max_context_tokens=getattr(args, "fineweb_max_context_tokens", 2000),
            stride=fw_stride,
            layers=MULTI_LAYERS,
            seed=args.seed,
        )
        raw_data.extend(fineweb_data)
        if rank == 0:
            print(f"  [data]   -> {len(fineweb_data)} FineWeb examples added (total: {len(raw_data)})")

    if not raw_data:
        if rank == 0:
            print("ERROR: No training data loaded!")
        cleanup_distributed()
        return

    # Tokenize cot_text → context_input_ids for items that don't have them yet
    from data_loading import prepare_context_ids
    stride_val = int(args.stride) if args.stride and args.stride != "punctuation" else 5
    prepare_context_ids(
        raw_data, tokenizer,
        stride=stride_val,
        layers=MULTI_LAYERS,
    )

    # Drop items that still lack context_input_ids (e.g. empty cot_text)
    before = len(raw_data)
    raw_data = [d for d in raw_data if d.get("context_input_ids")]
    if len(raw_data) < before and rank == 0:
        print(f"  [data] Dropped {before - len(raw_data)} items without context_input_ids")

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
