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
import sys

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
    """Materialize steering vectors from MULTI_LAYERS (configurable via --n-layers)."""
    N_LAYERS = len(MULTI_LAYERS)

    to_fill = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    layers = list(MULTI_LAYERS)

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


def construct_parallel_batch(
    batch_list: list[TrainingDataPoint],
    tokenizer,
    device: torch.device,
    max_ctx_tokens: int | None = None,
) -> dict:
    """Construct a parallel batch: CoT and oracle as separate batch items.

    Returns [2B, L_max] where first B items are CoT, last B items are oracle.
    Both left-padded to L_max = max(max_cot_len, max_oracle_len).
    CoT labels are all -100. Only oracle items contribute to loss.
    """
    pad_id = tokenizer.pad_token_id
    B = len(batch_list)

    cots = []
    for dp in batch_list:
        assert dp.context_input_ids is not None
        ctx = list(dp.context_input_ids)
        if max_ctx_tokens and len(ctx) > max_ctx_tokens:
            ctx = ctx[-max_ctx_tokens:]
        cots.append(ctx)

    max_cot_len = max(len(c) for c in cots)
    max_oracle_len = max(len(dp.input_ids) for dp in batch_list)
    L_max = max(max_cot_len, max_oracle_len)

    # CoT items: left-padded to L_max, labels all -100
    cot_ids, cot_masks, cot_kv_masks = [], [], []
    for c in cots:
        pad = L_max - len(c)
        cot_ids.append(torch.tensor([pad_id] * pad + c, dtype=torch.long))
        cot_masks.append(torch.tensor([False] * pad + [True] * len(c), dtype=torch.bool))
        cot_kv_masks.append(torch.tensor([False] * pad + [True] * len(c), dtype=torch.bool))
    cot_labels = [torch.full((L_max,), -100, dtype=torch.long) for _ in range(B)]

    # Oracle items: left-padded to L_max
    oracle_ids, oracle_masks, oracle_labels = [], [], []
    for dp in batch_list:
        pad = L_max - len(dp.input_ids)
        oracle_ids.append(torch.tensor([pad_id] * pad + dp.input_ids, dtype=torch.long))
        oracle_masks.append(torch.tensor([False] * pad + [True] * len(dp.input_ids), dtype=torch.bool))
        oracle_labels.append(torch.tensor([-100] * pad + dp.labels, dtype=torch.long))

    # Stack: [cot_0..cot_{B-1}, oracle_0..oracle_{B-1}]
    return {
        "input_ids": torch.stack(cot_ids + oracle_ids).to(device),
        "attention_mask": torch.stack(cot_masks + oracle_masks).to(device),
        "labels": torch.stack(cot_labels + oracle_labels).to(device),
        "parallel_B": B,
        "cot_mask": torch.stack(cot_kv_masks).to(device),
    }


# ── Data conversion ──
def dicts_to_training_data(
    raw_data: list[dict], tokenizer,
) -> list[TrainingDataPoint]:
    training_data = []
    n_layers_runtime = len(MULTI_LAYERS) if MULTI_LAYERS else 3
    _reexpand_warned = False

    for item in raw_data:
        ctx_pos = item["context_positions"]
        num_pos = item["num_positions"]

        # Re-expand positions if precomputed with fewer layers than runtime config.
        # Precomputed data stores context_positions = base_positions * n_layers_data.
        # We detect mismatch by checking if total_positions is divisible by the
        # runtime layer count.  If not, infer the original layer count, extract
        # base positions, and re-expand.
        if n_layers_runtime > 1 and len(ctx_pos) % n_layers_runtime != 0:
            # Try common old layer counts (3 is the most common)
            for old_n in [3, 1, 2, 4, 5, 6]:
                if len(ctx_pos) % old_n == 0:
                    k = len(ctx_pos) // old_n
                    base_positions = ctx_pos[:k]
                    ctx_pos = base_positions * n_layers_runtime
                    num_pos = len(ctx_pos)
                    if not _reexpand_warned:
                        print(f"  [data] Re-expanding positions: {old_n} layers -> {n_layers_runtime} layers (K={k})")
                        _reexpand_warned = True
                    break

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
            meta_info={"prompt": item["prompt"]},
        )
        training_data.append(dp)

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
    "answer_trajectory": {
        "arg": "answer_trajectory_n",
        "module": None,  # precompute-only (requires vLLM)
        "loader": None,
        "corpus": "main",
    },
    "atypical_answer": {
        "arg": "atypical_answer_n",
        "module": "dataset_classes.cot_atypical_answer",
        "loader": "load_cot_atypical_answer_data",
        "corpus": "atypical",  # loads from its own JSONL, not main corpus
    },
    "prompt_inversion": {
        "arg": "prompt_inversion_n",
        "module": "dataset_classes.cot_prompt_inversion",
        "loader": "load_cot_prompt_inversion_data",
        "corpus": "main",
    },
    "compqa": {
        "arg": "compqa_n",
        "module": None,
        "loader": None,
        "corpus": "compqa",  # precompute-only (needs compqa_raw.json)
    },
    "hint_admission": {
        "arg": "hint_admission_n",
        "module": "dataset_classes.cot_hint_admission",
        "loader": "load_cot_hint_admission_data",
        "corpus": "hint_admission",
    },
    "hinted_answer_pred": {
        "arg": "hinted_answer_pred_n",
        "module": "dataset_classes.cot_hinted_answer_pred",
        "loader": "load_cot_hinted_answer_pred_data",
        "corpus": "hint_admission",  # uses same HF dataset
    },
    "cotqa": {
        "arg": "cotqa_n",
        "module": "dataset_classes.cot_cotqa",
        "loader": "load_cot_cotqa_data",
        "corpus": "cotqa",  # loads from HF directly
    },
}


HF_TRAINING_REPO = "mats-10-sprint-cs-jb/cot-oracle-training-v6"

_HF_CACHE_DIR = Path(os.path.join(os.environ["CACHE_DIR"], "cot_oracle", ".hf_cache")) if os.environ.get("CACHE_DIR") else Path("data/.hf_cache")


def _resolve_hf_dataset(path_or_id: str) -> str:
    """Resolve an HF dataset ID to a local JSONL path; local paths returned as-is.

    HF IDs look like 'org/dataset-name' (has '/', no file extension).
    Downloads the dataset and exports to cached JSONL.
    """
    # Local path — return as-is
    if os.path.sep in path_or_id and os.path.exists(path_or_id):
        return path_or_id
    if "." in path_or_id.split("/")[-1]:  # has extension → local file
        return path_or_id

    # Looks like HF ID (has '/', no extension)
    if "/" not in path_or_id:
        return path_or_id  # bare name, treat as local

    cache_name = path_or_id.replace("/", "__") + ".jsonl"
    cache_path = _HF_CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"  [HF] Using cached: {cache_path}")
        return str(cache_path)

    print(f"  [HF] Downloading dataset: {path_or_id}")
    from datasets import load_dataset
    ds = load_dataset(path_or_id, split="train")
    _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(cache_path))
    print(f"  [HF] Cached {len(ds)} rows → {cache_path}")
    return str(cache_path)


def _download_precomputed_from_hf(task_name: str, pdir: Path) -> Path:
    """Download a precomputed JSONL file from HuggingFace."""
    from huggingface_hub import hf_hub_download

    filename = f"{task_name}.jsonl"
    print(f"  [train] Downloading {filename} from HuggingFace: {HF_TRAINING_REPO}")
    pdir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HF_TRAINING_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(pdir),
    )
    return Path(local_path)


def _live_load_task(task_name: str, info: dict, n: int, args, tokenizer) -> list[dict]:
    """Load a single task via its dataset loader (live, not precomputed)."""
    import importlib

    mod = importlib.import_module(info["module"])
    loader_fn = getattr(mod, info["loader"])

    if info["corpus"] == "atypical":
        atypical_path = getattr(args, "atypical_data_path",
                                "data/atypical_answer_training.jsonl")
        return loader_fn(
            atypical_path, tokenizer, args.model,
            num_examples=n, stride=args.stride,
            atypical_data_path=atypical_path,
        )
    elif info["corpus"] == "compqa":
        compqa_cache = str(Path(getattr(args, "precomputed_dir", "data/precomputed")) / "compqa_raw.json")
        return loader_fn(
            compqa_cache, tokenizer, args.model,
            num_examples=n, stride=args.stride,
        )
    elif info["corpus"] == "hint_admission":
        hint_path = getattr(args, "hint_admission_data_path",
                            "data/hint_admission_training.jsonl")
        return loader_fn(
            hint_path, tokenizer, args.model,
            num_examples=n, stride=args.stride,
            hint_admission_data_path=hint_path,
        )
    elif info["corpus"] == "cotqa":
        return loader_fn(
            "", tokenizer, args.model,
            num_examples=n, stride=args.stride,
        )
    else:
        return loader_fn(
            args.corpus, tokenizer, args.model,
            num_examples=n, stride=args.stride,
        )


def load_precomputed_tasks(precomputed_dir: str, args, tokenizer=None) -> list[dict]:
    """Load training data from precomputed JSONL files, downloading from HF if needed.

    Falls back to live loading for tasks whose JSONL isn't on HF.
    """
    pdir = Path(precomputed_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    all_data = []
    enabled = []

    for task_name, info in TASK_REGISTRY.items():
        n = getattr(args, info["arg"], 0)
        if n <= 0:
            continue

        jsonl_path = pdir / f"{task_name}.jsonl"
        if not jsonl_path.exists():
            try:
                jsonl_path = _download_precomputed_from_hf(task_name, pdir)
            except Exception as e:
                if tokenizer is not None:
                    print(f"  [fallback] No precomputed {task_name} on HF ({e}), live-loading from corpus...")
                    data = _live_load_task(task_name, info, n, args, tokenizer)
                    all_data.extend(data)
                    enabled.append(f"{task_name}({len(data)},live)")
                    print(f"    -> {len(data)} examples (live)")
                    continue
                else:
                    raise RuntimeError(
                        f"Task {task_name} (n={n}) has no precomputed data on HF and no tokenizer "
                        f"for live fallback. Either set n=0 to disable or provide a tokenizer."
                    ) from e

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

        if info["module"] is None:
            print(f"    Skipping {task_name} (precompute-only, no live loader)")
            continue

        mod = importlib.import_module(info["module"])
        loader_fn = getattr(mod, info["loader"])

        if info["corpus"] == "atypical":
            atypical_path = args.atypical_data_path
            data = loader_fn(
                atypical_path, tokenizer, args.model,
                num_examples=n,
                stride=args.stride,
                atypical_data_path=atypical_path,
            )
        elif info["corpus"] == "compqa":
            compqa_cache = str(Path(getattr(args, "precomputed_dir", "data/precomputed")) / "compqa_raw.json")
            data = loader_fn(
                compqa_cache, tokenizer, args.model,
                num_examples=n,
                stride=args.stride,
            )
        elif info["corpus"] == "hint_admission":
            hint_path = getattr(args, "hint_admission_data_path",
                                "data/hint_admission_training.jsonl")
            data = loader_fn(
                hint_path, tokenizer, args.model,
                num_examples=n,
                stride=args.stride,
                hint_admission_data_path=hint_path,
            )
        elif info["corpus"] == "cotqa":
            data = loader_fn(
                "", tokenizer, args.model,
                num_examples=n,
                stride=args.stride,
            )
        else:
            data = loader_fn(
                args.corpus, tokenizer, args.model,
                num_examples=n,
                stride=args.stride,
            )

        all_data.extend(data)
        print(f"    -> {len(data)} examples loaded")

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


def _run_unified_eval(model, tokenizer, model_name, global_step, args, eval_datasets, log_dir=None):
    """Run all evals (task + detection) in a single call."""
    global _GEMINI_BASELINES
    from evals.training_eval_hook import run_training_evals
    import wandb

    print(f"\n--- Evals at step {global_step} ---")
    eval_start = time.time()
    metrics = run_training_evals(
        model, tokenizer, model_name=model_name,
        step=global_step, device="cuda",
        eval_dir=args.eval_dir,
        max_items_per_eval=25,
        skip_rot13=(global_step < args.rot13_start_step),
        oracle_adapter_name="default",
        activation_cache_dir=args.activation_cache_dir,
        log_dir=log_dir,
        eval_names=getattr(args, "evals", None),
        stride=args.stride,
        eval_batch_size=args.eval_batch_size,
        task_eval_datasets=eval_datasets,
    )
    elapsed = time.time() - eval_start
    if metrics:
        wandb.log(metrics, step=global_step)
        # Print oracle vs Gemini baseline comparison table
        if _GEMINI_BASELINES is None:
            _GEMINI_BASELINES = _load_gemini_baselines()
        oracle_scores = {}
        for k, v in metrics.items():
            if k.endswith("_acc") and isinstance(v, float):
                name = k.removeprefix("eval/").removesuffix("_acc")
                oracle_scores[name] = ("acc", v)
            elif k.endswith("_token_f1") and isinstance(v, float):
                name = k.removeprefix("eval/").removesuffix("_token_f1")
                oracle_scores[name] = ("token_f1", v)
        if oracle_scores:
            print(f"\n  {'Eval':<30s} {'Metric':<10s} {'Oracle':>8s} {'Gemini':>8s} {'Δ':>8s}")
            print(f"  {'-'*66}")
            for name, (metric, val) in sorted(oracle_scores.items()):
                gem = _GEMINI_BASELINES.get(name)
                if gem and gem[0] == metric:
                    delta = val - gem[1]
                    print(f"  {name:<30s} {metric:<10s} {val:>8.3f} {gem[1]:>8.3f} {delta:>+8.3f}")
                else:
                    print(f"  {name:<30s} {metric:<10s} {val:>8.3f} {'—':>8s} {'':>8s}")
            # Mean acc comparison
            mean_acc = metrics.get("eval/mean_acc")
            if mean_acc is not None and _GEMINI_BASELINES:
                gem_accs = [v for _, (m, v) in _GEMINI_BASELINES.items() if m == "acc"]
                if gem_accs:
                    gem_mean = sum(gem_accs) / len(gem_accs)
                    print(f"  {'MEAN':<30s} {'acc':<10s} {mean_acc:>8.3f} {gem_mean:>8.3f} {mean_acc - gem_mean:>+8.3f}")
            print()
    print(f"  Eval took {elapsed:.1f}s")
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

    task_stage_idx = {}  # task_name -> int index (for wandb logging)

    if task_order == "sequential":
        # Group by task, respect YAML ordering for curriculum
        yaml_task_names = getattr(args, "_yaml_task_order", [])
        # Map YAML task name -> datapoint_type (module name, or "cot_{name}" for precompute-only)
        yaml_to_dtype = {k: v["module"].split(".")[-1] if v["module"] else f"cot_{k}" for k, v in TASK_REGISTRY.items()}
        # Order: YAML order first, then any remaining types alphabetically
        ordered_types = []
        for yt in yaml_task_names:
            dt = yaml_to_dtype.get(yt)
            if dt and dt in train_per_type:
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
    else:
        random.shuffle(final_training)
        if rank == 0:
            print(f"  Task order: SHUFFLED")

    # Data sharding for multi-GPU
    if world_size > 1:
        aligned = (len(final_training) // (args.batch_size * world_size)) * (args.batch_size * world_size)
        final_training = final_training[:aligned]
        final_training = final_training[rank::world_size]

    eval_datasets = eval_per_type
    if rank == 0:
        print(f"  Training: {len(final_training)}, Eval: {sum(len(v) for v in eval_datasets.values())}")
        print(f"  Eval tasks: {', '.join(sorted(eval_datasets.keys()))}")

    # Pre-materialize eval steering vectors once (avoids re-extraction every eval call)
    # Skip for Flamingo mode — eval uses cross-attention, not steering
    if rank == 0 and not args.flamingo:
        print(f"\n  Pre-materializing eval steering vectors...")
        mat_start = time.time()
        mat_batch_size = 8
        all_eval_items = [dp for dps in eval_datasets.values() for dp in dps]
        for i in range(0, len(all_eval_items), mat_batch_size):
            batch = all_eval_items[i : i + mat_batch_size]
            materialized = materialize_multilayer_steering_vectors(batch, tokenizer, model)
            for orig, mat in zip(batch, materialized):
                orig.steering_vectors = mat.steering_vectors
        print(f"  Pre-materialized {len(all_eval_items)} eval items in {time.time() - mat_start:.1f}s")
    if world_size > 1:
        dist.barrier()

    # Optimizer + scheduler
    num_batches = len(final_training) // args.batch_size
    total_steps = (num_batches // grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    # Precompute per-stage step boundaries for progress tracking
    stage_step_ranges = {}  # task_name -> (start_step, end_step)
    if task_order == "sequential" and task_blocks:
        cursor = 0
        for task_name, items in task_blocks:
            stage_steps = len(items) // (args.batch_size * grad_accum)
            stage_step_ranges[task_name] = (cursor, cursor + stage_steps)
            cursor += stage_steps

    # Dynamic eval/save cadence: ~10 evals over the relevant span
    if task_order == "sequential":
        reference_steps = max(len(items) // (args.batch_size * grad_accum) for items in train_per_type.values() if len(items) >= args.batch_size)
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
            "n_eval": sum(len(v) for v in eval_datasets.values()),
            "n_tasks": len(eval_datasets),
            "train/samples_seen": global_step * args.effective_batch_size,
        }, step=global_step)

        # Log task index legend to wandb config
        wandb.config.update({"task_index_legend": task_to_idx}, allow_val_change=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "eval_logs"

    if rank == 0:
        print(f"\n  LR: {args.lr}")
        print(f"  Batch: {args.batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {args.batch_size * grad_accum * world_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Steps: {total_steps}")
        print(f"  Warmup: {warmup_steps}")

    model.train()

    # Step-0 eval (baseline before any training)
    # Skip all evals in Flamingo mode — evals use steering injection, not cross-attention
    skip_step0 = getattr(args, "no_step0_eval", False)
    if global_step == 0 and rank == 0 and not skip_step0 and not args.flamingo:
        _run_unified_eval(eval_model, tokenizer, args.model, 0, args, eval_datasets, log_dir=log_dir)
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
        if task_order != "sequential":
            random.shuffle(final_training)
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
        accum_context_lengths = []

        for start in pbar:
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break

            batch_types = [dp.datapoint_type for dp in batch_list]

            if args.flamingo:
                # Parallel: CoT + oracle as separate batch items, connected via xattn
                batch = construct_parallel_batch(
                    batch_list, tokenizer, device,
                    max_ctx_tokens=args.flamingo_max_ctx_tokens,
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = ddp_model(**batch)
                    loss = outputs.loss / grad_accum
                loss.backward()
            else:
                # Standard steering path
                batch_list = materialize_multilayer_steering_vectors(
                    batch_list, tokenizer, model
                )
                batch = construct_batch(batch_list, tokenizer, device)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = train_features_batch(
                        batch, ddp_model, submodule,
                        args.steering_coefficient, device, dtype,
                    )
                    loss = outputs.loss / grad_accum
                loss.backward()

            # Per-task loss (use unscaled loss for logging)
            with torch.no_grad():
                logits = outputs.logits.detach()
                labels = batch["labels"] if isinstance(batch, dict) else batch.labels
                # Parallel flamingo: logits are oracle-only [B], labels are [2B] — slice to oracle
                if args.flamingo and "parallel_B" in batch:
                    labels = labels[batch["parallel_B"]:]
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

            for i, task_type in enumerate(batch_types):
                accum_task_losses[task_type].append(per_item_loss[i].item())
            accum_loss_sum += outputs.loss.item()
            accum_batch_types.extend(batch_types)
            accum_batch_tokens += batch_tokens
            accum_context_lengths.extend(len(dp.context_input_ids) for dp in batch_list if dp.context_input_ids is not None)

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

            # Phase checkpoint: save when dominant task changes in sequential mode
            if rank == 0 and task_order == "sequential" and prev_dominant_task is not None and dominant_task != prev_dominant_task:
                ckpt_path = save_dir / f"step_{global_step}_phase_{prev_dominant_task}"
                print(f"\n  Phase transition: {prev_dominant_task} -> {dominant_task}")
                print(f"  Saving phase checkpoint to {ckpt_path}")
                model.save_pretrained(str(ckpt_path))
                _save_training_state(ckpt_path, global_step, optimizer, scheduler)
            if world_size > 1 and task_order == "sequential" and prev_dominant_task is not None and dominant_task != prev_dominant_task:
                dist.barrier()
            prev_dominant_task = dominant_task

            # Unified eval (task + detection, rank 0 only)
            if global_step > 0 and global_step % args.eval_steps == 0 and not args.flamingo:
                if rank == 0:
                    _, elapsed = _run_unified_eval(eval_model, tokenizer, args.model, global_step, args, eval_datasets, log_dir=log_dir)
                    eval_time_total += elapsed
                    model.train()
                if world_size > 1:
                    dist.barrier()

            # Save checkpoint (rank 0 only)
            if global_step > 0 and global_step % args.save_steps == 0:
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
            accum_context_lengths = []

            global_step += 1

    # Final eval (rank 0 only)
    if rank == 0 and not args.flamingo:
        _run_unified_eval(eval_model, tokenizer, args.model, global_step, args, eval_datasets, log_dir=log_dir)

        # Save final
        final_path = save_dir / "final"
        print(f"  Saving final checkpoint to {final_path}")
        model.save_pretrained(str(final_path))
        _save_training_state(final_path, global_step, optimizer, scheduler)

    if world_size > 1:
        dist.barrier()

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
        # Preserve YAML ordering for sequential curriculum
        args._yaml_task_order = list(config["tasks"].keys())
        for task_name, task_cfg in config["tasks"].items():
            arg_name = f"{task_name}_n"
            if hasattr(args, arg_name) and getattr(args, f"_cli_{arg_name}", False) is False:
                setattr(args, arg_name, task_cfg.get("n", 0))

    # Training params
    if "training" in config:
        t = config["training"]
        _float_keys = {"lr", "warmup_fraction", "max_grad_norm", "steering_coefficient"}
        _int_keys = {"batch_size", "eval_batch_size", "epochs", "seed", "effective_batch_size"}
        for key in ["lr", "batch_size", "eval_batch_size", "epochs",
                     "warmup_fraction", "max_grad_norm", "steering_coefficient",
                     "gradient_checkpointing", "task_order", "seed",
                     "effective_batch_size"]:
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
        for key in ["stride", "position_encoding", "pe_alpha", "n_layers"]:
            if key in a and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, a[key])
        if "layers" in a and not getattr(args, "_cli_layers", False):
            args.layers = a["layers"]  # list of ints, e.g. [9, 18, 27]

    # Eval
    if "eval" in config:
        e = config["eval"]
        for key in ["eval_dir", "rot13_start_step", "eval_steps", "save_steps"]:
            if key in e and not getattr(args, f"_cli_{key}", False):
                val = e[key]
                if key in {"rot13_start_step", "eval_steps", "save_steps"}:
                    val = int(val)
                setattr(args, key, val)
        if "evals" in e and not getattr(args, "_cli_evals", False):
            raw_evals = e["evals"]
            eval_names = []
            eval_baselines = {}
            for entry in raw_evals:
                if isinstance(entry, str):
                    eval_names.append(entry)
                elif isinstance(entry, dict):
                    name = list(entry.keys())[0]
                    eval_names.append(name)
                    if isinstance(entry[name], dict) and "baselines" in entry[name]:
                        eval_baselines[name] = entry[name]["baselines"]
            args.evals = eval_names
            args.eval_baselines = eval_baselines

    # Data paths
    if "data" in config:
        d = config["data"]
        if "corpus" in d and not getattr(args, "_cli_corpus", False):
            args.corpus = d["corpus"]
        if "precomputed_dir" in d and not getattr(args, "_cli_precomputed_dir", False):
            args.precomputed_dir = d["precomputed_dir"]
        if "activation_cache_dir" in d and not getattr(args, "_cli_activation_cache_dir", False):
            args.activation_cache_dir = d["activation_cache_dir"]
        if "atypical_data_path" in d and not getattr(args, "_cli_atypical_data_path", False):
            args.atypical_data_path = d["atypical_data_path"]
        if "hint_admission_data_path" in d and not getattr(args, "_cli_hint_admission_data_path", False):
            args.hint_admission_data_path = d["hint_admission_data_path"]

    # Flamingo
    if "flamingo" in config:
        f = config["flamingo"]
        if f.get("enabled") and not getattr(args, "_cli_flamingo", False):
            args.flamingo = True
        if "xattn_interval" in f and not getattr(args, "_cli_flamingo_xattn_interval", False):
            args.flamingo_xattn_interval = int(f["xattn_interval"])
        if "xattn_lora_r" in f and not getattr(args, "_cli_flamingo_xattn_lora_r", False):
            args.flamingo_xattn_lora_r = int(f["xattn_lora_r"])

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
    local_rank, rank, world_size = setup_distributed()

    parser = argparse.ArgumentParser(description="Train CoT Oracle")
    parser.add_argument("--config", nargs="+", default=None,
                        help="YAML config file(s). Multiple configs are merged left-to-right (later overrides earlier)")
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

    # Per-task example counts — defaults are 0; set via --config (train.yaml is source of truth)
    parser.add_argument("--full-recon-n", type=int, default=0)
    parser.add_argument("--next-step-n", type=int, default=0)
    parser.add_argument("--answer-pred-n", type=int, default=0)
    parser.add_argument("--correctness-n", type=int, default=0)
    parser.add_argument("--decorative-n", type=int, default=0)
    parser.add_argument("--domain-n", type=int, default=0)
    parser.add_argument("--reasoning-term-n", type=int, default=0)
    parser.add_argument("--partial-answer-n", type=int, default=0)
    parser.add_argument("--answer-trajectory-n", type=int, default=0)
    parser.add_argument("--atypical-answer-n", type=int, default=0)
    parser.add_argument("--prompt-inversion-n", type=int, default=0)
    parser.add_argument("--compqa-n", type=int, default=0)
    parser.add_argument("--hint-admission-n", type=int, default=0)
    parser.add_argument("--atypical-data-path",
                        default="data/atypical_answer_training.jsonl",
                        help="Path to atypical answer JSONL (from precompute_atypical_training.py)")
    parser.add_argument("--hint-admission-data-path",
                        default="data/hint_admission_training.jsonl",
                        help="Path to hint admission JSONL (from precompute_hint_admission.py)")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
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
    parser.add_argument("--task-order", choices=["shuffled", "sequential"], default="shuffled",
                        help="'shuffled' mixes all tasks; 'sequential' trains tasks one at a time")
    parser.add_argument("--effective-batch-size", type=int, default=32,
                        help="Total effective batch size (invariant to GPU count). "
                             "gradient_accumulation_steps = effective_batch_size / (batch_size * world_size)")

    # Flamingo cross-attention
    parser.add_argument("--flamingo", action="store_true", default=False,
                        help="Use Flamingo-style gated cross-attention instead of additive steering")
    parser.add_argument("--flamingo-xattn-interval", type=int, default=4,
                        help="Insert cross-attention every N transformer blocks")
    parser.add_argument("--flamingo-xattn-lora-r", type=int, default=64,
                        help="LoRA rank for cross-attention projections")
    parser.add_argument("--flamingo-max-ctx-tokens", type=int, default=2048,
                        help="Max context tokens for flamingo activation extraction (truncates from left)")

    # Eval / save
    parser.add_argument("--eval-steps", type=int, default=2000,
                        help="Run evals every N steps (shuffled mode)")
    parser.add_argument("--save-steps", type=int, default=10000,
                        help="Save checkpoint every N steps (shuffled mode)")
    parser.add_argument("--no-step0-eval", action="store_true", default=False,
                        help="Skip evals at step 0 (for quick ablation launches)")
    parser.add_argument("--rot13-start-step", type=int, default=2000)
    parser.add_argument("--start-step", type=int, default=None,
                        help="Starting global step (for resuming; 0 = restart data from beginning)")
    parser.add_argument("--eval-dir", default="data/evals")
    _default_act_cache = os.path.join(os.environ["FAST_CACHE_DIR"], "cot_oracle", "eval_precomputed") if os.environ.get("FAST_CACHE_DIR") else "data/eval_precomputed"
    parser.add_argument("--activation-cache-dir", default=_default_act_cache,
                        help="Dir with precomputed activation bundles (.pt)")

    # Output
    _default_save = os.path.join(os.environ["CACHE_DIR"], "cot_oracle", "checkpoints") if os.environ.get("CACHE_DIR") else "checkpoints"
    parser.add_argument("--save-dir", default=_default_save)
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

    # Mark which args were explicitly provided on CLI so config doesn't override them
    _defaults = {action.dest: action.default for action in parser._actions}
    for key, val in list(vars(args).items()):
        if key == "config":
            continue
        if val != _defaults.get(key):
            setattr(args, f"_cli_{key}", True)

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

    # Resolve HF dataset IDs to local paths
    if rank == 0:
        args.corpus = _resolve_hf_dataset(args.corpus)
        args.atypical_data_path = _resolve_hf_dataset(args.atypical_data_path)
        if hasattr(args, "hint_admission_data_path") and getattr(args, "hint_admission_n", 0) > 0:
            args.hint_admission_data_path = _resolve_hf_dataset(args.hint_admission_data_path)

    # Validate stride is set
    if args.stride is None:
        raise ValueError(
            "stride must be set via config (activations.stride) or CLI (--stride). "
            "Use an integer for fixed-stride or 'punctuation' for punctuation-based extraction."
        )

    set_seed(args.seed)

    # Multi-layer config
    global MULTI_LAYERS
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
        print(f"Multi-layer injection: {MULTI_LAYERS}")
        print(f"Distributed: world_size={world_size}, rank={rank}, local_rank={local_rank}")

    # Position encoding config
    global _PE_CONFIG
    _PE_CONFIG["enabled"] = getattr(args, "position_encoding", False)
    _PE_CONFIG["alpha"] = getattr(args, "pe_alpha", 0.1)
    if rank == 0:
        if _PE_CONFIG["enabled"]:
            print(f"Position encoding: ON (alpha={_PE_CONFIG['alpha']})")
        else:
            print("Position encoding: OFF")

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
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation="sdpa",  # O(n) memory; "eager" was O(n²) and OOMed at batch>8
    )
    base_model.enable_input_require_grads()

    if args.gradient_checkpointing:
        base_model.use_cache = False
        base_model.gradient_checkpointing_enable()

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
            print("Starting with FRESH LoRA")
        lora_config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    else:
        if rank == 0:
            print(f"Loading Adam's AO checkpoint: {args.ao_checkpoint}")
        model = PeftModel.from_pretrained(
            base_model, args.ao_checkpoint,
            is_trainable=True, autocast_adapter_dtype=False,
        )

    # Flamingo wrapper (after LoRA, before DDP)
    if args.flamingo:
        from flamingo_oracle import FlamingoOracleWrapper
        xattn_lora_r = args.flamingo_xattn_lora_r
        if rank == 0:
            print(f"\nWrapping with Flamingo cross-attention (interval={args.flamingo_xattn_interval}, lora_r={xattn_lora_r})")
        model = FlamingoOracleWrapper(
            model, base_model.config,
            xattn_interval=args.flamingo_xattn_interval,
            lora_r=xattn_lora_r, lora_alpha=xattn_lora_r * 2,
        )

        # Load Flamingo modules from checkpoint if resuming
        if args.resume_from:
            flamingo_path = Path(args.resume_from) / "flamingo_modules.pt"
            if flamingo_path.exists():
                model.load_flamingo_modules(str(args.resume_from))
                if rank == 0:
                    print(f"  Loaded Flamingo modules from {args.resume_from}")

        if rank == 0:
            model.print_trainable_parameters()

    # Ensure trainable params are fp32 (optimizer states stay fp32; autocast handles forward pass)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if rank == 0:
        if not args.flamingo:
            model.print_trainable_parameters()

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

    if args.precomputed_dir:
        if rank == 0:
            print(f"  Using precomputed data from {args.precomputed_dir} (auto-downloads from HF if needed)")
        raw_data = load_precomputed_tasks(args.precomputed_dir, args, tokenizer=tokenizer)
    else:
        raw_data = load_all_tasks(args, tokenizer)

    if not raw_data:
        if rank == 0:
            print("ERROR: No training data loaded!")
        cleanup_distributed()
        return

    random.shuffle(raw_data)

    # ── Wandb (rank 0 only) ──
    if rank == 0:
        import wandb
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

        # Build a descriptive run name from enabled tasks
        enabled_tasks = []
        for task_name, info in TASK_REGISTRY.items():
            n = getattr(args, info["arg"], 0)
            if n > 0:
                enabled_tasks.append(task_name)

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
    else:
        enabled_tasks = [tn for tn, info in TASK_REGISTRY.items() if getattr(args, info["arg"], 0) > 0]

    save_dir = Path(args.save_dir)

    # ── Precompute eval activation caches (rank 0 only) ──
    # Skip in Flamingo mode — evals use steering injection, not cross-attention
    eval_names = getattr(args, "evals", None)
    if rank == 0 and eval_names and not args.flamingo:
        from evals.training_eval_hook import precache_eval_activations
        print(f"\n{'=' * 60}")
        print("PRECOMPUTING EVAL ACTIVATIONS")
        print(f"{'=' * 60}")
        # For Flamingo mode, use the underlying PeftModel for eval caching
        eval_model = model.base_model if args.flamingo else model
        precache_eval_activations(
            eval_model, tokenizer, model_name=args.model,
            device=f"cuda:{local_rank}", eval_dir=args.eval_dir,
            activation_cache_dir=args.activation_cache_dir,
            eval_names=eval_names,
            stride=args.stride,
        )
        eval_model.train()
        gc.collect()
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

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
