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
    python src/train.py --config configs/train.yaml --full-recon-n 40000 --correctness-n 15000 --conv-qa-n 0

    # Resume from checkpoint (step auto-detected from training_state.pt):
    python src/train.py --config configs/train.yaml --resume-from checkpoints/step_5000
"""

import argparse
import gc
import json
import logging
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

    for item in raw_data:
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
        "corpus": "main",  # concept corpus removed; conv_qa now precompute-only
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
        "module": "dataset_classes.cot_compqa",
        "loader": "load_cot_compqa_data",
        "corpus": "compqa",
    },
    "hint_admission": {
        "arg": "hint_admission_n",
        "module": "dataset_classes.cot_hint_admission",
        "loader": "load_cot_hint_admission_data",
        "corpus": "hint_admission",
    },
}


HF_TRAINING_REPO = "mats-10-sprint-cs-jb/cot-oracle-training-v6"

_HF_CACHE_DIR = Path("data/.hf_cache")


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
            max_positions_per_layer=getattr(args, "max_positions_per_layer", 20),
            atypical_data_path=atypical_path,
        )
    elif info["corpus"] == "compqa":
        compqa_cache = str(Path(getattr(args, "precomputed_dir", "data/precomputed")) / "compqa_raw.json")
        return loader_fn(
            compqa_cache, tokenizer, args.model,
            num_examples=n, stride=args.stride,
            max_positions_per_layer=getattr(args, "max_positions_per_layer", 20),
        )
    elif info["corpus"] == "hint_admission":
        hint_path = getattr(args, "hint_admission_data_path",
                            "data/hint_admission_training.jsonl")
        return loader_fn(
            hint_path, tokenizer, args.model,
            num_examples=n, stride=args.stride,
            max_positions_per_layer=getattr(args, "max_positions_per_layer", 20),
            hint_admission_data_path=hint_path,
        )
    else:
        return loader_fn(
            args.corpus, tokenizer, args.model,
            num_examples=n, stride=args.stride,
            max_positions_per_layer=getattr(args, "max_positions_per_layer", 20),
        )


def load_precomputed_tasks(precomputed_dir: str, args) -> list[dict]:
    """Load training data from precomputed JSONL files, downloading from HF if needed."""
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
            jsonl_path = _download_precomputed_from_hf(task_name, pdir)

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


def run_eval(
    eval_datasets, model, tokenizer, submodule, device, dtype,
    global_step, eval_batch_size, steering_coefficient, log_dir=None,
):
    """Run fuzzy eval with token F1 scoring + wandb table logging."""
    import wandb

    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
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
        columns = ["id", "type", "oracle_prompt", "prediction", "target", "token_f1", "pred_tokens", "target_tokens"]
        table = wandb.Table(columns=columns)
        rows = []

        for i, (resp, dp) in enumerate(zip(eval_responses, eval_datasets[ds])):
            pred = resp.api_response.strip()
            target = dp.target_output.strip()
            score = _token_f1(pred, target)
            scores.append(score)
            oracle_prompt = getattr(dp, 'prompt', '') or str(dp.meta_info.get('prompt', ''))[:300]
            row = [i, dp.datapoint_type, oracle_prompt[:300], pred[:500], target[:500], round(score, 3), len(pred.split()), len(target.split())]
            table.add_data(*row)
            rows.append(row)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        wandb.log({f"eval/{ds}": avg_score}, step=global_step)
        print(f"  Step {global_step} | {ds}: token_f1={avg_score:.3f}")

        if eval_responses:
            print(f"    pred='{eval_responses[0].api_response.strip()[:120]}'")
            print(f"    targ='{eval_datasets[ds][0].target_output.strip()[:120]}'")

        wandb.log({f"eval_table/{ds}": table}, step=global_step)
        if log_dir and rows:
            _save_table_to_disk(Path(log_dir), f"eval_table_{ds}", global_step, columns, rows)

    model.train()
    torch.cuda.empty_cache()


def run_unfaith_evals(model, tokenizer, model_name, global_step, args, log_dir=None):
    """Run unfaithfulness evals if available."""
    from evals.training_eval_hook import run_training_evals
    import wandb

    print(f"\n--- Unfaithfulness evals at step {global_step} ---")
    metrics = run_training_evals(
        model, tokenizer, model_name=model_name,
        step=global_step, device="cuda",
        eval_dir=args.eval_dir,
        max_items_per_eval=20,
        skip_rot13=(global_step < args.rot13_start_step),
        oracle_adapter_name="default",
        activation_cache_dir=args.activation_cache_dir,
        log_dir=log_dir,
        eval_names=getattr(args, "unfaith_evals", None),
    )
    if metrics:
        # Inject baseline reference lines
        baselines = getattr(args, "eval_baselines", {})
        for eval_name, methods in baselines.items():
            for method, score in methods.items():
                metrics[f"eval/{eval_name}_baseline_{method}"] = score

        wandb.log(metrics, step=global_step)
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)) and "sample" not in k:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return metrics


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

    # Dynamic eval/save cadence only for sequential mode (stage-relative).
    # In shuffled mode, respect the config values from YAML.
    if task_order == "sequential":
        min_stage_steps = min(len(items) // args.batch_size for items in train_per_type.values() if len(items) >= args.batch_size)
        args.eval_steps = min(min_stage_steps // 3, max(-(-min_stage_steps // 20), 1))
        args.save_steps = args.eval_steps * 5
        if rank == 0:
            print(f"\n  Stage-relative cadence (min stage = {min_stage_steps} steps):")
            print(f"    eval_steps: {args.eval_steps} (~{min_stage_steps // max(args.eval_steps, 1)}x per min stage)")
            print(f"    save_steps: {args.save_steps} (~{min_stage_steps // max(args.save_steps, 1)}x per min stage)")

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

        for start in pbar:
            batch_list = final_training[start : start + args.batch_size]
            if len(batch_list) < args.batch_size:
                break

            batch_types = [dp.datapoint_type for dp in batch_list]

            # Materialize (uses unwrapped model)
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

            micro_step += 1
            if micro_step % grad_accum != 0:
                continue

            # Optimizer step (only every grad_accum micro-batches)
            clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
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

            # Logging (rank 0 only)
            if rank == 0:
                now = time.time()
                log_dict = {
                    "train/loss": loss.item() * grad_accum,  # log unscaled loss
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/total_tokens": total_tokens,
                    "train/batch_tokens": batch_tokens,
                    "train/step_time": now - last_step_time,
                    "train/wallclock_hours": (now - train_start_time - eval_time_total) / 3600,
                    "eval/wallclock_hours": eval_time_total / 3600,
                }
                last_step_time = now
                for task, ema_val in task_loss_ema.items():
                    log_dict[f"train/loss_{task}"] = ema_val

                # Track dominant task for sequential mode phase transitions
                batch_task_counts = defaultdict(int)
                for t in batch_types:
                    batch_task_counts[t] += 1
                dominant_task = max(batch_task_counts, key=batch_task_counts.get)
                log_dict["train/stage"] = dominant_task
                log_dict["train/stage_idx"] = task_stage_idx.get(dominant_task, -1)
                log_dict["train/progress"] = global_step / max(total_steps, 1)
                if dominant_task in stage_step_ranges:
                    s_start, s_end = stage_step_ranges[dominant_task]
                    s_len = max(s_end - s_start, 1)
                    log_dict["train/stage_progress"] = min((global_step - s_start) / s_len, 1.0)

                wandb.log(log_dict, step=global_step)

            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

            # Track dominant task for sequential mode phase transitions (all ranks)
            if rank != 0:
                batch_task_counts = defaultdict(int)
                for t in batch_types:
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

            # Task-level eval (rank 0 only)
            if global_step > 0 and global_step % args.eval_steps == 0:
                if rank == 0:
                    print(f"\n--- Task eval at step {global_step} ---")
                    eval_start = time.time()
                    run_eval(
                        eval_datasets, model, tokenizer, submodule,
                        device, dtype, global_step, args.eval_batch_size,
                        args.steering_coefficient, log_dir=log_dir,
                    )
                    eval_time_total += time.time() - eval_start
                    model.train()
                if world_size > 1:
                    dist.barrier()

            # Unfaithfulness eval (rank 0 only)
            if global_step > 0 and global_step % args.eval_steps == 0:
                if rank == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    eval_start = time.time()
                    run_unfaith_evals(model, tokenizer, args.model, global_step, args, log_dir=log_dir)
                    eval_time_total += time.time() - eval_start
                    model.train()
                    gc.collect()
                    torch.cuda.empty_cache()
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

            global_step += 1

    # Final eval (rank 0 only)
    if rank == 0:
        print(f"\n--- Final eval at step {global_step} ---")
        run_eval(
            eval_datasets, model, tokenizer, submodule,
            device, dtype, global_step, args.eval_batch_size,
            args.steering_coefficient, log_dir=log_dir,
        )

        run_unfaith_evals(model, tokenizer, args.model, global_step, args, log_dir=log_dir)

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
        for key in ["stride", "position_encoding", "pe_alpha"]:
            if key in a and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, a[key])

    # Eval
    if "eval" in config:
        e = config["eval"]
        for key in ["eval_dir", "rot13_start_step"]:
            if key in e and not getattr(args, f"_cli_{key}", False):
                setattr(args, key, e[key])
        if "unfaith_evals" in e and not getattr(args, "_cli_unfaith_evals", False):
            raw_evals = e["unfaith_evals"]
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
            args.unfaith_evals = eval_names
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

    # Per-task example counts — defaults are 0; set via --config (train.yaml is source of truth)
    parser.add_argument("--full-recon-n", type=int, default=0)
    parser.add_argument("--next-step-n", type=int, default=0)
    parser.add_argument("--answer-pred-n", type=int, default=0)
    parser.add_argument("--correctness-n", type=int, default=0)
    parser.add_argument("--decorative-n", type=int, default=0)
    parser.add_argument("--domain-n", type=int, default=0)
    parser.add_argument("--reasoning-term-n", type=int, default=0)
    parser.add_argument("--partial-answer-n", type=int, default=0)
    parser.add_argument("--conv-qa-n", type=int, default=0)
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
    parser.add_argument("--stride", type=int, default=5)
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

    # Eval / save
    parser.add_argument("--rot13-start-step", type=int, default=2000)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Starting global step (for resuming)")
    parser.add_argument("--eval-dir", default="data/evals")
    _default_cache = os.path.join(os.environ["CACHE_DIR"], "cot_oracle", "eval_precomputed") if os.environ.get("CACHE_DIR") else "data/eval_precomputed"
    parser.add_argument("--activation-cache-dir", default=_default_cache,
                        help="Dir with precomputed activation bundles (.pt)")

    # Output
    parser.add_argument("--save-dir", default="checkpoints")
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

    # Apply config file (CLI flags override config values)
    if args.config:
        config = load_config(args.config)
        apply_config(args, config)
        if rank == 0:
            print(f"Loaded config from {args.config}")
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

    set_seed(args.seed)

    # Multi-layer config
    global MULTI_LAYERS
    MULTI_LAYERS = [layer_percent_to_layer(args.model, p) for p in [25, 50, 75]]
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

    # Ensure LoRA params are fp32 (optimizer states stay fp32; autocast handles forward pass)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if rank == 0:
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
        raw_data = load_precomputed_tasks(args.precomputed_dir, args)
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
        # Save raw YAML config to wandb for reproducibility
        if args.config and Path(args.config).exists():
            wandb.save(args.config)
    else:
        enabled_tasks = [tn for tn, info in TASK_REGISTRY.items() if getattr(args, info["arg"], 0) > 0]

    save_dir = Path(args.save_dir)

    # ── Precompute eval activation caches (rank 0 only) ──
    eval_names = getattr(args, "unfaith_evals", None)
    if rank == 0 and eval_names:
        from evals.training_eval_hook import precache_eval_activations
        print(f"\n{'=' * 60}")
        print("PRECOMPUTING EVAL ACTIVATIONS")
        print(f"{'=' * 60}")
        precache_eval_activations(
            model, tokenizer, model_name=args.model,
            device=f"cuda:{local_rank}", eval_dir=args.eval_dir,
            activation_cache_dir=args.activation_cache_dir,
            eval_names=eval_names,
        )
        model.train()
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

    global_step = args.start_step
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
