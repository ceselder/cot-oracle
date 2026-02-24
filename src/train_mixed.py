"""
Train CoT Oracle — Mixed Training with Random Layer Subsets

By default, samples random subsets of layers per example (Poisson mean=5)
with grouped prefix format: "L5: ? ? ? ? L11: ? ? ? ?\n".

By default, uses fixed [L25%, L50%, L75%] layers and sinusoidal position
encoding. Use --random-layers for Poisson-sampled random layer subsets.
Use --no-position-encoding to disable positional encoding.

Features:
  - Parallel corpus pre-tokenization (eliminates redundant work across loaders)
  - Persistent data cache (SHA-256 keyed, skips data build on re-runs)
  - Multi-GPU: torchrun-compatible, auto gradient accumulation
  - Curriculum learning: 2-stage (token prediction → classification)
  - Third-binned eval: accuracy by depth-third presence
  - Per-task × per-third loss decomposition

Usage:
    torchrun --nproc_per_node=1 src/train_mixed.py \
        --corpus data/cot_corpus_v4/corpus.jsonl \
        --persona-corpus data/cot_corpus_v4/corpus_persona.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tqdm.auto import tqdm

def _is_rank0():
    return os.environ.get("LOCAL_RANK", "0") == "0"

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path
ensure_ao_repo_on_path()

import torch

from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    SPECIAL_TOKEN,
    find_pattern_in_tokens,
    create_training_datapoint,
)
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
import nl_probes.sft as sft_module
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

# Our dataset loaders
from dataset_classes.cot_context_prediction import load_cot_context_prediction_data
from dataset_classes.cot_sentence_prediction import load_cot_sentence_prediction_data
from dataset_classes.cot_decorative import load_cot_decorative_data
from dataset_classes.cot_domain import load_cot_domain_data
from dataset_classes.cot_correctness import load_cot_correctness_data
from dataset_classes.cot_persona import load_cot_persona_data
from dataset_classes.cot_summary import load_cot_summary_data
from dataset_classes.cot_answer_tracking import load_cot_answer_tracking_data
from dataset_classes.cot_chainscope_faithfulness import load_chainscope_faithfulness_data
from dataset_classes.cot_eval_task import load_eval_task_data, EVAL_TRAINING_TASKS

from cot_utils import layer_percent_to_layer, LAYER_COUNTS, split_cot_into_sentences, find_sentence_boundary_positions
from layer_utils import (
    sample_layers,
    build_random_layer_prefix,
    find_all_special_positions,
    layers_to_third_bin,
)
from position_encoding import apply_position_encoding
from corpus_tokenize import load_corpus, pretokenize_corpus, ensure_boundary_positions
from data_cache import load_cached_data, save_cached_data

STAGE1_TASKS = {"cot_context_prediction", "cot_sentence_prediction"}
STAGE2_TASKS = {
    "cot_decorative", "cot_domain", "cot_correctness", "cot_persona", "cot_summary",
    *(f"eval_{t}" for t in EVAL_TRAINING_TASKS),
}

# Eval tasks used for training — excluded from unfaithfulness eval hook
TRAINING_EVAL_TASK_NAMES = {f"eval_{t}" for t in EVAL_TRAINING_TASKS}

# Display names: strip cot_/eval_ prefixes for HF-style naming in wandb and print
def _display_name(datapoint_type: str) -> str:
    if datapoint_type.startswith("cot_"):
        return datapoint_type[4:]
    if datapoint_type.startswith("eval_"):
        return datapoint_type[5:]
    return datapoint_type


# ---------------------------------------------------------------------------
# Datapoint creation
# ---------------------------------------------------------------------------

def _create_random_layer_datapoint(
    item: dict,
    tokenizer,
    layers: list[int],
) -> TrainingDataPoint:
    """Create a TrainingDataPoint with random-layer-subset prefix.

    Prefix format: "L5: ? ? ? ? L11: ? ? ? ?\n"
    Positions grouped by layer: all positions for L5, then all for L11, etc.
    """
    orig_positions = item["context_positions"]
    num_pos_per_layer = len(orig_positions)
    total_positions = num_pos_per_layer * len(layers)

    prefix = build_random_layer_prefix(layers, num_pos_per_layer)
    prompt = prefix + item["prompt"]

    # Tokenize
    input_messages = [{"role": "user", "content": prompt}]
    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True,
        return_tensors=None, padding=False, enable_thinking=False,
    )

    full_messages = input_messages + [{"role": "assistant", "content": item["target_response"]}]
    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_tensors=None, padding=False, enable_thinking=False,
    )

    # Labels: mask prompt tokens
    assistant_start_idx = len(input_prompt_ids)
    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    # Find placeholder positions (non-consecutive OK — L{n}: labels break runs)
    special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    positions = find_all_special_positions(full_prompt_ids, special_token_id, total_positions)

    # Expand context_positions grouped by layer:
    # [p1,p2,...,pN, p1,p2,...,pN, ...] — all positions for first layer, then second, etc.
    expanded_ctx_positions = []
    for _ in layers:
        expanded_ctx_positions.extend(orig_positions)

    meta = {"multi_layers": layers, "num_pos_per_layer": num_pos_per_layer}
    stride_val = item.get("_stride")
    if stride_val is not None:
        meta["stride"] = stride_val
        meta["n_positions"] = num_pos_per_layer

    return TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=layers[0],  # Primary layer for AO compat
        steering_vectors=None,
        positions=positions,
        feature_idx=-1,
        target_output=item["target_response"],
        datapoint_type=item["datapoint_type"],
        context_input_ids=item["context_input_ids"],
        context_positions=expanded_ctx_positions,
        ds_label=None,
        meta_info=meta,
    )


_STRIDE_EXEMPT_TASKS = {"cot_context_prediction", "cot_sentence_prediction"}


# ---------------------------------------------------------------------------
# Parallel conversion workers
# ---------------------------------------------------------------------------

_worker_tokenizer = None


def _init_convert_worker(model_name):
    """Pool initializer: each worker loads its own tokenizer."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _worker_tokenizer
    _worker_tokenizer = load_tokenizer(model_name)


def _convert_chunk(prepared_items):
    """Worker function: convert prepared (item, layers, is_multi) tuples to TrainingDataPoints."""
    tokenizer = _worker_tokenizer
    results = []
    for item, layers, is_multi in prepared_items:
        if is_multi:
            dp = _create_random_layer_datapoint(item, tokenizer, layers)
        else:
            layer = layers[0]
            meta = {"multi_layers": [layer], "num_pos_per_layer": item["num_positions"]}
            stride_val = item.get("_stride")
            if stride_val is not None:
                meta["stride"] = stride_val
                meta["n_positions"] = item["num_positions"]
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=layer,
                num_positions=item["num_positions"],
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=item["context_positions"],
                meta_info=meta,
            )
        results.append(dp)
    return results


def _convert_chunk_sequential(prepared_items, tokenizer):
    """Sequential fallback: same as _convert_chunk but uses passed tokenizer."""
    global _worker_tokenizer
    _worker_tokenizer = tokenizer
    return _convert_chunk(prepared_items)


def _prepare_items(
    raw_data: list[dict],
    max_layers: int,
    layer_mean: int,
    position_stride: int | None,
    max_positions: int,
    layer_repeats: int = 1,
    fixed_layers: list[int] | None = None,
) -> list[tuple]:
    """Pre-compute random decisions for all items (sequential, deterministic).

    A fixed pool of layer_repeats layer sets is generated once, then every
    item cycles through the same pool. This guarantees every layer set sees
    every example.

    If fixed_layers is provided, uses those layers for all multi-layer items
    and the middle layer for single-layer items (no random sampling).

    Returns list of (item_dict, layers_list, is_multi) tuples ready for conversion.
    """
    # Pre-generate fixed layer set pools (one for multi-layer, one for single-layer items)
    effective_max = max_layers * 3 // 4
    if fixed_layers:
        multi_pool = [fixed_layers] * layer_repeats
        mid_layer = fixed_layers[len(fixed_layers) // 2]
        single_pool = [mid_layer] * layer_repeats
    else:
        multi_pool = [sample_layers(max_layers, layer_mean) for _ in range(layer_repeats)]
        single_pool = [random.randint(0, effective_max - 1) for _ in range(layer_repeats)]

    prepared = []
    for item in raw_data:
        # Stride-based position resampling for eligible tasks (applied once per item)
        if position_stride and item["datapoint_type"] not in _STRIDE_EXEMPT_TASKS:
            total_length = len(item["context_input_ids"])
            stride = position_stride
            positions = list(range(0, total_length, stride))
            if len(positions) > max_positions:
                positions = positions[:max_positions]
            if len(positions) < 2:
                positions = list(range(0, total_length, max(1, total_length // 4)))
            item["prompt"] = re.sub(
                r'Activations from \d+ sentence boundaries',
                f'Activations from {len(positions)} positions',
                item["prompt"],
            )
            item["context_input_ids"] = item["context_input_ids"][:positions[-1] + 1]
            item["context_positions"] = positions
            item["num_positions"] = len(positions)
            item["_stride"] = stride
        else:
            item["_stride"] = None

        # Cycle through fixed layer set pool
        layers_field = item.get("layers")
        is_multi = layers_field and len(layers_field) > 1
        for r in range(layer_repeats):
            if is_multi:
                prepared.append((item, multi_pool[r], True))
            else:
                prepared.append((item, [single_pool[r]], False))

    return prepared


def dicts_to_multilayer_training_data(
    raw_data: list[dict],
    tokenizer,
    max_layers: int = 36,
    layer_mean: int = 5,
    position_stride: int | None = None,
    max_positions: int = 50,
    model_name: str | None = None,
    num_workers: int | None = None,
    layer_repeats: int = 1,
    fixed_layers: list[int] | None = None,
) -> list[TrainingDataPoint]:
    """Convert dataset dicts to TrainingDataPoints with layer subsets.

    By default, samples random layers via Poisson. With fixed_layers, uses
    the given layers for all multi-layer items.

    Each item is replicated layer_repeats times with different layer sets
    (or the same set when fixed_layers is given).

    When model_name is provided and num_workers > 1, conversion runs in parallel
    using multiprocessing (each worker loads its own tokenizer).
    """
    # Step 1: pre-compute random decisions (sequential, preserves determinism)
    prepared = _prepare_items(raw_data, max_layers, layer_mean, position_stride, max_positions, layer_repeats, fixed_layers=fixed_layers)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    # Step 2: convert to TrainingDataPoints
    if num_workers > 1 and model_name and len(prepared) > 1000:
        chunk_size = max(1, len(prepared) // num_workers)
        chunks = [prepared[i:i + chunk_size] for i in range(0, len(prepared), chunk_size)]
        print(f"  Converting {len(prepared)} items with {len(chunks)} workers...")
        with mp.Pool(num_workers, initializer=_init_convert_worker, initargs=(model_name,)) as pool:
            chunk_results = pool.map(_convert_chunk, chunks)
        training_data = []
        for chunk in chunk_results:
            training_data.extend(chunk)
    else:
        training_data = _convert_chunk_sequential(prepared, tokenizer)

    return training_data


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

def install_multilayer_materialization(position_encoding: bool = False):
    """Monkey-patch materialize_missing_steering_vectors for grouped layer ordering.

    Grouped ordering (num_pos_per_layer in meta_info):
      positions: [p1,p2,...,pN, p1,p2,...,pN, ...]
      layer for position i: multi_layers[i // num_pos_per_layer]

    Legacy cycling ordering (no num_pos_per_layer):
      positions: [p1,p1,p1, p2,p2,p2, ...]
      layer for position i: multi_layers[i % n_layers]

    If position_encoding=True, applies sinusoidal PE to vectors before injection.
    """
    from peft import PeftModel

    def patched_materialize(batch_points, tokenizer, model):
        to_fill = [
            (i, dp) for i, dp in enumerate(batch_points)
            if dp.steering_vectors is None
        ]
        if not to_fill:
            return batch_points

        assert isinstance(model, PeftModel), "Model must be a PeftModel"

        for _, dp in to_fill:
            if dp.context_input_ids is None or dp.context_positions is None:
                raise ValueError(
                    "Datapoint has steering_vectors=None but missing context_input_ids/context_positions"
                )

        # Collect ALL needed layers
        layers_needed = set()
        for _, dp in to_fill:
            multi_layers = dp.meta_info.get("multi_layers")
            if multi_layers:
                layers_needed.update(multi_layers)
            else:
                layers_needed.add(dp.layer)
        layers_needed = sorted(layers_needed)

        # Build padded input batch
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
                    [False] * pad_len + [True] * len(c),
                    dtype=torch.bool, device=device,
                )
            )
            left_offsets.append(pad_len)

        inputs_BL = {
            "input_ids": torch.stack(input_ids_tensors, dim=0),
            "attention_mask": torch.stack(attn_masks_tensors, dim=0),
        }

        # One forward pass collecting all needed layers
        submodules = {
            layer: get_hf_submodule(model, layer, use_lora=True)
            for layer in layers_needed
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

        # Build steering vectors for each item
        new_batch = list(batch_points)
        for b in range(len(to_fill)):
            idx, dp = to_fill[b]
            multi_layers = dp.meta_info.get("multi_layers")
            num_pos_per_layer = dp.meta_info.get("num_pos_per_layer")

            if multi_layers and num_pos_per_layer is not None:
                # Grouped ordering: position i -> layer = multi_layers[i // num_pos_per_layer]
                n_layers = len(multi_layers)
                vectors_list = []
                for i, pos in enumerate(positions_per_item[b]):
                    layer = multi_layers[i // num_pos_per_layer]
                    adj_pos = pos + left_offsets[b]
                    acts_BLD = acts_by_layer[layer]
                    L = acts_BLD.shape[1]
                    assert 0 <= adj_pos < L, (
                        f"Act index {adj_pos} out of range (L={L}) "
                        f"for item {b}, position {pos}, layer {layer}"
                    )
                    vectors_list.append(acts_BLD[b, adj_pos, :])
                vectors = torch.stack(vectors_list, dim=0).detach().contiguous()
            elif multi_layers:
                # Legacy cycling ordering: position i -> layer = multi_layers[i % n_layers]
                n_layers = len(multi_layers)
                vectors_list = []
                for i, pos in enumerate(positions_per_item[b]):
                    layer = multi_layers[i % n_layers]
                    adj_pos = pos + left_offsets[b]
                    acts_BLD = acts_by_layer[layer]
                    L = acts_BLD.shape[1]
                    assert 0 <= adj_pos < L, (
                        f"Act index {adj_pos} out of range (L={L}) "
                        f"for item {b}, position {pos}, layer {layer}"
                    )
                    vectors_list.append(acts_BLD[b, adj_pos, :])
                vectors = torch.stack(vectors_list, dim=0).detach().contiguous()
            else:
                # Single-layer: standard behavior
                layer = dp.layer
                acts_BLD = acts_by_layer[layer]
                idxs = [p + left_offsets[b] for p in positions_per_item[b]]
                L = acts_BLD.shape[1]
                assert all(0 <= i < L for i in idxs), (
                    f"Act index out of range for item {b}: {idxs} with L={L}"
                )
                vectors = acts_BLD[b, idxs, :].detach().contiguous()

            if position_encoding:
                source_positions = positions_per_item[b]
                total_length = len(dp.context_input_ids)
                vectors = apply_position_encoding(vectors, source_positions, total_length)

            assert len(vectors.shape) == 2
            dp_new = dp.model_copy(deep=True)
            dp_new.steering_vectors = vectors
            new_batch[idx] = dp_new

        return new_batch

    # Monkey-patch
    import nl_probes.utils.dataset_utils as du_module
    du_module.materialize_missing_steering_vectors = patched_materialize
    sft_module.materialize_missing_steering_vectors = patched_materialize
    print(f"Installed multi-layer materialization patch (position_encoding={position_encoding})")


def _build_full_text_with_source(dp: TrainingDataPoint, tokenizer) -> str:
    """Build full prompt text with ? placeholders replaced by actual source tokens.

    Each placeholder block 'L{n}: ? ? ? ?' becomes 'L{n}: <<tok1>> <<tok2>> ...'
    so wandb shows what text the activations came from.
    """
    if dp.context_input_ids is None or dp.context_positions is None:
        return tokenizer.decode(dp.input_ids, skip_special_tokens=False)

    num_pos_per_layer = dp.meta_info.get("num_pos_per_layer", len(dp.context_positions))
    multi_layers = dp.meta_info.get("multi_layers", [dp.layer])

    # Decode each source token individually
    ctx_ids = dp.context_input_ids
    # Group context_positions by layer — each layer block repeats the same positions
    positions_per_layer = dp.context_positions[:num_pos_per_layer]

    source_tokens = []
    for pos in positions_per_layer:
        if pos < len(ctx_ids):
            tok_text = tokenizer.decode([ctx_ids[pos]], skip_special_tokens=False).strip()
            source_tokens.append(tok_text if tok_text else "∅")
        else:
            source_tokens.append("∅")

    # Build the replacement prefix: L{n}: <<tok1>> <<tok2>> ...
    parts = []
    for layer in multi_layers:
        highlighted = " ".join(f"<<{t}>>" for t in source_tokens)
        parts.append(f"L{layer}: {highlighted}")
    new_prefix = " ".join(parts) + " \n"

    # Decode the full prompt and replace the original prefix
    full_text = tokenizer.decode(dp.input_ids, skip_special_tokens=False)

    # Find and replace the L{n}: ? ? ? ... pattern
    # The original prefix looks like "L5: ? ? ? ? L11: ? ? ? ?\n"
    placeholder_block = SPECIAL_TOKEN * num_pos_per_layer  # " ? ? ? ?"
    old_parts = []
    for layer in multi_layers:
        old_parts.append(f"L{layer}:{placeholder_block}")
    old_prefix = " ".join(old_parts) + " \n"

    if old_prefix in full_text:
        full_text = full_text.replace(old_prefix, new_prefix, 1)

    return full_text


def install_third_eval_hook(max_layers: int = 36, single_stride: bool = False):
    """Replace eval_all_datasets with a version that:
    1. Runs eval once per dataset (not twice)
    2. Logs standard accuracy + third-binned accuracy
    3. Logs a wandb Table with eval traces (prompt, label, prediction, layers, etc.)
    4. Skips stride-binned panels when single_stride=True
    """
    import gc
    import wandb
    from nl_probes.utils.eval import run_evaluation, score_eval_responses, parse_answer

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        model.eval()
        eval_metrics = {}
        table_rows = []

        for ds in eval_datasets:
            ds_display = _display_name(ds)
            eval_data = eval_datasets[ds]

            eval_responses = run_evaluation(
                eval_data=eval_data,
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                device=device,
                dtype=dtype,
                global_step=global_step,
                lora_path=None,
                eval_batch_size=cfg.eval_batch_size,
                steering_coefficient=cfg.steering_coefficient,
                generation_kwargs=cfg.generation_kwargs,
            )

            # Standard accuracy
            percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_data)
            eval_metrics[f"eval_format_correct/{ds_display}"] = percent_format_correct
            eval_metrics[f"eval_ans_correct/{ds_display}"] = percent_ans_correct
            print(f"Step {global_step} {ds_display} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")

            # Third-binned accuracy + table rows
            by_bin = defaultdict(lambda: {"correct": 0, "total": 0})
            for resp, dp in zip(eval_responses, eval_data, strict=True):
                multi_layers = dp.meta_info.get("multi_layers", [dp.layer])
                qbin = layers_to_third_bin(multi_layers, max_layers)

                cleaned = parse_answer(resp.api_response)
                target = parse_answer(dp.target_output)
                correct = cleaned == target
                by_bin[qbin]["total"] += 1
                if correct:
                    by_bin[qbin]["correct"] += 1

                ao_prompt = resp.prompt
                full_text = _build_full_text_with_source(dp, tokenizer)

                table_rows.append([
                    global_step,
                    ds_display,
                    dp.target_output,
                    resp.api_response.strip(),
                    correct,
                    str(multi_layers),
                    qbin,
                    ao_prompt,
                    full_text,
                ])

            for qbin, counts in sorted(by_bin.items()):
                n = counts["total"]
                acc = counts["correct"] / n if n > 0 else 0.0
                eval_metrics[f"eval_third/{ds_display}/{qbin}/accuracy"] = acc
                eval_metrics[f"eval_third/{ds_display}/{qbin}/n"] = n

        # Log metrics
        if wandb.run is not None:
            wandb.log(eval_metrics, step=global_step)
            wandb.summary.update(eval_metrics)

            # Log eval traces table
            table = wandb.Table(columns=[
                "step", "task", "label", "prediction", "correct",
                "layers", "third_bin", "ao_prompt", "full_prompt",
            ], data=table_rows)
            wandb.log({f"eval_traces/step_{global_step}": table}, step=global_step)
            print(f"  Logged {len(table_rows)} eval traces + {sum(1 for k in eval_metrics if 'third' in k)} third metrics")

        model.train()
        torch.cuda.empty_cache()
        gc.collect()

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print(f"Installed third eval hook (stride panels: {'disabled' if single_stride else 'enabled'})")


def install_third_task_loss_hook(max_layers: int = 36, single_stride: bool = False):
    """Monkey-patch AO's training loop to log per-task AND per-third loss to wandb.

    Logs:
      train/loss_{task}           — per-task average loss (HF display names)
      train/loss_{task}/{qbin}    — per-task x per-third-bin loss
      train/loss_third/{qbin}     — per-third average loss (across all tasks)
      train/wallclock_hours       — wall time since start
      train/examples_seen         — running total of training examples
    """
    import wandb
    import torch.nn.functional as F
    from nl_probes.sft import train_features_batch as _original_train
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

    import time
    _batch_state = {"types": [], "meta_infos": [], "start_time": time.time(), "last_time": time.time(), "examples_seen": 0}

    from nl_probes.sft import construct_batch as _original_construct
    def patched_construct_batch(batch_list, tokenizer, device):
        _batch_state["types"] = [dp.datapoint_type for dp in batch_list]
        _batch_state["meta_infos"] = [dp.meta_info for dp in batch_list]
        return _original_construct(batch_list, tokenizer, device)
    sft_module.construct_batch = patched_construct_batch

    def patched_train_features_batch(cfg, training_batch, model, submodule, device, dtype):
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

        batch_types = _batch_state["types"]
        meta_infos = _batch_state["meta_infos"]
        if batch_types and len(batch_types) == training_batch.input_ids.shape[0]:
            logits = outputs.logits
            labels = training_batch.labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            ).view(shift_labels.shape)
            mask = (shift_labels != -100).float()
            per_item_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Per-task losses (using HF display names)
            task_losses = defaultdict(list)
            # Per-task × per-third losses
            task_third_losses = defaultdict(list)
            # Per-third losses (across all tasks)
            third_losses = defaultdict(list)

            for i, task_type in enumerate(batch_types):
                loss_val = per_item_loss[i].item()
                display = _display_name(task_type)
                task_losses[display].append(loss_val)

                meta = meta_infos[i]
                multi_layers = meta.get("multi_layers", [])
                if multi_layers:
                    qbin = layers_to_third_bin(multi_layers, max_layers)
                    task_third_losses[f"{display}/{qbin}"].append(loss_val)
                    third_losses[qbin].append(loss_val)

            _batch_state["examples_seen"] += len(batch_types)
            now = time.time()
            log_dict = {}
            for task, losses in task_losses.items():
                log_dict[f"train/loss_{task}"] = sum(losses) / len(losses)
            for key, losses in task_third_losses.items():
                log_dict[f"train/loss_{key}"] = sum(losses) / len(losses)
            for qbin, losses in third_losses.items():
                log_dict[f"train/loss_third/{qbin}"] = sum(losses) / len(losses)

            # Stage indicator based on which tasks are present in batch
            stage2_tasks_present = [t for t in batch_types if t in STAGE2_TASKS]
            if not stage2_tasks_present:
                log_dict["train/stage"] = 1
            else:
                order = STAGE2_ORDER_DEFAULT
                max_idx = max(order.index(t) for t in stage2_tasks_present if t in order)
                log_dict["train/stage"] = max_idx + 2

            log_dict["train/step_time"] = now - _batch_state["last_time"]
            log_dict["train/wallclock_hours"] = (now - _batch_state["start_time"]) / 3600
            log_dict["train/examples_seen"] = _batch_state["examples_seen"]
            _batch_state["last_time"] = now

            if wandb.run is not None:
                wandb.log(log_dict, commit=False)

        return outputs.loss

    sft_module.train_features_batch = patched_train_features_batch
    print(f"Installed per-task loss logging hook (stride panels: {'disabled' if single_stride else 'enabled'})")


STAGE2_ORDER_DEFAULT = [
    "cot_decorative", "cot_domain", "cot_correctness", "cot_persona", "cot_summary",
    *(f"eval_{t}" for t in EVAL_TRAINING_TASKS),
]


def build_curriculum(
    stage1_data: list[TrainingDataPoint],
    stage2_data: list[TrainingDataPoint],
    stage1_reg: float = 0.2,
    seed: int = 42,
    stage2_order: list[str] | None = None,
) -> tuple[list[TrainingDataPoint], list[int]]:
    """Build a multi-stage curriculum: stage 1 (token prediction), then one stage per classification task.

    Stage 1: only stage1_data (context pred, sentence pred), shuffled.
    Stage 2+: each classification task in order, with regularization from all prior stages.

    Returns (ordered_data, stage_boundaries) where stage_boundaries[i] is the
    index where stage i+1 starts (i.e. stage_boundaries[0] = start of stage 2).
    """
    if stage2_order is None:
        stage2_order = STAGE2_ORDER_DEFAULT

    rng = random.Random(seed)

    # Group stage 2 data by task
    by_task: dict[str, list] = defaultdict(list)
    for dp in stage2_data:
        by_task[dp.datapoint_type].append(dp)

    # Stage 1
    s1 = list(stage1_data)
    rng.shuffle(s1)

    ordered = list(s1)
    stage_boundaries = [len(ordered)]  # boundary between stage 1 and first stage-2 sub-stage
    prior_pool = list(stage1_data)  # pool of all prior-stage data for regularization

    print(f"\nCurriculum staging:")
    print(f"  Stage 1: {len(s1)} examples (token prediction)")

    for i, task in enumerate(stage2_order):
        task_data = by_task.get(task, [])
        if not task_data:
            continue
        n_reg = int(len(task_data) * stage1_reg)
        reg_subset = rng.sample(prior_pool, min(n_reg, len(prior_pool)))
        stage = list(task_data) + reg_subset
        rng.shuffle(stage)
        ordered.extend(stage)
        stage_boundaries.append(len(ordered))
        prior_pool.extend(task_data)
        print(f"  Stage {i + 2}: {len(stage)} examples ({len(task_data)} {task} + {len(reg_subset)} reg)")

    return ordered, stage_boundaries


# ---------------------------------------------------------------------------
# Data building (reuses train_mixed loaders, different conversion)
# ---------------------------------------------------------------------------

def build_training_mixture(
    corpus_path: str,
    persona_corpus_path: str | None,
    labels_dir: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    max_layers: int,
    layer_mean: int,
    task_sizes: dict[str, int] | None = None,
    position_stride: int | None = None,
    max_positions: int = 50,
    num_workers: int | None = None,
    corpus_entries: list[dict] | None = None,
    persona_entries: list[dict] | None = None,
    layer_repeats: int = 1,
    fixed_layers: list[int] | None = None,
) -> list[TrainingDataPoint]:
    """Build the mixed training data from up to 7 tasks, with layer subsets.

    If corpus_entries (pre-tokenized with _ctx_ids) is provided, skips all corpus
    loading and tokenization. Otherwise loads and pre-tokenizes internally.
    Then converts everything to TrainingDataPoints in one parallel batch.
    """

    if task_sizes is None:
        task_sizes = {
            "cot_context_prediction": 100000,
            "cot_sentence_prediction": 30000,
            "cot_decorative": 10000,
            "cot_domain": 15000,
            "cot_correctness": 15000,
            "cot_persona": 15000,
        }

    # --- Pre-tokenize corpus once if not already done ---
    if corpus_entries is None:
        print("\nPre-tokenizing main corpus...")
        corpus_entries = load_corpus(corpus_path)
        pretokenize_corpus(corpus_entries, model_name, num_workers=num_workers)

    if persona_entries is None and persona_corpus_path and Path(persona_corpus_path).exists():
        print("Pre-tokenizing persona corpus...")
        persona_entries = load_corpus(persona_corpus_path)
        pretokenize_corpus(persona_entries, model_name, num_workers=num_workers)

    all_raw = []

    # --- Load raw dicts from each task (fast: no tokenization, just sampling) ---

    print("\n=== Task 1: Context Prediction — Random Positions ===")
    raw = load_cot_context_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_context_prediction", 100000),
        corpus_entries=corpus_entries,
    )
    print(f"  Loaded {len(raw)} raw examples")
    all_raw.extend(raw)

    print("\n=== Task 2: Context Prediction — Sentence Boundaries ===")
    raw = load_cot_sentence_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_sentence_prediction", 30000),
        corpus_entries=corpus_entries,
    )
    print(f"  Loaded {len(raw)} raw examples")
    all_raw.extend(raw)

    print("\n=== Task 3: Decorative CoT ===")
    raw = load_cot_decorative_data(
        corpus_path, tokenizer, model_name,
        num_examples=task_sizes.get("cot_decorative", 10000),
        corpus_entries=corpus_entries,
    )
    print(f"  Loaded {len(raw)} raw examples")
    all_raw.extend(raw)

    print("\n=== Task 4: Domain Classification ===")
    raw = load_cot_domain_data(
        corpus_path, tokenizer, model_name,
        num_examples=task_sizes.get("cot_domain", 15000),
        corpus_entries=corpus_entries,
    )
    print(f"  Loaded {len(raw)} raw examples")
    all_raw.extend(raw)

    print("\n=== Task 5: Correctness Prediction ===")
    raw = load_cot_correctness_data(
        corpus_path, tokenizer, model_name,
        num_examples=task_sizes.get("cot_correctness", 15000),
        corpus_entries=corpus_entries,
    )
    print(f"  Loaded {len(raw)} raw examples")
    all_raw.extend(raw)

    if persona_entries is not None or (persona_corpus_path and Path(persona_corpus_path).exists()):
        print("\n=== Task 6: Persona Detection ===")
        raw = load_cot_persona_data(
            persona_corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("cot_persona", 15000),
            corpus_entries=persona_entries,
        )
        print(f"  Loaded {len(raw)} raw examples")
        all_raw.extend(raw)
    else:
        print(f"\n  Skipping Task 6 (no persona corpus at {persona_corpus_path})")

    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    if Path(summaries_path).exists():
        print("\n=== Task 7: CoT Summary ===")
        raw = load_cot_summary_data(
            corpus_path, summaries_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("cot_summary", 15000),
            corpus_entries=corpus_entries,
        )
        print(f"  Loaded {len(raw)} raw examples")
        all_raw.extend(raw)
    else:
        print(f"\n  Skipping Task 7 (no summaries at {summaries_path})")

    # --- Eval-derived training tasks ---
    eval_train_dir = task_sizes.get("_eval_train_dir", "data/evals_train")
    n_per_eval_task = task_sizes.get("_n_eval_task", 3000)
    if Path(eval_train_dir).exists():
        for eval_task_name in sorted(EVAL_TRAINING_TASKS.keys()):
            eval_train_path = Path(eval_train_dir) / f"{eval_task_name}.json"
            if not eval_train_path.exists():
                print(f"\n  Skipping eval task {eval_task_name} (no data at {eval_train_path})")
                continue
            print(f"\n=== Eval Task: {eval_task_name} ===")
            raw = load_eval_task_data(
                str(eval_train_path), eval_task_name, tokenizer, model_name,
                num_examples=n_per_eval_task,
            )
            print(f"  Loaded {len(raw)} raw examples")
            all_raw.extend(raw)
    else:
        print(f"\n  Skipping eval training tasks (no dir at {eval_train_dir})")

    # --- Convert all at once (parallel) ---
    print(f"\n{'=' * 60}")
    print(f"Converting {len(all_raw)} raw dicts to TrainingDataPoints...")
    all_data = dicts_to_multilayer_training_data(
        all_raw, tokenizer, max_layers, layer_mean,
        position_stride, max_positions,
        model_name=model_name, num_workers=num_workers,
        layer_repeats=layer_repeats, fixed_layers=fixed_layers,
    )

    print(f"Total training examples: {len(all_data)}")
    type_counts = Counter(dp.datapoint_type for dp in all_data)
    for dpt, count in sorted(type_counts.items()):
        pct = count / len(all_data) * 100
        print(f"  {_display_name(dpt)}: {count} ({pct:.1f}%)")

    return all_data


def build_eval_datasets(
    corpus_path: str,
    labels_dir: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    max_layers: int,
    layer_mean: int,
    position_stride: int | None = None,
    max_positions: int = 50,
    corpus_entries: list[dict] | None = None,
    fixed_layers: list[int] | None = None,
) -> dict[str, list[TrainingDataPoint]]:
    """Build held-out eval datasets with layer subsets (fixed seed for reproducibility)."""
    eval_datasets = {}

    # Use a fixed seed for eval so layer subsets are identical across checkpoints
    saved_state = random.getstate()
    random.seed(12345)

    # Zero-shot: Answer Tracking
    if labels_dir:
        tracking_path = Path(labels_dir) / "labels_answer_tracking.jsonl"
        if tracking_path.exists():
            print("\n=== Eval: Answer Tracking (zero-shot, 100 items) ===")
            raw = load_cot_answer_tracking_data(
                corpus_path, str(tracking_path), tokenizer, model_name, layer_percents,
                num_examples=100, corpus_entries=corpus_entries,
            )
            data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean, position_stride, max_positions, fixed_layers=fixed_layers)
            eval_datasets["cot_answer_tracking"] = data
            print(f"  Generated {len(data)} eval examples")

    # Summary eval
    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    if Path(summaries_path).exists():
        print("\n=== Eval: CoT Summary (100 items) ===")
        raw = load_cot_summary_data(
            corpus_path, summaries_path, tokenizer, model_name, layer_percents,
            num_examples=100, seed=999, corpus_entries=corpus_entries,
        )
        data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean, position_stride, max_positions, fixed_layers=fixed_layers)
        eval_datasets["cot_summary"] = data
        print(f"  Generated {len(data)} eval examples")

    # Chainscope faithfulness eval split (100 held-out items, different seed)
    chainscope_path = "data/chainscope_qwen3_8b_cots.json"
    if Path(chainscope_path).exists():
        print("\n=== Eval: Chainscope Faithfulness (100 items) ===")
        raw = load_chainscope_faithfulness_data(
            chainscope_path, tokenizer, model_name,
            num_examples=100, seed=12345,
        )
        data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean, position_stride, max_positions, fixed_layers=fixed_layers)
        eval_datasets["chainscope_faithfulness"] = data
        print(f"  Generated {len(data)} eval examples")

    random.setstate(saved_state)
    return eval_datasets


def install_unfaithfulness_eval_hook(model_name, eval_dir="data/evals", fast_n=5):
    """Monkey-patch AO's eval to run unfaithfulness evals alongside training evals."""
    import wandb
    from evals.common import load_eval_items
    from evals.score_oracle import score_eval, EVAL_PARSING
    from evals.run_evals import run_eval_batched

    eval_dir = Path(eval_dir)
    act_layer = layer_percent_to_layer(model_name, 50)

    # Evals used for training — skip these to avoid evaluating on training data
    # Also skip raw step_importance files (different schema, not EvalItem)
    _skip_evals = (
        {"decorative_cot", "sentence_insertion", "step_importance_raw", "step_importance_faithfulness_raw"}
        | set(EVAL_TRAINING_TASKS.keys())
    )

    fast_items = {}
    for eval_file in sorted(eval_dir.glob("*.json")):
        eval_name = eval_file.stem
        if eval_name in _skip_evals:
            continue
        items = load_eval_items(eval_file)
        fast_items[eval_name] = items[:fast_n]

    total_items = sum(len(v) for v in fast_items.values())
    print(f"Unfaithfulness eval hook: {len(fast_items)} evals, {total_items} total items")

    _original_eval = sft_module.eval_all_datasets

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        _original_eval(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)

        print(f"\n--- Unfaithfulness Evals (step {global_step}) ---")

        for eval_name, items in fast_items.items():
            completed = run_eval_batched(
                model, tokenizer, items, act_layer,
                model_name=model_name, device=str(device),
                oracle_max_new_tokens=30,
            )

            parsing_config = EVAL_PARSING.get(eval_name)
            if parsing_config:
                metrics = score_eval(eval_name, completed, parsing_config)
                if metrics and "accuracy" in metrics:
                    display = _display_name(eval_name)
                    wandb.log({
                        f"unfaith/{display}/accuracy": metrics["accuracy"],
                        f"unfaith/{display}/n_scored": metrics.get("n_items", 0),
                    }, step=global_step)
                    print(f"  {display}: acc={metrics['accuracy']:.3f} ({metrics.get('n_items', 0)} scored)")

        print("--- End Unfaithfulness Evals ---\n")

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed unfaithfulness eval hook")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid fork warning from HF tokenizers
    parser = argparse.ArgumentParser(description="Train CoT Oracle — Mixed Training")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--persona-corpus", default=None, help="Path to persona corpus.jsonl")
    parser.add_argument("--labels-dir", default=None, help="Directory with label files (for eval)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU micro-batch size")
    parser.add_argument("--effective-batch-size", type=int, default=128, help="Global effective batch size (controls gradient accumulation)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_randlayer")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--wandb-run-id", default="", help="Resume a specific wandb run by ID")
    parser.add_argument("--min-evals-per-stage", type=int, default=3, help="Minimum evals per sequential stage")
    parser.add_argument("--max-evals-per-stage", type=int, default=10, help="Maximum evals per sequential stage")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--fast-eval-n", type=int, default=10)
    parser.add_argument("--no-unfaith-evals", action="store_true")
    parser.add_argument("--layer-mean", type=int, default=5, help="Poisson mean for number of layers per example")
    parser.add_argument("--layer-repeats", type=int, default=3, help="Replicate each example N times with different layer sets")
    parser.add_argument("--quantize-4bit", action="store_true", help="Load model in 4-bit (for <24GB GPUs)")
    parser.add_argument("--random-layers", action="store_true", help="Use Poisson-sampled random layer subsets instead of fixed [L25%%, L50%%, L75%%]")
    parser.add_argument("--no-position-encoding", action="store_true", help="Disable sinusoidal positional encoding on activation vectors")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable 2-stage curriculum (shuffle all tasks together)")
    parser.add_argument("--stage1-reg", type=float, default=0.2, help="Fraction of stage 2 size to mix in as stage 1 regularization")
    parser.add_argument("--position-stride", type=int, default=5, help="Fixed stride for position sampling from whole sequence")
    parser.add_argument("--max-positions", type=int, default=50, help="Cap positions per example (prevents huge prefixes)")
    parser.add_argument("--stage2-order", nargs="+", default=STAGE2_ORDER_DEFAULT, help="Order of stage-2 tasks (each becomes its own sub-stage)")
    parser.add_argument("--no-data-cache", action="store_true", help="Force rebuild training data (ignore persistent cache)")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint dir (reads training_state.json for examples_seen)")
    parser.add_argument("--num-workers", type=int, default=None, help="Workers for parallel data conversion (default: min(cpu_count, 8))")
    # Task size overrides
    parser.add_argument("--n-context-pred", type=int, default=100000)
    parser.add_argument("--n-sentence-pred", type=int, default=30000)
    parser.add_argument("--n-decorative", type=int, default=10000)
    parser.add_argument("--n-domain", type=int, default=15000)
    parser.add_argument("--n-correctness", type=int, default=15000)
    parser.add_argument("--n-persona", type=int, default=15000)
    parser.add_argument("--n-summary", type=int, default=15000)
    # Eval task args
    parser.add_argument("--eval-train-dir", default="data/evals_train", help="Directory with precomputed eval training JSONs")
    parser.add_argument("--n-eval-task", type=int, default=3000, help="Examples per eval training task")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]  # Still passed to dataset loaders (they need some layers to produce dicts)
    max_layers = LAYER_COUNTS[args.model]

    # Fixed layers by default; --random-layers opts into Poisson sampling
    fixed_layers = None
    if not args.random_layers:
        fixed_layers = [layer_percent_to_layer(args.model, p) for p in layer_percents]
        print(f"Fixed-layer mode: using layers {fixed_layers}")
    else:
        print(f"Random-layer mode: Poisson(mean={args.layer_mean})")

    task_sizes = {
        "cot_context_prediction": args.n_context_pred,
        "cot_sentence_prediction": args.n_sentence_pred,
        "cot_decorative": args.n_decorative,
        "cot_domain": args.n_domain,
        "cot_correctness": args.n_correctness,
        "cot_persona": args.n_persona,
        "cot_summary": args.n_summary,
        "_eval_train_dir": args.eval_train_dir,
        "_n_eval_task": args.n_eval_task,
    }

    # Initialize distributed early so we can gate data building on rank 0
    import torch.distributed as dist
    from datetime import timedelta
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    # Compute gradient accumulation to keep effective batch size constant across GPU counts
    grad_accum = args.effective_batch_size // (args.batch_size * world_size)
    grad_accum = max(1, grad_accum)
    effective_bs = args.batch_size * world_size * grad_accum
    if rank == 0:
        print(f"GPU-independent training: effective_batch={effective_bs}, "
              f"micro_batch={args.batch_size}, world_size={world_size}, grad_accum={grad_accum}")

    # Suppress tqdm on non-rank-0 processes
    if rank != 0:
        os.environ["TQDM_DISABLE"] = "1"

    # Persistent data cache params (everything that affects generated data)
    cache_extra = dict(
        layer_mean=args.layer_mean,
        max_layers=max_layers,
        layer_percents=layer_percents,
        position_stride=args.position_stride,
        max_positions=args.max_positions,
        labels_dir=args.labels_dir,
        layer_repeats=args.layer_repeats,
        fixed_layers=str(fixed_layers) if fixed_layers else None,
        eval_train_dir=args.eval_train_dir,
        n_eval_task=args.n_eval_task,
    )

    # Build data on rank 0, share via temp pickle to other ranks
    import pickle
    data_pickle = Path(args.save_dir) / "_data_cache.pkl"
    data_pickle.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        # Try persistent cache first
        cached = None
        if not args.no_data_cache:
            cached = load_cached_data(
                args.corpus, args.persona_corpus, task_sizes, args.model, **cache_extra,
            )

        if cached is not None:
            final_training, eval_datasets = cached
        else:
            # Ensure boundary positions + pre-tokenize once
            print("Ensuring boundary_positions are computed...")
            corpus_entries = ensure_boundary_positions(args.corpus, args.model, num_workers=args.num_workers)
            pretokenize_corpus(corpus_entries, args.model, num_workers=args.num_workers)

            persona_entries = None
            if args.persona_corpus and Path(args.persona_corpus).exists():
                persona_entries = ensure_boundary_positions(args.persona_corpus, args.model, num_workers=args.num_workers)
                pretokenize_corpus(persona_entries, args.model, num_workers=args.num_workers)

            mode = "fixed-layer" if fixed_layers else f"random-layer (mean={args.layer_mean})"
            print(f"Building {mode} training data (max_layers={max_layers})...")
            training_data = build_training_mixture(
                args.corpus, args.persona_corpus, args.labels_dir,
                tokenizer, args.model, layer_percents, max_layers, args.layer_mean,
                task_sizes, args.position_stride, args.max_positions,
                num_workers=args.num_workers,
                corpus_entries=corpus_entries, persona_entries=persona_entries,
                layer_repeats=args.layer_repeats, fixed_layers=fixed_layers,
            )

            assert training_data, "No training data generated!"

            eval_datasets = build_eval_datasets(
                args.corpus, args.labels_dir, tokenizer, args.model, layer_percents,
                max_layers, args.layer_mean, args.position_stride, args.max_positions,
                corpus_entries=corpus_entries, fixed_layers=fixed_layers,
            )

            # Split off 100 examples per training task as eval
            by_type = defaultdict(list)
            for dp in training_data:
                by_type[dp.datapoint_type].append(dp)

            final_training = []
            for dpt, dps in by_type.items():
                if len(dps) > 100:
                    eval_datasets[dpt] = dps[-100:]
                    final_training.extend(dps[:-100])
                else:
                    final_training.extend(dps)

            print(f"\nTraining: {len(final_training)}, Eval: {sum(len(v) for v in eval_datasets.values())}")
            for name, items in eval_datasets.items():
                print(f"  eval/{name}: {len(items)} items")

            # Save to persistent cache
            save_cached_data(
                final_training, eval_datasets,
                args.corpus, args.persona_corpus, task_sizes, args.model, **cache_extra,
            )

        # Share with other ranks via temp pickle
        with open(data_pickle, "wb") as f:
            pickle.dump((final_training, eval_datasets), f)

    dist.barrier()  # wait for rank 0 to finish

    if rank != 0:
        with open(data_pickle, "rb") as f:
            final_training, eval_datasets = pickle.load(f)

    dist.barrier()  # all ranks loaded
    if rank == 0:
        data_pickle.unlink(missing_ok=True)

    # Download AO checkpoint
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
        eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run or f"cot_oracle_randlayer_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_local_path,
        eval_on_start=True,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    model_kwargs = {"attn_implementation": "sdpa"}
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Check if all data uses a single stride value (skip stride panels if so)
    all_strides = {dp.meta_info.get("stride") for dp in final_training if dp.meta_info.get("stride") is not None}
    single_stride = len(all_strides) <= 1

    # Install hooks
    install_multilayer_materialization(position_encoding=not args.no_position_encoding)
    install_third_eval_hook(max_layers, single_stride=single_stride)

    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
        )

    install_third_task_loss_hook(max_layers, single_stride=single_stride)

    # Login to wandb (supports WANDB_API_KEY env var or .netrc)
    import wandb
    wandb.login()

    # Support resuming a wandb run by ID (AO lib doesn't expose this in config)
    if args.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"

    # Build curriculum or sequential ordering
    import numpy as np

    # Sequential priority: context_prediction and sentence_prediction lead
    SEQUENTIAL_PRIORITY = ["cot_context_prediction", "cot_sentence_prediction"]

    stage_boundaries = []
    task_order = []  # (task_name, count) for stage boundary labels
    if args.no_curriculum:
        # Sequential ordering: large generative tasks first, then smaller tasks by count
        rng = random.Random(42)
        by_type = defaultdict(list)
        for dp in final_training:
            by_type[dp.datapoint_type].append(dp)

        ordered = []
        seen = set()
        # Priority tasks first (context_prediction, sentence_prediction)
        for dpt in SEQUENTIAL_PRIORITY:
            if dpt in by_type:
                dps = by_type[dpt]
                rng.shuffle(dps)
                ordered.extend(dps)
                seen.add(dpt)
                task_order.append((dpt, len(dps)))

        # Remaining tasks sorted by count (largest first)
        remaining = [(dpt, dps) for dpt, dps in by_type.items() if dpt not in seen]
        remaining.sort(key=lambda x: -len(x[1]))
        for dpt, dps in remaining:
            rng.shuffle(dps)
            ordered.extend(dps)
            task_order.append((dpt, len(dps)))

        # Compute stage boundaries (cumulative example counts at task transitions)
        cumulative = 0
        for dpt, count in task_order[:-1]:  # no boundary needed after last task
            cumulative += count
            stage_boundaries.append(cumulative)

        final_training = ordered
        print(f"Sequential ordering: {len(final_training)} examples (large tasks first)")
    else:
        stage1 = [dp for dp in final_training if dp.datapoint_type in STAGE1_TASKS]
        stage2 = [dp for dp in final_training if dp.datapoint_type in STAGE2_TASKS]
        final_training, stage_boundaries = build_curriculum(
            stage1, stage2, stage1_reg=args.stage1_reg, stage2_order=args.stage2_order,
        )

    # Print comprehensive training summary
    print(f"\n{'=' * 72}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 72}")
    print(f"  Model:            {cfg.model_name}")
    print(f"  AO checkpoint:    {cfg.load_lora_path}")
    print(f"  LR:               {cfg.lr}")
    print(f"  Batch:            effective={args.effective_batch_size} (micro={args.batch_size} x gpus={world_size} x accum={grad_accum})")
    print(f"  Epochs:           {cfg.num_epochs}")
    print(f"  Total steps:      ~{len(final_training) // effective_bs}")
    print(f"  Save dir:         {cfg.save_dir}")
    print(f"  Layers:           {fixed_layers or f'random Poisson(mean={args.layer_mean})'}")
    print(f"  Position stride:  {args.position_stride}, max positions: {args.max_positions}")

    # Per-task statistics table
    by_type_train = defaultdict(list)
    for dp in final_training:
        by_type_train[dp.datapoint_type].append(len(dp.input_ids))

    print(f"\n{'TRAINING TASKS':^72}")
    print(f"{'─' * 72}")
    print(f"  {'Task':<30} {'N':>8} {'Avg Len':>10} {'Std':>8}")
    print(f"  {'─' * 60}")

    # Print in order of appearance (preserves sequential ordering)
    seen_types = []
    seen_set = set()
    for dp in final_training:
        if dp.datapoint_type not in seen_set:
            seen_set.add(dp.datapoint_type)
            seen_types.append(dp.datapoint_type)

    for dpt in seen_types:
        lengths = by_type_train[dpt]
        name = _display_name(dpt)
        avg = np.mean(lengths)
        std = np.std(lengths)
        print(f"  {name:<30} {len(lengths):>8} {avg:>10.0f} {std:>8.0f}")

    total_train = len(final_training)
    all_lengths = [l for lengths in by_type_train.values() for l in lengths]
    print(f"  {'─' * 60}")
    print(f"  {'TOTAL':<30} {total_train:>8} {np.mean(all_lengths):>10.0f} {np.std(all_lengths):>8.0f}")

    # Eval datasets table
    print(f"\n{'EVAL DATASETS':^72}")
    print(f"{'─' * 72}")
    print(f"  {'Task':<30} {'N':>8} {'Avg Len':>10} {'Std':>8}")
    print(f"  {'─' * 60}")
    for name in sorted(eval_datasets.keys()):
        items = eval_datasets[name]
        lengths = [len(dp.input_ids) for dp in items]
        dname = _display_name(name)
        avg = np.mean(lengths)
        std = np.std(lengths)
        print(f"  {dname:<30} {len(items):>8} {avg:>10.0f} {std:>8.0f}")
    print(f"{'=' * 72}")

    # Build adaptive per-stage eval schedule in EXAMPLES-SEEN space (GPU-independent).
    # Schedules use cumulative example counts, not steps, so they're valid across any GPU count.
    total_examples = len(final_training)
    # Stage boundaries already in example-space; add 0 and total
    example_boundaries = [0] + list(stage_boundaries) + [total_examples]
    eval_step_set = set()  # these are examples_seen values, not optimizer steps
    for i in range(len(example_boundaries) - 1):
        stage_start = example_boundaries[i]
        stage_end = example_boundaries[i + 1]
        stage_len = stage_end - stage_start
        # Scale n_evals by stage size: ~1 eval per 2000 examples, clamped [3, 10]
        n_evals = max(args.min_evals_per_stage, min(args.max_evals_per_stage, stage_len // 2000))
        stride = max(1, stage_len // n_evals)
        for j in range(n_evals):
            eval_step_set.add(stage_start + j * stride)
        eval_step_set.add(stage_end)  # always eval at stage boundary
    eval_step_set.discard(0)  # step 0 handled by eval_on_start
    # Repeat for each epoch
    epoch_eval_points = set(eval_step_set)
    for epoch in range(1, cfg.num_epochs):
        for s in epoch_eval_points:
            eval_step_set.add(s + epoch * total_examples)
    cfg.eval_step_set = eval_step_set
    # Save checkpoints only at stage boundaries + epoch ends (not every eval)
    save_step_set = set()
    for b in example_boundaries[1:]:  # skip 0, include all stage boundaries + total
        for epoch in range(cfg.num_epochs):
            save_step_set.add(b + epoch * total_examples)
    save_step_set.discard(0)
    cfg.save_step_set = save_step_set

    # Store curriculum info for wandb
    stages = []
    if task_order:
        cumulative = 0
        for name, count in task_order:
            stages.append({"task": _display_name(name), "examples": count, "start": cumulative, "end": cumulative + count})
            cumulative += count
    cfg.curriculum_info = {
        "stages": stages,
        "stage_boundaries": list(stage_boundaries),
        "eval_schedule_epoch1": sorted(s for s in eval_step_set if s <= total_examples),
        "save_schedule_epoch1": sorted(s for s in save_step_set if s <= total_examples),
        "n_evals_total": len(eval_step_set),
        "n_saves_total": len(save_step_set),
        "examples_per_epoch": total_examples,
    }

    # Handle resume from checkpoint
    if args.resume_from:
        state_path = os.path.join(args.resume_from, "training_state.json")
        with open(state_path) as f:
            state = json.load(f)
        cfg.resume_from_examples = state["examples_seen"]
        cfg.load_lora_path = args.resume_from
        # Auto-resume wandb run if not explicitly set
        if not args.wandb_run_id and state.get("wandb_run_id"):
            os.environ["WANDB_RUN_ID"] = state["wandb_run_id"]
            os.environ["WANDB_RESUME"] = "allow"
        print(f"  Resuming from {args.resume_from}: {cfg.resume_from_examples} examples seen"
              f" (wandb run: {state.get('wandb_run_id', 'new')})")

    if stage_boundaries and task_order:
        for i, boundary in enumerate(stage_boundaries):
            print(f"  Stage boundary: {_display_name(task_order[i][0])}→{_display_name(task_order[i+1][0])} at example {boundary}")
    elif stage_boundaries:
        for i, boundary in enumerate(stage_boundaries):
            print(f"  Curriculum: stage {i+1}→{i+2} at example {boundary}")
    print(f"  Eval schedule: {len(eval_step_set)} evals across {len(example_boundaries)-1} stages × {cfg.num_epochs} epochs")
    print(f"  Save schedule: {len(save_step_set)} checkpoints (stage boundaries + epoch ends)")
    print(f"  Eval at (epoch 1): {sorted(s for s in eval_step_set if s <= total_examples)}")
    print(f"  Save at (epoch 1): {sorted(s for s in save_step_set if s <= total_examples)}")

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
