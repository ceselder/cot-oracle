"""
Train CoT Oracle: Random Layer Subset Training

Like train_mixed.py but samples random subsets of all 36 layers per example
(Poisson(5) layers on average) instead of the fixed 3 layers [25%, 50%, 75%].

New prefix format: "L5: ? ? ? ? L11: ? ? ? ?\n" (grouped per layer)
instead of "Layer: 9, 18, 27\n @ ? # @ ? # @ ? # \n" (cycling per position).

Quartile-binned eval: accuracy broken down by which depth quartiles are
present, logged as eval_quartile/{ds}/{bin}/accuracy to wandb.

Usage:
    torchrun --nproc_per_node=1 src/train_random_layers.py \
        --corpus data/cot_corpus_v4/corpus.jsonl \
        --persona-corpus data/cot_corpus_v4/corpus_persona.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tqdm.auto import tqdm

def _is_rank0():
    return os.environ.get("LOCAL_RANK", "0") == "0"

sys.path.insert(0, str(Path(__file__).parent))

# AO repo imports -- detect environment
_ao_candidates = [
    Path(__file__).resolve().parent.parent / "ao_reference",  # project-relative
    Path("/workspace/ao_reference"),  # vast.ai
    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),  # local
]
AO_REPO = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
sys.path.insert(0, str(AO_REPO))

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

from signs_of_life.ao_lib import layer_percent_to_layer, LAYER_COUNTS
from cot_utils import split_cot_into_sentences, find_sentence_boundary_positions
from layer_utils import (
    sample_layers,
    build_random_layer_prefix,
    find_all_special_positions,
    layers_to_quartile_bin,
)
from train_mixed import (
    ensure_boundary_positions,
    install_unfaithfulness_eval_hook,
)

STAGE1_TASKS = {"cot_context_prediction", "cot_sentence_prediction"}
STAGE2_TASKS = {"cot_decorative", "cot_domain", "cot_correctness", "cot_persona", "cot_summary"}


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
        meta_info={"multi_layers": layers, "num_pos_per_layer": num_pos_per_layer},
    )


def dicts_to_multilayer_training_data(
    raw_data: list[dict],
    tokenizer,
    max_layers: int = 36,
    layer_mean: int = 5,
) -> list[TrainingDataPoint]:
    """Convert dataset dicts to TrainingDataPoints with random layer subsets.

    Multi-layer items ('layers' key): sample random layers via Poisson.
    Single-layer items ('layer' key, e.g. context prediction): sample 1 random
    layer from the full range [0, max_layers) instead of the fixed 3 percentiles.
    """
    training_data = []
    skipped = 0

    for item in raw_data:
        layers_field = item.get("layers")  # list[int] for multi-layer, None for single
        if layers_field and len(layers_field) > 1:
            # Multi-layer item: sample random layers
            layers = sample_layers(max_layers, layer_mean)
            dp = _create_random_layer_datapoint(item, tokenizer, layers)
            training_data.append(dp)
        else:
            # Single-layer item: sample 1 random layer from full range
            layer = random.randint(0, max_layers - 1)
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
                meta_info={"multi_layers": [layer], "num_pos_per_layer": item["num_positions"]},
            )
            training_data.append(dp)

    if skipped > 0:
        print(f"  Skipped {skipped} datapoints during conversion")

    return training_data


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

def install_multilayer_materialization():
    """Monkey-patch materialize_missing_steering_vectors for grouped layer ordering.

    Grouped ordering (num_pos_per_layer in meta_info):
      positions: [p1,p2,...,pN, p1,p2,...,pN, ...]
      layer for position i: multi_layers[i // num_pos_per_layer]

    Legacy cycling ordering (no num_pos_per_layer):
      positions: [p1,p1,p1, p2,p2,p2, ...]
      layer for position i: multi_layers[i % n_layers]
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

            assert len(vectors.shape) == 2
            dp_new = dp.model_copy(deep=True)
            dp_new.steering_vectors = vectors
            new_batch[idx] = dp_new

        return new_batch

    # Monkey-patch
    import nl_probes.utils.dataset_utils as du_module
    du_module.materialize_missing_steering_vectors = patched_materialize
    sft_module.materialize_missing_steering_vectors = patched_materialize
    print("Installed multi-layer materialization patch (grouped ordering)")


def install_quartile_eval_hook(max_layers: int = 36):
    """Replace eval_all_datasets with a version that:
    1. Runs eval once per dataset (not twice)
    2. Logs standard accuracy + quartile-binned accuracy
    3. Logs a wandb Table with eval traces (prompt, label, prediction, layers, etc.)
    """
    import gc
    import wandb
    from nl_probes.utils.eval import run_evaluation, score_eval_responses, parse_answer

    def patched_eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step):
        model.eval()
        eval_metrics = {}
        table_rows = []

        for ds in eval_datasets:
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
            eval_metrics[f"eval_format_correct/{ds}"] = percent_format_correct
            eval_metrics[f"eval_ans_correct/{ds}"] = percent_ans_correct
            print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")

            # Quartile-binned accuracy + table rows
            by_bin = defaultdict(lambda: {"correct": 0, "total": 0})
            for resp, dp in zip(eval_responses, eval_data, strict=True):
                multi_layers = dp.meta_info.get("multi_layers", [dp.layer])
                qbin = layers_to_quartile_bin(multi_layers, max_layers)

                cleaned = parse_answer(resp.api_response)
                target = parse_answer(dp.target_output)
                correct = cleaned == target
                by_bin[qbin]["total"] += 1
                if correct:
                    by_bin[qbin]["correct"] += 1

                # Build the masked prompt (what AO actually sees — input_ids with labels=-100 kept)
                ao_prompt = resp.prompt  # decoded prompt from eval (already stripped of target)

                table_rows.append([
                    global_step,
                    ds,
                    dp.target_output,
                    resp.api_response.strip(),
                    correct,
                    str(multi_layers),
                    qbin,
                    ao_prompt,
                ])

            for qbin, counts in sorted(by_bin.items()):
                n = counts["total"]
                acc = counts["correct"] / n if n > 0 else 0.0
                eval_metrics[f"eval_quartile/{ds}/{qbin}/accuracy"] = acc
                eval_metrics[f"eval_quartile/{ds}/{qbin}/n"] = n

        # Log metrics
        if wandb.run is not None:
            wandb.log(eval_metrics, step=global_step)
            wandb.summary.update(eval_metrics)

            # Log eval traces table
            table = wandb.Table(columns=[
                "step", "task", "label", "prediction", "correct",
                "layers", "quartile_bin", "ao_prompt",
            ], data=table_rows)
            wandb.log({f"eval_traces/step_{global_step}": table}, step=global_step)
            print(f"  Logged {len(table_rows)} eval traces + {sum(1 for k in eval_metrics if 'quartile' in k)} quartile metrics")

        model.train()
        torch.cuda.empty_cache()
        gc.collect()

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed quartile eval hook (with trace table)")


def install_quartile_task_loss_hook(max_layers: int = 36):
    """Monkey-patch AO's training loop to log per-task AND per-quartile loss to wandb.

    Logs:
      train/loss_{task}           — per-task average loss (as before)
      train/loss_{task}/{qbin}    — per-task × per-quartile-bin loss
      train/loss_quartile/{qbin}  — per-quartile average loss (across all tasks)
    """
    import wandb
    import torch.nn.functional as F
    from nl_probes.sft import train_features_batch as _original_train
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

    _batch_state = {"types": [], "meta_infos": []}

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

            # Per-task losses
            task_losses = defaultdict(list)
            # Per-task × per-quartile losses
            task_quartile_losses = defaultdict(list)
            # Per-quartile losses (across all tasks)
            quartile_losses = defaultdict(list)

            for i, task_type in enumerate(batch_types):
                loss_val = per_item_loss[i].item()
                task_losses[task_type].append(loss_val)

                multi_layers = meta_infos[i].get("multi_layers", [])
                if multi_layers:
                    qbin = layers_to_quartile_bin(multi_layers, max_layers)
                    task_quartile_losses[f"{task_type}/{qbin}"].append(loss_val)
                    quartile_losses[qbin].append(loss_val)

            log_dict = {}
            for task, losses in task_losses.items():
                log_dict[f"train/loss_{task}"] = sum(losses) / len(losses)
            for key, losses in task_quartile_losses.items():
                log_dict[f"train/loss_{key}"] = sum(losses) / len(losses)
            for qbin, losses in quartile_losses.items():
                log_dict[f"train/loss_quartile/{qbin}"] = sum(losses) / len(losses)

            # Stage indicator: 1 if all tasks are stage 1, 2 if any stage 2 present
            has_stage2 = any(t in STAGE2_TASKS for t in batch_types)
            log_dict["train/stage"] = 2 if has_stage2 else 1

            if wandb.run is not None:
                wandb.log(log_dict, commit=False)

        return outputs.loss

    sft_module.train_features_batch = patched_train_features_batch
    print("Installed per-task × per-quartile loss logging hook")


def build_curriculum(
    stage1_data: list[TrainingDataPoint],
    stage2_data: list[TrainingDataPoint],
    stage1_reg: float = 0.2,
    seed: int = 42,
) -> list[TrainingDataPoint]:
    """Build a two-stage curriculum: stage 1 (token prediction) then stage 2 (classification).

    Stage 1: only stage1_data (context pred, sentence pred), shuffled.
    Stage 2: all stage2_data + a random subset of stage1_data as regularization, shuffled.

    Returns a single list where stage 1 examples come first, then stage 2.
    The AO training loop will iterate through them in order.
    """
    rng = random.Random(seed)

    s1 = list(stage1_data)
    rng.shuffle(s1)

    # Stage 2: classification tasks + some stage 1 for regularization
    n_reg = int(len(stage2_data) * stage1_reg)
    reg_subset = rng.sample(stage1_data, min(n_reg, len(stage1_data)))
    s2 = list(stage2_data) + reg_subset
    rng.shuffle(s2)

    print(f"\nCurriculum staging:")
    print(f"  Stage 1: {len(s1)} examples (token prediction)")
    print(f"  Stage 2: {len(s2)} examples ({len(stage2_data)} classification + {len(reg_subset)} regularization)")
    print(f"  Stage transition at step ~{len(s1) // 8}")  # rough, depends on batch size and DDP

    return s1 + s2


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
) -> list[TrainingDataPoint]:
    """Build the mixed training data from up to 7 tasks, with random layer subsets."""

    if task_sizes is None:
        task_sizes = {
            "cot_context_prediction": 100000,
            "cot_sentence_prediction": 30000,
            "cot_decorative": 10000,
            "cot_domain": 15000,
            "cot_correctness": 15000,
            "cot_persona": 15000,
        }

    all_data = []

    # Task 1: Context Prediction — Random Positions
    print("\n=== Task 1: Context Prediction — Random Positions ===")
    raw = load_cot_context_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_context_prediction", 100000),
    )
    data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 2: Context Prediction — Sentence Boundaries
    print("\n=== Task 2: Context Prediction — Sentence Boundaries ===")
    raw = load_cot_sentence_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_sentence_prediction", 30000),
    )
    data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 3: Decorative CoT
    print("\n=== Task 3: Decorative CoT ===")
    raw = load_cot_decorative_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_decorative", 10000),
    )
    data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 4: Domain Classification
    print("\n=== Task 4: Domain Classification ===")
    raw = load_cot_domain_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_domain", 15000),
    )
    data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 5: Correctness Prediction
    print("\n=== Task 5: Correctness Prediction ===")
    raw = load_cot_correctness_data(
        corpus_path, tokenizer, model_name, layer_percents,
        num_examples=task_sizes.get("cot_correctness", 15000),
    )
    data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
    print(f"  Generated {len(data)} examples")
    all_data.extend(data)

    # Task 6: Persona Detection
    if persona_corpus_path and Path(persona_corpus_path).exists():
        print("\n=== Task 6: Persona Detection ===")
        raw = load_cot_persona_data(
            persona_corpus_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("cot_persona", 15000),
        )
        data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
        print(f"  Generated {len(data)} examples")
        all_data.extend(data)
    else:
        print(f"\n  Skipping Task 6 (no persona corpus at {persona_corpus_path})")

    # Task 7: CoT Summary
    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    if Path(summaries_path).exists():
        print("\n=== Task 7: CoT Summary ===")
        raw = load_cot_summary_data(
            corpus_path, summaries_path, tokenizer, model_name, layer_percents,
            num_examples=task_sizes.get("cot_summary", 15000),
        )
        data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
        print(f"  Generated {len(data)} examples")
        all_data.extend(data)
    else:
        print(f"\n  Skipping Task 7 (no summaries at {summaries_path})")

    print(f"\n{'=' * 60}")
    print(f"Total training examples: {len(all_data)}")

    type_counts = Counter(dp.datapoint_type for dp in all_data)
    for dtype, count in sorted(type_counts.items()):
        pct = count / len(all_data) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    return all_data


def build_eval_datasets(
    corpus_path: str,
    labels_dir: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    max_layers: int,
    layer_mean: int,
) -> dict[str, list[TrainingDataPoint]]:
    """Build held-out eval datasets with random layer subsets (fixed seed for reproducibility)."""
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
                num_examples=100,
            )
            data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
            eval_datasets["cot_answer_tracking"] = data
            print(f"  Generated {len(data)} eval examples")

    # Summary eval
    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    if Path(summaries_path).exists():
        print("\n=== Eval: CoT Summary (100 items) ===")
        raw = load_cot_summary_data(
            corpus_path, summaries_path, tokenizer, model_name, layer_percents,
            num_examples=100, seed=999,
        )
        data = dicts_to_multilayer_training_data(raw, tokenizer, max_layers, layer_mean)
        eval_datasets["cot_summary"] = data
        print(f"  Generated {len(data)} eval examples")

    random.setstate(saved_state)
    return eval_datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle — Random Layer Subsets")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--persona-corpus", default=None, help="Path to persona corpus.jsonl")
    parser.add_argument("--labels-dir", default=None, help="Directory with label files (for eval)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_randlayer")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--fast-eval-n", type=int, default=10)
    parser.add_argument("--no-unfaith-evals", action="store_true")
    parser.add_argument("--layer-mean", type=int, default=5, help="Poisson mean for number of layers per example")
    parser.add_argument("--quantize-4bit", action="store_true", help="Load model in 4-bit (for <24GB GPUs)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable 2-stage curriculum (shuffle all tasks together)")
    parser.add_argument("--stage1-reg", type=float, default=0.2, help="Fraction of stage 2 size to mix in as stage 1 regularization")
    # Task size overrides
    parser.add_argument("--n-context-pred", type=int, default=100000)
    parser.add_argument("--n-sentence-pred", type=int, default=30000)
    parser.add_argument("--n-decorative", type=int, default=10000)
    parser.add_argument("--n-domain", type=int, default=15000)
    parser.add_argument("--n-correctness", type=int, default=15000)
    parser.add_argument("--n-persona", type=int, default=15000)
    parser.add_argument("--n-summary", type=int, default=15000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]  # Still passed to dataset loaders (they need some layers to produce dicts)
    max_layers = LAYER_COUNTS[args.model]

    task_sizes = {
        "cot_context_prediction": args.n_context_pred,
        "cot_sentence_prediction": args.n_sentence_pred,
        "cot_decorative": args.n_decorative,
        "cot_domain": args.n_domain,
        "cot_correctness": args.n_correctness,
        "cot_persona": args.n_persona,
        "cot_summary": args.n_summary,
    }

    # Initialize distributed early so we can gate data building on rank 0
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # Suppress tqdm on non-rank-0 processes
    if rank != 0:
        os.environ["TQDM_DISABLE"] = "1"

    # Build data on rank 0, save to temp pickle, other ranks load it
    import pickle
    import tempfile
    data_pickle = Path(args.save_dir) / "_data_cache.pkl"
    data_pickle.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("Ensuring boundary_positions are computed...")
        ensure_boundary_positions(args.corpus, tokenizer)
        if args.persona_corpus and Path(args.persona_corpus).exists():
            ensure_boundary_positions(args.persona_corpus, tokenizer)

        print(f"Building random-layer training data (layer_mean={args.layer_mean}, max_layers={max_layers})...")
        training_data = build_training_mixture(
            args.corpus, args.persona_corpus, args.labels_dir,
            tokenizer, args.model, layer_percents, max_layers, args.layer_mean,
            task_sizes,
        )

        assert training_data, "No training data generated!"

        eval_datasets = build_eval_datasets(
            args.corpus, args.labels_dir, tokenizer, args.model, layer_percents,
            max_layers, args.layer_mean,
        )

        # Split off 100 examples per training task as eval
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

        with open(data_pickle, "wb") as f:
            pickle.dump((final_training, eval_datasets), f)
        print(f"Saved data cache to {data_pickle}")

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
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
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

    # Install hooks
    install_multilayer_materialization()
    install_quartile_eval_hook(max_layers)

    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
        )

    install_quartile_task_loss_hook(max_layers)

    # Login to wandb (supports WANDB_API_KEY env var or .netrc)
    import wandb
    wandb.login()

    # Build curriculum or shuffle
    stage_transition_step = None
    if args.no_curriculum:
        random.seed(42)
        random.shuffle(final_training)
        print(f"Shuffled {len(final_training)} training examples (no curriculum)")
    else:
        stage1 = [dp for dp in final_training if dp.datapoint_type in STAGE1_TASKS]
        stage2 = [dp for dp in final_training if dp.datapoint_type in STAGE2_TASKS]
        final_training = build_curriculum(stage1, stage2, stage1_reg=args.stage1_reg)
        # Approximate step where stage 2 begins (before DDP sharding, so rough)
        stage_transition_step = len(stage1) // args.batch_size

    # Print training summary
    print(f"\nStarting training:")
    print(f"  Model: {cfg.model_name}")
    print(f"  AO checkpoint: {cfg.load_lora_path}")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Total steps: ~{len(final_training) // cfg.train_batch_size}")
    print(f"  Save dir: {cfg.save_dir}")
    print(f"  Layer mean: {args.layer_mean}, Max layers: {max_layers}")
    print(f"  Tasks: {sorted(set(dp.datapoint_type for dp in final_training))}")
    if stage_transition_step is not None:
        print(f"  Curriculum: stage 1→2 at ~step {stage_transition_step}")

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
