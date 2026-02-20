"""
Train CoT Oracle: Single-Run Mixed Training

Configurable task mixture (via --tasks / --task-size), including:
  - Context prediction — random positions
  - Context prediction — sentence boundaries
  - Decorative CoT
  - Correctness prediction
  - Conversational CoT Q/A
  - Domain classification (optional)
  - Persona detection (optional)
  - CoT summary (optional)

Context-prediction random-position items use 1 random layer per example.
Sentence-structured items use 3 activations per boundary (L25%, L50%, L75%).
Each sentence-structured example is DOUBLED: once with all 3 layers,
once with L50% only. This teaches the oracle to work with both formats.

Monkey-patches AO's materialize_missing_steering_vectors for multi-layer.
Starts from Adam's AO checkpoint.

Usage:
    # Generate corpus first (OpenRouter, no GPU)
    python src/data_pipeline/generate_cots.py --openrouter \
        --output data/cot_corpus_v4/corpus.jsonl

    # Generate persona corpus (OpenRouter, no GPU)
    python src/data_pipeline/generate_cots.py --openrouter --personas \
        --n-problems 1000 --output data/cot_corpus_v4/corpus_persona.jsonl

    # Train (requires torchrun even on single GPU)
    torchrun --nproc_per_node=1 src/train_mixed.py \
        --corpus data/cot_corpus_v4/corpus.jsonl \
        --tasks cot_context_prediction,cot_sentence_prediction,cot_decorative,cot_correctness,cot_conversation \
        --model Qwen/Qwen3-8B
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path

ensure_ao_repo_on_path()

import torch

from nl_probes.utils.dataset_utils import (
    create_training_datapoint,
    TrainingDataPoint,
    SPECIAL_TOKEN,
    find_pattern_in_tokens,
)

# Per-layer placeholder tokens: each layer gets a distinct token so the model
# knows which depth an activation came from before injection even happens.
# " ?" is kept for L50% (backward compat with AO pretrained checkpoint).
LAYER_TOKENS = [" @", " ?", " #"]  # L25%, L50%, L75%
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
from dataset_classes.cot_conversation import load_cot_conversation_data

# Held-out eval tasks
from dataset_classes.cot_answer_tracking import load_cot_answer_tracking_data

from cot_utils import (
    find_sentence_boundary_positions,
    layer_percent_to_layer,
    split_cot_into_sentences,
)


def ensure_boundary_positions(corpus_path: str, tokenizer) -> str:
    """Ensure all corpus entries have boundary_positions computed.

    If entries are missing boundary_positions (e.g., OpenRouter-generated corpus),
    compute them using the tokenizer. Rewrites the corpus file in-place.
    Returns the (possibly updated) corpus path.
    """
    import json

    entries = []
    needs_update = False
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append(entry)
                if not entry.get("boundary_positions"):
                    needs_update = True

    if not needs_update:
        print(f"All {len(entries)} corpus entries already have boundary_positions")
        return corpus_path

    print(f"Computing boundary_positions for {len(entries)} entries...")
    updated = 0
    for entry in entries:
        if entry.get("boundary_positions"):
            continue

        sentences = entry.get("sentences") or split_cot_into_sentences(entry["cot_response"])
        if len(sentences) < 2:
            continue

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)

        entry["boundary_positions"] = boundary_positions
        entry["sentences"] = sentences
        entry["n_sentences"] = len(sentences)
        updated += 1

    # Write back
    with open(corpus_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"  Updated {updated} entries with boundary_positions")
    return corpus_path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    title: str
    default_size: int
    requires_persona_corpus: bool = False
    requires_summaries: bool = False


def _load_task_context_prediction(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_context_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_sentence_prediction(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_sentence_prediction_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_decorative(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_decorative_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_domain(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_domain_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_correctness(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_correctness_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_persona(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del corpus_path, summaries_path
    if not persona_corpus_path:
        raise ValueError("persona corpus missing")
    return load_cot_persona_data(
        persona_corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


def _load_task_summary(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path
    if not summaries_path:
        raise ValueError("summaries missing")
    return load_cot_summary_data(
        corpus_path,
        summaries_path,
        tokenizer,
        model_name,
        layer_percents,
        num_examples=num_examples,
    )


def _load_task_conversation(
    corpus_path: str,
    persona_corpus_path: str | None,
    summaries_path: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int,
) -> list[dict]:
    del persona_corpus_path, summaries_path
    return load_cot_conversation_data(
        corpus_path, tokenizer, model_name, layer_percents, num_examples=num_examples
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "cot_context_prediction": TaskSpec(
        name="cot_context_prediction",
        title="Context Prediction — Random Positions",
        default_size=100000,
    ),
    "cot_sentence_prediction": TaskSpec(
        name="cot_sentence_prediction",
        title="Context Prediction — Sentence Boundaries",
        default_size=30000,
    ),
    "cot_decorative": TaskSpec(
        name="cot_decorative",
        title="Decorative CoT",
        default_size=10000,
    ),
    "cot_domain": TaskSpec(
        name="cot_domain",
        title="Domain Classification",
        default_size=15000,
    ),
    "cot_correctness": TaskSpec(
        name="cot_correctness",
        title="Correctness Prediction",
        default_size=15000,
    ),
    "cot_persona": TaskSpec(
        name="cot_persona",
        title="Persona Detection",
        default_size=15000,
        requires_persona_corpus=True,
    ),
    "cot_summary": TaskSpec(
        name="cot_summary",
        title="CoT Summary",
        default_size=15000,
        requires_summaries=True,
    ),
    "cot_conversation": TaskSpec(
        name="cot_conversation",
        title="Conversational CoT Q/A",
        default_size=20000,
    ),
}

TASK_LOADERS = {
    "cot_context_prediction": _load_task_context_prediction,
    "cot_sentence_prediction": _load_task_sentence_prediction,
    "cot_decorative": _load_task_decorative,
    "cot_domain": _load_task_domain,
    "cot_correctness": _load_task_correctness,
    "cot_persona": _load_task_persona,
    "cot_summary": _load_task_summary,
    "cot_conversation": _load_task_conversation,
}

TASK_ORDER = [
    "cot_context_prediction",
    "cot_sentence_prediction",
    "cot_decorative",
    "cot_domain",
    "cot_correctness",
    "cot_persona",
    "cot_summary",
    "cot_conversation",
]

# Opinionated default set for the current sprint.
DEFAULT_ENABLED_TASKS = [
    "cot_context_prediction",
    "cot_sentence_prediction",
    "cot_decorative",
    "cot_correctness",
    "cot_conversation",
]


def parse_enabled_tasks(tasks_arg: str) -> list[str]:
    raw = (tasks_arg or "").strip()
    if not raw or raw.lower() == "default":
        return list(DEFAULT_ENABLED_TASKS)
    if raw.lower() == "all":
        return list(TASK_ORDER)

    tasks = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {TASK_ORDER}")

    deduped = []
    for t in tasks:
        if t not in deduped:
            deduped.append(t)
    return deduped


def parse_task_size_overrides(task_size_args: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for raw in task_size_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --task-size '{raw}'. Expected format task_name=count")
        task_name, value_str = raw.split("=", 1)
        task_name = task_name.strip()
        value_str = value_str.strip()
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unknown task in --task-size: {task_name}")
        try:
            value = int(value_str)
        except ValueError as exc:
            raise ValueError(f"Invalid count in --task-size '{raw}'") from exc
        if value < 0:
            raise ValueError(f"Task size must be >= 0: {raw}")
        overrides[task_name] = value
    return overrides


def resolve_task_sizes(
    enabled_tasks: list[str],
    task_size_overrides: dict[str, int] | None = None,
    legacy_task_sizes: dict[str, int] | None = None,
) -> dict[str, int]:
    task_sizes = {task: TASK_SPECS[task].default_size for task in enabled_tasks}

    # Keep backwards compatibility with old --n-* flags.
    if legacy_task_sizes:
        for task, size in legacy_task_sizes.items():
            if task in task_sizes and size >= 0:
                task_sizes[task] = size

    if task_size_overrides:
        for task, size in task_size_overrides.items():
            if task in task_sizes:
                task_sizes[task] = size

    return task_sizes


def print_task_help() -> None:
    print("Available tasks:")
    for task in TASK_ORDER:
        spec = TASK_SPECS[task]
        reqs = []
        if spec.requires_persona_corpus:
            reqs.append("needs --persona-corpus")
        if spec.requires_summaries:
            reqs.append("needs summaries.jsonl")
        req_str = f" ({', '.join(reqs)})" if reqs else ""
        print(f"  {task:<24} default={spec.default_size:<6} {spec.title}{req_str}")

def _find_mixed_token_positions(
    token_ids: list[int],
    layer_token_ids: list[int],
    num_positions: int,
) -> list[int]:
    """Find positions of a repeating pattern of mixed tokens (e.g., [@, ?, #, @, ?, #, ...]).

    layer_token_ids: list of token IDs to cycle through (one per layer).
    num_positions: total number of placeholder tokens to find.
    Returns list of token positions in token_ids.
    """
    n_layers = len(layer_token_ids)
    token_set = set(layer_token_ids)
    positions = []

    for i, tid in enumerate(token_ids):
        if len(positions) == num_positions:
            break
        expected_tid = layer_token_ids[len(positions) % n_layers]
        if tid == expected_tid:
            positions.append(i)
        elif tid in token_set and not positions:
            # Haven't started matching yet, skip
            continue

    assert len(positions) == num_positions, (
        f"Expected {num_positions} mixed-token positions, found {len(positions)}"
    )
    return positions


def _create_multilayer_datapoint(
    item: dict,
    tokenizer,
    layers: list[int],
    use_per_layer_tokens: bool = True,
) -> TrainingDataPoint:
    """Create a TrainingDataPoint with multi-layer prefix (3 acts per sentence boundary).

    If use_per_layer_tokens=True (default):
      Prefix: "Layer: 9, 18, 27\n @ ? # @ ? # @ ? # \n"
      Each layer gets a distinct placeholder token so the model knows
      which depth each activation came from.

    If use_per_layer_tokens=False (legacy):
      Prefix: "Layer: 9, 18, 27\n ? ? ? ? ? ? ? ? ? \n"
      All positions use the same " ?" token.
    """
    orig_positions = item["context_positions"]
    num_positions = len(orig_positions) * len(layers)

    # Build custom multi-layer prefix
    layers_str = ", ".join(str(l) for l in layers)
    prefix = f"Layer: {layers_str}\n"

    if use_per_layer_tokens:
        # Cycle through per-layer tokens: @?#@?#@?#...
        for i in range(num_positions):
            prefix += LAYER_TOKENS[i % len(layers)]
    else:
        prefix += SPECIAL_TOKEN * num_positions

    prefix += " \n"

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

    # Find placeholder positions
    if use_per_layer_tokens:
        layer_token_ids = [
            tokenizer.encode(lt, add_special_tokens=False)[0] for lt in LAYER_TOKENS
        ]
        positions = _find_mixed_token_positions(full_prompt_ids, layer_token_ids, num_positions)
    else:
        positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    # Expand context_positions: [p1,p1,p1, p2,p2,p2, ...]
    expanded_ctx_positions = []
    for p in orig_positions:
        expanded_ctx_positions.extend([p] * len(layers))

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
        meta_info={"multi_layers": layers},
    )


def dicts_to_training_data(
    raw_data: list[dict],
    tokenizer,
    use_per_layer_tokens: bool = True,
) -> list[TrainingDataPoint]:
    """Convert dataset loader output to AO TrainingDataPoint objects.

    Handles both single-layer (context prediction) and multi-layer
    (sentence-structured tasks) formats.

    Single-layer items have 'layer' (int) — standard AO format.
    Multi-layer items have 'layers' (list[int]) — for each, we create:
      1. A 3-layer version with per-layer tokens (@?#) and 3*N placeholder tokens
      2. A single-layer L50% duplicate with standard "Layer: 18" prefix and N ? tokens
    This effectively doubles the sentence-structured training data.
    """
    training_data = []
    skipped = 0

    for item in raw_data:
        try:
            layers = item.get("layers")  # list[int] for multi-layer, None for single
            if layers and len(layers) > 1:
                # 1) Multi-layer version (3 acts per sentence boundary)
                dp_multi = _create_multilayer_datapoint(
                    item, tokenizer, layers,
                    use_per_layer_tokens=use_per_layer_tokens,
                )
                training_data.append(dp_multi)

                # 2) Single-layer L50% duplicate (standard AO format)
                mid_layer = layers[len(layers) // 2]  # Middle layer = 50%
                dp_single = create_training_datapoint(
                    datapoint_type=item["datapoint_type"],
                    prompt=item["prompt"],
                    target_response=item["target_response"],
                    layer=mid_layer,
                    num_positions=item["num_positions"],
                    tokenizer=tokenizer,
                    acts_BD=None,
                    feature_idx=-1,
                    context_input_ids=item["context_input_ids"],
                    context_positions=item["context_positions"],
                )
                training_data.append(dp_single)
            else:
                # Single-layer item (Task 1: context prediction)
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


def build_training_mixture(
    corpus_path: str,
    persona_corpus_path: str | None,
    labels_dir: str | None,
    tokenizer,
    model_name: str,
    layer_percents: list[int],
    enabled_tasks: list[str] | None = None,
    task_sizes: dict[str, int] | None = None,
    use_per_layer_tokens: bool = True,
) -> list[TrainingDataPoint]:
    """Build mixed training data from enabled task registry."""
    del labels_dir  # kept for call-site compatibility

    if enabled_tasks is None:
        enabled_tasks = list(DEFAULT_ENABLED_TASKS)
    if task_sizes is None:
        task_sizes = resolve_task_sizes(enabled_tasks)

    all_data = []
    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    summaries_exists = Path(summaries_path).exists()
    persona_exists = bool(persona_corpus_path and Path(persona_corpus_path).exists())

    print("\nSelected tasks:")
    for task in enabled_tasks:
        print(f"  - {task} (n={task_sizes.get(task, 0)})")

    for task_idx, task_name in enumerate(enabled_tasks, start=1):
        spec = TASK_SPECS[task_name]
        num_examples = task_sizes.get(task_name, spec.default_size)
        if num_examples <= 0:
            print(f"\n=== Task {task_idx}: {spec.title} ===")
            print("  Skipped (size set to 0)")
            continue

        if spec.requires_persona_corpus and not persona_exists:
            print(f"\n=== Task {task_idx}: {spec.title} ===")
            print(f"  Skipped (no persona corpus at {persona_corpus_path})")
            continue
        if spec.requires_summaries and not summaries_exists:
            print(f"\n=== Task {task_idx}: {spec.title} ===")
            print(f"  Skipped (no summaries at {summaries_path})")
            continue

        print(f"\n=== Task {task_idx}: {spec.title} ===")
        try:
            raw = TASK_LOADERS[task_name](
                corpus_path=corpus_path,
                persona_corpus_path=persona_corpus_path,
                summaries_path=summaries_path if summaries_exists else None,
                tokenizer=tokenizer,
                model_name=model_name,
                layer_percents=layer_percents,
                num_examples=num_examples,
            )
            data = dicts_to_training_data(raw, tokenizer, use_per_layer_tokens=use_per_layer_tokens)
            print(f"  Generated {len(data)} examples")
            all_data.extend(data)
        except ValueError as e:
            print(f"  Skipped ({e})")
        except Exception as e:
            print(f"  FAILED: {e}")

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
    use_per_layer_tokens: bool = True,
) -> dict[str, list[TrainingDataPoint]]:
    """Build held-out eval datasets."""
    eval_datasets = {}

    # Zero-shot: Answer Tracking (held out from training)
    if labels_dir:
        tracking_path = Path(labels_dir) / "labels_answer_tracking.jsonl"
        if tracking_path.exists():
            print("\n=== Eval: Answer Tracking (zero-shot, 100 items) ===")
            try:
                raw = load_cot_answer_tracking_data(
                    corpus_path, str(tracking_path), tokenizer, model_name, layer_percents,
                    num_examples=100,
                )
                data = dicts_to_training_data(
                    raw,
                    tokenizer,
                    use_per_layer_tokens=use_per_layer_tokens,
                )
                eval_datasets["cot_answer_tracking"] = data
                print(f"  Generated {len(data)} eval examples")
            except Exception as e:
                print(f"  Failed: {e}")

    # Summary eval (100 held-out items)
    summaries_path = str(Path(corpus_path).parent / "summaries.jsonl")
    if Path(summaries_path).exists():
        print("\n=== Eval: CoT Summary (100 items) ===")
        try:
            raw = load_cot_summary_data(
                corpus_path, summaries_path, tokenizer, model_name, layer_percents,
                num_examples=100, seed=999,  # Different seed for eval split
            )
            data = dicts_to_training_data(
                raw,
                tokenizer,
                use_per_layer_tokens=use_per_layer_tokens,
            )
            eval_datasets["cot_summary"] = data
            print(f"  Generated {len(data)} eval examples")
        except Exception as e:
            print(f"  Failed: {e}")

    return eval_datasets


def install_multilayer_materialization():
    """Monkey-patch materialize_missing_steering_vectors to handle multi-layer items.

    Multi-layer items have meta_info["multi_layers"] = [L25%, L50%, L75%].
    Their context_positions are expanded: [p1,p1,p1, p2,p2,p2, ...] where
    each position repeats len(multi_layers) times.

    During materialization, we cycle through multi_layers when picking
    activations: position i uses layer multi_layers[i % len(multi_layers)].
    """
    from peft import PeftModel
    from nl_probes.utils.dataset_utils import materialize_missing_steering_vectors as _orig_mat

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

        # Collect ALL needed layers (single-layer .layer + multi-layer meta_info)
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

            if multi_layers:
                # Multi-layer: cycle through layers for each position
                # context_positions = [p1,p1,p1, p2,p2,p2, ...]
                # layers cycle:       [L1,L2,L3, L1,L2,L3, ...]
                n_layers = len(multi_layers)
                vectors_list = []
                for i, pos in enumerate(positions_per_item[b]):
                    layer = multi_layers[i % n_layers]
                    adj_pos = pos + left_offsets[b]
                    acts_BLD = acts_by_layer[layer]
                    L = acts_BLD.shape[1]
                    if adj_pos < 0 or adj_pos >= L:
                        raise IndexError(
                            f"Multi-layer act index {adj_pos} out of range (L={L}) "
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
                if any(i < 0 or i >= L for i in idxs):
                    raise IndexError(
                        f"Act index out of range for item {b}: {idxs} with L={L}"
                    )
                vectors = acts_BLD[b, idxs, :].detach().contiguous()

            assert len(vectors.shape) == 2
            dp_new = dp.model_copy(deep=True)
            dp_new.steering_vectors = vectors
            new_batch[idx] = dp_new

        return new_batch

    # Monkey-patch in AO's sft module and dataset_utils
    import nl_probes.utils.dataset_utils as du_module
    du_module.materialize_missing_steering_vectors = patched_materialize
    sft_module.materialize_missing_steering_vectors = patched_materialize
    print("Installed multi-layer materialization patch")


def install_per_task_loss_hook():
    """Monkey-patch AO's training loop to log per-task loss to wandb."""
    import wandb
    import torch.nn.functional as F
    from nl_probes.sft import train_features_batch as _original_train
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook

    _batch_state = {"types": []}

    from nl_probes.sft import construct_batch as _original_construct
    def patched_construct_batch(batch_list, tokenizer, device):
        _batch_state["types"] = [dp.datapoint_type for dp in batch_list]
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
    """Monkey-patch AO's eval to run unfaithfulness evals alongside training evals."""
    import wandb
    from evals.common import load_eval_items
    from evals.score_oracle import score_eval, EVAL_PARSING
    from evals.run_evals import run_single_item

    eval_dir = Path(eval_dir)
    act_layer = layer_percent_to_layer(model_name, 50)

    fast_items = {}
    for eval_file in sorted(eval_dir.glob("*.json")):
        eval_name = eval_file.stem
        if eval_name in ("decorative_cot", "sentence_insertion"):
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
            try:
                completed = []
                for item in items:
                    result = run_single_item(
                        model, tokenizer, item, act_layer,
                        model_name=model_name, device=str(device),
                    )
                    completed.append(result)

                parsing_config = EVAL_PARSING.get(eval_name)
                if parsing_config:
                    metrics = score_eval(eval_name, completed, parsing_config)
                    if metrics:
                        wandb.log({
                            f"unfaith/{eval_name}/accuracy": metrics["accuracy"],
                            f"unfaith/{eval_name}/n_scored": metrics.get("n_items", 0),
                        }, step=global_step)
                        print(f"  {eval_name}: acc={metrics['accuracy']:.3f} ({metrics.get('n_items', 0)} scored)")

            except Exception as e:
                print(f"  {eval_name}: FAILED ({e})")

        print("--- End Unfaithfulness Evals ---\n")

    sft_module.eval_all_datasets = patched_eval_all_datasets
    print("Installed unfaithfulness eval hook")


def main():
    parser = argparse.ArgumentParser(description="Train CoT Oracle — Mixed Training")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--persona-corpus", default=None, help="Path to persona corpus.jsonl")
    parser.add_argument("--labels-dir", default=None, help="Directory with label files (for eval)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-dir", default="checkpoints/cot_oracle_mixed")
    parser.add_argument("--wandb-project", default="cot_oracle")
    parser.add_argument("--wandb-run", default="")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--fast-eval-n", type=int, default=10)
    parser.add_argument("--no-unfaith-evals", action="store_true")
    parser.add_argument(
        "--tasks",
        default="default",
        help=(
            "Comma-separated task list, or 'default', or 'all'. "
            "Example: cot_context_prediction,cot_sentence_prediction,cot_decorative,cot_correctness,cot_conversation"
        ),
    )
    parser.add_argument(
        "--task-size",
        action="append",
        default=[],
        help="Per-task size override. Format: task_name=count. Repeatable.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="Print available tasks and exit.")
    parser.add_argument("--per-layer-tokens", action="store_true", default=False,
                        help="Use distinct placeholder tokens per layer (@?#) instead of all ?")
    # Task size overrides
    parser.add_argument("--n-context-pred", type=int, default=100000)
    parser.add_argument("--n-sentence-pred", type=int, default=30000)
    parser.add_argument("--n-decorative", type=int, default=10000)
    parser.add_argument("--n-domain", type=int, default=15000)
    parser.add_argument("--n-correctness", type=int, default=15000)
    parser.add_argument("--n-persona", type=int, default=15000)
    parser.add_argument("--n-summary", type=int, default=15000)
    parser.add_argument("--n-conversation", type=int, default=20000)
    args = parser.parse_args()

    if args.list_tasks:
        print_task_help()
        return

    tokenizer = load_tokenizer(args.model)
    layer_percents = [25, 50, 75]

    enabled_tasks = parse_enabled_tasks(args.tasks)
    legacy_task_sizes = {
        "cot_context_prediction": args.n_context_pred,
        "cot_sentence_prediction": args.n_sentence_pred,
        "cot_decorative": args.n_decorative,
        "cot_domain": args.n_domain,
        "cot_correctness": args.n_correctness,
        "cot_persona": args.n_persona,
        "cot_summary": args.n_summary,
        "cot_conversation": args.n_conversation,
    }
    task_size_overrides = parse_task_size_overrides(args.task_size)
    task_sizes = resolve_task_sizes(
        enabled_tasks,
        task_size_overrides=task_size_overrides,
        legacy_task_sizes=legacy_task_sizes,
    )

    # Ensure boundary_positions are computed (needed for sentence-structured tasks)
    print("Ensuring boundary_positions are computed...")
    ensure_boundary_positions(args.corpus, tokenizer)
    if args.persona_corpus and Path(args.persona_corpus).exists():
        ensure_boundary_positions(args.persona_corpus, tokenizer)

    # Build training data
    print(f"Building mixed training data (per_layer_tokens={args.per_layer_tokens})...")
    training_data = build_training_mixture(
        args.corpus, args.persona_corpus, args.labels_dir,
        tokenizer, args.model, layer_percents,
        enabled_tasks=enabled_tasks,
        task_sizes=task_sizes,
        use_per_layer_tokens=args.per_layer_tokens,
    )

    if not training_data:
        print("ERROR: No training data generated!")
        return

    # Build eval datasets
    eval_datasets = build_eval_datasets(
        args.corpus,
        args.labels_dir,
        tokenizer,
        args.model,
        layer_percents,
        use_per_layer_tokens=args.per_layer_tokens,
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
        wandb_run_name=args.wandb_run or f"cot_oracle_mixed_{args.model.split('/')[-1]}",
        gradient_checkpointing=args.gradient_checkpointing,
        load_lora_path=lora_local_path,
        eval_on_start=True,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Initialize distributed (required by AO's train_model)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Install hooks
    install_multilayer_materialization()

    if not args.no_unfaith_evals and Path(args.eval_dir).exists():
        install_unfaithfulness_eval_hook(
            model_name=args.model,
            eval_dir=args.eval_dir,
            fast_n=args.fast_eval_n,
        )

    install_per_task_loss_hook()

    # Login to wandb via env var (newer wandb requires 40-char key format)
    import os
    assert os.environ.get("WANDB_API_KEY"), "Set WANDB_API_KEY env var"
    import wandb

    # Shuffle training data!
    random.seed(42)
    random.shuffle(final_training)
    print(f"Shuffled {len(final_training)} training examples")

    # Print training summary
    print(f"\nStarting training:")
    print(f"  Model: {cfg.model_name}")
    print(f"  AO checkpoint: {cfg.load_lora_path}")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Total steps: ~{len(final_training) // cfg.train_batch_size}")
    print(f"  Save dir: {cfg.save_dir}")
    print(f"  Tasks: {sorted(set(dp.datapoint_type for dp in final_training))}")

    train_model(
        cfg=cfg,
        training_data=final_training,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs={"attn_implementation": "sdpa"},
    )


if __name__ == "__main__":
    main()
