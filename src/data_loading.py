"""
Unified data loading for the CoT Oracle.

Downloads precomputed JSONL from HuggingFace, caches locally, and returns
lists of dicts in the standard format expected by dicts_to_training_data().

Also provides load_fineweb_data() for PastLens-style context prediction
from FineWeb + LMSYS streaming (togglable auxiliary training data).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Generator

from tasks import TASKS, TaskDef, get_trainable_tasks


_HF_CACHE_DIR = Path(os.environ.get("COT_ORACLE_CACHE_DIR", "data/hf_cache"))


def load_task_data(
    task_name: str,
    split: str = "train",
    n: int | None = None,
) -> list[dict]:
    """Download {split}.jsonl from task's HF repo, return list of dicts.

    Args:
        task_name: Key in TASKS dict.
        split: "train" or "test".
        n: Max examples to return. None = all.

    Returns:
        List of dicts with keys: task, prompt, target_response,
        context_input_ids, context_positions, layers, (+ datapoint_type).
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {sorted(TASKS.keys())}")

    task_def = TASKS[task_name]
    local_path = _download_from_hf(task_def, split)

    data = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Normalize: ensure 'task' field exists
            if "task" not in item and "datapoint_type" in item:
                item["task"] = item["datapoint_type"]
            elif "task" not in item:
                item["task"] = task_name
            # Ensure datapoint_type exists (needed by dicts_to_training_data)
            if "datapoint_type" not in item:
                item["datapoint_type"] = task_def.legacy_datapoint_type or task_name
            # Ensure num_positions exists
            if "num_positions" not in item:
                item["num_positions"] = len(item.get("context_positions", []))
            # Ensure layer exists
            if "layer" not in item:
                layers = item.get("layers", [9, 18, 27])
                item["layer"] = layers[0] if layers else 9
            data.append(item)

    if n is not None and len(data) > n:
        random.shuffle(data)
        data = data[:n]

    return data


def load_all_training_data(task_config: dict[str, dict]) -> list[dict]:
    """Load train splits for all enabled tasks.

    Args:
        task_config: Maps task_name -> {"n": int, ...}.
            Set n: 0 to disable a task.

    Returns:
        Combined list of dicts from all enabled tasks, shuffled.
    """
    trainable = get_trainable_tasks()
    all_data: list[dict] = []

    for task_name, cfg in task_config.items():
        n = cfg.get("n", 0)
        if n <= 0:
            continue

        if task_name not in trainable:
            raise ValueError(
                f"Task {task_name!r} is not trainable. "
                f"Trainable tasks: {sorted(trainable.keys())}"
            )

        print(f"  [data] Loading {task_name} (n={n})...")
        items = load_task_data(task_name, split="train", n=n)

        if not items:
            raise RuntimeError(
                f"Task {task_name!r} is enabled (n={n}) but returned 0 items. "
                f"Check HF repo: {TASKS[task_name].hf_repo}"
            )

        print(f"  [data]   -> {len(items)} examples")
        all_data.extend(items)

    random.shuffle(all_data)
    if all_data:
        print(f"  [data] Total task examples: {len(all_data)}")
    return all_data


def _download_from_hf(task_def: TaskDef, split: str) -> Path:
    """Download a split from the task's HF repo, cache locally. Return local path."""
    cache_dir = _HF_CACHE_DIR / task_def.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{split}.jsonl"

    if local_path.exists():
        return local_path

    print(f"  [data] Downloading {task_def.hf_repo}/{split}.jsonl ...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=task_def.hf_repo,
            filename=f"{split}.jsonl",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        # hf_hub_download may place it in a subdirectory — symlink if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path and downloaded_path.exists():
            if not local_path.exists():
                local_path.symlink_to(downloaded_path)
    except Exception as e:
        # Try loading via HuggingFace datasets library as fallback
        try:
            _download_via_datasets_lib(task_def, split, local_path)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download {task_def.hf_repo}/{split}.jsonl: {e}\n"
                f"Fallback also failed: {e2}"
            ) from e2

    if not local_path.exists():
        raise FileNotFoundError(
            f"Download completed but file not found at {local_path}"
        )

    return local_path


def _download_via_datasets_lib(task_def: TaskDef, split: str, local_path: Path) -> None:
    """Fallback: use HuggingFace `datasets` library to load and save as JSONL."""
    from datasets import load_dataset

    ds = load_dataset(task_def.hf_repo, split=split)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")


# ── FineWeb context prediction (PastLens-style) ──


def _fineweb_text_generator(tokenizer) -> Generator[str, None, None]:
    """Stream a 50/50 mix of FineWeb pretrain and LMSYS chat text."""
    from datasets import load_dataset

    pretrain_ds = iter(load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    ))
    chat_ds = iter(load_dataset(
        "lmsys/lmsys-chat-1m", split="train", streaming=True,
    ))

    bos = tokenizer.bos_token or ""

    while True:
        # FineWeb sample
        try:
            yield bos + next(pretrain_ds)["text"]
        except StopIteration:
            pretrain_ds = iter(load_dataset(
                "HuggingFaceFW/fineweb", name="sample-10BT",
                split="train", streaming=True,
            ))
            continue

        # LMSYS chat sample
        try:
            conv = next(chat_ds)["conversation"]
            yield tokenizer.apply_chat_template(conv, tokenize=False)
        except (StopIteration, Exception):
            chat_ds = iter(load_dataset(
                "lmsys/lmsys-chat-1m", split="train", streaming=True,
            ))
            continue


def load_fineweb_data(
    tokenizer,
    model_name: str,
    n: int = 50000,
    max_context_tokens: int = 2000,
    stride: int = 5,
    layers: list[int] | None = None,
    min_target_tokens: int = 5,
    max_target_tokens: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Generate PastLens-style context prediction from FineWeb + LMSYS streaming.

    Each example: tokenize web/chat text (max max_context_tokens), extract
    stride-based positions, predict next or previous K tokens.
    Returns dicts compatible with dicts_to_training_data().

    Args:
        tokenizer: HuggingFace tokenizer.
        model_name: Model name for layer calculation.
        n: Number of examples to generate.
        max_context_tokens: Max tokens for activation context.
        stride: Position stride (same as CoT tasks).
        layers: Layer indices for activation extraction.
        min_target_tokens: Minimum tokens to predict.
        max_target_tokens: Maximum tokens to predict.
        seed: Random seed for reproducibility.
    """
    from cot_utils import layer_percent_to_layer

    rng = random.Random(seed)

    if layers is None:
        layers = [layer_percent_to_layer(model_name, p) for p in [25, 50, 75]]
    n_layers = len(layers)

    print(f"  [fineweb] Streaming FineWeb + LMSYS, generating {n} examples...")
    gen = _fineweb_text_generator(tokenizer)

    datapoints: list[dict] = []
    skipped = 0
    min_context_tokens = stride * 3  # need at least a few stride positions

    while len(datapoints) < n:
        text = next(gen)

        # Tokenize with headroom for target tokens
        input_ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_context_tokens + max_target_tokens,
        )["input_ids"]
        L = len(input_ids)

        if L < min_context_tokens + min_target_tokens:
            skipped += 1
            continue

        k_target = rng.randint(
            min_target_tokens, min(max_target_tokens, max(min_target_tokens, L // 4)),
        )
        direction = rng.choice(["future", "past"])

        if direction == "future":
            # Context = tokens[:cutoff+1], target = tokens after cutoff
            max_cutoff = min(L - k_target - 1, max_context_tokens - 1)
            if max_cutoff < min_context_tokens:
                skipped += 1
                continue
            cutoff = rng.randint(min_context_tokens, max_cutoff)

            context_ids = input_ids[:cutoff + 1]
            positions = list(range(0, len(context_ids), stride))
            if positions[-1] != len(context_ids) - 1:
                positions.append(len(context_ids) - 1)

            target_ids = input_ids[cutoff + 1: cutoff + 1 + k_target]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            prompt = f"Predict the next {k_target} tokens of this text."

        else:  # past
            # Activations start at act_start; target = k_target tokens before it
            act_start_max = min(L - stride, max_context_tokens) - 1
            if act_start_max < k_target:
                skipped += 1
                continue
            act_start = rng.randint(k_target, act_start_max)

            # Context includes everything up to the end of the activation span
            context_end = min(act_start + max_context_tokens, L)
            context_ids = input_ids[:context_end]
            positions = list(range(act_start, len(context_ids), stride))
            if positions[-1] != len(context_ids) - 1:
                positions.append(len(context_ids) - 1)

            target_ids = input_ids[act_start - k_target: act_start]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            prompt = f"Predict the {k_target} tokens that came before the activation region."

        # Repeat positions for each layer (same format as CoT tasks)
        all_positions = positions * n_layers

        datapoints.append({
            "datapoint_type": "fineweb_context_prediction",
            "task": "fineweb",
            "prompt": prompt,
            "target_response": target_text,
            "layer": layers[0],
            "num_positions": len(all_positions),
            "context_input_ids": context_ids,
            "context_positions": all_positions,
            "n_prompt_positions": 0,  # web text has no prompt/question positions
        })

        if len(datapoints) % 10000 == 0:
            print(f"  [fineweb] {len(datapoints)}/{n} examples...")

    if skipped > 0:
        print(f"  [fineweb] Skipped {skipped} short texts")

    print(f"  [fineweb] Generated {len(datapoints)} examples")
    return datapoints[:n]
