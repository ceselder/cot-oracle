"""
Unified data loading for the CoT Oracle.

Downloads precomputed JSONL from HuggingFace, caches locally, and returns
lists of dicts in the standard format expected by dicts_to_training_data().

Replaces the 30+ dataset loaders in dataset_classes/ and the task loading
machinery in train.py (TASK_REGISTRY, load_precomputed_tasks, etc.).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

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

    if not all_data:
        raise RuntimeError("No training data loaded — all tasks disabled or empty.")

    random.shuffle(all_data)
    print(f"  [data] Total training examples: {len(all_data)}")
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
