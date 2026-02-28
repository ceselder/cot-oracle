#!/usr/bin/env python3
"""
Upload infogap training dataset to HuggingFace in parquet format.

Reads JSONL files from data/precomputed/ for all 8 infogap tasks and uploads
as a single HF dataset with a task-type column.

Usage:
    python scripts/upload_infogap_dataset.py --dry-run
    python scripts/upload_infogap_dataset.py
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv(os.path.expanduser("~/.env"))

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ID = "mats-10-sprint-cs-jb/cot-oracle-infogap-v1"

INFOGAP_TASKS = [
    "early_answer_pred",
    "backtrack_pred",
    "error_pred",
    "self_correction",
    "verification",
    "branch_pred",
    "completion_pred",
    "remaining_strategy",
]


def load_all_tasks(input_dir: Path) -> list[dict]:
    all_items = []
    for task_name in INFOGAP_TASKS:
        jsonl_path = input_dir / f"{task_name}.jsonl"
        if not jsonl_path.exists():
            print(f"  WARNING: {jsonl_path} not found, skipping")
            continue
        count = 0
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))
                    count += 1
        print(f"  {task_name}: {count} examples")
    return all_items


def save_and_upload(items: list[dict], dry_run: bool):
    output_dir = PROJECT_ROOT / "data" / "hf_uploads" / "infogap"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "train.parquet"
    df = pd.DataFrame(items)
    df.to_parquet(parquet_path, index=False)

    n_items = len(items)
    num_bytes = parquet_path.stat().st_size

    type_dist = Counter(item["datapoint_type"] for item in items)
    type_table = "\n".join(f"| `{t}` | {c} |" for t, c in sorted(type_dist.items()))

    readme = f"""---
tags:
  - cot-oracle
  - chain-of-thought
  - information-gap
  - reasoning-analysis
license: mit
dataset_info:
  splits:
    - name: train
      num_bytes: {num_bytes}
      num_examples: {n_items}
---

# CoT Oracle: Information Gap Training Data

Partial-CoT training data designed to measure the gap between what text reveals
and what activations encode. Each example truncates a CoT at a strategic point
and asks a question that is hard for text-only monitors but answerable from
the model's internal representations.

## Task Types

| Task | Count |
|------|-------|
{type_table}

**Total: {n_items} examples**

## Tiers

- **Tier 1 (Binary Classification):** early_answer_pred, backtrack_pred, error_pred, self_correction, verification
- **Tier 2 (MCQ):** branch_pred, completion_pred
- **Tier 3 (Open-ended):** remaining_strategy

## Schema

Standard CoT Oracle training format: `datapoint_type`, `prompt`, `target_response`,
`layer`, `layers`, `num_positions`, `context_input_ids`, `context_positions`.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}", split="train")
```
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    print(f"\n  Saved {n_items} items to {parquet_path} ({num_bytes / 1e6:.1f} MB)")

    if not dry_run:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=token)
        api.upload_folder(folder_path=str(output_dir), repo_id=REPO_ID, repo_type="dataset")
        print(f"  Uploaded to https://huggingface.co/datasets/{REPO_ID}")
    else:
        print(f"  Dry run — would upload to https://huggingface.co/datasets/{REPO_ID}")


def main():
    parser = argparse.ArgumentParser(description="Upload infogap dataset to HF")
    parser.add_argument("--input-dir", default="data/precomputed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading infogap task data...")
    items = load_all_tasks(Path(args.input_dir))

    if not items:
        print("  No data found! Run generate_infogap_dataset.py first.")
        return

    type_dist = Counter(item["datapoint_type"] for item in items)
    print(f"\n  Task distribution:")
    for t, c in sorted(type_dist.items()):
        print(f"    {t}: {c}")

    save_and_upload(items, args.dry_run)
    print("\nDone!")


if __name__ == "__main__":
    main()
