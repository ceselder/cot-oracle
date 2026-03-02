#!/usr/bin/env python3
"""
Upload chunked CompQA dataset to HuggingFace with train/test split.

Single dataset, two splits. Training loader pulls train split,
eval handler pulls test split.

Usage:
    python scripts/upload_chunked_compqa.py --dry-run
    python scripts/upload_chunked_compqa.py
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv(os.path.expanduser("~/.env"))

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ID = "mats-10-sprint-cs-jb/cot-oracle-compqa-chunked"


def main():
    parser = argparse.ArgumentParser(description="Upload chunked CompQA to HF")
    parser.add_argument("--input", default="data/chunked_compqa/chunked_compqa.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load JSONL
    rows = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {args.input}")

    df = pd.DataFrame(rows)

    # Stats
    print(f"\n  By category:")
    for cat, cnt in sorted(Counter(df["datapoint_type"]).items()):
        print(f"    {cat}: {cnt}")

    print(f"\n  Split position stats:")
    fracs = df["split_index"] / df["num_sentences"]
    print(f"    mean: {fracs.mean():.2f}, median: {fracs.median():.2f}, min: {fracs.min():.2f}, max: {fracs.max():.2f}")

    if "target_label" in df.columns:
        labeled = df[df["target_label"].notna()]
        if len(labeled) > 0:
            true_count = labeled["target_label"].sum()
            print(f"\n  Binary labels: {int(true_count)} True / {len(labeled) - int(true_count)} False ({true_count/len(labeled):.1%} positive)")

    # Train/test split by CoT text hash (stable across runs)
    # Use hash of cot_text to deterministically assign
    df["_hash"] = df["cot_text"].apply(lambda x: hash(x) % 10000)
    rng = random.Random(args.seed)
    all_hashes = sorted(df["_hash"].unique())
    rng.shuffle(all_hashes)
    test_end = int(len(all_hashes) * args.test_fraction)
    test_hashes = set(all_hashes[:test_end])

    df_train = df[~df["_hash"].isin(test_hashes)].drop(columns=["_hash"])
    df_test = df[df["_hash"].isin(test_hashes)].drop(columns=["_hash"])
    print(f"\n  Split: {len(df_train)} train, {len(df_test)} test")

    # Save parquets
    output_dir = PROJECT_ROOT / "data" / "hf_uploads" / "compqa_chunked"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    print(f"  Saved {len(df_train)} train -> {train_path}")
    print(f"  Saved {len(df_test)} test -> {test_path}")

    # Build README
    cat_dist = Counter(df["datapoint_type"])
    cat_table = "\n".join(f"| {cat} | {cnt} |" for cat, cnt in sorted(cat_dist.items()))

    label_info = ""
    if "target_label" in df.columns:
        labeled = df[df["target_label"].notna()]
        if len(labeled) > 0:
            true_count = int(labeled["target_label"].sum())
            false_count = len(labeled) - true_count
            label_info = f"\n**Binary label distribution:** {true_count} Yes / {false_count} No ({true_count/len(labeled):.0%} positive)\n"

    readme = f"""---
tags:
  - cot-oracle
  - chain-of-thought
  - reasoning-analysis
license: mit
---

# CoT Oracle: Chunked CompQA (Natural Split Points)

Computational QA dataset for training and evaluating chain-of-thought oracles.
Each Qwen3-8B chain-of-thought is split at a **natural, category-appropriate point**
chosen by Gemini 2.5 Flash Lite, with free-form answers generated from the suffix.

## Pipeline

- **Round 1:** Gemini sees the full CoT + category, picks a semantically meaningful split point.
- **Round 2:** Gemini sees the suffix + category-specific question, answers in free form.

Binary tasks (backtrack, self-correction, verification) include a `target_label` boolean.

## Categories

| Category | Description | Count |
|----------|-------------|-------|
| `cot_backtrack_pred` | Does the model revise or backtrack? | {cat_dist.get('cot_backtrack_pred', 0)} |
| `cot_self_correction` | Does the model notice and correct an error? | {cat_dist.get('cot_self_correction', 0)} |
| `cot_verification` | Does the model verify its work? | {cat_dist.get('cot_verification', 0)} |
| `cot_remaining_strategy` | Describe the remaining reasoning approach | {cat_dist.get('cot_remaining_strategy', 0)} |
{label_info}
## Splits

| Split | Rows |
|-------|------|
| train | {len(df_train)} |
| test | {len(df_test)} |

Split by CoT hash (all data for a given CoT goes to the same split).

## Schema

| Field | Description |
|-------|-------------|
| `question` | Original problem |
| `cot_text` | Full chain-of-thought text |
| `cot_prefix` | Prefix text (up to split point) |
| `cot_suffix` | Suffix text (after split point) |
| `prompt` | Oracle question (category-specific) |
| `target_response` | Free-form answer from suffix (Gemini Round 2) |
| `target_label` | Boolean for binary tasks (Yes->True, No->False); null for strategy |
| `datapoint_type` | Category identifier |
| `split_index` | Sentence index of split (0-based, last sentence of prefix) |
| `num_sentences` | Total sentences in the CoT |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}")
train = ds["train"]
test = ds["test"]
```
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    if not args.dry_run:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=token)
        api.upload_folder(folder_path=str(output_dir), repo_id=REPO_ID, repo_type="dataset")
        print(f"  Uploaded to https://huggingface.co/datasets/{REPO_ID}")
    else:
        print(f"  Dry run -- would upload to https://huggingface.co/datasets/{REPO_ID}")

    print("\nDone!")


if __name__ == "__main__":
    main()
