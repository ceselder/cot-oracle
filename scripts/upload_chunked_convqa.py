#!/usr/bin/env python3
"""
Upload chunked ConvQA dataset to HuggingFace with train/test split.

Single dataset, two splits. Training loader pulls train split,
eval handler pulls test split.

Usage:
    python scripts/upload_chunked_convqa.py --dry-run
    python scripts/upload_chunked_convqa.py
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
REPO_ID = "mats-10-sprint-cs-jb/cot-oracle-convqa-chunked"


def main():
    parser = argparse.ArgumentParser(description="Upload chunked ConvQA to HF")
    parser.add_argument("--input", default="data/chunked_qa/chunked_convqa.jsonl")
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
    print(f"\n  By query_type:")
    for qt, cnt in sorted(Counter(df["query_type"]).items()):
        print(f"    {qt}: {cnt}")
    print(f"\n  By source:")
    for src, cnt in sorted(Counter(df["source"]).items()):
        print(f"    {src}: {cnt}")

    scored = df[df["bb_correct"].notna()]
    if len(scored) > 0:
        n_correct = scored["bb_correct"].sum()
        print(f"\n  BB correctness: {int(n_correct)}/{len(scored)} = {n_correct/len(scored):.1%}")

    # Train/test split by cot_id (all queries for a CoT go to same split)
    cot_ids = sorted(df["cot_id"].unique())
    rng = random.Random(args.seed)
    rng.shuffle(cot_ids)
    test_end = int(len(cot_ids) * args.test_fraction)
    test_ids = set(cot_ids[:test_end])
    train_ids = set(cot_ids[test_end:])

    df_train = df[df["cot_id"].isin(train_ids)]
    df_test = df[df["cot_id"].isin(test_ids)]
    print(f"\n  Split: {len(df_train)} train ({len(train_ids)} CoTs), {len(df_test)} test ({len(test_ids)} CoTs)")

    # Save parquets
    output_dir = PROJECT_ROOT / "data" / "hf_uploads" / "convqa_chunked"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    print(f"  Saved {len(df_train)} train → {train_path}")
    print(f"  Saved {len(df_test)} test → {test_path}")

    # Build README
    type_dist = Counter(df["query_type"])
    type_table = "\n".join(f"| `{qt}` | {cnt} |" for qt, cnt in sorted(type_dist.items()))
    source_dist = Counter(df["source"])
    source_table = "\n".join(f"| {src} | {cnt} |" for src, cnt in sorted(source_dist.items()))

    readme = f"""---
tags:
  - cot-oracle
  - chain-of-thought
  - reasoning-analysis
  - eval
license: mit
---

# CoT Oracle: Chunked ConvQA

Sentence-chunked conversational QA where queries ask about the **suffix** (future reasoning)
of a chain-of-thought. Designed to test whether an oracle with activation access can answer
questions about unseen future reasoning that a black-box text monitor cannot.

Each CoT is truncated at 25%, 50%, and 75% of sentences. For each truncation point, 8 queries
across 4 categories probe the suffix.

## Splits

| Split | CoTs | Rows |
|-------|------|------|
| train | {len(train_ids)} | {len(df_train)} |
| test | {len(test_ids)} | {len(df_test)} |

## Query Types

| Type | Count |
|------|-------|
{type_table}

## Sources

| Source | Count |
|--------|-------|
{source_table}

**Total: {len(rows)} rows**

## Schema

| Field | Description |
|-------|-------------|
| `cot_id` | Source corpus ID |
| `source` | Task source (MATH, GSM8K, etc.) |
| `question` | Original problem |
| `cot_text` | Full chain-of-thought |
| `chunk_index` | Truncation point (0-based sentence index) |
| `num_chunks` | Total number of sentences |
| `next_chunk` | The sentence right after the truncation point |
| `query_type` | Category: general, thematic, behavior, reasoning |
| `query` | Question about the suffix |
| `gt_response` | Ground truth (Gemini sees prefix + suffix) |
| `bb_response` | Black-box monitor guess (Gemini sees only prefix) |
| `bb_correct` | Whether BB response matches ground truth |

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
        print(f"  Dry run — would upload to https://huggingface.co/datasets/{REPO_ID}")

    print("\nDone!")


if __name__ == "__main__":
    main()
