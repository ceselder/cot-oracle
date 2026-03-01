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
    print(f"\n  By source:")
    for src, cnt in sorted(Counter(df["source"]).items()):
        print(f"    {src}: {cnt}")

    print(f"\n  Split position stats:")
    fracs = df["split_index"] / df["num_sentences"]
    print(f"    mean: {fracs.mean():.2f}, median: {fracs.median():.2f}, min: {fracs.min():.2f}, max: {fracs.max():.2f}")

    scored = df[df["bb_correct"].notna()]
    if len(scored) > 0:
        n_correct = scored["bb_correct"].sum()
        print(f"\n  BB correctness: {int(n_correct)}/{len(scored)} = {n_correct/len(scored):.1%}")

    # Train/test split by cot_id (all rows for a CoT go to same split)
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
    source_dist = Counter(df["source"])
    source_table = "\n".join(f"| {src} | {cnt} |" for src, cnt in sorted(source_dist.items()))

    scored = df[df["bb_correct"].notna()]
    bb_acc_str = f"{scored['bb_correct'].sum() / len(scored):.1%}" if len(scored) > 0 else "N/A"

    # Per-source BB accuracy
    source_acc_rows = []
    for src in sorted(source_dist.keys()):
        src_scored = scored[scored["source"] == src]
        if len(src_scored) > 0:
            acc = src_scored["bb_correct"].sum() / len(src_scored)
            source_acc_rows.append(f"| {src} | {len(df[df['source'] == src])} | {acc:.0%} |")
        else:
            source_acc_rows.append(f"| {src} | {len(df[df['source'] == src])} | — |")
    source_acc_table = "\n".join(source_acc_rows)

    readme = f"""---
tags:
  - cot-oracle
  - chain-of-thought
  - reasoning-analysis
  - eval
license: mit
---

# CoT Oracle: Chunked ConvQA (Natural Split Points)

Information-gap QA dataset for training and evaluating chain-of-thought oracles.

Each Qwen3-8B chain-of-thought is split at a **natural turning point** chosen by
Gemini 2.5 Flash Lite. At the split, a tailored open-ended question is generated
about the suffix content that is not obvious from reading the prefix alone.
An oracle with activation access to the prefix should outperform a black-box
text monitor that can only read the prefix.

- **Round 1:** Gemini sees the full CoT, picks a natural split, generates a question.
- **Round 2:** Gemini sees only the prefix, answers the question (BB baseline).
- **Round 3:** Gemini sees only the suffix, answers the question (ground truth).
- **Scoring:** Gemini judges whether the BB answer is substantively correct given the GT.

**Overall BB accuracy: {bb_acc_str}** — the oracle should beat this by reading latent
information from the prefix activations.

## Splits

| Split | CoTs | Rows |
|-------|------|------|
| train | {len(train_ids)} | {len(df_train)} |
| test | {len(test_ids)} | {len(df_test)} |

Split by `cot_id` (all data for a given CoT goes to the same split).

## Sources ({len(source_dist)} task domains)

| Source | Count | BB accuracy |
|--------|-------|-------------|
{source_acc_table}

**Total: {len(rows)} rows** (1 per CoT)

## Schema

| Field | Description |
|-------|-------------|
| `question` | Original problem |
| `cot_text` | Full chain-of-thought text |
| `prompt` | Generated question about the suffix |
| `target_response` | GT answer (Gemini sees suffix, Round 3) |
| `bb_response` | BB answer (Gemini sees prefix, Round 2) |
| `bb_correct` | Whether BB answer is substantively correct (Gemini-judged) |
| `cot_prefix` | Prefix text (sentences 0..split_index) |
| `cot_suffix` | Suffix text (sentences split_index+1..end) |
| `split_index` | Sentence index chosen by Gemini (0-based, last sentence of prefix) |
| `num_sentences` | Total sentences in the CoT |
| `cot_id` | Source corpus ID |
| `source` | Task domain (MATH, GSM8K, etc.) |
| `generation_prompt` | Round 1 system+user prompt (for regeneration) |

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
