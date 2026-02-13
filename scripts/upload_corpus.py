"""
Upload CoT corpus to HuggingFace as a dataset.

Usage:
    python3 scripts/upload_corpus.py \
        --corpus data/cot_corpus_8b/corpus.jsonl \
        --repo ceselder/qwen3-8b-math-cot-corpus \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. ceselder/qwen3-8b-math-cot-corpus)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model used for generation")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    assert corpus_path.exists(), f"Corpus not found: {corpus_path}"

    # Read corpus stats
    records = []
    with open(corpus_path) as f:
        for line in f:
            records.append(json.loads(line))

    categories = {}
    sources = {}
    for r in records:
        cat = r.get("category", "unknown")
        src = r.get("source", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1

    print(f"Corpus: {len(records)} records")
    print(f"Categories: {categories}")
    print(f"Sources: {sources}")

    # Create README
    readme = f"""---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: correct_answer
    dtype: string
  - name: cot_response
    dtype: string
  - name: direct_response
    dtype: string
  - name: category
    dtype: string
  - name: source
    dtype: string
  - name: sentences
    sequence: string
  - name: n_sentences
    dtype: int64
  splits:
  - name: train
    num_examples: {len(records)}
license: mit
task_categories:
- text-generation
language:
- en
tags:
- chain-of-thought
- math
- reasoning
- activation-oracles
---

# {args.model} Math CoT Corpus

Chain-of-thought reasoning traces from {args.model} on MATH and GSM8K problems.

## Overview

- **Model:** {args.model}
- **Total examples:** {len(records)}
- **Sources:** {json.dumps(sources)}
- **Categories:** {json.dumps(categories)}

## Categories

- **load_bearing**: CoT got the answer right, direct answer got it wrong → CoT actually helped
- **both_correct**: Both CoT and direct answer correct → CoT may be decorative
- **both_wrong**: Neither approach worked → problem too hard for model
- **cot_hurt**: Direct answer correct but CoT led to wrong answer

## Fields

- `question`: The math problem
- `correct_answer`: Ground truth answer
- `cot_response`: Full CoT response including `<think>` tags
- `direct_response`: Direct answer without CoT
- `category`: One of load_bearing, both_correct, both_wrong, cot_hurt
- `source`: Dataset source (math or gsm8k)
- `sentences`: CoT split into sentences
- `n_sentences`: Number of sentences in CoT

## Use Case

Training data for activation oracles that read CoT reasoning trajectories.
Part of the [CoT Oracle](https://github.com/ceselder/cot-oracle) project.
"""

    # Create repo
    api = HfApi()
    try:
        create_repo(args.repo, repo_type="dataset", exist_ok=True)
        print(f"Created/found repo: {args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload files
    readme_path = corpus_path.parent / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    api.upload_file(
        path_or_fileobj=str(corpus_path),
        path_in_repo="corpus.jsonl",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Uploaded corpus.jsonl")

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Uploaded README.md")

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
