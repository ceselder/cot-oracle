#!/usr/bin/env python3
"""
Upload importance++ resampling data to HuggingFace.

Usage:
    python scripts/upload_resampling_importancepp.py --dry-run
    python scripts/upload_resampling_importancepp.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ID = "mats-10-sprint-cs-jb/cot-oracle-resampling-importancepp"


def build_dataset(resampling_path: Path, corpus_path: Path) -> list[dict]:
    """Load resampling JSONL, join with corpus CoTs, and build upload rows."""
    # Load corpus into lookup by id
    corpus_by_id = {}
    with open(corpus_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            corpus_by_id[entry["id"]] = entry
    print(f"  Loaded corpus: {len(corpus_by_id)} entries")

    items = []
    n_joined = 0
    with open(resampling_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            cot = corpus_by_id.get(entry["id"], {})
            if cot:
                n_joined += 1

            # Compute summary stats
            imp = entry["sentence_importance"]
            n_important = sum(1 for s in imp if s["important"])
            kl_scores = [s["resampling_importance_kl_binary"] for s in imp]
            acc_scores = [s["resampling_importance_accuracy"] for s in imp]
            top_k = sorted(range(len(kl_scores)), key=lambda i: kl_scores[i], reverse=True)[:3]

            # Filter: only include entries where at least one sentence has KL > 0.1
            if max(kl_scores) <= 0.1:
                continue

            # Build SFT input/label
            sentences = cot.get("sentences", [])
            question = cot.get("question", entry["question"])
            numbered_cot = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
            sft_input = (
                f"Identify the most causally important reasoning steps in this chain of thought.\n\n"
                f"Problem: {question}\n\n"
                f"Chain of thought:\n{numbered_cot}"
            )
            # Only include sentences above threshold, up to 3
            important_indices = [i for i in range(len(kl_scores)) if kl_scores[i] > 0.1]
            important_indices.sort(key=lambda i: kl_scores[i], reverse=True)
            important_indices = important_indices[:3]
            top_k = important_indices
            sft_label_lines = [f"[{idx+1}] {sentences[idx]}" for idx in important_indices if idx < len(sentences)]
            sft_label = "\n".join(sft_label_lines)

            items.append({
                # SFT columns first
                "sft_input": sft_input,
                "sft_label": sft_label,
                # Core fields from corpus
                "id": entry["id"],
                "source": entry["source"],
                "domain": cot.get("domain", ""),
                "question": cot.get("question", entry["question"]),
                "correct_answer": cot.get("correct_answer", entry["correct_answer"]),
                "subject": cot.get("subject", ""),
                "level": str(cot.get("level", "")),
                "cot_response": cot.get("cot_response", ""),
                "cot_content": cot.get("cot_content", ""),
                "direct_response": cot.get("direct_response"),
                "cot_answer": cot.get("cot_answer", ""),
                "direct_answer": cot.get("direct_answer"),
                "cot_correct": cot.get("cot_correct"),
                "direct_correct": cot.get("direct_correct"),
                "category": cot.get("category"),
                "sentences": json.dumps(cot.get("sentences", [])),
                "n_sentences": entry["n_sentences"],
                "rollout_idx": cot.get("rollout_idx", 0),
                # Importance++ fields
                "n_important": n_important,
                "top_k_indices": json.dumps(top_k),
                "truncations": json.dumps(entry["truncations"]),
                "sentence_importance": json.dumps(entry["sentence_importance"]),
                "importance_kl_scores": json.dumps(kl_scores),
                "importance_acc_scores": json.dumps(acc_scores),
            })

    print(f"  Loaded {len(items)} resampling entries ({n_joined} joined with corpus)")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resampling", default="data/pipeline_medium_20260224_003836/resampling.jsonl")
    parser.add_argument("--corpus", default="data/pipeline_medium_20260224_003836/corpus_subset_650.jsonl")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resampling_path = PROJECT_ROOT / args.resampling
    corpus_path = PROJECT_ROOT / args.corpus
    items = build_dataset(resampling_path, corpus_path)

    # Stats for README
    sources = {}
    total_sentences = 0
    total_important = 0
    all_kl = []
    for item in items:
        sources[item["source"]] = sources.get(item["source"], 0) + 1
        total_sentences += item["n_sentences"]
        total_important += item["n_important"]
        all_kl.extend(json.loads(item["importance_kl_scores"]))

    kl_nonzero = [k for k in all_kl if k > 0]
    kl_nonzero.sort()
    kl_median = kl_nonzero[len(kl_nonzero) // 2] if kl_nonzero else 0.0
    kl_gt01 = sum(1 for k in all_kl if k > 0.1)

    # Save locally
    output_dir = PROJECT_ROOT / "data" / "hf_uploads" / "resampling-importancepp"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "train.parquet"
    pd.DataFrame(items).to_parquet(parquet_path, index=False)

    source_table = "\n".join(f"| {src} | {count} |" for src, count in sorted(sources.items(), key=lambda x: -x[1]))

    readme = f"""---
tags:
  - cot-oracle
  - thought-anchors
  - chain-of-thought
  - importance
  - resampling
license: mit
dataset_info:
  features:
    - name: sft_input
      dtype: string
    - name: sft_label
      dtype: string
    - name: id
      dtype: string
    - name: source
      dtype: string
    - name: domain
      dtype: string
    - name: question
      dtype: string
    - name: correct_answer
      dtype: string
    - name: subject
      dtype: string
    - name: level
      dtype: string
    - name: cot_response
      dtype: string
    - name: cot_content
      dtype: string
    - name: direct_response
      dtype: string
    - name: cot_answer
      dtype: string
    - name: direct_answer
      dtype: string
    - name: cot_correct
      dtype: bool
    - name: direct_correct
      dtype: bool
    - name: category
      dtype: string
    - name: sentences
      dtype: string
    - name: n_sentences
      dtype: int64
    - name: rollout_idx
      dtype: int64
    - name: n_important
      dtype: int64
    - name: top_k_indices
      dtype: string
    - name: truncations
      dtype: string
    - name: sentence_importance
      dtype: string
    - name: importance_kl_scores
      dtype: string
    - name: importance_acc_scores
      dtype: string
  splits:
    - name: train
      num_examples: {len(items)}
---

# CoT Oracle: Importance++ Resampling Data

On-policy Qwen3-8B chain-of-thought traces with **importance++ scores** computed using the thought-anchors resampling methodology (Bogdan et al., 2025).

## Overview

- **Model:** Qwen/Qwen3-8B
- **Entries:** {len(items)} (filtered from 565 resampled entries, requiring max KL > 0.1)
- **Total sentences scored:** {total_sentences}
- **Important sentences:** {total_important} ({total_important/max(total_sentences,1)*100:.1f}%)
- **Rollouts per truncation point:** 20
- **Sampling strategy:** Frugal (sparse every 3rd sentence + fill-in for accuracy jumps + interpolation)

## Methodology

This follows the **thought-anchors** (arXiv:2506.19143) resampling methodology:

1. For each CoT, truncate at sentence boundary `t` (keeping `<think>` open)
2. Let Qwen3-8B **continue reasoning** from the truncation point (full rollout, up to 4096 tokens)
3. Repeat 20 times per truncation point to build answer distributions
4. Compute importance metrics between adjacent truncation points:
   - **`resampling_importance_accuracy`**: accuracy(with sentence) - accuracy(without)
   - **`resampling_importance_kl_binary`**: KL divergence over P(correct) — the key metric from the paper
   - **`resampling_importance_kl_categorical`**: KL divergence over full answer distributions

A sentence is marked **important** if `importance_accuracy > 0.3` or `kl_binary > 0.1`.

## KL Divergence Stats

| Metric | Value |
|--------|-------|
| Sentences with KL > 0 | {len(kl_nonzero)} / {len(all_kl)} |
| Median KL (nonzero) | {kl_median:.4f} |
| Max KL | {max(all_kl):.4f} |
| KL > 0.1 | {kl_gt01} / {len(all_kl)} ({kl_gt01/max(len(all_kl),1)*100:.1f}%) |

## Sources

| Dataset | Count |
|---------|-------|
{source_table}

## Schema

| Field | Description |
|-------|-------------|
| `sft_input` | **SFT input**: problem + numbered CoT sentences |
| `sft_label` | **SFT label**: up to 3 most important sentences (KL > 0.1) |
| `id` | Entry ID (format: `source_idx_rN`) |
| `source` | Dataset source (math, gsm8k, aqua_rat, etc.) |
| `domain` | Problem domain |
| `question` | Full problem text |
| `correct_answer` | Ground truth answer |
| `subject` | Subject area |
| `level` | Difficulty level |
| `cot_response` | Full chain-of-thought reasoning (inside `<think>` tags) |
| `cot_content` | Model's final answer text (after `</think>`) |
| `direct_response` | Direct answer without CoT |
| `cot_answer` | Extracted answer from CoT response |
| `direct_answer` | Extracted answer from direct response |
| `cot_correct` | Whether CoT answer is correct |
| `direct_correct` | Whether direct answer is correct |
| `category` | load_bearing / both_correct / both_wrong / cot_hurt |
| `sentences` | JSON list of CoT sentences |
| `n_sentences` | Number of CoT sentences |
| `rollout_idx` | Rollout index |
| `n_important` | Count of important sentences |
| `top_k_indices` | JSON list of top-3 sentence indices by KL |
| `truncations` | JSON list of per-truncation-point stats (accuracy, n_rollouts, interpolated) |
| `sentence_importance` | JSON list of per-sentence importance++ metrics |
| `importance_kl_scores` | JSON list of KL binary scores (one per sentence) |
| `importance_acc_scores` | JSON list of accuracy delta scores (one per sentence) |

## Usage

```python
from datasets import load_dataset
import json

ds = load_dataset("{args.repo}", split="train")

# Get importance scores for first entry
entry = ds[0]
scores = json.loads(entry["importance_kl_scores"])
sentences = json.loads(entry["sentence_importance"])

# Find thought anchors (high KL sentences)
for s in sentences:
    s = json.loads(s) if isinstance(s, str) else s
    if s["resampling_importance_kl_binary"] > 0.1:
        print(f"Anchor (KL={{s['resampling_importance_kl_binary']:.3f}}): {{s['sentence_text']}}")
```

## References

- Thought Anchors (Bogdan et al., 2025): [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)
- Thought Branches (Macar, Bogdan et al., 2025): [arXiv:2510.27484](https://arxiv.org/abs/2510.27484)
- CoT Oracle project: [GitHub](https://github.com/japhba/cot-oracle)
- Full corpus: [mats-10-sprint-cs-jb/cot-oracle-corpus-v5](https://huggingface.co/datasets/mats-10-sprint-cs-jb/cot-oracle-corpus-v5)
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    print(f"  Saved {len(items)} items to {parquet_path}")
    print(f"  README at {readme_path}")

    if not args.dry_run:
        api = HfApi()
        create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.repo, repo_type="dataset")
        print(f"\n  Uploaded to https://huggingface.co/datasets/{args.repo}")
    else:
        print(f"\n  Dry run — would upload to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
