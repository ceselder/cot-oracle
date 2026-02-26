#!/usr/bin/env python3
"""Upload fixed forced_answer_entropy_riya to HuggingFace."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_ORG = "mats-10-sprint-cs-jb"
api = HfApi(token=HF_TOKEN)


def flatten_metadata(items):
    flat = []
    for item in items:
        row = {k: v for k, v in item.items() if k != "metadata"}
        meta = item.get("metadata") or {}
        for mk, mv in meta.items():
            if mv is None or not isinstance(mv, (str, int, float, bool)):
                row[f"meta_{mk}"] = str(mv)
            else:
                row[f"meta_{mk}"] = mv
        flat.append(row)
    if flat:
        all_keys = set()
        for row in flat:
            all_keys.update(row.keys())
        for key in all_keys:
            types = set()
            for row in flat:
                v = row.get(key)
                if v is not None:
                    types.add(type(v).__name__)
            if len(types) > 1:
                for row in flat:
                    if key in row:
                        row[key] = str(row[key])
    return flat


def upload_forced_entropy():
    repo_id = f"{HF_ORG}/cot-oracle-eval-forced-answer-entropy-riya"
    print(f"Uploading to {repo_id}...")

    with open("data/evals/forced_answer_entropy_riya.json") as f:
        raw = json.load(f)

    flat = flatten_metadata(raw)
    ds = Dataset.from_list(flat)

    card = """---
tags:
  - cot-oracle
  - unfaithfulness-detection
  - chain-of-thought
  - eval
license: mit
---

# CoT Oracle Eval: forced_answer_entropy_riya

Forced answer entropy eval (oracle logprob method). Ground truth: at each sentence boundary during CoT, source model is forced to answer via "So, the answer is:" — logprobs over A/B/C/D extracted via HF forward pass (exact logits, no top-k truncation), entropy H(p) computed. Oracle evaluation: oracle reads activations at same boundary, its own logprobs over A/B/C/D are extracted and entropy computed. Metrics: R²/correlation between oracle and GT entropy, KL(GT||oracle), top-1 agreement. Source: allenai/ai2_arc ARC-Challenge.

**v2 (2026-02-26):** Recomputed ground truth using HF transformers forward pass instead of vLLM logprobs=20, which missed answer tokens outside top-20 (89% of items had fallback -20.0 logprobs). Now 0% fallback rate.

**Known limitation:** 35.5% of eval items (71/200) use ARC-Challenge questions that also appear in the training corpus (both load from split="test"). These items are NOT held-out. Consider filtering to the 129 clean items for uncontaminated evaluation.

Part of the [CoT Oracle Evals collection](https://huggingface.co/collections/mats-10-sprint-cs-jb/cot-oracle-evals).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-eval-forced-answer-entropy-riya", split="train")
```
"""

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Update card: v2 with HF forward pass ground truth",
    )

    ds.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        private=False,
        commit_message="v2: Recompute GT with HF forward pass (fix logprob fallback bug)",
    )
    print(f"  Done: {len(raw)} items uploaded")


def update_sentence_insertion_card():
    repo_id = f"{HF_ORG}/cot-oracle-eval-sentence-insertion"
    print(f"\nUpdating model card for {repo_id}...")

    card = """---
tags:
  - cot-oracle
  - unfaithfulness-detection
  - chain-of-thought
  - eval
license: mit
---

# CoT Oracle Eval: sentence_insertion

Needle-in-haystack: detect foreign sentence inserted into CoT. 50 items with a sentence from a different problem's CoT spliced in; 50 clean items. Oracle sees activations at sentence boundaries and must identify which step was inserted (or say "none" for clean items). Off-by-one tolerance in scoring. Source: ceselder/qwen3-8b-math-cot-corpus.

**Known limitation — training data overlap:** ~60% of host CoTs (60/100) and ~43% of donor CoTs (22/51) appear in the training corpus (`mats-10-sprint-cs-jb/cot-oracle-corpus-v5`). While there is no direct `sentence_insertion` training task, the oracle is trained on these same CoTs via `full_recon` (40K examples), `next_step` (30K), `answer_pred` (20K), and other tasks. This means the oracle may detect insertions via memorized activation patterns rather than genuine coherence checking. When interpreting results, compare accuracy on the ~60 contaminated items vs ~40 clean items to assess memorization effects.

Part of the [CoT Oracle Evals collection](https://huggingface.co/collections/mats-10-sprint-cs-jb/cot-oracle-evals).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-eval-sentence-insertion", split="train")
```
"""

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Document training data overlap (60% host CoT contamination)",
    )
    print("  Done: model card updated")


def delete_decorative_cot():
    repo_id = f"{HF_ORG}/cot-oracle-eval-decorative-cot"
    print(f"\nDeleting {repo_id}...")
    try:
        api.delete_repo(repo_id, repo_type="dataset", token=HF_TOKEN)
        print("  Done: repo deleted")
    except Exception as e:
        print(f"  Delete failed: {e}")
        print("  Trying to update card to mark as deprecated...")
        card = """---
tags:
  - cot-oracle
  - deprecated
license: mit
---

# DEPRECATED: CoT Oracle Eval: decorative_cot

**This eval has been removed from the active eval suite.**

Reason: 100% training data contamination — all 56 eval questions appear in the training corpus (`mats-10-sprint-cs-jb/cot-oracle-corpus-v5`). Additionally, training labels disagree with eval labels 53% of the time (single greedy vs 10-run labels), and n=56 provides unusable statistical power.
"""
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Mark as deprecated: 100% training data contamination",
        )
        print("  Marked as deprecated instead")


if __name__ == "__main__":
    upload_forced_entropy()
    update_sentence_insertion_card()
    delete_decorative_cot()
