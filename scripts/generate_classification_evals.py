#!/usr/bin/env python3
"""Generate 10 classification eval datasets from Adam's NL probe loaders and upload to HF.

Each eval produces ~250 EvalItem-compatible records with:
  - test_prompt = clean_prompt = per-item binary yes/no question
  - correct_answer = "yes" or "no"
  - metadata["cot_text"] = source context text (used for activation extraction)

Usage:
    python scripts/generate_classification_evals.py [--dry-run] [--n-items 250]
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root and ao_reference to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ao_reference"))

# Override the default data dir so Adam's loaders find the datasets
import nl_probes.dataset_classes.classification_dataset_manager as cdm
cdm.DEFAULT_DATA_DIR = str(ROOT / "ao_reference" / "datasets" / "classification_datasets")

from datasets import Dataset
from huggingface_hub import HfApi

HF_ORG = "mats-10-sprint-cs-jb"
HF_COLLECTION = "mats-10-sprint-cs-jb/evals-cot-oracle-working-699d15ecbba7e43452853440"


# ── Loader configs ──────────────────────────────────────────────────
# Each entry: (eval_name, loader_factory, question_template, answer_mapping)
# question_template: a string with {} for per-item label substitution, OR None if
# the loader's own questions should be used directly.

def _sst2_loader():
    return cdm.SstDatasetLoader()

def _snli_loader():
    return cdm.SnliDatasetLoader()

def _md_gender_loader():
    return cdm.MdGenderDatasetLoader()

def _ag_news_loader():
    return cdm.AgNewsDatasetLoader()

def _ner_loader():
    return cdm.NerDatasetLoader()

def _tense_loader():
    return cdm.TenseDatasetLoader()

def _language_id_loader():
    return cdm.LanguageIDDatasetLoader()

def _singular_plural_loader():
    return cdm.SingularPluralDatasetLoader()


# For geometry_of_truth and relations, we aggregate multiple sub-datasets
def _geometry_of_truth_loaders():
    return cdm.GeometryOfTruthDatasetLoader.get_all_loaders()

def _relation_loaders():
    return cdm.RelationDatasetLoader.get_all_loaders()


EVAL_CONFIGS = [
    # (eval_name, loader_or_loaders_factory, is_multi)
    ("cls_sst2", _sst2_loader, False),
    ("cls_snli", _snli_loader, False),
    ("cls_md_gender", _md_gender_loader, False),
    ("cls_ag_news", _ag_news_loader, False),
    ("cls_ner", _ner_loader, False),
    ("cls_tense", _tense_loader, False),
    ("cls_language_id", _language_id_loader, False),
    ("cls_singular_plural", _singular_plural_loader, False),
    ("cls_geometry_of_truth", _geometry_of_truth_loaders, True),
    ("cls_relations", _relation_loaders, True),
]


def samples_to_eval_items(eval_name: str, samples: list[cdm.ContextQASample], n_items: int, seed: int = 42) -> list[dict]:
    """Convert ContextQASamples to EvalItem-compatible dicts.

    Each sample has multiple question paraphrases; we pick one per sample to
    create a single EvalItem. We subsample to n_items with balanced yes/no.
    """
    rng = random.Random(seed)

    # Flatten: one (context, question, answer) per sample, picking first Q/A pair
    flat = []
    for sample in samples:
        # Use the first question/answer pair
        q = sample.questions[0]
        a = sample.answers[0]
        flat.append((sample.context, q, a))

    rng.shuffle(flat)

    # Balance yes/no
    yes_items = [x for x in flat if x[2] == "Yes"]
    no_items = [x for x in flat if x[2] == "No"]
    half = n_items // 2
    yes_items = yes_items[:half]
    no_items = no_items[:half]
    balanced = yes_items + no_items
    rng.shuffle(balanced)

    items = []
    for i, (context, question, answer) in enumerate(balanced):
        items.append({
            "eval_name": eval_name,
            "example_id": f"{eval_name}_{i:04d}",
            "clean_prompt": question,
            "test_prompt": question,
            "correct_answer": "yes" if answer == "Yes" else "no",
            "nudge_answer": None,
            "metadata": {"cot_text": context},
        })

    return items


def flatten_metadata(items: list[dict]) -> list[dict]:
    """Flatten metadata dict into top-level meta_ columns for HF upload."""
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
    return flat


def upload_eval(eval_name: str, items: list[dict], api: HfApi, token: str, dry_run: bool = False):
    """Upload a classification eval dataset to HuggingFace."""
    repo_id = f"{HF_ORG}/cot-oracle-eval-{eval_name.replace('_', '-')}"

    flat = flatten_metadata(items)
    ds = Dataset.from_list(flat)

    if dry_run:
        print(f"  [dry-run] Would upload {len(items)} items to {repo_id}")
        print(f"  Sample: {json.dumps(items[0], indent=2, default=str)[:500]}")
        return

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    card = f"""---
tags:
  - cot-oracle
  - classification
  - eval
license: mit
---

# CoT Oracle Classification Eval: {eval_name}

Binary classification eval from Adam's Activation Oracles NL probes.
{len(items)} items, 50/50 yes/no balanced.

Used as OOD generalization test for the CoT-trained oracle.

## Schema

| Field | Description |
|-------|-------------|
| `eval_name` | `{eval_name}` |
| `test_prompt` / `clean_prompt` | Per-item yes/no classification question |
| `correct_answer` | `"yes"` or `"no"` |
| `meta_cot_text` | Source text for activation extraction |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
```
"""
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", token=token,
        commit_message="Update dataset card",
    )

    ds.push_to_hub(repo_id, token=token, private=False,
                   commit_message=f"Upload {eval_name} classification eval ({len(items)} items)")

    # Add to collection
    try:
        api.add_collection_item(collection_slug=HF_COLLECTION, item_id=repo_id, item_type="dataset", token=token)
    except Exception:
        pass  # Already in collection

    print(f"  Uploaded {len(items)} items to {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't upload, just print stats")
    parser.add_argument("--n-items", type=int, default=250, help="Items per eval (balanced yes/no)")
    parser.add_argument("--evals", nargs="*", help="Specific evals to generate (default: all)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run:
        assert token, "Set HF_TOKEN env var"

    api = HfApi(token=token) if not args.dry_run else None

    succeeded = []
    failed = []

    for eval_name, loader_factory, is_multi in EVAL_CONFIGS:
        if args.evals and eval_name not in args.evals:
            continue

        print(f"\n{'='*60}")
        print(f"Generating {eval_name}...")

        try:
            if is_multi:
                loaders = loader_factory()
                all_samples = []
                for loader in loaders:
                    samples = loader.load(num_qa_per_sample=1)
                    all_samples.extend(samples)
                    print(f"  {loader.group}/{loader.name}: {len(samples)} samples")
            else:
                loader = loader_factory()
                all_samples = loader.load(num_qa_per_sample=1)
                print(f"  {loader.group}/{loader.name}: {len(all_samples)} raw samples")
        except Exception as e:
            print(f"  FAILED to load {eval_name}: {e}")
            failed.append(eval_name)
            continue

        # Strip leading "# " from questions (Adam's loaders prepend it)
        for s in all_samples:
            for j in range(len(s.questions)):
                if s.questions[j].startswith("# "):
                    s.questions[j] = s.questions[j][2:]

        items = samples_to_eval_items(eval_name, all_samples, n_items=args.n_items)
        print(f"  {eval_name}: {len(items)} items (yes={sum(1 for x in items if x['correct_answer']=='yes')}, no={sum(1 for x in items if x['correct_answer']=='no')})")

        if items:
            upload_eval(eval_name, items, api, token, dry_run=args.dry_run)
            succeeded.append(eval_name)

    print(f"\n{'='*60}")
    print(f"Done! {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"FAILED: {', '.join(failed)}")


if __name__ == "__main__":
    main()
