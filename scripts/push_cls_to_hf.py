#!/usr/bin/env python3
"""Push classification datasets to HF in our standard schema.

Generates train.jsonl and test.jsonl for each cls_ task from Adam's
NL probe loaders, then uploads to mats-10-sprint-cs-jb/cls-{name}.

Schema matches load_task_data() expectations:
    {task, datapoint_type, prompt, target_response, excerpt}

supervisor_context="excerpt" in TaskDef → load_task_data() maps excerpt → cot_text.

Usage:
    python scripts/push_cls_to_hf.py [--dry-run] [--tasks cls_sst2 cls_ag_news ...]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ao_reference"))

import nl_probes.dataset_classes.classification_dataset_manager as cdm
cdm.DEFAULT_DATA_DIR = str(ROOT / "ao_reference" / "datasets" / "classification_datasets")

from huggingface_hub import HfApi

HF_ORG = "mats-10-sprint-cs-jb"

# ── Loader configs ──
# (task_name, loader_factory, is_multi)

TASK_CONFIGS = [
    ("cls_sst2", lambda: cdm.SstDatasetLoader(), False),
    ("cls_snli", lambda: cdm.SnliDatasetLoader(), False),
    ("cls_ag_news", lambda: cdm.AgNewsDatasetLoader(), False),
    ("cls_ner", lambda: cdm.NerDatasetLoader(), False),
    ("cls_tense", lambda: cdm.TenseDatasetLoader(), False),
    ("cls_language_id", lambda: cdm.LanguageIDDatasetLoader(), False),
    ("cls_singular_plural", lambda: cdm.SingularPluralDatasetLoader(), False),
    ("cls_geometry_of_truth", lambda: cdm.GeometryOfTruthDatasetLoader.get_all_loaders, True),
    ("cls_relations", lambda: cdm.RelationDatasetLoader.get_all_loaders, True),
]


def load_samples(loader_factory, is_multi):
    """Load ContextQASamples from Adam's loaders."""
    if is_multi:
        factory = loader_factory()
        loaders = factory()
        all_samples = []
        for loader in loaders:
            samples = loader.load(num_qa_per_sample=1)
            all_samples.extend(samples)
            print(f"    {loader.group}/{loader.name}: {len(samples)} samples")
        return all_samples
    else:
        loader = loader_factory()
        samples = loader.load(num_qa_per_sample=1)
        print(f"    {loader.group}/{loader.name}: {len(samples)} samples")
        return samples


def samples_to_items(task_name: str, samples: list, seed: int = 42) -> list[dict]:
    """Convert ContextQASamples to our standard schema dicts."""
    rng = random.Random(seed)

    flat = []
    for s in samples:
        q = s.questions[0]
        a = s.answers[0]
        # Strip leading "# " from Adam's loaders
        if q.startswith("# "):
            q = q[2:]
        flat.append((s.context, q, a))

    rng.shuffle(flat)

    # Balance yes/no
    yes_items = [x for x in flat if x[2] == "Yes"]
    no_items = [x for x in flat if x[2] == "No"]
    n_balanced = min(len(yes_items), len(no_items))
    balanced = yes_items[:n_balanced] + no_items[:n_balanced]
    rng.shuffle(balanced)

    items = []
    for context, question, answer in balanced:
        items.append({
            "task": task_name,
            "datapoint_type": task_name,
            "prompt": f"Answer with 'Yes' or 'No' only. {question}",
            "target_response": "Yes" if answer == "Yes" else "No",
            "excerpt": context,
        })

    return items


def split_train_test(items: list[dict], test_frac: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 train/test split."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_frac)
    return shuffled[n_test:], shuffled[:n_test]


def upload_jsonl(api: HfApi, repo_id: str, items: list[dict], filename: str, token: str):
    """Upload a list of dicts as JSONL to HF."""
    content = "\n".join(json.dumps(item) for item in items) + "\n"
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {filename} ({len(items)} items)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tasks", nargs="*", help="Specific tasks (default: all)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run:
        assert token, "Set HF_TOKEN in ~/.env"

    api = HfApi(token=token) if not args.dry_run else None

    for task_name, loader_factory, is_multi in TASK_CONFIGS:
        if args.tasks and task_name not in args.tasks:
            continue

        print(f"\n{'='*60}")
        print(f"Generating {task_name}...")

        samples = load_samples(loader_factory, is_multi)
        items = samples_to_items(task_name, samples)
        train, test = split_train_test(items)

        print(f"  {task_name}: {len(items)} total → {len(train)} train / {len(test)} test")
        print(f"  Sample: {json.dumps(items[0], default=str)[:200]}")

        if args.dry_run:
            continue

        # Derive repo name from task name: cls_sst2 → cls-sst2
        repo_id = f"{HF_ORG}/{task_name.replace('_', '-')}"
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        upload_jsonl(api, repo_id, train, "train.jsonl", token)
        upload_jsonl(api, repo_id, test, "test.jsonl", token)
        print(f"  Uploaded to {repo_id}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
