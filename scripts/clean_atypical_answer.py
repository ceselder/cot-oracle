#!/usr/bin/env python3
"""
Clean the atypical_answer dataset: re-label using 95% Wilson confidence
intervals, "typical"/"atypical" labels, 80/20 question-level split, balanced.

Labeling (CI-based, symmetric):
  typical:   rollout matches majority AND majority_ci_lo > 0.50
             (95% confident this IS the dominant answer)
  atypical:  rollout differs from majority AND majority_ci_lo > 0.50
             (95% confident majority exists, this rollout deviates)
  dropped:   majority_ci_lo <= 0.50 (no confident majority → indeterminate)

Reads from: mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts-50
Writes to:  mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned
"""

import json
import math
import random
from collections import Counter
from pathlib import Path


# ── Config ──

RAW_HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts-50"
CLEAN_HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned"
COLLECTION_SLUG = "mats-10-sprint-cs-jb/cleaned-datasets-69a365228a41b50e0e1e9af4"
SEED = 42

ORACLE_PROMPT = (
    "Will the model give the typical or atypical answer? "
    "Describe the answer typicality."
)

METADATA_FIELDS = [
    "question_id", "source", "question", "choices", "correct_answer",
    "cot_text", "model_answer", "majority_answer",
    "minority_rate", "majority_ci_lo", "majority_ci_hi",
    "n_valid_rollouts", "rollout_idx",
]


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean atypical_answer dataset")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--output-dir", default="data/cleaned_atypical_answer")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    random.seed(SEED)

    # ── Load raw rollouts ──
    print("Loading raw rollouts from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(RAW_HF_REPO, split="train")
    print(f"  {len(ds)} rows")

    # ── Group by question to compute per-question CIs ──
    print("Grouping by question_id...")
    by_question: dict[str, list[dict]] = {}
    for entry in ds:
        entry = dict(entry)
        qid = entry["question_id"]
        by_question.setdefault(qid, []).append(entry)
    print(f"  {len(by_question)} questions")

    # ── Compute per-question stats and filter ──
    print("Computing Wilson CIs and filtering...")
    processed = []
    kept_questions = 0
    dropped_questions = 0
    no_valid_questions = 0

    for qid, rollouts in by_question.items():
        # Only valid rollouts (have a model_answer)
        valid = [r for r in rollouts if r.get("model_answer") is not None]
        if len(valid) < 10:
            no_valid_questions += 1
            continue

        # Find majority answer
        answer_counts = Counter(r["model_answer"] for r in valid)
        majority_answer = answer_counts.most_common(1)[0][0]
        majority_count = answer_counts[majority_answer]
        n_valid = len(valid)
        minority_rate = 1.0 - majority_count / n_valid

        # Wilson CI on the majority proportion
        majority_ci_lo, majority_ci_hi = wilson_ci(majority_count, n_valid)

        # Filter: 95% confident that majority is truly > 50%
        if majority_ci_lo <= 0.50:
            dropped_questions += 1
            continue

        kept_questions += 1

        for r in valid:
            is_majority = r["model_answer"] == majority_answer
            label = "typical" if is_majority else "atypical"

            result = {
                "task": "atypical_answer",
                "datapoint_type": "cot_atypical_answer",
                "prompt": ORACLE_PROMPT,
                "target_response": label,
                "label": label,
                "question_id": qid,
                "source": r.get("source", ""),
                "question": r.get("question", ""),
                "choices": r.get("choices", ""),
                "correct_answer": r.get("correct_answer", ""),
                "cot_text": r.get("cot_text", ""),
                "model_answer": r.get("model_answer"),
                "majority_answer": majority_answer,
                "minority_rate": round(minority_rate, 4),
                "majority_ci_lo": round(majority_ci_lo, 4),
                "majority_ci_hi": round(majority_ci_hi, 4),
                "n_valid_rollouts": n_valid,
                "rollout_idx": r.get("rollout_idx", 0),
            }
            processed.append(result)

    print(f"  Kept: {kept_questions} questions ({len(processed)} rollouts)")
    print(f"  Dropped (no confident majority): {dropped_questions} questions")
    print(f"  Dropped (too few valid): {no_valid_questions} questions")

    # ── Stats ──
    labels = Counter(p["label"] for p in processed)
    print("\nLabel distribution (before balancing):")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v} ({v / len(processed):.1%})")

    unique_cots = len(set(p["cot_text"] for p in processed))
    unique_qs = len(set(p["question_id"] for p in processed))
    print(f"  Unique CoTs: {unique_cots}")
    print(f"  Unique questions: {unique_qs}")

    rates = [p["minority_rate"] for p in processed]
    print(f"  minority_rate range: {min(rates):.0%} - {max(rates):.0%} (mean {sum(rates)/len(rates):.0%})")

    ci_los = [p["majority_ci_lo"] for p in processed]
    print(f"  majority_ci_lo range: {min(ci_los):.3f} - {max(ci_los):.3f}")

    # ── 80/20 train/test split by question_id ──
    print("\nSplitting 80/20 by question_id...")
    by_q: dict[str, list[dict]] = {}
    for p in processed:
        by_q.setdefault(p["question_id"], []).append(p)

    question_ids = sorted(by_q.keys())
    split_rng = random.Random(42)
    split_rng.shuffle(question_ids)
    train_end = int(len(question_ids) * 0.8)
    train_qids = set(question_ids[:train_end])

    train_raw = [p for p in processed if p["question_id"] in train_qids]
    test_raw = [p for p in processed if p["question_id"] not in train_qids]

    # ── Balance both splits to 50/50 ──
    print("\nBalancing to 50/50...")

    def balance_split(data, seed_rng):
        pools = {}
        for item in data:
            pools.setdefault(item["label"], []).append(item)
        min_pool = min(len(p) for p in pools.values())
        balanced = []
        for label, pool in pools.items():
            seed_rng.shuffle(pool)
            balanced.extend(pool[:min_pool])
        seed_rng.shuffle(balanced)
        return balanced

    train_data = balance_split(train_raw, rng)
    test_data = balance_split(test_raw, random.Random(SEED + 1))

    # ── Verify no overlap ──
    train_cots = set(p["cot_text"] for p in train_data)
    test_overlap = sum(1 for p in test_data if p["cot_text"] in train_cots)
    train_q = set(p["question_id"] for p in train_data)
    test_q = set(p["question_id"] for p in test_data)

    print(f"\n  Train: {len(train_data)} ({len(train_q)} questions)")
    print(f"  Test:  {len(test_data)} ({len(test_q)} questions)")
    print(f"  Question overlap: {len(train_q & test_q)}")
    print(f"  CoT overlap: {test_overlap}")

    # ── Save locally ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in [("train", train_data), ("test", test_data)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {split_name}: {path} ({len(data)} items)")

        split_labels = Counter(p["label"] for p in data)
        for k, v in sorted(split_labels.items()):
            print(f"    {k}: {v} ({v / len(data):.1%})")

    # ── Upload ──
    if args.upload:
        print(f"\nUploading to {CLEAN_HF_REPO}...")
        from huggingface_hub import HfApi
        import os

        token = args.hf_token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        api.create_repo(CLEAN_HF_REPO, repo_type="dataset", exist_ok=True)

        for split_name in ["train", "test"]:
            path = out_dir / f"{split_name}.jsonl"
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{split_name}.jsonl",
                repo_id=CLEAN_HF_REPO,
                repo_type="dataset",
            )
            print(f"  Uploaded {split_name}.jsonl")

        readme_path = out_dir / "README.md"
        if readme_path.exists():
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=CLEAN_HF_REPO,
                repo_type="dataset",
            )
            print("  Uploaded README.md")

        print(f"  Dataset: https://huggingface.co/datasets/{CLEAN_HF_REPO}")

        try:
            api.add_collection_item(
                collection_slug=COLLECTION_SLUG,
                item_id=CLEAN_HF_REPO,
                item_type="dataset",
            )
            print(f"  Added to collection")
        except Exception as e:
            print(f"  Collection: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
