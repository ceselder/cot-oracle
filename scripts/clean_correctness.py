#!/usr/bin/env python3
"""
Create a correctness prediction dataset from MMLU-Pro STEM rollouts.

Single-source (MMLU-Pro math/physics/chemistry/engineering) to avoid
source-as-shortcut confound. Length-balanced to prevent length-as-shortcut.

Reads from: data/mmlu_pro_correctness_rollouts.jsonl (local)
Writes to:  mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


# ── Config ──

INPUT_FILE = "data/mmlu_pro_correctness_rollouts.jsonl"
CLEAN_HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned"
COLLECTION_SLUG = "mats-10-sprint-cs-jb/cleaned-datasets-69a365228a41b50e0e1e9af4"
SEED = 42
TRUNCATION_LIMIT = 8000
N_LENGTH_BINS = 10

ORACLE_PROMPT = (
    "Did the model's chain-of-thought reasoning lead to the correct answer?"
)


def build_target_response(cot_correct: bool) -> str:
    if cot_correct:
        return "Yes, the model reached the correct answer."
    else:
        return "No, the model did not reach the correct answer."


def length_balanced_sample(
    correct: list[dict], incorrect: list[dict], rng: random.Random,
) -> list[dict]:
    """Balance correct/incorrect by CoT token length using decile bins."""
    all_lens = [r["cot_token_length"] for r in correct + incorrect]
    bin_edges = np.percentile(all_lens, np.linspace(0, 100, N_LENGTH_BINS + 1))
    bin_edges[0] = 0
    bin_edges[-1] = float("inf")

    correct_bins: dict[int, list[dict]] = {i: [] for i in range(N_LENGTH_BINS)}
    incorrect_bins: dict[int, list[dict]] = {i: [] for i in range(N_LENGTH_BINS)}

    for item in correct:
        tlen = item["cot_token_length"]
        for b in range(N_LENGTH_BINS):
            if bin_edges[b] <= tlen < bin_edges[b + 1]:
                correct_bins[b].append(item)
                break

    for item in incorrect:
        tlen = item["cot_token_length"]
        for b in range(N_LENGTH_BINS):
            if bin_edges[b] <= tlen < bin_edges[b + 1]:
                incorrect_bins[b].append(item)
                break

    balanced = []
    for b in range(N_LENGTH_BINS):
        c_pool = correct_bins[b]
        ic_pool = incorrect_bins[b]
        n = min(len(c_pool), len(ic_pool))
        rng.shuffle(c_pool)
        rng.shuffle(ic_pool)
        balanced.extend(c_pool[:n])
        balanced.extend(ic_pool[:n])

    rng.shuffle(balanced)
    return balanced


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create correctness dataset")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--output-dir", default="data/cleaned_correctness")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    random.seed(SEED)

    # ── Load ──
    print(f"Loading rollouts from {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        raw = [json.loads(l) for l in f]
    print(f"  {len(raw)} rows")

    # ── Process ──
    print(f"Processing: drop truncated (>={TRUNCATION_LIMIT}), drop unlabeled...")
    processed = []
    dropped_truncated = 0
    dropped_no_label = 0

    for entry in raw:
        if entry["cot_token_length"] >= TRUNCATION_LIMIT:
            dropped_truncated += 1
            continue
        if entry.get("cot_correct") is None:
            dropped_no_label += 1
            continue

        cot_correct = entry["cot_correct"]
        result = {
            "task": "correctness",
            "datapoint_type": "cot_correctness",
            "prompt": ORACLE_PROMPT,
            "target_response": build_target_response(cot_correct),
            "label": "correct" if cot_correct else "incorrect",
            "question_id": entry["question_id"],
            "question": entry["question"],
            "correct_answer": entry["correct_answer"],
            "cot_text": entry["cot_text"],
            "cot_answer": entry.get("cot_answer", ""),
            "cot_token_length": entry["cot_token_length"],
            "source": entry["source"],
            "subject": entry.get("subject", ""),
        }
        processed.append(result)

    print(f"  Kept: {len(processed)}")
    print(f"  Dropped truncated: {dropped_truncated}")
    print(f"  Dropped no label: {dropped_no_label}")

    labels = Counter(p["label"] for p in processed)
    print(f"\nLabel distribution (before balancing):")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v} ({v / len(processed):.1%})")

    # ── 80/20 train/test split by question_id ──
    print("\nSplitting 80/20 by question_id...")
    by_question: dict[str, list[dict]] = {}
    for p in processed:
        by_question.setdefault(p["question_id"], []).append(p)

    question_ids = sorted(by_question.keys())
    split_rng = random.Random(42)
    split_rng.shuffle(question_ids)
    train_end = int(len(question_ids) * 0.8)
    train_qids = set(question_ids[:train_end])

    train_raw = [p for p in processed if p["question_id"] in train_qids]
    test_raw = [p for p in processed if p["question_id"] not in train_qids]

    print(f"  Train pool: {len(train_raw)}")
    print(f"  Test pool:  {len(test_raw)}")

    # ── Length-balanced sampling ──
    print("\nLength-balanced sampling (10 decile bins)...")

    train_correct = [p for p in train_raw if p["label"] == "correct"]
    train_incorrect = [p for p in train_raw if p["label"] == "incorrect"]
    print(f"  [train] Before: {len(train_correct)} correct, {len(train_incorrect)} incorrect")
    train_data = length_balanced_sample(train_correct, train_incorrect, rng)

    test_correct = [p for p in test_raw if p["label"] == "correct"]
    test_incorrect = [p for p in test_raw if p["label"] == "incorrect"]
    print(f"  [test]  Before: {len(test_correct)} correct, {len(test_incorrect)} incorrect")
    test_data = length_balanced_sample(test_correct, test_incorrect, random.Random(SEED + 1))

    print(f"\n  Train: {len(train_data)}")
    print(f"  Test:  {len(test_data)}")

    # Verify balance
    for split_name, data in [("train", train_data), ("test", test_data)]:
        split_labels = Counter(p["label"] for p in data)
        print(f"\n  {split_name}:")
        for k, v in sorted(split_labels.items()):
            print(f"    {k}: {v} ({v / len(data):.1%})")

    # Check CoT overlap
    train_cots = set(t["cot_text"] for t in train_data)
    test_overlap = sum(1 for t in test_data if t["cot_text"] in train_cots)
    print(f"\n  CoT train/test overlap: {test_overlap}")

    # Verify length balance
    import statistics
    for split_name, data in [("train", train_data), ("test", test_data)]:
        c_lens = [p["cot_token_length"] for p in data if p["label"] == "correct"]
        ic_lens = [p["cot_token_length"] for p in data if p["label"] == "incorrect"]
        print(f"\n  {split_name} length check:")
        print(f"    correct:   median={statistics.median(c_lens):.0f}, mean={statistics.mean(c_lens):.0f}")
        print(f"    incorrect: median={statistics.median(ic_lens):.0f}, mean={statistics.mean(ic_lens):.0f}")

    # Subject distribution
    for split_name, data in [("train", train_data), ("test", test_data)]:
        print(f"\n  {split_name} subject balance:")
        for subj in sorted(set(p["subject"] for p in data)):
            c = sum(1 for p in data if p["subject"] == subj and p["label"] == "correct")
            ic = sum(1 for p in data if p["subject"] == subj and p["label"] == "incorrect")
            t = c + ic
            print(f"    {subj:<12} correct={c:>4} ({c/t:.0%})  incorrect={ic:>4} ({ic/t:.0%})")

    # Sample target responses
    print("\nSample target responses:")
    for label in ["correct", "incorrect"]:
        examples = [p for p in train_data if p["label"] == label]
        if examples:
            print(f"  [{label}] {examples[0]['target_response']}")

    # ── Save locally ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in [("train", train_data), ("test", test_data)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {split_name}: {path} ({len(data)} items)")

    # ── Upload ──
    if args.upload:
        print(f"\nUploading to {CLEAN_HF_REPO}...")
        from huggingface_hub import HfApi
        import os

        token = args.hf_token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        api.create_repo(CLEAN_HF_REPO, repo_type="dataset", exist_ok=True)

        for fname in out_dir.iterdir():
            if fname.suffix in (".jsonl", ".md"):
                api.upload_file(
                    path_or_fileobj=str(fname),
                    path_in_repo=fname.name,
                    repo_id=CLEAN_HF_REPO,
                    repo_type="dataset",
                )
                print(f"  Uploaded {fname.name}")

        print(f"  Dataset: https://huggingface.co/datasets/{CLEAN_HF_REPO}")

        try:
            api.add_collection_item(
                collection_slug=COLLECTION_SLUG,
                item_id=CLEAN_HF_REPO,
                item_type="dataset",
            )
            print(f"  Added to collection: {COLLECTION_SLUG}")
        except Exception as e:
            print(f"  Collection: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
