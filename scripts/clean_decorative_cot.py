#!/usr/bin/env python3
"""
Clean the decorative_cot dataset: CI-based labels, "load_bearing"/"decorative",
one rollout per question (unique prompts), 80/20 question-level split,
length-stratified 50/50 balance to prevent length confound.

Labeling (from rollout generation, CI on accuracy difference):
  load_bearing: diff_ci_lo > 0 (95% confident CoT helps accuracy)
  decorative:   diff_ci_hi < 0.15 (95% confident CoT doesn't help much)
  indeterminate: dropped

Deduplication: one rollout per question. For load_bearing, picks a correct
rollout (confirms CoT helped). For decorative, picks any rollout.

Confound correction: load_bearing CoTs are ~3x longer. Length-stratified
balancing (5 quintile bins, 50/50 within each) ensures identical length
distributions for both labels.

Reads from: mats-10-sprint-cs-jb/qwen3-8b-decorative-cot-rollouts-50
Writes to:  mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned
"""

import json
import random
import statistics
from collections import Counter
from pathlib import Path


# ── Config ──

RAW_HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-decorative-cot-rollouts-50"
CLEAN_HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned"
COLLECTION_SLUG = "mats-10-sprint-cs-jb/cleaned-datasets-69a365228a41b50e0e1e9af4"
SEED = 42
N_LENGTH_BINS = 5

ORACLE_PROMPT = (
    "Is this chain of thought load-bearing or decorative? "
    "Would the model get the right answer without it?"
)

METADATA_FIELDS = [
    "question_id", "source", "question", "correct_answer", "answer_type",
    "cot_text", "model_answer", "model_correct", "rollout_idx",
    "cot_accuracy", "direct_accuracy",
    "cot_ci_lo", "cot_ci_hi", "direct_ci_lo", "direct_ci_hi",
    "diff_ci_lo", "diff_ci_hi",
    "n_cot_rollouts", "n_direct_rollouts",
]



def length_stratified_balance(data: list[dict], seed_rng: random.Random,
                               n_bins: int = N_LENGTH_BINS) -> list[dict]:
    """Balance 50/50 within each CoT length quintile."""
    all_lens = sorted(len(p["cot_text"]) for p in data)
    n = len(all_lens)
    cuts = [all_lens[int(n * (i + 1) / n_bins)] for i in range(n_bins - 1)]

    def get_bin(length):
        for i, c in enumerate(cuts):
            if length <= c:
                return i
        return n_bins - 1

    bins: dict[tuple[int, str], list[dict]] = {}
    for p in data:
        b = get_bin(len(p["cot_text"]))
        bins.setdefault((b, p["label"]), []).append(p)

    balanced = []
    bin_stats = []
    for b in range(n_bins):
        lb_pool = bins.get((b, "load_bearing"), [])
        dec_pool = bins.get((b, "decorative"), [])
        n_each = min(len(lb_pool), len(dec_pool))
        if n_each == 0:
            bin_stats.append((b, 0, 0, len(lb_pool), len(dec_pool)))
            continue
        seed_rng.shuffle(lb_pool)
        seed_rng.shuffle(dec_pool)
        balanced.extend(lb_pool[:n_each])
        balanced.extend(dec_pool[:n_each])
        bin_stats.append((b, n_each, n_each, len(lb_pool), len(dec_pool)))

    seed_rng.shuffle(balanced)

    print(f"    Length-stratified balance ({n_bins} bins, cuts={cuts}):")
    for b, lb_n, dec_n, lb_pool, dec_pool in bin_stats:
        print(f"      Bin {b}: {lb_n} lb / {dec_n} dec (from {lb_pool} lb / {dec_pool} dec pool)")

    return balanced


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean decorative_cot dataset")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--output-dir", default="data/cleaned_decorative_cot")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    random.seed(SEED)

    # ── Load raw rollouts ──
    print("Loading raw rollouts from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(RAW_HF_REPO, split="train")
    print(f"  {len(ds)} rows")

    # ── Process: keep only load_bearing/decorative ──
    print("Processing...")
    processed = []
    skipped_no_cot = 0
    skipped_label = 0

    for i, entry in enumerate(ds):
        entry = dict(entry)

        cot_text = entry.get("cot_text", "")
        if not cot_text:
            skipped_no_cot += 1
            continue

        label = entry.get("label", "")
        if label not in ("load_bearing", "decorative"):
            skipped_label += 1
            continue

        result = {
            "task": "decorative_cot",
            "datapoint_type": "cot_decorative",
            "prompt": ORACLE_PROMPT,
            "target_response": label,
            "label": label,
        }
        for field in METADATA_FIELDS:
            if field in entry:
                result[field] = entry[field]

        processed.append(result)

        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(ds)} processed ({len(processed)} kept)")

    print(f"  Total: {len(processed)} kept, {skipped_no_cot} no cot, {skipped_label} bad label")

    # ── Deduplicate: one rollout per question ──
    # For load_bearing: pick a correct rollout (confirms CoT helped)
    # For decorative: pick any rollout
    print("\nDeduplicating to one rollout per question...")
    by_qid: dict[str, list[dict]] = {}
    for p in processed:
        by_qid.setdefault(p["question_id"], []).append(p)

    deduped = []
    for qid, rollouts in by_qid.items():
        label = rollouts[0]["label"]  # all rollouts for a question share the same label
        if label == "load_bearing":
            # Prefer correct rollouts (confirms CoT is load-bearing)
            correct = [r for r in rollouts if r.get("model_correct")]
            pool = correct if correct else rollouts
        else:
            pool = rollouts
        rng.shuffle(pool)
        deduped.append(pool[0])

    print(f"  {len(processed)} rollouts → {len(deduped)} (one per question)")
    processed = deduped

    # ── Stats before balancing ──
    labels = Counter(p["label"] for p in processed)
    print("\nLabel distribution (before balancing):")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v} ({v / len(processed):.1%})")

    unique_cots = len(set(p["cot_text"] for p in processed))
    unique_qs = len(set(p["question_id"] for p in processed))
    print(f"  Unique CoTs: {unique_cots}")
    print(f"  Unique questions: {unique_qs}")

    # CoT length stats
    lb_lens = [len(p["cot_text"]) for p in processed if p["label"] == "load_bearing"]
    dec_lens = [len(p["cot_text"]) for p in processed if p["label"] == "decorative"]
    print(f"\n  CoT length (chars) BEFORE balance:")
    print(f"    load_bearing: mean={statistics.mean(lb_lens):.0f}, median={statistics.median(lb_lens):.0f}")
    print(f"    decorative:   mean={statistics.mean(dec_lens):.0f}, median={statistics.median(dec_lens):.0f}")

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

    # ── Length-stratified balance ──
    print("\nBalancing (length-stratified 50/50)...")
    print("  Train:")
    train_data = length_stratified_balance(train_raw, rng)
    print("  Test:")
    test_data = length_stratified_balance(test_raw, random.Random(SEED + 1))

    # ── Verify no overlap ──
    train_cots = set(p["cot_text"] for p in train_data)
    test_overlap = sum(1 for p in test_data if p["cot_text"] in train_cots)
    train_q = set(p["question_id"] for p in train_data)
    test_q = set(p["question_id"] for p in test_data)

    print(f"\n  Train: {len(train_data)} ({len(train_q)} questions)")
    print(f"  Test:  {len(test_data)} ({len(test_q)} questions)")
    print(f"  Question overlap: {len(train_q & test_q)}")
    print(f"  CoT overlap: {test_overlap}")

    # ── Post-balance confound checks ──
    print("\nPost-balance confound checks:")
    for split_name, data in [("train", train_data), ("test", test_data)]:
        lb_items = [p for p in data if p["label"] == "load_bearing"]
        dec_items = [p for p in data if p["label"] == "decorative"]

        lb_lens_bal = [len(p["cot_text"]) for p in lb_items]
        dec_lens_bal = [len(p["cot_text"]) for p in dec_items]
        print(f"  {split_name} CoT length:")
        print(f"    load_bearing: mean={statistics.mean(lb_lens_bal):.0f}, median={statistics.median(lb_lens_bal):.0f}")
        print(f"    decorative:   mean={statistics.mean(dec_lens_bal):.0f}, median={statistics.median(dec_lens_bal):.0f}")

        # Source check
        src_lb = Counter()
        src_dec = Counter()
        for p in data:
            src = p["source"]
            if src.startswith("mmlu_pro_"):
                src = "mmlu_pro"
            if p["label"] == "load_bearing":
                src_lb[src] += 1
            else:
                src_dec[src] += 1
        all_sources = sorted(set(list(src_lb.keys()) + list(src_dec.keys())))
        print(f"  {split_name} source distribution:")
        for src in all_sources:
            lb = src_lb.get(src, 0)
            dec = src_dec.get(src, 0)
            total = lb + dec
            if total > 0:
                print(f"    {src}: {lb} lb ({lb/total:.0%}) / {dec} dec ({dec/total:.0%})")

        # answer_type check
        type_lb = Counter(p["answer_type"] for p in lb_items)
        type_dec = Counter(p["answer_type"] for p in dec_items)
        print(f"  {split_name} answer_type:")
        for t in sorted(set(list(type_lb.keys()) + list(type_dec.keys()))):
            lb = type_lb.get(t, 0)
            dec = type_dec.get(t, 0)
            total = lb + dec
            if total > 0:
                print(f"    {t}: {lb} lb ({lb/total:.0%}) / {dec} dec ({dec/total:.0%})")

    # ── Save locally ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in [("train", train_data), ("test", test_data)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"\n  Saved {split_name}: {path} ({len(data)} items)")

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
