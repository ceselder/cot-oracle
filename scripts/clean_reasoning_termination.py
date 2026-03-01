#!/usr/bin/env python3
"""
Clean the reasoning_termination dataset:
  - Decode context_input_ids back to text, extract question + cot_text
  - Remove "Activations from N positions" prompt prefix
  - Change target response from will_terminate/will_continue to Yes/No format
  - Strip precomputed tokenization (context_input_ids, context_positions)
  - Add question-level train/test split (80/20)
  - Balance 50/50 will_terminate / will_continue

Reads from: ceselder/cot-oracle-reasoning-termination-balanced
Writes to:  mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned
"""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


# ── Config ──

RAW_HF_REPO = "ceselder/cot-oracle-reasoning-termination-balanced"
CLEAN_HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned"
COLLECTION_SLUG = "mats-10-sprint-cs-jb/cleaned-datasets-69a365228a41b50e0e1e9af4"
SEED = 42

LABEL_RATIOS = {
    "will_terminate": 0.50,
    "will_continue": 0.50,
}

ORACLE_PROMPT = (
    "Will the model terminate reasoning (emit </think>) soon? "
    "If yes, estimate how many tokens remain."
)


def parse_target_response(raw_target: str) -> tuple[str, int] | None:
    """Parse raw target like 'will_terminate, in 25 tokens' or 'will_continue, 549 tokens remain'."""
    m = re.match(r"will_terminate, in (\d+) tokens", raw_target)
    if m:
        return "will_terminate", int(m.group(1))

    m = re.match(r"will_continue, (\d+) tokens remain", raw_target)
    if m:
        return "will_continue", int(m.group(1))

    return None


def estimate_confidence(label: str, tokens_remaining: int) -> int:
    """Confidence based on distance from termination boundary.
    Close to boundary (~50 tokens) = less confident. Far away = very confident."""
    if label == "will_terminate":
        # 5 tokens left → 95%, 25 → 75%, 45 → 55%
        return max(55, min(99, 100 - tokens_remaining))
    else:
        # 50 remaining → 60%, 200 → 90%, 500+ → 99%
        return max(55, min(99, 50 + tokens_remaining // 5))


def build_target_response(label: str, tokens_remaining: int) -> str:
    conf = estimate_confidence(label, tokens_remaining)
    if label == "will_terminate":
        return f"Yes, the model will stop reasoning in approximately {tokens_remaining} tokens. (confidence: {conf}%)"
    else:
        return f"No, the model will continue reasoning for approximately {tokens_remaining} more tokens. (confidence: {conf}%)"


def extract_question_and_cot(text: str) -> tuple[str, str] | None:
    """Split decoded text into question (user turn) and cot_text (assistant turn)."""
    # Format: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{cot}
    parts = text.split("<|im_start|>assistant\n", 1)
    if len(parts) != 2:
        return None

    user_part = parts[0]
    cot_text = parts[1].rstrip()

    # Extract question from user turn
    user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", user_part, re.DOTALL)
    if not user_match:
        return None

    question = user_match.group(1).strip()
    return question, cot_text


def question_id_from_text(question: str) -> str:
    """Generate a stable question_id from question text."""
    return hashlib.md5(question.encode()).hexdigest()[:12]


def balance_split(data: list[dict], rng: random.Random) -> list[dict]:
    pools: dict[str, list[dict]] = {}
    for item in data:
        pools.setdefault(item["label"], []).append(item)

    max_total = float("inf")
    for label, ratio in LABEL_RATIOS.items():
        pool_size = len(pools.get(label, []))
        if ratio > 0:
            max_total = min(max_total, pool_size / ratio)
    max_total = int(max_total)

    balanced = []
    for label, ratio in LABEL_RATIOS.items():
        pool = pools.get(label, [])
        target_n = int(max_total * ratio)
        rng.shuffle(pool)
        balanced.extend(pool[:target_n])

    rng.shuffle(balanced)
    return balanced


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean reasoning_termination dataset")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--output-dir", default="data/cleaned_reasoning_termination")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    random.seed(SEED)

    # ── Load raw data ──
    print("Loading raw data from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(RAW_HF_REPO, split="train")
    print(f"  {len(ds)} rows")

    # ── Load tokenizer for decoding ──
    print("Loading Qwen3-8B tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # ── Process ──
    print("Processing: decode tokens, extract text, reformat targets...")
    processed = []
    skipped_parse = 0
    skipped_extract = 0

    for i, entry in enumerate(ds):
        # Parse raw target response
        parsed = parse_target_response(entry["target_response"])
        if parsed is None:
            skipped_parse += 1
            continue
        label, tokens_remaining = parsed

        # Decode token IDs to text
        text = tokenizer.decode(entry["context_input_ids"], skip_special_tokens=False)

        # Extract question and CoT
        extracted = extract_question_and_cot(text)
        if extracted is None:
            skipped_extract += 1
            continue
        question, cot_text = extracted

        qid = question_id_from_text(question)
        target = build_target_response(label, tokens_remaining)

        result = {
            "task": "reasoning_termination",
            "datapoint_type": "cot_reasoning_termination",
            "prompt": ORACLE_PROMPT,
            "target_response": target,
            "label": label,
            "question_id": qid,
            "question": question,
            "cot_text": cot_text,
            "tokens_remaining": tokens_remaining,
            "cot_token_length": len(entry["context_input_ids"]),
        }
        processed.append(result)

        if (i + 1) % 3000 == 0:
            print(f"  {i + 1}/{len(ds)} processed ({len(processed)} kept)")

    print(f"  Total: {len(processed)} kept, {skipped_parse} bad target, {skipped_extract} bad extraction")

    # ── Stats ──
    labels = Counter(p["label"] for p in processed)
    print("\nLabel distribution (before balancing):")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v} ({v / len(processed):.1%})")

    unique_questions = len(set(p["question_id"] for p in processed))
    print(f"  Unique questions: {unique_questions}")

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

    # ── Balance both splits ──
    print("Balancing to 50/50 ratio...")
    train_data = balance_split(train_raw, rng)
    test_data = balance_split(test_raw, random.Random(SEED + 1))

    print(f"  Train: {len(train_data)} (from {len(train_raw)} pool)")
    print(f"  Test:  {len(test_data)} (from {len(test_raw)} pool)")

    # Check CoT overlap
    train_cots = set(t["cot_text"] for t in train_data)
    test_overlap = sum(1 for t in test_data if t["cot_text"] in train_cots)
    print(f"  CoT train/test overlap: {test_overlap}")

    # Show sample target responses
    print("\nSample target responses:")
    for label in ["will_terminate", "will_continue"]:
        examples = [p for p in train_data if p["label"] == label]
        if examples:
            print(f"  [{label}] {examples[0]['target_response']}")

    # Show split label distributions
    for split_name, data in [("train", train_data), ("test", test_data)]:
        split_labels = Counter(p["label"] for p in data)
        print(f"\n  {split_name}:")
        for k, v in sorted(split_labels.items()):
            print(f"    {k}: {v} ({v / len(data):.1%})")

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
