#!/usr/bin/env python3
"""
Generate atypical answer training data for the CoT Oracle.

Downloads hard MCQ datasets, generates N rollouts per question from Qwen3-8B
with temperature sampling, and labels each rollout as "majority" or "minority"
based on the answer distribution across all rollouts for that question.

Sources: MMLU-Pro (~12k, 10 choices), ARC-Challenge (~2.5k), CommonsenseQA (~11k), MMLU (~15.5k)
Total: ~41k questions x 25 rollouts = ~1M completions

Usage (on GPU):
    python scripts/precompute_atypical_training.py \
        --model Qwen/Qwen3-8B \
        --n-rollouts 25 \
        --temperature 0.6

    # Resume from checkpoint:
    python scripts/precompute_atypical_training.py --resume

    # Re-label from existing checkpoint without re-generating:
    python scripts/precompute_atypical_training.py --split-only

    # Upload to HuggingFace:
    python scripts/precompute_atypical_training.py --split-only --upload
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


# ── Dataset loaders ──

def load_mmlu_pro() -> list[dict]:
    """MMLU-Pro: ~12k hard questions with 10 choices (A-J)."""
    from datasets import load_dataset
    print("  Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = []
    for i, row in enumerate(ds):
        options = row["options"]
        labels = [chr(65 + j) for j in range(len(options))]
        choices = "\n".join(f"{l}) {o}" for l, o in zip(labels, options))
        items.append({
            "question_id": f"mmlupro_{i}",
            "source": f"mmlu_pro_{row.get('category', 'unknown')}",
            "question": row["question"],
            "choices": choices,
            "n_choices": len(options),
            "correct_answer": row["answer"],
        })
    print(f"    {len(items)} questions")
    return items


def load_arc_challenge() -> list[dict]:
    """ARC-Challenge: ~2.5k hard science questions, 3-5 choices."""
    from datasets import load_dataset
    print("  Loading ARC-Challenge...")
    items = []
    for split in ["train", "test"]:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        for row in ds:
            raw_labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            # Normalize numeric labels (1,2,3,4) to letters (A,B,C,D)
            labels = []
            for l in raw_labels:
                if l.isdigit():
                    labels.append(chr(64 + int(l)))
                else:
                    labels.append(l)
            choices = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
            correct = row["answerKey"]
            if correct.isdigit():
                correct = chr(64 + int(correct))
            items.append({
                "question_id": f"arc_{row['id']}",
                "source": "arc_challenge",
                "question": row["question"],
                "choices": choices,
                "n_choices": len(labels),
                "correct_answer": correct,
            })
    print(f"    {len(items)} questions")
    return items


def load_commonsense_qa() -> list[dict]:
    """CommonsenseQA: ~11k commonsense questions, 5 choices (A-E)."""
    from datasets import load_dataset
    print("  Loading CommonsenseQA...")
    items = []
    for split in ["train", "validation"]:
        ds = load_dataset("tau/commonsense_qa", split=split)
        for row in ds:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choices = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
            items.append({
                "question_id": f"csqa_{row['id']}",
                "source": "commonsense_qa",
                "question": row["question"],
                "choices": choices,
                "n_choices": len(labels),
                "correct_answer": row.get("answerKey", ""),
            })
    print(f"    {len(items)} questions")
    return items


def load_mmlu() -> list[dict]:
    """MMLU: ~15.5k questions across 57 subjects, 4 choices (A-D)."""
    from datasets import load_dataset
    print("  Loading MMLU...")
    items = []
    for split in ["test", "validation"]:
        ds = load_dataset("cais/mmlu", "all", split=split)
        for i, row in enumerate(ds):
            choices_list = row["choices"]
            labels = ["A", "B", "C", "D"]
            choices = "\n".join(f"{l}) {c}" for l, c in zip(labels, choices_list))
            answer_letter = chr(65 + row["answer"])
            items.append({
                "question_id": f"mmlu_{split}_{i}",
                "source": f"mmlu_{row.get('subject', 'unknown')}",
                "question": row["question"],
                "choices": choices,
                "n_choices": 4,
                "correct_answer": answer_letter,
            })
    print(f"    {len(items)} questions")
    return items


def load_hellaswag() -> list[dict]:
    """HellaSwag: ~10k commonsense completion, 4 choices."""
    from datasets import load_dataset
    print("  Loading HellaSwag...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for i, row in enumerate(ds):
        ctx = row["ctx"]
        endings = row["endings"]
        labels = ["A", "B", "C", "D"]
        choices = "\n".join(f"{l}) {e}" for l, e in zip(labels, endings))
        correct = chr(65 + int(row["label"]))
        items.append({
            "question_id": f"hellaswag_{i}",
            "source": "hellaswag",
            "question": f"Complete the following:\n{ctx}",
            "choices": choices,
            "n_choices": 4,
            "correct_answer": correct,
        })
    print(f"    {len(items)} questions")
    return items


DATASET_LOADERS = {
    "mmlu_pro": load_mmlu_pro,
    "arc": load_arc_challenge,
    "csqa": load_commonsense_qa,
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
}

ALL_DATASETS = ["mmlu_pro", "arc", "csqa", "mmlu"]  # default set


def load_all_questions(datasets: list[str]) -> list[dict]:
    """Load questions from selected datasets."""
    items = []
    for name in datasets:
        loader = DATASET_LOADERS[name]
        items.extend(loader())
    print(f"  Total: {len(items)} questions from {len(datasets)} datasets")
    return items


# ── Prompt formatting ──

PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step, then give your final answer as just the letter.

Question: {question}

{choices}"""


def format_prompt(item: dict, tokenizer) -> str:
    """Format as chat message with thinking disabled for reliable answer extraction."""
    content = PROMPT_TEMPLATE.format(
        question=item["question"], choices=item["choices"]
    )
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── Answer extraction ──

def extract_answer(response: str, n_choices: int = 10) -> str | None:
    """Extract answer letter from model response.

    Looks for the last standalone letter (A-J depending on n_choices)
    in the response text. Prefers text after </think> if present.
    """
    if not response:
        return None

    valid = "".join(chr(65 + i) for i in range(min(n_choices, 10)))
    pattern = r'\b([' + valid + r'])\b'

    # Prefer text after </think> tag
    think_end = response.rfind("</think>")
    if think_end >= 0:
        after = response[think_end + len("</think>"):]
        matches = list(re.finditer(pattern, after))
        if matches:
            return matches[-1].group(1)

    # Also look for "answer is X" or "Answer: X" patterns
    ans_pattern = r'(?:answer\s+is|answer:\s*)\s*([' + valid + r'])\b'
    ans_matches = list(re.finditer(ans_pattern, response, re.IGNORECASE))
    if ans_matches:
        return ans_matches[-1].group(1)

    # Fallback: last standalone letter
    matches = list(re.finditer(pattern, response))
    if matches:
        return matches[-1].group(1)

    return None


# ── vLLM generation ──

def generate_rollouts(
    all_items: list[dict],
    model_name: str,
    n_rollouts: int,
    temperature: float,
    max_tokens: int,
    checkpoint_path: Path,
    batch_size: int = 500,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 4096,
):
    """Generate rollouts using vLLM with batched generation and checkpointing."""
    from vllm import LLM, SamplingParams

    # Load checkpoint to find completed question IDs
    completed_ids = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    completed_ids.add(data["question_id"])
                except json.JSONDecodeError:
                    continue
    print(f"  Checkpoint: {len(completed_ids)} questions already done")

    pending = [item for item in all_items if item["question_id"] not in completed_ids]
    if not pending:
        print("  All questions completed!")
        return

    print(f"  Pending: {len(pending)} questions ({len(pending) * n_rollouts} completions)")

    # Load model
    print(f"  Loading vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        seed=42,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # vLLM defaults max_tokens=16 if not set — use max_model_len for "unlimited"
    effective_max_tokens = max_tokens if max_tokens is not None else max_model_len
    params = SamplingParams(
        temperature=temperature,
        max_tokens=effective_max_tokens,
        n=n_rollouts,
        seed=42,
    )

    # Format all prompts
    print("  Formatting prompts...")
    prompts = [format_prompt(item, tokenizer) for item in pending]

    # Process in batches for checkpointing
    n_batches = (len(pending) + batch_size - 1) // batch_size
    t0 = time.time()
    total_done = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pending))
        batch_prompts = prompts[start:end]
        batch_items = pending[start:end]

        batch_n = len(batch_prompts)
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: "
              f"{batch_n} questions ({batch_n * n_rollouts} completions)")

        bt0 = time.time()
        outputs = llm.generate(batch_prompts, params)
        bt = time.time() - bt0

        # Write results to checkpoint (append mode)
        with open(checkpoint_path, "a") as f:
            for item, output in zip(batch_items, outputs):
                for ri, completion in enumerate(output.outputs):
                    answer = extract_answer(completion.text, item["n_choices"])
                    record = {
                        "question_id": item["question_id"],
                        "source": item["source"],
                        "question": item["question"],
                        "choices": item["choices"],
                        "n_choices": item["n_choices"],
                        "correct_answer": item["correct_answer"],
                        "rollout_idx": ri,
                        "cot_text": completion.text,
                        "model_answer": answer,
                    }
                    f.write(json.dumps(record) + "\n")
                f.flush()

        total_done += batch_n
        elapsed = time.time() - t0
        comp_done = total_done * n_rollouts
        comp_total = len(pending) * n_rollouts
        rate = comp_done / elapsed
        eta = (comp_total - comp_done) / rate if rate > 0 else 0

        print(f"    {bt:.0f}s ({batch_n * n_rollouts / bt:.0f} completions/s)")
        print(f"    Progress: {total_done}/{len(pending)} questions "
              f"({comp_done}/{comp_total} completions)")
        print(f"    Elapsed: {elapsed / 3600:.1f}h, ETA: {eta / 3600:.1f}h")

    # Cleanup
    del llm
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\n  Generation complete! {total_done * n_rollouts} total completions "
          f"in {(time.time() - t0) / 3600:.1f}h")


# ── Labeling ──

def label_rollouts(checkpoint_path: Path, output_path: Path, min_valid_frac: float = 0.5):
    """Read checkpoint, compute answer distributions, label each rollout."""
    print("  Loading checkpoint...")

    # Group rollouts by question
    questions: dict[str, dict] = {}
    n_lines = 0
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1

            qid = record["question_id"]
            if qid not in questions:
                questions[qid] = {
                    "question_id": qid,
                    "source": record["source"],
                    "question": record["question"],
                    "choices": record["choices"],
                    "n_choices": record["n_choices"],
                    "correct_answer": record["correct_answer"],
                    "rollouts": [],
                }
            questions[qid]["rollouts"].append({
                "rollout_idx": record["rollout_idx"],
                "cot_text": record["cot_text"],
                "model_answer": record["model_answer"],
            })

    print(f"  Loaded {n_lines} rollout lines across {len(questions)} questions")

    # Label each rollout
    stats = Counter()
    output_items = []

    for qid, data in questions.items():
        rollouts = data["rollouts"]
        n_total = len(rollouts)

        # Count valid answers
        answers = [r["model_answer"] for r in rollouts if r["model_answer"] is not None]
        if len(answers) < n_total * min_valid_frac:
            stats["skipped_extraction_fail"] += 1
            continue

        answer_counts = Counter(answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        n_valid = len(answers)
        minority_rate = 1.0 - answer_counts[majority_answer] / n_valid

        if minority_rate == 0:
            stats["skipped_no_variance"] += 1
            continue

        stats["questions_with_variance"] += 1

        # Label each rollout
        for rollout in rollouts:
            if rollout["model_answer"] is None:
                stats["skipped_null_answer"] += 1
                continue

            label = "minority" if rollout["model_answer"] != majority_answer else "majority"
            output_items.append({
                "question_id": qid,
                "source": data["source"],
                "question": data["question"],
                "choices": data["choices"],
                "correct_answer": data["correct_answer"],
                "cot_text": rollout["cot_text"],
                "model_answer": rollout["model_answer"],
                "label": label,
                "majority_answer": majority_answer,
                "minority_rate": round(minority_rate, 4),
                "n_valid_rollouts": n_valid,
                "rollout_idx": rollout["rollout_idx"],
            })
            stats[f"label_{label}"] += 1

    # Print stats
    print(f"\n  Labeling stats:")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")

    maj = stats.get("label_majority", 0)
    mino = stats.get("label_minority", 0)
    total = maj + mino
    if total > 0:
        print(f"    majority/minority split: {maj/total:.1%} / {mino/total:.1%}")
    print(f"    Total output items: {total}")

    # Save
    with open(output_path, "w") as f:
        for item in output_items:
            f.write(json.dumps(item) + "\n")
    print(f"\n  Saved {len(output_items)} items to {output_path}")

    return output_items


# ── Upload ──

def upload_to_hf(path: Path, repo_id: str, token: str | None = None,
                 collection_slug: str | None = None):
    """Upload dataset to HuggingFace and optionally add to collection."""
    from datasets import Dataset
    from huggingface_hub import HfApi

    print(f"  Loading {path}...")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"  Creating dataset with {len(items)} items...")
    ds = Dataset.from_list(items)

    print(f"  Pushing to {repo_id}...")
    ds.push_to_hub(repo_id, token=token, private=False)
    print(f"  Uploaded to https://huggingface.co/datasets/{repo_id}")

    # Add to collection if specified
    if collection_slug:
        try:
            api = HfApi(token=token)
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="dataset",
            )
            print(f"  Added to collection: {collection_slug}")
        except Exception as e:
            print(f"  Warning: could not add to collection: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate atypical answer training data"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-rollouts", type=int, default=25,
                        help="Rollouts per question (default: 25)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per completion (default: unlimited, uses max-model-len)")
    parser.add_argument("--output", default="atypical_answer_training.jsonl")
    parser.add_argument("--checkpoint", default="atypical_training_checkpoint.jsonl")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Questions per vLLM batch for checkpointing")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Datasets to use (default: mmlu_pro arc csqa mmlu)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--split-only", action="store_true",
                        help="Skip generation, just label from checkpoint")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace after labeling")
    parser.add_argument("--hf-repo",
                        default="mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-collection",
                        default="mats-10-sprint-cs-jb/cot-oracle-training-data-68539f06e54a8e04c1e79d78")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Phase 1: Load questions
    print("=" * 60)
    print("Phase 1: Loading questions")
    print("=" * 60)
    all_items = load_all_questions(args.datasets)

    if not args.split_only:
        # Phase 2: Generate rollouts
        print("\n" + "=" * 60)
        print("Phase 2: Generating rollouts")
        print("=" * 60)

        if not args.resume and checkpoint_path.exists():
            print(f"  WARNING: checkpoint exists at {checkpoint_path}")
            print(f"  Use --resume to continue, or delete it to start fresh")
            sys.exit(1)

        generate_rollouts(
            all_items=all_items,
            model_name=args.model,
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    # Phase 3: Label
    print("\n" + "=" * 60)
    print("Phase 3: Labeling rollouts")
    print("=" * 60)

    if not checkpoint_path.exists():
        print(f"  ERROR: No checkpoint at {checkpoint_path}")
        print(f"  Run without --split-only first.")
        sys.exit(1)

    items = label_rollouts(checkpoint_path, output_path)

    # Phase 4: Upload
    if args.upload:
        print("\n" + "=" * 60)
        print("Phase 4: Uploading to HuggingFace")
        print("=" * 60)
        token = args.hf_token
        upload_to_hf(
            output_path, args.hf_repo, token=token,
            collection_slug=args.hf_collection,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
