#!/usr/bin/env python3
"""
Generate 50 rollouts per question for atypical answer training data.

Uses the same question set as the existing 25-rollout dataset but generates
50 rollouts for tighter Wilson CIs on majority/minority classification.

Sources: MMLU-Pro (law, business, psychology, biology, chemistry)
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

# ── Config ──

MODEL = "Qwen/Qwen3-8B"
N_ROLLOUTS = 50
TEMPERATURE = 0.6
MAX_TOKENS = 4096
BATCH_SIZE = 500
GPU_MEM = 0.92
SEED = 42
CHECKPOINT = "atypical_rollouts_50.jsonl"
OUTPUT = "atypical_rollouts_50_labeled.jsonl"

MCQ_TEMPLATE = """Answer the following multiple choice question. Think step by step, then give your final answer as just the letter.

Question: {question}

{choices}"""


def load_questions():
    """Load questions from MMLU-Pro (same sources as 25-rollout dataset)."""
    from datasets import load_dataset

    items = []

    # MMLU-Pro
    print("  Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # Filter to same categories as existing dataset
    keep_cats = {"law", "business", "psychology", "biology", "chemistry"}
    for i, row in enumerate(ds):
        cat = row.get("category", "unknown")
        if cat not in keep_cats:
            continue
        options = row["options"]
        labels = [chr(65 + j) for j in range(len(options))]
        choices = "\n".join(f"{l}) {o}" for l, o in zip(labels, options))
        items.append({
            "question_id": f"mmlupro_{i}",
            "source": f"mmlu_pro_{cat}",
            "question": row["question"],
            "choices": choices,
            "n_choices": len(options),
            "correct_answer": row["answer"],
        })

    print(f"  Total: {len(items)} questions")
    return items


def extract_answer(text: str, n_choices: int = 10) -> str | None:
    """Extract letter answer from CoT output."""
    valid = set(chr(65 + i) for i in range(n_choices))

    # Try "answer is X" patterns
    patterns = [
        r"(?:final answer|answer is|I choose|I\'ll go with|select)\s*[:\-]?\s*\(?([A-J])\)?",
        r"\b([A-J])\s*\)?\s*$",  # letter at end of text
        r"^\s*\(?([A-J])\)?\s*$",  # letter on its own line
        r"\*\*([A-J])\*\*",  # bold letter
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            letter = m.group(1).upper()
            if letter in valid:
                return letter

    # Last single letter in valid set
    for ch in reversed(text):
        if ch.upper() in valid:
            return ch.upper()

    return None


def generate_rollouts(items, checkpoint_path):
    """Generate N_ROLLOUTS per question using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Load checkpoint
    completed = {}
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        with open(ckpt) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item["question_id"]
                    completed.setdefault(qid, []).append(item)
        print(f"  Loaded {sum(len(v) for v in completed.values())} rollouts from checkpoint ({len(completed)} questions)")

    # Filter to questions needing more rollouts
    todo = []
    for item in items:
        qid = item["question_id"]
        existing = len(completed.get(qid, []))
        if existing < N_ROLLOUTS:
            todo.append((item, N_ROLLOUTS - existing))

    if not todo:
        print("  All questions done!")
        return completed

    print(f"  {len(todo)} questions need rollouts")

    # Build prompts
    all_prompts = []
    all_meta = []
    for item, n_needed in todo:
        prompt_text = MCQ_TEMPLATE.format(
            question=item["question"],
            choices=item["choices"],
        )
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        for rollout_idx in range(n_needed):
            all_prompts.append(formatted)
            all_meta.append({
                "item": item,
                "rollout_idx": len(completed.get(item["question_id"], [])) + rollout_idx,
            })

    total_prompts = len(all_prompts)
    print(f"  Total prompts to generate: {total_prompts}")

    # Initialize vLLM
    print(f"  Loading {MODEL}...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_TOKENS + 512,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Generate in batches
    ckpt_f = open(ckpt, "a")
    for batch_start in range(0, total_prompts, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_prompts)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_meta = all_meta[batch_start:batch_end]

        t0 = time.time()
        outputs = llm.generate(batch_prompts, sampling)
        elapsed = time.time() - t0

        for output, meta in zip(outputs, batch_meta):
            cot_text = output.outputs[0].text
            # Strip think tags if present
            cot_text = re.sub(r"<think>.*?</think>", "", cot_text, flags=re.DOTALL).strip()

            item = meta["item"]
            answer = extract_answer(cot_text, item.get("n_choices", 10))

            rollout = {
                "question_id": item["question_id"],
                "source": item["source"],
                "question": item["question"],
                "choices": item["choices"],
                "correct_answer": item["correct_answer"],
                "cot_text": cot_text,
                "model_answer": answer,
                "rollout_idx": meta["rollout_idx"],
            }
            ckpt_f.write(json.dumps(rollout) + "\n")
            completed.setdefault(item["question_id"], []).append(rollout)

        ckpt_f.flush()
        done = batch_end
        print(f"  [{done}/{total_prompts}] {elapsed:.1f}s ({len(batch_prompts)/elapsed:.0f} prompts/s)")

    ckpt_f.close()
    return completed


def label_and_save(completed, output_path):
    """Label rollouts and save final dataset."""
    output = []
    stats = Counter()

    for qid, rollouts in completed.items():
        # Count valid answers
        valid = [r for r in rollouts if r["model_answer"] is not None]
        if len(valid) < 20:
            stats["too_few_valid"] += 1
            continue

        # Find majority answer
        answer_counts = Counter(r["model_answer"] for r in valid)
        majority_answer = answer_counts.most_common(1)[0][0]
        majority_count = answer_counts[majority_answer]
        minority_rate = 1.0 - majority_count / len(valid)

        for r in valid:
            label = "majority" if r["model_answer"] == majority_answer else "minority"
            output.append({
                **r,
                "label": label,
                "majority_answer": majority_answer,
                "minority_rate": round(minority_rate, 4),
                "n_valid_rollouts": len(valid),
            })
            stats[label] += 1

    print(f"\nLabeling stats:")
    print(f"  Questions: {len(completed)}")
    print(f"  Skipped (too few valid): {stats['too_few_valid']}")
    print(f"  majority: {stats['majority']}")
    print(f"  minority: {stats['minority']}")
    print(f"  Total labeled: {stats['majority'] + stats['minority']}")

    random.shuffle(output)
    with open(output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved to {output_path}")


def upload(output_path, repo_id, token):
    """Upload to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo", default="mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts-50")
    args = parser.parse_args()

    random.seed(SEED)

    if not args.label_only:
        print("Loading questions...")
        items = load_questions()

        print(f"\nGenerating {N_ROLLOUTS} rollouts per question...")
        completed = generate_rollouts(items, CHECKPOINT)
    else:
        print("Loading from checkpoint...")
        completed = {}
        with open(CHECKPOINT) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed.setdefault(item["question_id"], []).append(item)
        print(f"  {sum(len(v) for v in completed.values())} rollouts, {len(completed)} questions")

    print("\nLabeling...")
    label_and_save(completed, OUTPUT)

    if args.upload:
        import os
        token = os.environ.get("HF_TOKEN")
        upload(OUTPUT, args.hf_repo, token)

    print("\nDone!")


if __name__ == "__main__":
    main()
