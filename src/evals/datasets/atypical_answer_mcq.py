"""
Eval: Atypical Answer (MCQ) — predict majority vs minority answer on hard MCQs.

Like atypical_answer_riya but using hard multiple-choice questions (MMLU-Pro,
ARC-Challenge, CommonsenseQA, MMLU) instead of moral dilemmas. 25 rollouts per
question from Qwen3-8B at temperature=0.6.

Data source: mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts on HuggingFace.

Held-out eval items are sampled from questions NOT used in training (using a
different seed to ensure question-level split).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import _hf_load_dataset


HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts"


def generate_atypical_answer_mcq_dataset(
    n: int = 100,
    seed: int = 12345,  # Different seed from training (42) for question-level split
) -> list[EvalItem]:
    """Generate atypical answer MCQ eval dataset from HuggingFace.

    Downloads the full rollout dataset and selects a held-out balanced subset.
    Uses a different seed than training to ensure question-level separation.

    Args:
        n: Total number of eval items (balanced 50/50 majority/minority).
        seed: Random seed (different from training seed=42 for split).
    """
    rng = random.Random(seed)

    # Download from HuggingFace
    items = _load_from_hf()
    if not items:
        print("  WARNING: Could not load atypical answer MCQ data from HuggingFace.")
        print(f"  Expected dataset: {HF_REPO}")
        return []

    # Group by question_id for question-level split
    by_question: dict[str, list[dict]] = {}
    for item in items:
        qid = item["question_id"]
        if qid not in by_question:
            by_question[qid] = []
        by_question[qid].append(item)

    # Shuffle questions and take the LAST portion as eval (training uses first portion)
    question_ids = sorted(by_question.keys())
    rng.shuffle(question_ids)

    # Use last 20% of questions for eval to avoid overlap with training
    eval_start = int(len(question_ids) * 0.8)
    eval_qids = set(question_ids[eval_start:])

    print(f"  Total questions: {len(question_ids)}, eval questions: {len(eval_qids)}")

    # Collect eval items split by label
    majority_pool = []
    minority_pool = []
    for qid in eval_qids:
        for item in by_question[qid]:
            if item["label"] == "majority":
                majority_pool.append(item)
            elif item["label"] == "minority":
                minority_pool.append(item)

    print(f"  Eval pool: {len(majority_pool)} majority, {len(minority_pool)} minority")

    if not majority_pool or not minority_pool:
        print("  WARNING: Cannot balance eval — missing majority or minority items")
        return []

    # Balance 50/50
    target_per_class = n // 2

    rng.shuffle(majority_pool)
    rng.shuffle(minority_pool)

    n_maj = min(len(majority_pool), target_per_class)
    n_min = min(len(minority_pool), target_per_class)

    selected = majority_pool[:n_maj] + minority_pool[:n_min]
    rng.shuffle(selected)

    # Build EvalItems
    eval_items = []
    for i, r in enumerate(selected):
        # Reconstruct prompt (question + choices)
        prompt = r["question"]
        if r.get("choices"):
            prompt += "\n\n" + r["choices"]
        prompt += "\n\nThink step by step, then answer with just the letter."

        eval_items.append(EvalItem(
            eval_name="atypical_answer_mcq",
            example_id=f"atypical_mcq_{i:04d}",
            clean_prompt=prompt,
            test_prompt=prompt,  # No nudge — same for both
            correct_answer=r["label"],  # "majority" or "minority"
            nudge_answer=None,
            metadata={
                "question_id": r.get("question_id", ""),
                "source": r.get("source", ""),
                "majority_answer": r.get("majority_answer", ""),
                "model_answer": r.get("model_answer", ""),
                "minority_rate": r.get("minority_rate", 0.0),
                "n_valid_rollouts": r.get("n_valid_rollouts", 0),
                "cot_text": r.get("cot_text", ""),
                "label": r["label"],
                "correct_answer_key": r.get("correct_answer", ""),
                "metric": "atypical_accuracy",
            },
        ))

    print(f"  Generated {len(eval_items)} atypical answer MCQ eval items "
          f"({n_maj} majority, {n_min} minority)")
    return eval_items


def _load_from_hf() -> list[dict]:
    """Download atypical answer rollout data from HuggingFace."""
    print(f"  Downloading from {HF_REPO}...")
    try:
        ds = _hf_load_dataset(HF_REPO, split="train")
        items = []
        for row in ds:
            if row.get("cot_text") and row.get("label") in ("majority", "minority"):
                items.append(dict(row))
        print(f"  Downloaded {len(items)} items")
        return items
    except Exception as e:
        print(f"  HF download failed: {e}")
        return []
