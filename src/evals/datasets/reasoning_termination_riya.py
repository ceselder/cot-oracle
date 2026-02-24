"""
Eval: Reasoning Termination Prediction

Given a prefix of a chain of thought, predict whether the model will end its
thinking (emit </think>) within the next 100 tokens.

Labels:
  - "will_terminate": model is about to stop reasoning
  - "will_continue":  model has substantial reasoning left

**Training labels** (heuristic, used in dataset_classes/cot_reasoning_termination.py):
  Positive: prefix cut 25-55 tokens from CoT end
  Negative: prefix cut 300+ tokens from CoT end

**Eval labels** (resampled, produced by scripts/precompute_reasoning_termination.py):
  Positive if </think> appears in range 20-60 tokens after prefix in >=45/50 resamples
  Negative if </think> appears beyond token 200 in >=45/50 resamples

This file generates lightweight placeholder EvalItems from AMC + AIME data.
The precompute script fills in the actual CoT prefixes and resampled labels.

Source datasets:
  - AI-MO/aimo-validation-aime (90 items)
  - AI-MO/aimo-validation-amc  (83 items)

Neither AMC nor AIME overlap with GSM8K or MATH training data.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import _hf_load_dataset


def _load_aime_amc(seed: int = 42) -> list[dict]:
    """Load problems from AIME and AMC validation sets."""
    rng = random.Random(seed)
    pool = []

    try:
        ds_aime = _hf_load_dataset("AI-MO/aimo-validation-aime", split="train")
        for row in ds_aime:
            answer = str(row.get("answer", "")).strip()
            question = row.get("problem", "").strip()
            if question and answer:
                pool.append({
                    "question": question,
                    "correct_answer": answer,
                    "source": "aime",
                })
    except Exception as e:
        print(f"  Warning: could not load AIME dataset: {e}")

    try:
        ds_amc = _hf_load_dataset("AI-MO/aimo-validation-amc", split="train")
        for row in ds_amc:
            answer = str(row.get("answer", "")).strip()
            question = row.get("problem", "").strip()
            if question and answer:
                pool.append({
                    "question": question,
                    "correct_answer": answer,
                    "source": "amc",
                })
    except Exception as e:
        print(f"  Warning: could not load AMC dataset: {e}")

    rng.shuffle(pool)
    return pool


def generate_reasoning_termination_dataset(
    n: int = 100,
    seed: int = 42,
) -> list[EvalItem]:
    """Generate reasoning termination eval items from AMC + AIME problems.

    These are placeholder items. The precompute script
    (scripts/precompute_reasoning_termination.py) generates the actual CoT
    prefixes, resamples, and fills in the labels.

    clean_prompt = the math question (will be used to generate full CoT)
    test_prompt  = same question (prefix will be set by precompute)
    correct_answer = "will_terminate" or "will_continue" (set by precompute)

    At generation time, correct_answer is "pending_precompute" since we
    don't know the label until resampling is done on GPU.
    """
    random.seed(seed)

    problems = _load_aime_amc(seed=seed)
    if not problems:
        print("  Warning: no AIME/AMC problems loaded, returning empty dataset")
        return []

    # We want more problems than n because the precompute step will discard
    # some (short CoTs, ambiguous resamples). Request 2x.
    selected = problems[:min(n * 2, len(problems))]

    items = []
    for i, problem in enumerate(selected):
        items.append(EvalItem(
            eval_name="reasoning_termination_riya",
            example_id=f"reason_term_{i:04d}",
            clean_prompt=problem["question"],
            test_prompt=problem["question"],  # Same — prefix set by precompute
            correct_answer="pending_precompute",
            nudge_answer=None,
            metadata={
                "source": problem["source"],
                "math_answer": problem["correct_answer"],
            },
        ))

    return items
