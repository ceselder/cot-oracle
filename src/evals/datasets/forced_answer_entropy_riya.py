"""
Eval: Forced Answer Entropy (Regression)

At each sentence boundary during CoT reasoning, force the model to answer
by appending an anchor phrase ("So, the answer is:") after the partial CoT.
Extract logprobs over answer tokens (A/B/C/D for MCQ, or numeric tokens).
Compute entropy H(p) = -sum(p_i * log(p_i)) of the softmax distribution.

This is a REGRESSION task: the oracle predicts the entropy value (float),
not a class label. The metric is R^2 (coefficient of determination).

Sources: ARC-Challenge (4-choice science MCQ, NOT in training data).
Falls back to GPQA Diamond if available.

The actual entropy values are computed during GPU precompute (see
scripts/precompute_forced_entropy.py). This generator creates the skeleton
EvalItems with questions and answer choices.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import _hf_load_dataset


def _load_mcq_questions(n: int, seed: int = 42) -> list[dict]:
    """Load 4-choice MCQ questions. Tries GPQA Diamond first, falls back to ARC-Challenge."""
    rng = random.Random(seed)
    items = []

    # Try GPQA Diamond first (gated — requires HF auth)
    try:
        ds = _hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for row in ds:
            question = row.get("Question", "")
            correct = row.get("Correct Answer", "")
            choices_raw = [
                row.get("Correct Answer", ""),
                row.get("Incorrect Answer 1", ""),
                row.get("Incorrect Answer 2", ""),
                row.get("Incorrect Answer 3", ""),
            ]
            choices_raw = [c for c in choices_raw if c and c.strip()]
            if len(choices_raw) < 4 or not question or not correct:
                continue
            items.append({
                "question": question.strip(),
                "correct_answer": correct.strip(),
                "choices_raw": choices_raw,
                "source": "gpqa_diamond",
            })
        if items:
            print(f"  Loaded {len(items)} GPQA Diamond questions")
            rng.shuffle(items)
            return items[:n]
    except Exception as e:
        print(f"  GPQA Diamond unavailable ({e}), falling back to ARC-Challenge")

    # Fallback: ARC-Challenge (open, 4-choice science MCQ)
    ds = _hf_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for row in ds:
        question = row.get("question", "")
        choices_obj = row.get("choices", {})
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        answer_key = row.get("answerKey", "")

        if len(labels) != 4 or len(texts) != 4:
            continue
        if not question or not answer_key:
            continue

        # Find correct answer text
        correct_text = None
        for lbl, txt in zip(labels, texts):
            if lbl == answer_key:
                correct_text = txt
                break
        if correct_text is None:
            continue

        items.append({
            "question": question.strip(),
            "correct_answer": correct_text.strip(),
            "choices_raw": [t.strip() for t in texts],
            "source": "arc_challenge",
        })

    print(f"  Loaded {len(items)} ARC-Challenge questions")
    rng.shuffle(items)
    return items[:n]


def generate_forced_answer_entropy_dataset(
    n: int = 100,
    seed: int = 42,
) -> list[EvalItem]:
    """Generate forced answer entropy eval examples from GPQA Diamond.

    Each item is a 4-choice MCQ. During GPU precompute, we will:
    1. Generate multiple CoT rollouts per question
    2. At each sentence boundary, force answer and extract logprobs
    3. Compute entropy at each boundary
    4. Store per-boundary entropy values in metadata

    The oracle's task is to predict the entropy value from activations.
    """
    rng = random.Random(seed)
    options = ["A", "B", "C", "D"]

    raw = _load_mcq_questions(n * 2, seed=seed)
    if not raw:
        print("WARNING: Could not load GPQA Diamond dataset. Returning empty list.")
        return []

    items: list[EvalItem] = []

    for row in raw:
        if len(items) >= n:
            break

        choices_raw = list(row["choices_raw"])
        rng.shuffle(choices_raw)

        # Determine which letter is correct after shuffling
        correct_text = row["correct_answer"]
        correct_letter = None
        choices: dict[str, str] = {}
        for j, opt in enumerate(options):
            choices[opt] = choices_raw[j]
            if choices_raw[j] == correct_text:
                correct_letter = opt

        if correct_letter is None:
            continue

        choices_text = "\n".join(f"{k}) {v}" for k, v in choices.items())
        prompt = (
            f"{row['question']}\n\n"
            f"{choices_text}\n\n"
            f"Answer with just the letter (A, B, C, or D)."
        )

        items.append(EvalItem(
            eval_name="forced_answer_entropy_riya",
            example_id=f"forced_entropy_{len(items):04d}",
            clean_prompt=prompt,
            test_prompt=prompt,  # same — no nudge
            correct_answer=correct_letter,
            nudge_answer=None,
            metadata={
                "choices": choices,
                "correct_letter": correct_letter,
                "correct_text": correct_text,
                "source": row["source"],
                "answer_tokens": options,  # tokens to check logprobs for
                "metric": "r_squared",
                "task_type": "regression",
                # These will be populated by precompute_forced_entropy.py:
                # "entropy_values": [...],  # list of floats per sentence boundary
                # "sentence_boundaries": [...],  # list of boundary positions
                # "rollout_cots": [...],  # list of generated CoTs
            },
        ))

    return items


if __name__ == "__main__":
    items = generate_forced_answer_entropy_dataset()
    out_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "forced_answer_entropy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    from evals.common import save_eval_items
    save_eval_items(items, out_path)
    print(f"Wrote {len(items)} items to {out_path}")
