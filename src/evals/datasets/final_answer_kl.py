"""Final-answer prediction eval (KL metric) dataset.

Generates MCQ-style items where the oracle must predict final answer distribution.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.hinted_mcq import _extract_gsm8k_answer, _generate_distractors


def generate_final_answer_kl_dataset(n: int = 100, seed: int = 42) -> list[EvalItem]:
    random.seed(seed)

    rows: list[dict] = []
    try:
        from datasets import load_dataset

        gsm = load_dataset("openai/gsm8k", "main", split="test")
        for row in gsm:
            answer_str = _extract_gsm8k_answer(row["answer"])
            if answer_str is None:
                continue
            try:
                float(answer_str)
            except ValueError:
                continue
            rows.append({"question": row["question"], "answer": answer_str, "source": "GSM8K"})
    except Exception:
        from data_generation import MATH_PROBLEMS

        for p in MATH_PROBLEMS:
            rows.append({"question": p["q"], "answer": p["a"], "source": "MATH_SYN"})

    if not rows:
        return []

    random.shuffle(rows)
    options = ["A", "B", "C", "D"]

    items: list[EvalItem] = []
    for i, row in enumerate(rows):
        if len(items) >= n:
            break

        try:
            correct_num = float(row["answer"])
        except ValueError:
            continue

        correct_str = str(int(correct_num)) if correct_num == int(correct_num) else row["answer"]
        distractors = _generate_distractors(correct_num, n=3)

        correct_pos = random.randint(0, 3)
        choices: dict[str, str] = {}
        d_idx = 0
        for j, opt in enumerate(options):
            if j == correct_pos:
                choices[opt] = correct_str
            else:
                choices[opt] = distractors[d_idx]
                d_idx += 1

        correct_letter = options[correct_pos]
        choices_text = "\\n".join(f"{k}) {v}" for k, v in choices.items())
        prompt = f"{row['question']}\\n\\n{choices_text}\\n\\nAnswer with just the letter (A, B, C, or D)."

        items.append(
            EvalItem(
                eval_name="final_answer_kl",
                example_id=f"final_answer_kl_{len(items):04d}",
                clean_prompt=prompt,
                test_prompt=prompt,
                correct_answer=correct_letter,
                nudge_answer=None,
                metadata={
                    "choices": choices,
                    "source": row.get("source", "unknown"),
                    "metric": "avg_kl_divergence",
                },
            )
        )

    return items
