"""
Eval 5: Answer Correctness (sanity check)

Can the oracle predict if the model's final answer is correct?
Uses clean math problems — no nudging.

If this fails, something fundamental is wrong with the oracle.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_generation import MATH_PROBLEMS
from evals.common import EvalItem


def generate_answer_correctness_dataset(n: int = 50, seed: int = 42) -> list[EvalItem]:
    """Generate answer correctness eval examples."""
    items = []

    for i in range(min(n, len(MATH_PROBLEMS))):
        problem = MATH_PROBLEMS[i]
        items.append(EvalItem(
            eval_name="answer_correctness",
            example_id=f"correctness_{i:04d}",
            clean_prompt=problem["q"],
            test_prompt=problem["q"],  # Same — no nudge
            correct_answer=problem["a"],
            nudge_answer=None,
            metadata={},
        ))

    # If we need more than hardcoded problems, try GSM8K
    if len(items) < n:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for row in ds:
                if len(items) >= n:
                    break
                parts = row["answer"].split("####")
                if len(parts) < 2:
                    continue
                answer = parts[-1].strip().replace(",", "")
                items.append(EvalItem(
                    eval_name="answer_correctness",
                    example_id=f"correctness_{len(items):04d}",
                    clean_prompt=row["question"],
                    test_prompt=row["question"],
                    correct_answer=answer,
                    nudge_answer=None,
                    metadata={"source": "gsm8k"},
                ))
        except Exception:
            pass  # Use what we have

    return items
