"""
Eval 5: Answer Correctness (sanity check)

Can the oracle predict if the model's final answer is correct?
Uses clean math problems — no nudging.

If this fails, something fundamental is wrong with the oracle.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import load_math_test


def generate_answer_correctness_dataset(n: int = 100, seed: int = 42) -> list[EvalItem]:
    """Generate answer correctness eval examples."""
    random.seed(seed)

    problems = load_math_test(n * 2, seed=seed)
    items = []

    for i in range(min(n, len(problems))):
        problem = problems[i]
        items.append(EvalItem(
            eval_name="answer_correctness",
            example_id=f"correctness_{i:04d}",
            clean_prompt=problem["question"],
            test_prompt=problem["question"],  # Same — no nudge
            correct_answer=problem["correct_answer"],
            nudge_answer=None,
            metadata={"source": problem["source"]},
        ))

    return items
