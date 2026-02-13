"""
Eval 4: Decorative CoT

Is the model's chain-of-thought load-bearing or decorative?
- Decorative: model gets it right WITH and WITHOUT CoT
- Load-bearing: model only gets it right WITH CoT

Requires running the model N times per condition (expensive).
Ground truth is determined during run_evals.py, not at dataset generation time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_generation import MATH_PROBLEMS
from evals.common import EvalItem


def generate_decorative_cot_dataset(n: int = 50, seed: int = 42) -> list[EvalItem]:
    """Generate decorative CoT eval examples.

    clean_prompt and test_prompt are the same question — the difference
    is that clean_prompt will be run WITHOUT CoT (enable_thinking=False)
    and test_prompt will be run WITH CoT (enable_thinking=True).
    """
    items = []

    for i in range(min(n, len(MATH_PROBLEMS))):
        problem = MATH_PROBLEMS[i]
        items.append(EvalItem(
            eval_name="decorative_cot",
            example_id=f"decorative_{i:04d}",
            clean_prompt=problem["q"],
            test_prompt=problem["q"],  # Same question, different inference mode
            correct_answer=problem["a"],
            nudge_answer=None,
            metadata={"n_runs": 5},  # How many times to run each condition
        ))

    # Extend with GSM8K if needed
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
                    eval_name="decorative_cot",
                    example_id=f"decorative_{len(items):04d}",
                    clean_prompt=row["question"],
                    test_prompt=row["question"],
                    correct_answer=answer,
                    nudge_answer=None,
                    metadata={"n_runs": 5, "source": "gsm8k"},
                ))
        except Exception:
            pass

    return items
