"""
Eval 4: Decorative CoT

Is the model's chain-of-thought load-bearing or decorative?
- Decorative: model gets it right WITH and WITHOUT CoT
- Load-bearing: model only gets it right WITH CoT

If precomputed rollouts exist (from precompute_decorative_rollouts.py),
uses those labels and representative CoTs. Otherwise generates skeleton
items that require live computation during eval.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import load_math_test


def generate_decorative_cot_dataset(n: int = 100, seed: int = 42) -> list[EvalItem]:
    """Generate decorative CoT eval examples.

    If data/evals/decorative_cot_v2.json exists (from precompute_decorative_rollouts.py),
    uses the precomputed labels and representative CoTs. Otherwise falls back to
    skeleton generation from MATH problems.
    """
    random.seed(seed)

    # Try precomputed v2 file first
    precomputed_path = Path("data/evals/decorative_cot_v2.json")
    if precomputed_path.exists():
        with open(precomputed_path) as f:
            precomputed = json.load(f)
        # Filter to decorative/load_bearing only (skip indeterminate)
        valid = [r for r in precomputed if r.get("label") in ("decorative", "load_bearing")]
        if valid:
            random.shuffle(valid)
            items = []
            for i, r in enumerate(valid[:n]):
                items.append(EvalItem(
                    eval_name="decorative_cot",
                    example_id=f"decorative_{i:04d}",
                    clean_prompt=r["question"],
                    test_prompt=r["question"],
                    correct_answer=r.get("correct_answer", ""),
                    nudge_answer=None,
                    metadata={
                        "decorative_label": r["label"],
                        "representative_cot": r.get("representative_cot", ""),
                        "with_cot_acc": r.get("cot_accuracy", 0.0),
                        "without_cot_acc": r.get("direct_accuracy", 0.0),
                        "source": r.get("source", "MATH"),
                        "n_runs": r.get("n_runs", 10),
                    },
                ))
            print(f"  decorative_cot: loaded {len(items)} precomputed items from {precomputed_path}")
            return items

    # Fallback: skeleton items from MATH (requires live computation during eval)
    problems = load_math_test(n * 2, seed=seed)
    items = []

    for i in range(min(n, len(problems))):
        problem = problems[i]
        items.append(EvalItem(
            eval_name="decorative_cot",
            example_id=f"decorative_{i:04d}",
            clean_prompt=problem["question"],
            test_prompt=problem["question"],
            correct_answer=problem["correct_answer"],
            nudge_answer=None,
            metadata={"n_runs": 5, "source": problem["source"]},
        ))

    return items
