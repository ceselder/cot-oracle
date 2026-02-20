"""Hint/prompt influence yes/no eval dataset.

Reuses hinted MCQ construction but scored as yes/no influence detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.hinted_mcq import generate_hinted_mcq_dataset


def generate_hint_influence_yesno_dataset(n: int = 100, seed: int = 42) -> list[EvalItem]:
    base_items = generate_hinted_mcq_dataset(n=n, seed=seed)

    out: list[EvalItem] = []
    for i, item in enumerate(base_items):
        out.append(
            EvalItem(
                eval_name="hint_influence_yesno",
                example_id=f"hint_influence_{i:04d}",
                clean_prompt=item.clean_prompt,
                test_prompt=item.test_prompt,
                correct_answer=item.correct_answer,
                nudge_answer=item.nudge_answer,
                metadata={**item.metadata, "metric": "yes_no_accuracy"},
            )
        )

    return out
