"""ROT13 model-organism reconstruction eval dataset.

Provides math problems for the ROT13 model organism eval. At eval time:
1. The ROT13 LoRA adapter generates obfuscated CoT
2. Activations are extracted from the base model (adapters disabled)
3. The oracle reconstructs the plain-English reasoning from activations
4. A normal (non-ROT13) CoT is generated as ground truth for scoring
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import load_math_test


def generate_rot13_reconstruction_dataset(
    n: int = 100,
    seed: int = 42,
    **_kwargs,
) -> list[EvalItem]:
    """Generate eval items for ROT13 model organism reconstruction.

    Each item is a math problem. The model organism (ROT13 LoRA) generates
    obfuscated CoT at eval time; no pre-encoding needed here.
    """
    problems = load_math_test(n, seed=seed)

    items: list[EvalItem] = []
    for i, prob in enumerate(problems):
        items.append(
            EvalItem(
                eval_name="rot13_reconstruction",
                example_id=f"rot13_{i:04d}",
                clean_prompt=prob["question"],
                test_prompt=prob["question"],
                correct_answer=str(prob["correct_answer"]),
                nudge_answer=None,
                metadata={
                    "source": prob.get("source", ""),
                    "subject": prob.get("subject", ""),
                    "metric": "token_match_rate",
                },
            )
        )

    return items
