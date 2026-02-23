"""Held-out CoT reconstruction eval dataset.

Goal: reconstruct CoT from saved activations and score with averaged token KL.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.corpus_helpers import load_corpus_rows, get_cot_text


def generate_held_out_cot_reconstruction_dataset(
    n: int = 100,
    seed: int = 42,
    corpus_path: str = "data/cot_corpus_v5/corpus_medium.jsonl",
) -> list[EvalItem]:
    random.seed(seed)
    rows = load_corpus_rows(corpus_path)
    if not rows:
        return []

    random.shuffle(rows)

    items: list[EvalItem] = []
    for row in rows:
        if len(items) >= n:
            break

        question = str(row.get("question", "")).strip()
        cot = get_cot_text(row)
        if len(cot) < 32:
            continue

        items.append(
            EvalItem(
                eval_name="held_out_cot_reconstruction",
                example_id=f"held_out_cot_{len(items):04d}",
                clean_prompt=question,
                test_prompt=question,
                correct_answer=str(row.get("correct_answer", "")),
                nudge_answer=None,
                metadata={
                    "reference_cot": cot,
                    "source": row.get("source"),
                    "domain": row.get("domain"),
                    "row_id": row.get("id"),
                    "metric": "avg_token_kl",
                },
            )
        )

    return items
