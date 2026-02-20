"""ROT13 model-organism reconstruction eval dataset."""

from __future__ import annotations

import codecs
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.corpus_helpers import load_corpus_rows, get_cot_text


def _rot13(text: str) -> str:
    return codecs.decode(text, "rot_13")


def generate_rot13_reconstruction_dataset(
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
        decoded_cot = get_cot_text(row)
        if len(decoded_cot) < 32:
            continue

        rot13_cot = _rot13(decoded_cot)
        if rot13_cot == decoded_cot:
            continue

        items.append(
            EvalItem(
                eval_name="rot13_reconstruction",
                example_id=f"rot13_{len(items):04d}",
                clean_prompt=question,
                test_prompt=question,
                correct_answer=str(row.get("correct_answer", "")),
                nudge_answer=None,
                metadata={
                    "decoded_cot": decoded_cot,
                    "rot13_cot": rot13_cot,
                    "source": row.get("source"),
                    "domain": row.get("domain"),
                    "row_id": row.get("id"),
                    "metric": "tokens_successfully_inverted",
                },
            )
        )

    return items
