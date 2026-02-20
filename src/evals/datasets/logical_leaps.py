"""CoT logical-leaps eval dataset.

Preferred source is a JSONL file with Gemini/annotator labels.
Fallback uses a weak heuristic label so pipeline can run end-to-end.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.corpus_helpers import load_corpus_rows, get_cot_text


YES_TOKENS = ["obviously", "clearly", "therefore", "thus", "hence", "it follows"]


def _heuristic_has_logical_leap(cot: str) -> bool:
    text = cot.lower()
    signal = sum(tok in text for tok in YES_TOKENS)
    # Weak heuristic only for fallback mode.
    return signal >= 2 and len(re.findall(r"[.!?]", text)) <= 6


def _load_gemini_labels(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(row.get("question", "")).strip()
            cot = str(row.get("cot") or row.get("cot_response") or row.get("reasoning") or "").strip()
            label = row.get("has_logical_leap")
            if isinstance(label, str):
                label = label.strip().lower() in {"1", "true", "yes", "y"}
            if not question or not cot or not isinstance(label, bool):
                continue

            rows.append(
                {
                    "question": question,
                    "cot": cot,
                    "has_logical_leap": bool(label),
                    "label_source": "gemini",
                    "row_id": row.get("id"),
                }
            )

    return rows


def generate_logical_leaps_dataset(
    n: int = 100,
    seed: int = 42,
    corpus_path: str = "data/cot_corpus_v5/corpus_medium.jsonl",
    gemini_labels_path: str = "data/evals/logical_leaps_gemini.jsonl",
) -> list[EvalItem]:
    random.seed(seed)

    gemini_rows = _load_gemini_labels(Path(gemini_labels_path))
    rows: list[dict]

    if gemini_rows:
        rows = gemini_rows
        random.shuffle(rows)
    else:
        corpus_rows = load_corpus_rows(corpus_path)
        random.shuffle(corpus_rows)
        rows = []
        for row in corpus_rows:
            cot = get_cot_text(row)
            rows.append(
                {
                    "question": str(row.get("question", "")).strip(),
                    "cot": cot,
                    "has_logical_leap": _heuristic_has_logical_leap(cot),
                    "label_source": "heuristic_fallback",
                    "row_id": row.get("id"),
                    "source": row.get("source"),
                }
            )

    items: list[EvalItem] = []
    for row in rows:
        if len(items) >= n:
            break

        question = str(row.get("question", "")).strip()
        cot = str(row.get("cot", "")).strip()
        if not question or len(cot) < 24:
            continue

        has_leap = bool(row.get("has_logical_leap"))
        label_yes_no = "yes" if has_leap else "no"

        items.append(
            EvalItem(
                eval_name="logical_leaps",
                example_id=f"logical_leaps_{len(items):04d}",
                clean_prompt=question,
                test_prompt=question,
                correct_answer=label_yes_no,
                nudge_answer=None,
                metadata={
                    "reference_cot": cot,
                    "has_logical_leap": has_leap,
                    "label_source": row.get("label_source", "unknown"),
                    "source": row.get("source"),
                    "row_id": row.get("row_id"),
                    "metric": "yes_no_accuracy",
                },
            )
        )

    return items
