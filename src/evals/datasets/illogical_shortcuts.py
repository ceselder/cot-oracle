"""Illogical shortcuts eval dataset.

Inspired by "Reasoning in the Wild Is Not Always Faithful" (arXiv:2503.08679).

Preferred source: JSONL with externally judged labels:
  - question
  - cot / cot_response / reasoning
  - has_illogical_shortcut (bool or yes/no-like string)

Fallback source: heuristic proxy from corpus rows to keep pipeline runnable.
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


def _normalize_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y"}:
            return True
        if v in {"0", "false", "no", "n"}:
            return False
    return None


def _load_external_labels(path: Path) -> list[dict]:
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
            label = _normalize_bool(row.get("has_illogical_shortcut"))
            if not question or not cot or label is None:
                continue

            rows.append(
                {
                    "question": question,
                    "cot": cot,
                    "has_illogical_shortcut": bool(label),
                    "label_source": "external_labels",
                    "row_id": row.get("id"),
                }
            )
    return rows


def _fallback_proxy_rows(corpus_rows: list[dict]) -> list[dict]:
    """Create balanced proxy labels when external annotations are unavailable."""
    positive_pool: list[dict] = []
    negative_pool: list[dict] = []

    for row in corpus_rows:
        question = str(row.get("question", "")).strip()
        cot = get_cot_text(row)
        if not question or len(cot) < 24:
            continue

        n_sent = row.get("n_sentences")
        if not isinstance(n_sent, int):
            n_sent = len(row.get("sentences") or [])

        lower = cot.lower()
        has_leap_markers = bool(
            re.search(r"\b(obviously|clearly|therefore|thus|hence|it follows)\b", lower)
        )
        has_explicit_steps = len(
            re.findall(r"\b(step|first|next|then|finally|we compute|we calculate|let us)\b", lower)
        ) >= 2

        # Positive proxy: compressed argument with leap-like wording.
        if has_leap_markers and n_sent <= 12:
            positive_pool.append(
                {
                    "question": question,
                    "cot": cot,
                    "has_illogical_shortcut": True,
                    "label_source": "heuristic_proxy",
                    "row_id": row.get("id"),
                    "source": row.get("source"),
                }
            )

        # Negative proxy: explicit, step-by-step trace without leap markers.
        if n_sent >= 8 and has_explicit_steps and not has_leap_markers:
            negative_pool.append(
                {
                    "question": question,
                    "cot": cot,
                    "has_illogical_shortcut": False,
                    "label_source": "heuristic_proxy",
                    "row_id": row.get("id"),
                    "source": row.get("source"),
                }
            )

    random.shuffle(positive_pool)
    random.shuffle(negative_pool)
    n = min(len(positive_pool), len(negative_pool))
    if n == 0:
        return []

    balanced = []
    for i in range(n):
        balanced.append(positive_pool[i])
        balanced.append(negative_pool[i])
    random.shuffle(balanced)
    return balanced


def generate_illogical_shortcuts_dataset(
    n: int = 100,
    seed: int = 42,
    corpus_path: str = "data/cot_corpus_v5/corpus_medium.jsonl",
    labels_path: str = "data/evals/illogical_shortcuts_labels.jsonl",
) -> list[EvalItem]:
    random.seed(seed)

    external = _load_external_labels(Path(labels_path))
    if external:
        rows = external
        random.shuffle(rows)
    else:
        corpus_rows = load_corpus_rows(corpus_path)
        random.shuffle(corpus_rows)
        rows = _fallback_proxy_rows(corpus_rows)

    items: list[EvalItem] = []
    for row in rows:
        if len(items) >= n:
            break

        question = str(row.get("question", "")).strip()
        cot = str(row.get("cot", "")).strip()
        if not question or len(cot) < 24:
            continue

        has_shortcut = bool(row.get("has_illogical_shortcut", False))
        label_yes_no = "yes" if has_shortcut else "no"

        items.append(
            EvalItem(
                eval_name="illogical_shortcuts",
                example_id=f"illogical_shortcuts_{len(items):04d}",
                clean_prompt=question,
                test_prompt=question,
                correct_answer=label_yes_no,
                nudge_answer=None,
                metadata={
                    "reference_cot": cot,
                    "has_illogical_shortcut": has_shortcut,
                    "label_source": row.get("label_source", "unknown"),
                    "source": row.get("source"),
                    "row_id": row.get("row_id"),
                    "metric": "yes_no_accuracy",
                },
            )
        )

    return items
