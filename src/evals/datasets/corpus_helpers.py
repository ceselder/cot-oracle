"""Helpers for loading CoT corpus rows used by eval dataset generators."""

from __future__ import annotations

import json
from pathlib import Path


def load_corpus_rows(corpus_path: str | Path, limit: int | None = None) -> list[dict]:
    """Load JSONL corpus rows, skipping malformed rows.

    Expected fields (best-effort):
      - question
      - cot_response or cot_content
      - correct_answer (optional)
      - source/domain (optional)
    """
    path = Path(corpus_path)
    if not path.exists():
        return []

    rows: list[dict] = []
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
            cot = str(row.get("cot_content") or row.get("cot_response") or "").strip()
            if not question or not cot:
                continue

            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    return rows


def get_cot_text(row: dict) -> str:
    return str(row.get("cot_content") or row.get("cot_response") or "").strip()
