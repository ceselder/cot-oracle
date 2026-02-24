"""Shared test-split loaders for eval dataset generators.

All eval datasets must draw questions from held-out test splits to avoid
contamination with training data. This module provides standardised loaders.

Available sources:
  - GSM8K test (1319 items, all numeric answers)
  - MATH-500 test (500 items, ~325 with numeric answers)
  - Scruples test (2500 moral dilemma items)
"""

from __future__ import annotations

import importlib
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the HuggingFace `datasets` library at import time.
# The local evals/datasets/ package shadows it, so we temporarily strip
# conflicting sys.path entries to find the real one.
# ---------------------------------------------------------------------------
_hf_datasets = None

def _get_hf_datasets():
    global _hf_datasets
    if _hf_datasets is not None:
        return _hf_datasets

    # Save and strip local path entries that shadow HF datasets
    _this = str(Path(__file__).resolve().parent)
    _evals = str(Path(__file__).resolve().parent.parent)
    _src = str(Path(__file__).resolve().parent.parent.parent)
    exclude = {_this, _evals, _src}

    original_path = sys.path[:]
    sys.path = [p for p in sys.path if p not in exclude]

    # Temporarily remove local 'datasets' package from sys.modules
    local_keys = [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]
    saved = {k: sys.modules.pop(k) for k in local_keys}

    try:
        importlib.invalidate_caches()
        _hf_datasets = importlib.import_module("datasets")
    finally:
        sys.path = original_path
        # Restore local package refs but keep HF modules loaded too
        for k, v in saved.items():
            if k not in sys.modules:
                sys.modules[k] = v

    return _hf_datasets


def _hf_load_dataset(*args, **kwargs):
    """Load a HuggingFace dataset."""
    return _get_hf_datasets().load_dataset(*args, **kwargs)


# ------------------------------------------------------------------
# Math problem loaders
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_gsm8k_raw() -> list[dict]:
    """Load all GSM8K test items (cached)."""
    ds = _hf_load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        parts = row["answer"].split("####")
        if len(parts) < 2:
            continue
        answer = parts[-1].strip().replace(",", "")
        try:
            float(answer)
        except ValueError:
            continue
        items.append({
            "question": row["question"],
            "correct_answer": answer,
            "source": "gsm8k_test",
            "subject": "arithmetic",
        })
    return items


@lru_cache(maxsize=1)
def _load_math500_raw() -> list[dict]:
    """Load MATH-500 test items with numeric answers (cached)."""
    ds = _hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds:
        answer = row["answer"].strip()
        try:
            float(answer.replace(",", ""))
        except ValueError:
            continue
        items.append({
            "question": row["problem"],
            "correct_answer": answer,
            "source": "math500_test",
            "subject": row.get("subject", "math"),
        })
    return items


def load_math_test(n: int, seed: int = 42) -> list[dict]:
    """Load n math problems from GSM8K + MATH-500 test splits.

    Returns list of dicts with keys: question, correct_answer, source, subject.
    Deterministic given seed.
    """
    rng = random.Random(seed)
    pool = list(_load_gsm8k_raw()) + list(_load_math500_raw())
    rng.shuffle(pool)
    return pool[:n]


def load_gsm8k_test(n: int, seed: int = 42) -> list[dict]:
    """Load n items from GSM8K test split only."""
    rng = random.Random(seed)
    pool = list(_load_gsm8k_raw())
    rng.shuffle(pool)
    return pool[:n]


# ------------------------------------------------------------------
# Scruples loader
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_scruples_raw() -> list[dict]:
    """Load Scruples test split (cached)."""
    ds = _hf_load_dataset("metaeval/scruples", split="test")
    items = []
    for row in ds:
        text = row.get("text", "")
        if not text or len(text) < 50:
            continue
        binarized = row.get("binarized_label", "")
        if binarized not in ("RIGHT", "WRONG"):
            continue
        # Truncate very long texts
        if len(text) > 500:
            text = text[:500] + "..."
        items.append({
            "text": text,
            "correct_judgment": binarized,
            "source": "scruples_test",
        })
    return items


@lru_cache(maxsize=1)
def _load_arc_challenge_raw() -> list[dict]:
    """Load ARC-Challenge test items (cached).

    Already MCQ format with A-E choices. Harder than GSM8K —
    Qwen3-8B is ~75% accurate, making it susceptible to hints.
    """
    ds = _hf_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in ds:
        question = row.get("question", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if len(labels) < 3 or len(labels) != len(texts):
            continue
        choice_map = dict(zip(labels, texts))
        items.append({
            "question": question,
            "choices": choice_map,
            "correct_letter": answer_key,
            "source": "arc_challenge_test",
            "subject": "science",
        })
    return items


def load_arc_challenge_test(n: int, seed: int = 42) -> list[dict]:
    """Load n items from ARC-Challenge test split.

    Returns list of dicts with keys: question, choices, correct_letter, source, subject.
    """
    rng = random.Random(seed)
    pool = list(_load_arc_challenge_raw())
    rng.shuffle(pool)
    return pool[:n]


def load_scruples_test(n: int, seed: int = 42) -> list[dict]:
    """Load n moral dilemmas from Scruples test split.

    Returns list of dicts with keys: text, correct_judgment, source.
    """
    rng = random.Random(seed)
    pool = list(_load_scruples_raw())
    rng.shuffle(pool)
    return pool[:n]


# ------------------------------------------------------------------
# Shared helpers (moved from data_generation.py / hinted_mcq.py)
# ------------------------------------------------------------------

def generate_wrong_answer(correct: str) -> str:
    """Generate a plausible but wrong numeric answer."""
    try:
        val = int(correct)
    except ValueError:
        try:
            val = int(float(correct))
        except ValueError:
            return correct + "0"

    options = [
        val + random.randint(1, 20),
        val - random.randint(1, 20),
        val + random.choice([10, 100, -10, -100]),
        int(val * random.choice([1.1, 0.9, 1.2, 0.8])),
    ]
    wrong = random.choice(options)
    while wrong == val:
        wrong = val + random.randint(1, 50)
    return str(wrong)


def generate_distractors(correct: float, n: int = 3) -> list[str]:
    """Generate n plausible wrong numeric answers for MCQ."""
    distractors: set[str] = set()
    attempts = 0
    while len(distractors) < n and attempts < 100:
        attempts += 1
        offset = random.choice([
            random.randint(1, 20),
            -random.randint(1, 20),
            random.choice([10, -10, 100, -100]),
            int(correct * random.choice([0.1, -0.1, 0.5, -0.5])),
        ])
        candidate = int(correct + offset)
        if candidate != int(correct) and candidate > 0:
            distractors.add(str(candidate))

    while len(distractors) < n:
        distractors.add(str(int(correct) + len(distractors) + 7))

    return list(distractors)[:n]
