from __future__ import annotations

import json
import os
import re
from typing import Any


QA_GEMINI_SCORE_TASKS = frozenset({"sqa", "chunked_convqa", "chunked_compqa"})
QA_GEMINI_SCORE_MODEL = os.environ["COT_ORACLE_QA_EVAL_MODEL"] if "COT_ORACLE_QA_EVAL_MODEL" in os.environ else "google/gemini-2.5-flash"
OPENROUTER_API_BASE = os.environ["OPENROUTER_API_BASE"] if "OPENROUTER_API_BASE" in os.environ else "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_COMPLETIONS_URL = f"{OPENROUTER_API_BASE.rstrip('/')}/chat/completions"
QA_GEMINI_SCORE_MAX_TOKENS = 160
QA_GEMINI_SCORE_SYSTEM = (
    "You grade QA answers against a reference answer. "
    "Return ONLY JSON with keys `score` and `reason`. "
    "`score` must be a float from 0.0 to 1.0 where 1.0 means the candidate answer is fully correct and 0.0 means it is fully wrong or irrelevant. "
    "`reason` must be one very short sentence, at most 12 words."
)


def is_gemini_qa_task(task_name: str) -> bool:
    return task_name in QA_GEMINI_SCORE_TASKS


def extract_judge_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Judge did not return JSON: {cleaned[:200]}")
    return json.loads(cleaned[start:end + 1])


def build_qa_gemini_score_prompt(task_name: str, qa_prompt: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Question:\n{qa_prompt}\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Candidate answer:\n{prediction}\n\n"
        "Score the candidate answer against the reference answer."
    )


def compute_token_f1_scores(predictions: list[str], targets: list[str]) -> list[float]:
    scores = []
    for prediction, target in zip(predictions, targets):
        pred_tokens = prediction.lower().split()
        target_tokens = target.lower().split()
        if not target_tokens:
            scores.append(1.0 if not pred_tokens else 0.0)
            continue
        pred_set = set(pred_tokens)
        target_set = set(target_tokens)
        if not pred_set or not target_set:
            scores.append(0.0)
            continue
        common = pred_set & target_set
        precision = len(common) / len(pred_set)
        recall = len(common) / len(target_set)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return scores
