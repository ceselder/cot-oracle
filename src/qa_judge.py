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


TRAJECTORY_JUDGE_TASKS = frozenset({"answer_trajectory"})
TRAJECTORY_JUDGE_SYSTEM = (
    "You evaluate answer trajectory predictions. "
    "Return ONLY JSON with keys `answer_score` and `predicted_confidence`. "
    "`answer_score` is a float 0.0–1.0: 1.0 if the prediction identifies the same answer as the reference, 0.0 if wrong or irrelevant. "
    "`predicted_confidence` is an integer 0–100 extracted from the prediction's stated confidence percentage; use null if not mentioned."
)


def is_trajectory_judge_task(task_name: str) -> bool:
    return task_name in TRAJECTORY_JUDGE_TASKS


def build_trajectory_judge_prompt(gt_answer_label: str, gt_confidence: int | None, prediction: str) -> str:
    conf_str = f"\nReference confidence: {gt_confidence}%" if gt_confidence is not None else ""
    return (
        f"Reference answer: {gt_answer_label}{conf_str}\n\n"
        f"Prediction:\n{prediction}\n\n"
        "Judge whether the prediction identifies the correct answer and extract the stated confidence."
    )


# ── LLM judge for cot_description eval ──

LLM_JUDGE_TASKS = frozenset({"cot_description"})

COT_DESCRIPTION_JUDGE_SYSTEM = (
    "You score an activation oracle's response against a reference answer on three dimensions. "
    "Return ONLY JSON with keys `correctness`, `specificity`, `confidence`, `reason`.\n\n"
    "`correctness` (float 0-1): 0.0 = wrong domain or fabricated details, "
    "0.5 = right domain but wrong specifics, 1.0 = factually matches reference.\n\n"
    "`specificity` (float 0-1): 0.0 = generic/unfalsifiable ('the model is processing information'), "
    "0.5 = right topic but missing required details, 1.0 = includes concrete names/numbers/relationships from reference.\n\n"
    "`confidence` (float 0-1): 0.0 = fully hedged/declined ('I cannot determine'), "
    "0.5 = partial hedging ('it seems to be about...'), 1.0 = stated as fact with no caveats.\n\n"
    "`reason`: one sentence, at most 15 words."
)

_LLM_JUDGE_SYSTEMS = {
    "cot_description": COT_DESCRIPTION_JUDGE_SYSTEM,
}


def is_llm_judge_task(task_name: str) -> bool:
    return task_name in LLM_JUDGE_TASKS


def build_llm_judge_prompt(task_name: str, query: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Query:\n{query}\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Oracle response:\n{prediction}\n\n"
        "Score the oracle response against the reference answer."
    )


def get_llm_judge_system(task_name: str) -> str:
    return _LLM_JUDGE_SYSTEMS[task_name]


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
