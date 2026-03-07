from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"


@lru_cache(maxsize=1)
def load_train_config() -> dict[str, Any]:
    return yaml.safe_load(TRAIN_CONFIG_PATH.read_text())


def get_score_model(cfg: dict[str, Any] | None = None) -> str:
    if cfg is not None and "eval" in cfg and "score_model" in cfg["eval"]:
        return str(cfg["eval"]["score_model"])
    return str(load_train_config()["eval"]["score_model"])


SCORE_MODEL = get_score_model()
LLM_SCORE_LABEL = "llm-score"
LLM_SCORE_VALUES = tuple(f"{i / 10:.1f}" for i in range(11))
LLM_SCORE_VALUES_TEXT = ", ".join(LLM_SCORE_VALUES)
LLM_SCORE_GRANULARITY_TEXT = f"Use only 0.1 increments: {LLM_SCORE_VALUES_TEXT}."
SCORE_PROMPT_VERSION = "20260307_decile_scores_v2"
QA_SCORE_TASKS = frozenset({"sqa", "chunked_convqa", "chunked_compqa"})
OPENROUTER_API_BASE = os.environ["OPENROUTER_API_BASE"] if "OPENROUTER_API_BASE" in os.environ else "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_COMPLETIONS_URL = f"{OPENROUTER_API_BASE.rstrip('/')}/chat/completions"
QA_SCORE_MAX_TOKENS = 256


def llm_score_field_requirement(field_name: str) -> str:
    return f"`{field_name}` must be a float from 0.0 to 1.0 in 0.1 increments only (choose one of {LLM_SCORE_VALUES_TEXT})"


def llm_score_axis_requirement(field_name: str) -> str:
    return f"`{field_name}` (float 0-1, 0.1 increments only; choose one of {LLM_SCORE_VALUES_TEXT})"


QA_SCORE_SYSTEM = (
    "You grade QA answers against a reference answer. "
    "Return ONLY JSON with keys `score` and `reason`. "
    f"{llm_score_field_requirement('score')}, where 1.0 means the candidate answer is fully correct and 0.0 means it is fully wrong or irrelevant. "
    "`reason` must be one very short sentence, at most 12 words."
)

REFERENCE_SCORE_MAX_TOKENS = 128
REFERENCE_SCORE_SYSTEM = (
    "You grade a candidate response against a reference response for an evaluation benchmark. "
    "Return ONLY JSON with keys `score` and `reason`. "
    f"{llm_score_field_requirement('score')}, where 1.0 means the candidate matches the reference response essentially perfectly, "
    "0.5 means it is partially correct or missing important details, and 0.0 means it is wrong, irrelevant, or contradicted by the reference. "
    "Focus on semantic correctness rather than wording overlap. "
    "`reason` must be one short sentence, at most 15 words."
)


def is_qa_score_task(task_name: str) -> bool:
    return task_name in QA_SCORE_TASKS


def extract_judge_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        score_match = re.match(r"^\s*(?:score\s*:\s*)?([01](?:\.\d+)?)\s*(?:\n+|$)(.*)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if score_match is None:
            raise ValueError(f"Judge did not return JSON: {cleaned[:200]}")
        reason = score_match.group(2).strip()
        return {"score": float(score_match.group(1)), "reason": reason}
    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        # Truncated JSON — try to close it
        truncated = cleaned[start:]
        # Close any open string, then close the object
        truncated = re.sub(r',\s*"[^"]*$', '', truncated)  # remove trailing partial key
        truncated = re.sub(r':\s*"[^"]*$', ': ""', truncated)  # close partial string value
        if not truncated.rstrip().endswith("}"):
            truncated = truncated.rstrip().rstrip(",") + "}"
        return json.loads(truncated)
    return json.loads(cleaned[start:end + 1])


def build_qa_score_prompt(task_name: str, qa_prompt: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Question:\n{qa_prompt}\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Candidate answer:\n{prediction}\n\n"
        "Score the candidate answer against the reference answer."
    )


def build_openended_reference_score_prompt(task_name: str, query: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Prompt:\n{query}\n\n"
        f"Reference response:\n{target}\n\n"
        f"Candidate response:\n{prediction}\n\n"
        "Score the candidate response against the reference response."
    )


TRAJECTORY_JUDGE_TASKS = frozenset({"answer_trajectory"})
TRAJECTORY_JUDGE_SYSTEM = (
    "You evaluate answer trajectory predictions. "
    "Return ONLY JSON with keys `answer_score` and `predicted_confidence`. "
    f"{llm_score_field_requirement('answer_score')}: 1.0 if the prediction identifies the same answer as the reference, 0.0 if wrong or irrelevant. "
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

LLM_JUDGE_TASKS = frozenset({"cot_description", "cot_metacognition", "sae_unverbalized"})

COT_DESCRIPTION_JUDGE_SYSTEM = (
    "You score an activation oracle's response against a reference answer on three dimensions. "
    "Return ONLY JSON with keys `correctness`, `specificity`, `confidence`, `reason`.\n\n"
    f"{llm_score_axis_requirement('correctness')}: 0.0 = wrong domain or fabricated details, "
    "0.5 = right domain but wrong specifics, 1.0 = factually matches reference.\n\n"
    f"{llm_score_axis_requirement('specificity')}: 0.0 = generic/unfalsifiable ('the model is processing information'), "
    "0.5 = right topic but missing required details, 1.0 = includes concrete names/numbers/relationships from reference.\n\n"
    f"{llm_score_axis_requirement('confidence')}: 0.0 = fully hedged/declined ('I cannot determine'), "
    "0.5 = partial hedging ('it seems to be about...'), 1.0 = stated as fact with no caveats.\n\n"
    "`reason`: one sentence, at most 15 words."
)

COT_METACOGNITION_JUDGE_SYSTEM = (
    "You evaluate whether an activation oracle correctly identified a model's metacognitive state "
    "(uncertainty, backtracking, error recognition, alternative paths, missing info, or number prediction). "
    "Return ONLY JSON with keys `correctness`, `specificity`, `vagueness_penalty`, `confidence`, `reason`.\n\n"
    f"{llm_score_axis_requirement('correctness')}: 0.0 = fabricated/wrong metacognitive state, "
    "0.5 = right category but wrong specifics, 1.0 = accurately identifies what the model is thinking about its own reasoning.\n\n"
    f"{llm_score_axis_requirement('specificity')}: 0.0 = generic boilerplate ('the model is uncertain'), "
    "0.5 = right topic but missing concrete values/steps, "
    "1.0 = includes exact numbers, variable names, or specific reasoning steps from the reference.\n\n"
    f"{llm_score_axis_requirement('vagueness_penalty')}: 1.0 = response is pure boilerplate that could apply to any problem "
    "('the model is processing information', 'thinking about the problem'), "
    "0.5 = somewhat generic but mentions the right domain, "
    "0.0 = clearly problem-specific with concrete details.\n\n"
    f"{llm_score_axis_requirement('confidence')}: 0.0 = fully hedged/declined ('I cannot determine'), "
    "0.5 = partial hedging ('it seems to be...'), 1.0 = stated as fact with no caveats.\n\n"
    "`reason`: one sentence, at most 15 words."
)

SAE_UNVERBALIZED_JUDGE_SYSTEM = (
    "You evaluate whether an activation oracle correctly identified unverbalized content — "
    "information present in the model's internal activations but NOT stated in the chain-of-thought text. "
    "Return ONLY JSON with keys `correctness`, `specificity`, `confidence`, `reason`.\n\n"
    f"{llm_score_axis_requirement('correctness')}: 0.0 = fabricated or wrong content, "
    "0.5 = right domain but wrong specifics, 1.0 = accurately identifies the same unverbalized content as reference.\n\n"
    f"{llm_score_axis_requirement('specificity')}: 0.0 = generic/unfalsifiable ('the model is processing information'), "
    "0.5 = right topic but missing concrete details, "
    "1.0 = includes specific concepts, domains, or signals matching the reference.\n\n"
    f"{llm_score_axis_requirement('confidence')}: 0.0 = fully hedged/declined ('I cannot determine'), "
    "0.5 = partial hedging ('it seems to be...'), 1.0 = stated as fact with no caveats.\n\n"
    "`reason`: one sentence, at most 15 words."
)

_LLM_JUDGE_SYSTEMS = {
    "cot_description": COT_DESCRIPTION_JUDGE_SYSTEM,
    "cot_metacognition": COT_METACOGNITION_JUDGE_SYSTEM,
    "sae_unverbalized": SAE_UNVERBALIZED_JUDGE_SYSTEM,
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
