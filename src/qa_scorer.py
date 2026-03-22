from __future__ import annotations

import json
import os
import re

import httpx
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "eval.yaml"


@lru_cache(maxsize=1)
def _load_eval_config() -> dict[str, Any]:
    return yaml.safe_load(EVAL_CONFIG_PATH.read_text())


def get_score_model(cfg: dict[str, Any] | None = None) -> str:
    if cfg is not None:
        if "score_model" in cfg:
            return str(cfg["score_model"])
        if "eval" in cfg and "score_model" in cfg["eval"]:
            return str(cfg["eval"]["score_model"])
    return str(_load_eval_config()["score_model"])


SCORE_MODEL = get_score_model()
LLM_SCORE_LABEL = "llm-score"
LLM_SCORE_VALUES = tuple(f"{i / 10:.1f}" for i in range(11))
LLM_SCORE_VALUES_TEXT = ", ".join(LLM_SCORE_VALUES)
LLM_SCORE_GRANULARITY_TEXT = f"Use only 0.1 increments: {LLM_SCORE_VALUES_TEXT}."

QA_SCORE_TASKS = frozenset({"sqa", "chunked_convqa", "chunked_compqa"})
OPENROUTER_API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_COMPLETIONS_URL = f"{OPENROUTER_API_BASE.rstrip('/')}/chat/completions"
QA_SCORE_MAX_TOKENS = 256

_scorer_endpoint_cache: tuple[str, dict] | None = None


def _get_scorer_endpoint() -> tuple[str, dict]:
    """Return (chat_completions_url, headers) for the scorer via OpenRouter."""
    global _scorer_endpoint_cache
    if _scorer_endpoint_cache is not None:
        return _scorer_endpoint_cache

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    _scorer_endpoint_cache = (OPENROUTER_CHAT_COMPLETIONS_URL, headers)
    return _scorer_endpoint_cache


def check_openrouter_available() -> bool:
    """Check if the OpenRouter scorer endpoint is reachable."""
    url, headers = _get_scorer_endpoint()
    base = url.rsplit("/chat/completions", 1)[0]
    try:
        resp = httpx.get(f"{base}/models", headers=headers, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


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

def is_qa_score_task(task_name: str) -> bool:
    return task_name in QA_SCORE_TASKS



def _json_loads_lenient(s: str) -> dict:
    """json.loads with fallback for invalid escape sequences and control chars."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
        return json.loads(fixed, strict=False)


def extract_scorer_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        score_match = re.match(r"^\s*(?:score\s*:\s*)?([01](?:\.\d+)?)\s*(?:\n+|$)(.*)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if score_match is None:
            raise ValueError(f"Scorer did not return JSON: {cleaned[:200]}")
        return {"score": float(score_match.group(1)), "reason": score_match.group(2).strip()}
    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        truncated = cleaned[start:]
        truncated = re.sub(r',\s*"[^"]*$', '', truncated)
        truncated = re.sub(r':\s*"[^"]*$', ': ""', truncated)
        if not truncated.rstrip().endswith("}"):
            truncated = truncated.rstrip().rstrip(",") + "}"
        return _json_loads_lenient(truncated)
    return _json_loads_lenient(cleaned[start:end + 1])


def _scored_api_call(client: httpx.Client, headers: dict, body: dict, url: str | None = None, max_retries: int = 3) -> str:
    """Make a scorer API call with retry on 429 and empty/non-JSON responses."""
    if url is None:
        url = OPENROUTER_CHAT_COMPLETIONS_URL
    # Disable thinking for local Qwen models (they output <think> blocks that eat max_tokens)
    if "localhost" in url and "chat_template_kwargs" not in body:
        body = {**body, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(max_retries):
        try:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"    [scorer] 429 rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not raw:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return raw
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"    [scorer] 429 rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    return raw


# ── Binary LLM scorer (replaces keyword matching) ──

BINARY_SCORE_SYSTEM = (
    "You judge whether a candidate answer matches a reference answer in meaning. "
    "Output ONLY a JSON object: {\"match\": 1} if they agree, {\"match\": 0} if they disagree. "
    "Ignore differences in phrasing, length, or style — only check semantic agreement."
)


def score_binary_llm(
    predictions: list[str],
    targets: list[str],
    score_model: str | None = None,
) -> list[tuple[float, str]]:
    """Score binary predictions via LLM. Returns [(score, raw_response), ...]."""
    if score_model is None:
        score_model = get_score_model()

    url, headers = _get_scorer_endpoint()

    results = []
    with httpx.Client(timeout=90.0) as client:
        for prediction, target in zip(predictions, targets):
            user_msg = f"Reference: {target}\nCandidate: {prediction}"
            body = {
                "model": score_model,
                "messages": [
                    {"role": "system", "content": BINARY_SCORE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 32,
            }
            raw = _scored_api_call(client, headers, body, url=url)
            parsed = _json_loads_lenient(raw) if "{" in raw else {}
            score = float(parsed.get("match", 0))
            results.append((score, raw))
    return results


def build_qa_score_prompt(task_name: str, qa_prompt: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Question:\n{qa_prompt}\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Candidate answer:\n{prediction}\n\n"
        "Score the candidate answer against the reference answer."
    )



# ── Trajectory scorer ──

TRAJECTORY_SCORER_TASKS = frozenset({"answer_trajectory"})
TRAJECTORY_SCORER_SYSTEM = (
    "You evaluate answer trajectory predictions. "
    "Return ONLY JSON with keys `answer_score` and `predicted_confidence`. "
    f"{llm_score_field_requirement('answer_score')}: 1.0 if the prediction identifies the same answer as the reference, 0.0 if wrong or irrelevant. "
    "`predicted_confidence` is an integer 0-100 extracted from the prediction's stated confidence percentage; use null if not mentioned."
)


def is_trajectory_scorer_task(task_name: str) -> bool:
    return task_name in TRAJECTORY_SCORER_TASKS


def build_trajectory_scorer_prompt(gt_answer_label: str, gt_confidence: int | None, prediction: str) -> str:
    conf_str = f"\nReference confidence: {gt_confidence}%" if gt_confidence is not None else ""
    return (
        f"Reference answer: {gt_answer_label}{conf_str}\n\n"
        f"Prediction:\n{prediction}\n\n"
        "Score whether the prediction identifies the correct answer and extract the stated confidence."
    )


# ── LLM scorer for cot_description / cot_metacognition / sae_unverbalized ──

LLM_SCORER_TASKS = frozenset({"cot_description", "cot_metacognition", "sae_unverbalized"})

COT_DESCRIPTION_SCORER_SYSTEM = (
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

COT_METACOGNITION_SCORER_SYSTEM = (
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

SAE_UNVERBALIZED_SCORER_SYSTEM = (
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

_LLM_SCORER_SYSTEMS = {
    "cot_description": COT_DESCRIPTION_SCORER_SYSTEM,
    "cot_metacognition": COT_METACOGNITION_SCORER_SYSTEM,
    "sae_unverbalized": SAE_UNVERBALIZED_SCORER_SYSTEM,
}


def is_llm_scorer_task(task_name: str) -> bool:
    return task_name in LLM_SCORER_TASKS


def build_llm_scorer_prompt(task_name: str, query: str, target: str, prediction: str) -> str:
    return (
        f"Task: {task_name}\n\n"
        f"Query:\n{query}\n\n"
        f"Reference answer:\n{target}\n\n"
        f"Oracle response:\n{prediction}\n\n"
        "Score the oracle response against the reference answer."
    )


def get_llm_scorer_system(task_name: str) -> str:
    return _LLM_SCORER_SYSTEMS[task_name]


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
