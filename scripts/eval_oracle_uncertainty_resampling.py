#!/usr/bin/env python3
"""Evaluate activation-oracle uncertainty via activation-position resampling on cot_description."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
AO_REF_DIR = ROOT / "ao_reference"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(AO_REF_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import httpx
import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.ao import choose_attn_implementation
from data_loading import load_task_data, prepare_context_ids
from eval_loop import _extract_base_positions, _materialize_activations, _batched_oracle_generate
from qa_judge import OPENROUTER_CHAT_COMPLETIONS_URL, QA_GEMINI_SCORE_MODEL, extract_judge_json, compute_token_f1_scores

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

VARIANCE_JUDGE_SYSTEM = (
    "You evaluate variance across multiple answers to the same question. "
    "Return ONLY JSON with keys `variance`, `reason`, `canonical_answer`.\n\n"
    "`variance` is a float in [0,1].\n"
    "Minimum allowed value is 0.01; NEVER output 0.0.\n"
    "0.01 = all answers semantically equivalent.\n"
    "0.25 = minor detail/wording differences.\n"
    "0.5 = materially different details but same general topic.\n"
    "0.75 = conflicting key facts.\n"
    "1.0 = mutually contradictory or unrelated answers.\n\n"
    "Judge semantic disagreement, not style/length differences.\n"
    "`reason` must be <= 15 words.\n"
    "`canonical_answer` should be one short line summarizing the majority answer."
)

TOKEN_HARDNESS_SYSTEM = (
    "You are given short left/right context around a single masked token. "
    "Infer whether the exact missing token is recoverable from context alone.\n"
    "Return ONLY JSON with keys `guess`, `confidence`, `reason`.\n"
    "`guess` must be one token (or UNKNOWN). `confidence` must be float in [0,1]. "
    "`reason` must be <= 12 words."
)

TOKEN_RECALL_ORACLE_QUESTION = (
    "What is the exact single token at the referenced focus position in the reasoning trace? "
    "Return exactly one token."
)

PANEL_B_PROBE_RERANK_SYSTEM = (
    "You are selecting a single answer-bearing token from a reasoning trace for a probe-position experiment. "
    "Choose the token that is maximally salient for answering the question.\n"
    "Prefer concrete names, numbers, dates, ratios, and technical terms that directly answer the question.\n"
    "Avoid generic discourse words, weak filler tokens, and tokens copied from the question unless they are clearly the answer.\n"
    "If the question asks 'who', prefer a person name over an institution or role word.\n"
    "If the question asks for a quantity/date/ratio, prefer the exact numeric token.\n"
    "Return ONLY JSON with keys `candidate_id`, `candidate_token`, and `reason`. "
    "`candidate_id` must be the 1-based candidate number. `candidate_token` must exactly copy the selected token."
)

PANEL_B_ORACLE_QUESTION_SYSTEM = (
    "You write a natural-language probe question for an activation oracle experiment. "
    "The question must ask about the same underlying information as the original question, "
    "but zoom in on one specific answer token or short span.\n"
    "Write a concise natural question whose correct answer is exactly the provided answer text. "
    "Do not mention tokens, spans, masking, reasoning traces, or positions.\n"
    "If needed, add a short answer-format instruction so the answer is exactly the provided text "
    "(for example: answer with only the number, first name, first word, or short phrase).\n"
    "Return ONLY JSON with keys `question` and `reason`."
)

SINGLE_TOKEN_CANDIDATE_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
WORD_TOKEN_CANDIDATE_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")
COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "was", "is", "are", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "at", "by", "from", "into",
    "over", "under", "within", "during", "after", "before", "between",
}
LOW_INFO_WORDS = {
    "however", "therefore", "overall", "ultimately", "subsequently", "effectively",
    "additionally", "furthermore", "moreover", "currently", "historically",
    "exactly", "directly", "primarily", "typically", "generally", "because",
    "following", "regardless",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-oracle uncertainty via resampling")
    parser.add_argument("--checkpoint", default="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n-per-class", type=int, default=40, help="Answerable + unanswerable items each")
    parser.add_argument("--diversity-fields", nargs="+", default=["domain", "tier"], help="Fields used for diverse sampling")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--resamples-per-item", type=int, default=6, help="Panel A resamples per item per temperature")
    parser.add_argument("--panel-b-resamples", type=int, default=0, help="0 = use --resamples-per-item")
    parser.add_argument("--panel-b-temperature", type=float, default=0.8)
    parser.add_argument("--panel-b-layer", type=int, default=18)
    parser.add_argument("--panel-b-num-questions", type=int, default=5)
    parser.add_argument("--panel-b-relative-window", type=int, default=40, help="Evaluate positions in [-window, +window] around informative token")
    parser.add_argument("--panel-b-position-stride", type=int, default=4, help="Stride over relative offsets when selecting panel-B activation positions")
    parser.add_argument("--panel-b-target-rel-fraction", type=float, default=0.5, help="Target relative location for informative token")
    parser.add_argument("--panel-b-target-rel-tolerance", type=float, default=0.2, help="Tolerance around target relative location")
    parser.add_argument("--panel-b-min-rel-fraction", type=float, default=0.0, help="Soft floor for informative token relative location")
    parser.add_argument("--panel-b-require-numeric", type=int, default=0, help="1=prefer/require numeric informative tokens")
    parser.add_argument("--panel-b-word-fallback", type=int, default=1, help="1=allow non-numeric fallback if not enough numeric probes")
    parser.add_argument("--panel-b-question-mode", default="token_recall", choices=["token_recall", "original", "natural"])
    parser.add_argument("--panel-b-question-model", default=QA_GEMINI_SCORE_MODEL)
    parser.add_argument("--panel-b-rerank", default="llm", choices=["none", "llm"])
    parser.add_argument("--panel-b-rerank-model", default=QA_GEMINI_SCORE_MODEL)
    parser.add_argument("--panel-b-rerank-top-k", type=int, default=8)
    parser.add_argument("--panel-b-rerank-context-chars", type=int, default=72)
    parser.add_argument("--panel-b-num-shards", type=int, default=1)
    parser.add_argument("--panel-b-shard-index", type=int, default=0)
    parser.add_argument("--panel-b-selected-probes-in", default="")
    parser.add_argument("--panel-b-stop-after-selection", type=int, default=0)
    parser.add_argument("--skip-panel-a", type=int, default=0)
    parser.add_argument("--panel-b-hard-token-filter", default="llm", choices=["none", "llm"])
    parser.add_argument("--panel-b-hard-token-model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--panel-b-hard-token-max-confidence", type=float, default=0.45)
    parser.add_argument("--panel-b-hard-token-context-tokens", type=int, default=18)
    parser.add_argument("--panel-b-cache-dir", default="", help="Cache directory for per-question panel-B results")
    parser.add_argument("--panel-b-cache-version", default="mid_align_v1")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--activation-extract-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--variance-metric", default="f1", choices=["f1", "llm"])
    parser.add_argument("--min-variance-floor", type=float, default=1e-3)
    parser.add_argument("--judge-model", default=QA_GEMINI_SCORE_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=140)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_logs/oracle_uncertainty_resampling")
    parser.add_argument("--panel-b-max-positions", type=int, default=0, help="0 = all positions")
    args = parser.parse_args()
    assert args.panel_b_num_shards > 0
    assert 0 <= args.panel_b_shard_index < args.panel_b_num_shards
    return args


def _resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    return str(path.resolve()) if path.exists() else checkpoint


def _resolve_model_device(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def _load_model_and_tokenizer(model_name: str, checkpoint: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_device = _resolve_model_device(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": model_device},
        trust_remote_code=True,
        attn_implementation=choose_attn_implementation(model_name),
    )
    model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
    model.eval()
    return model, tokenizer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _get_panel_b_cache_dir(args: argparse.Namespace) -> Path:
    if args.panel_b_cache_dir:
        cache_root = Path(args.panel_b_cache_dir)
    else:
        cache_base = os.environ["FAST_CACHE_DIR"] if "FAST_CACHE_DIR" in os.environ else os.environ["CACHE_DIR"]
        cache_root = Path(cache_base) / "oracle_uncertainty_resampling_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _get_panel_b_hard_token_cache_dir(args: argparse.Namespace) -> Path:
    cache_dir = _get_panel_b_cache_dir(args) / "hard_token_filter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_panel_b_probe_rerank_cache_dir(args: argparse.Namespace) -> Path:
    cache_dir = _get_panel_b_cache_dir(args) / "probe_reranker"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_panel_b_oracle_question_cache_dir(args: argparse.Namespace) -> Path:
    cache_dir = _get_panel_b_cache_dir(args) / "oracle_questions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _normalize_single_token(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


def _panel_b_hard_token_cache_path(args: argparse.Namespace, checkpoint: str, probe: dict, left_ctx: str, right_ctx: str) -> Path:
    payload = {
        "cache_version": args.panel_b_cache_version,
        "model": args.model,
        "checkpoint": checkpoint,
        "item_id": probe["item"]["item_id"],
        "answer_token": probe["answer_token"],
        "focus_position": probe["focus_position"],
        "panel_b_hard_token_model": args.panel_b_hard_token_model,
        "panel_b_hard_token_context_tokens": args.panel_b_hard_token_context_tokens,
        "left_ctx": left_ctx,
        "right_ctx": right_ctx,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return _get_panel_b_hard_token_cache_dir(args) / f"hard_token_{probe['item']['item_id']}_{key[:24]}.json"


def _panel_b_question_cache_path(args: argparse.Namespace, checkpoint: str, probe: dict, rel_indices: list[int]) -> Path:
    payload = {
        "cache_version": args.panel_b_cache_version,
        "model": args.model,
        "checkpoint": checkpoint,
        "variance_metric": args.variance_metric,
        "min_variance_floor": args.min_variance_floor,
        "item_id": probe["item"]["item_id"],
        "source_question": probe["item"]["prompt"],
        "oracle_question": probe["oracle_question"],
        "target_response": probe["item"]["target_response"],
        "answer_token": probe["answer_token"],
        "candidate_type": probe["candidate_type"],
        "focus_position": probe["focus_position"],
        "focus_position_rel": probe["focus_position_rel"],
        "panel_b_layer": args.panel_b_layer,
        "panel_b_temperature": args.panel_b_temperature,
        "panel_b_resamples": args.panel_b_resamples if args.panel_b_resamples > 0 else args.resamples_per_item,
        "panel_b_relative_window": args.panel_b_relative_window,
        "panel_b_position_stride": args.panel_b_position_stride,
        "panel_b_target_rel_fraction": args.panel_b_target_rel_fraction,
        "panel_b_target_rel_tolerance": args.panel_b_target_rel_tolerance,
        "panel_b_max_positions": args.panel_b_max_positions,
        "panel_b_require_numeric": args.panel_b_require_numeric,
        "panel_b_word_fallback": args.panel_b_word_fallback,
        "panel_b_question_mode": args.panel_b_question_mode,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "activation_extract_batch_size": args.activation_extract_batch_size,
        "rel_indices": rel_indices,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    fname = f"panel_b_q_{probe['item']['item_id']}_{key[:24]}.json"
    return _get_panel_b_cache_dir(args) / fname


def _panel_b_probe_rerank_cache_path(args: argparse.Namespace, checkpoint: str, item: dict, candidates: list[dict]) -> Path:
    payload = {
        "cache_version": args.panel_b_cache_version,
        "model": args.model,
        "checkpoint": checkpoint,
        "item_id": item["item_id"],
        "prompt": item["prompt"],
        "target_response": item["target_response"],
        "panel_b_rerank": args.panel_b_rerank,
        "panel_b_rerank_model": args.panel_b_rerank_model,
        "panel_b_rerank_top_k": args.panel_b_rerank_top_k,
        "panel_b_rerank_context_chars": args.panel_b_rerank_context_chars,
        "candidates": [
            {
                "answer_token": row["answer_token"],
                "candidate_type": row["candidate_type"],
                "is_named_entity": row["is_named_entity"],
                "in_question": row["in_question"],
                "focus_position": row["focus_position"],
                "focus_position_rel": row["focus_position_rel"],
                "focus_position_frac": row["focus_position_frac"],
                "priority": row["priority"],
            }
            for row in candidates
        ],
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return _get_panel_b_probe_rerank_cache_dir(args) / f"probe_rerank_{item['item_id']}_{key[:24]}.json"


def _panel_b_oracle_question_cache_path(args: argparse.Namespace, checkpoint: str, probe: dict) -> Path:
    payload = {
        "cache_version": args.panel_b_cache_version,
        "model": args.model,
        "checkpoint": checkpoint,
        "item_id": probe["item"]["item_id"],
        "panel_b_question_mode": args.panel_b_question_mode,
        "panel_b_question_model": args.panel_b_question_model,
        "source_question": probe["item"]["prompt"],
        "target_response": probe["item"]["target_response"],
        "answer_token": probe["answer_token"],
        "candidate_type": probe["candidate_type"],
        "target_context": probe["target_context"],
        "cot_context": probe["cot_context"],
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return _get_panel_b_oracle_question_cache_dir(args) / f"oracle_question_{probe['item']['item_id']}_{key[:24]}.json"


def _extract_base_positions_for_item(item: dict, n_layers: int) -> list[int]:
    base_positions = _extract_base_positions(item["context_positions"], n_layers)
    assert len(base_positions) > 0
    return base_positions


def _balanced_subset(items: list[dict], n_per_class: int, seed: int) -> list[dict]:
    answerable_items = [item for item in items if item["answerable"] is True]
    unanswerable_items = [item for item in items if item["answerable"] is False]
    assert len(answerable_items) >= n_per_class
    assert len(unanswerable_items) >= n_per_class
    rng = random.Random(seed)
    rng.shuffle(answerable_items)
    rng.shuffle(unanswerable_items)
    subset = answerable_items[:n_per_class] + unanswerable_items[:n_per_class]
    rng.shuffle(subset)
    return subset


def _diversity_key(item: dict, fields: list[str]) -> tuple[str, ...]:
    return tuple(str(item.get(field, "<missing>")) for field in fields)


def _select_diverse_class_subset(
    items: list[dict],
    n: int,
    seed: int,
    diversity_fields: list[str],
) -> list[dict]:
    assert len(items) >= n
    rng = random.Random(seed)
    if not diversity_fields:
        pool = list(items)
        rng.shuffle(pool)
        return pool[:n]

    groups: dict[tuple[str, ...], list[dict]] = {}
    for item in items:
        key = _diversity_key(item, diversity_fields)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(groups[key])

    selected: list[dict] = []
    exhausted = False
    while len(selected) < n and not exhausted:
        exhausted = True
        for key in group_keys:
            bucket = groups[key]
            if len(bucket) == 0:
                continue
            selected.append(bucket.pop())
            exhausted = False
            if len(selected) >= n:
                break

    if len(selected) < n:
        remainder = []
        for key in group_keys:
            remainder.extend(groups[key])
        rng.shuffle(remainder)
        selected.extend(remainder[: n - len(selected)])
    assert len(selected) == n
    return selected


def _balanced_diverse_subset(items: list[dict], n_per_class: int, seed: int, diversity_fields: list[str]) -> list[dict]:
    answerable_items = [item for item in items if item["answerable"] is True]
    unanswerable_items = [item for item in items if item["answerable"] is False]
    assert len(answerable_items) >= n_per_class
    assert len(unanswerable_items) >= n_per_class

    answerable_sel = _select_diverse_class_subset(
        items=answerable_items,
        n=n_per_class,
        seed=seed + 11,
        diversity_fields=diversity_fields,
    )
    unanswerable_sel = _select_diverse_class_subset(
        items=unanswerable_items,
        n=n_per_class,
        seed=seed + 29,
        diversity_fields=diversity_fields,
    )
    subset = answerable_sel + unanswerable_sel
    rng = random.Random(seed + 47)
    rng.shuffle(subset)
    return subset


def _extract_activations_batched(
    model,
    tokenizer,
    items: list[dict],
    layers: list[int],
    activation_extract_batch_size: int,
    device: str,
) -> list[torch.Tensor]:
    activations = []
    for i in tqdm(range(0, len(items), activation_extract_batch_size), desc="extract acts", leave=False):
        batch = items[i : i + activation_extract_batch_size]
        batch_acts = _materialize_activations(model, tokenizer, batch, layers=layers, device=device)
        activations.extend(batch_acts)
    return activations


def _oracle_generate(
    model,
    tokenizer,
    activations: list[torch.Tensor],
    prompts: list[str],
    layers: list[int],
    args: argparse.Namespace,
    generation_temperature: float,
) -> list[str]:
    items = [(a, p) for a, p in zip(activations, prompts)]
    predictions = _batched_oracle_generate(
        model=model,
        tokenizer=tokenizer,
        items=items,
        layers=layers,
        device=args.device,
        injection_layer=1,
        max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.eval_batch_size,
        generation_temperature=generation_temperature,
        oracle_adapter_name="default",
    )
    return predictions


def _openrouter_json_response(
    client: httpx.Client,
    body: dict,
    required_keys: list[str],
    label: str,
) -> tuple[dict, str]:
    last_raw = ""
    for attempt_idx in range(4):
        response = client.post(OPENROUTER_CHAT_COMPLETIONS_URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        last_raw = raw
        try:
            parsed = extract_judge_json(raw)
        except Exception as exc:
            print(f"[{label}] parse retry attempt={attempt_idx + 1} err={type(exc).__name__} raw={raw[:220]!r}")
            continue
        missing = [key for key in required_keys if key not in parsed]
        if len(missing) > 0:
            print(f"[{label}] missing keys retry attempt={attempt_idx + 1} missing={missing} raw={raw[:220]!r}")
            continue
        return parsed, raw
    raise AssertionError(f"[{label}] failed after retries; last_raw={last_raw!r}")


def _judge_variance_llm(
    client: httpx.Client,
    question: str,
    answers: list[str],
    judge_model: str,
    judge_max_tokens: int,
    min_variance_floor: float,
) -> dict:
    answers_blob = "\n".join([f"{i + 1}. {answer}" for i, answer in enumerate(answers)])
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answers from repeated activation resamples:\n{answers_blob}\n\n"
        "Score semantic disagreement across answers."
    )
    body = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": VARIANCE_JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": judge_max_tokens,
    }
    parsed, raw = _openrouter_json_response(
        client=client,
        body=body,
        required_keys=["variance", "reason", "canonical_answer"],
        label="variance_judge",
    )
    variance = max(min_variance_floor, float(parsed["variance"]))
    assert 0.0 <= variance <= 1.0
    return {
        "variance": variance,
        "reason": str(parsed["reason"]).strip(),
        "canonical_answer": str(parsed["canonical_answer"]).strip(),
        "raw": raw,
    }


def _judge_variance_f1(
    answers: list[str],
    min_variance_floor: float,
) -> dict:
    n = len(answers)
    assert n > 0

    if n == 1:
        mean_pairwise_f1 = 1.0
    else:
        pairwise = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise.append(compute_token_f1_scores([answers[i]], [answers[j]])[0])
        mean_pairwise_f1 = float(np.mean(pairwise))

    per_answer_mean_f1 = []
    for i in range(n):
        if n == 1:
            per_answer_mean_f1.append(1.0)
            continue
        scores = []
        for j in range(n):
            if i == j:
                continue
            scores.append(compute_token_f1_scores([answers[i]], [answers[j]])[0])
        per_answer_mean_f1.append(float(np.mean(scores)))
    canonical_idx = int(np.argmax(np.array(per_answer_mean_f1)))
    variance = max(min_variance_floor, 1.0 - mean_pairwise_f1)
    return {
        "variance": variance,
        "reason": f"f1_disagreement=1-{mean_pairwise_f1:.3f}",
        "canonical_answer": answers[canonical_idx],
        "raw": json.dumps({"metric": "f1", "mean_pairwise_f1": mean_pairwise_f1}),
    }


def _judge_variance(
    question: str,
    answers: list[str],
    args: argparse.Namespace,
    client: httpx.Client | None,
) -> dict:
    if args.variance_metric == "f1":
        return _judge_variance_f1(
            answers=answers,
            min_variance_floor=args.min_variance_floor,
        )
    assert client is not None
    return _judge_variance_llm(
        client=client,
        question=question,
        answers=answers,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        min_variance_floor=args.min_variance_floor,
    )


def _build_masked_token_context(item: dict, token_position: int, tokenizer, context_tokens: int) -> tuple[str, str]:
    ids = item["context_input_ids"]
    left_ids = ids[max(0, token_position - context_tokens):token_position]
    right_ids = ids[token_position + 1:min(len(ids), token_position + 1 + context_tokens)]
    left_ctx = tokenizer.decode(left_ids, skip_special_tokens=False).replace("\n", " ").strip()
    right_ctx = tokenizer.decode(right_ids, skip_special_tokens=False).replace("\n", " ").strip()
    return left_ctx, right_ctx


def _judge_token_hardness_llm(
    client: httpx.Client,
    left_ctx: str,
    right_ctx: str,
    target_token: str,
    model_name: str,
) -> dict:
    user_prompt = (
        "Masked context:\n"
        f"LEFT: {left_ctx}\n"
        "MASK: [[MASK]]\n"
        f"RIGHT: {right_ctx}\n\n"
        "Infer the exact missing token."
    )
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": TOKEN_HARDNESS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    parsed, raw = _openrouter_json_response(
        client=client,
        body=body,
        required_keys=["guess", "confidence", "reason"],
        label="hard_token_judge",
    )
    guess = str(parsed["guess"]).strip()
    confidence = float(parsed["confidence"])
    assert 0.0 <= confidence <= 1.0
    target_norm = _normalize_single_token(target_token)
    guess_norm = _normalize_single_token(guess)
    is_correct = target_norm == guess_norm
    return {
        "guess": guess,
        "confidence": confidence,
        "is_correct": is_correct,
        "reason": str(parsed["reason"]).strip(),
        "raw": raw,
    }


def _attach_panel_b_oracle_questions(
    probes: list[dict],
    mode: str,
    args: argparse.Namespace,
    checkpoint: str,
    run_dir: Path,
) -> list[dict]:
    if mode == "natural":
        return _attach_panel_b_natural_oracle_questions(probes=probes, args=args, checkpoint=checkpoint, run_dir=run_dir)
    out = []
    for probe in probes:
        probe_row = dict(probe)
        if mode == "token_recall":
            probe_row["oracle_question"] = TOKEN_RECALL_ORACLE_QUESTION
        else:
            probe_row["oracle_question"] = probe["item"]["prompt"]
        out.append(probe_row)
    return out


def _apply_panel_b_hard_token_filter(
    probes: list[dict],
    tokenizer,
    args: argparse.Namespace,
    checkpoint: str,
    run_dir: Path,
) -> list[dict]:
    if args.panel_b_hard_token_filter == "none":
        return probes

    assert args.panel_b_hard_token_filter == "llm"
    kept = []
    logs = []
    with httpx.Client(timeout=90.0) as client:
        for probe in tqdm(probes, desc="panel B: llm hard-token filter", leave=False):
            left_ctx, right_ctx = _build_masked_token_context(
                item=probe["item"],
                token_position=probe["focus_position"],
                tokenizer=tokenizer,
                context_tokens=args.panel_b_hard_token_context_tokens,
            )
            cache_path = _panel_b_hard_token_cache_path(args=args, checkpoint=checkpoint, probe=probe, left_ctx=left_ctx, right_ctx=right_ctx)
            if cache_path.exists():
                with open(cache_path) as f:
                    judged = json.load(f)
                print(f"[panel B hard] cache hit item={probe['item']['item_id']} path={cache_path.name}")
            else:
                judged = _judge_token_hardness_llm(
                    client=client,
                    left_ctx=left_ctx,
                    right_ctx=right_ctx,
                    target_token=probe["answer_token"],
                    model_name=args.panel_b_hard_token_model,
                )
                with open(cache_path, "w") as f:
                    json.dump(judged, f)
                print(f"[panel B hard] cache save item={probe['item']['item_id']} path={cache_path.name}")

            hardness_score = float(judged["is_correct"] is False) + (1.0 - judged["confidence"])
            hard = (judged["is_correct"] is False) or (judged["confidence"] <= args.panel_b_hard_token_max_confidence)
            log_row = {
                "item_id": probe["item"]["item_id"],
                "domain": probe["item"]["domain"],
                "tier": probe["item"]["tier"],
                "answer_token": probe["answer_token"],
                "focus_position_abs": probe["focus_position"],
                "focus_position_rel": probe["focus_position_rel"],
                "focus_position_frac": probe["focus_position_frac"],
                "left_ctx": left_ctx,
                "right_ctx": right_ctx,
                "guess": judged["guess"],
                "confidence": judged["confidence"],
                "is_correct": judged["is_correct"],
                "hardness_score": hardness_score,
                "is_hard": hard,
                "reason": judged["reason"],
                "judge_raw": judged["raw"],
            }
            logs.append(log_row)
            if hard:
                probe_row = dict(probe)
                probe_row["hardness_score"] = hardness_score
                probe_row["hard_guess"] = judged["guess"]
                probe_row["hard_confidence"] = judged["confidence"]
                probe_row["hard_is_correct"] = judged["is_correct"]
                kept.append(probe_row)

    _write_jsonl(run_dir / "panel_b_hard_token_filter.jsonl", logs)
    assert len(kept) > 0
    return kept


def _get_user_message(item: dict) -> str:
    if "hinted_prompt" in item and item["hinted_prompt"]:
        return item["hinted_prompt"]
    if "question" in item:
        return item["question"]
    return ""


def _prepare_item_token_alignment(item: dict, tokenizer) -> dict:
    user_msg = _get_user_message(item)
    prompt_msgs = [{"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    full_msgs = prompt_msgs + [{"role": "assistant", "content": item["cot_text"]}]
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    input_ids = encoded["input_ids"]
    cot_start = full_text.index(item["cot_text"])
    assert input_ids == item["context_input_ids"]
    base_positions = list(item["_base_positions"])
    return {
        "prompt_len": prompt_len,
        "offsets": offsets,
        "input_ids": input_ids,
        "cot_start": cot_start,
        "base_positions": base_positions,
        "base_position_set": set(base_positions),
    }


def _token_in_text(token: str, text_lower: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", text_lower) is not None


def _candidate_is_named_entity(candidate: str) -> bool:
    letters = [ch for ch in candidate if ch.isalpha()]
    return len(letters) > 0 and any(ch.isupper() for ch in letters)


def _question_wants_numeric(question_lower: str) -> bool:
    measure_terms = [
        "percentage", "percent", "ratio", "quantity", "amount", "weight", "mass",
        "capacity", "concentration", "date", "year", "time", "duration", "period",
        "temperature", "pressure", "current", "voltage", "distance", "speed",
        "count", "number", "dose", "dosage", "energy density", "peak concentration",
        "mean reduction", "participants", "planets", "sensor nodes", "tons",
        "kilograms", "grams", "minutes", "hours", "days", "weeks", "months",
    ]
    numeric_cues = ["how many", "how much", "what percentage", "what percent", "what ratio", "what date", "on what date", "which year", "what year", "how long"]
    return any(cue in question_lower for cue in numeric_cues) or any(term in question_lower for term in measure_terms)


def _question_wants_entity(question_lower: str) -> bool:
    entity_patterns = [
        r"^who\b",
        r"\bwhich (?:two )?(?:scientists?|experts?|explorers?|commanders?|leaders?|engineers?|researchers?|professors?|specialists?|ceos?)\b",
    ]
    return any(re.search(pattern, question_lower) is not None for pattern in entity_patterns)


def _question_keywords(question_lower: str) -> set[str]:
    keywords = set()
    for token in re.findall(r"[a-z0-9-]+", question_lower):
        if token in COMMON_STOPWORDS:
            continue
        if token in LOW_INFO_WORDS:
            continue
        if len(token) < 4:
            continue
        keywords.add(token)
    return keywords


def _candidate_question_overlap(item: dict, candidate: str) -> int:
    question_keywords = _question_keywords(item["prompt"].lower())
    if len(question_keywords) == 0:
        return 0
    target_lower = item["target_response"].lower()
    span = _find_unique_candidate_span(target_lower, candidate.lower())
    if span is None:
        match = re.search(re.escape(candidate.lower()), target_lower)
        if match is None:
            return 0
        span = (match.start(), match.end())
    start, end = span
    context = target_lower[max(0, start - 48):min(len(target_lower), end + 48)]
    return sum(1 for kw in question_keywords if kw in context)


def _candidate_priority(
    item: dict,
    question_lower: str,
    candidate: str,
    is_numeric: bool,
    in_question: bool,
    focus_position_frac: float,
    target_rel_fraction: float,
    target_rel_tolerance: float,
    min_rel_fraction: float,
) -> tuple:
    wants_numeric = _question_wants_numeric(question_lower)
    wants_entity = _question_wants_entity(question_lower)
    wants_content = question_lower.startswith("why")
    is_named_entity = _candidate_is_named_entity(candidate)
    distance = abs(focus_position_frac - target_rel_fraction)
    question_overlap = _candidate_question_overlap(item, candidate)
    numeric_specificity = len(re.sub(r"[^0-9]", "", candidate)) if is_numeric else 0

    question_match = 0
    if wants_numeric and is_numeric:
        question_match += 2
    if wants_entity and is_named_entity:
        question_match += 2
    if wants_content and (not is_numeric) and (not is_named_entity):
        question_match += 1

    candidate_specificity = 2 if is_numeric else 1 if is_named_entity else 0
    return (
        question_match,
        int(not in_question),
        question_overlap,
        numeric_specificity,
        candidate_specificity,
        int(focus_position_frac >= min_rel_fraction),
        int(distance <= target_rel_tolerance),
        -distance,
        len(candidate),
    )


def _extract_candidate_context(text: str, candidate: str, window_chars: int) -> str:
    span = _find_unique_candidate_span(text, candidate)
    if span is None:
        match = re.search(re.escape(candidate), text, flags=re.IGNORECASE)
        if match is None:
            return ""
        span = (match.start(), match.end())
    start, end = span
    left = text[max(0, start - window_chars):start]
    mid = text[start:end]
    right = text[end:min(len(text), end + window_chars)]
    return f"{left}[{mid}]{right}".replace("\n", " ").strip()


def _judge_panel_b_probe_candidate_llm(client: httpx.Client, item: dict, candidates: list[dict], model_name: str) -> dict:
    candidate_lines = []
    for idx, row in enumerate(candidates):
        candidate_lines.append(
            f"{idx + 1}. token={row['answer_token']!r} type={row['candidate_type']} named={row['is_named_entity']} "
            f"in_question={row['in_question']} rel_frac={row['focus_position_frac']:.3f}\n"
            f"   target_context: {row['target_context']}\n"
            f"   cot_context: {row['cot_context']}"
        )
    user_prompt = (
        f"Question:\n{item['prompt']}\n\n"
        f"Ground-truth answer description:\n{item['target_response']}\n\n"
        "Choose exactly one candidate token for probing. Pick the token that best represents the answer detail the oracle should lock onto.\n\n"
        "Candidates:\n"
        f"{chr(10).join(candidate_lines)}"
    )
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": PANEL_B_PROBE_RERANK_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    parsed, raw = _openrouter_json_response(client=client, body=body, required_keys=["candidate_id", "candidate_token", "reason"], label="panel_b_probe_rerank")
    return {
        "candidate_id": int(parsed["candidate_id"]),
        "candidate_token": str(parsed["candidate_token"]).strip(),
        "reason": str(parsed["reason"]).strip(),
        "raw": raw,
    }


def _judge_panel_b_oracle_question_llm(client: httpx.Client, probe: dict, model_name: str) -> dict:
    user_prompt = (
        f"Original question:\n{probe['item']['prompt']}\n\n"
        f"Ground-truth answer description:\n{probe['item']['target_response']}\n\n"
        f"Selected answer text:\n{probe['answer_token']}\n\n"
        f"Candidate type:\n{probe['candidate_type']}\n\n"
        f"Answer context from target response:\n{probe['target_context']}\n\n"
        f"Answer context from reasoning trace:\n{probe['cot_context']}\n\n"
        "Write one natural question that asks for exactly this answer text."
    )
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": PANEL_B_ORACLE_QUESTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }
    parsed, raw = _openrouter_json_response(client=client, body=body, required_keys=["question", "reason"], label="panel_b_oracle_question")
    question = str(parsed["question"]).strip()
    assert len(question) > 0
    return {
        "question": question,
        "reason": str(parsed["reason"]).strip(),
        "raw": raw,
    }


def _attach_panel_b_natural_oracle_questions(
    probes: list[dict],
    args: argparse.Namespace,
    checkpoint: str,
    run_dir: Path,
) -> list[dict]:
    out = []
    logs = []
    with httpx.Client(timeout=90.0) as client:
        for probe in tqdm(probes, desc="panel B: natural oracle questions", leave=False):
            cache_path = _panel_b_oracle_question_cache_path(args=args, checkpoint=checkpoint, probe=probe)
            if cache_path.exists():
                with open(cache_path) as f:
                    judged = json.load(f)
                print(f"[panel B question] cache hit item={probe['item']['item_id']} path={cache_path.name}")
            else:
                judged = _judge_panel_b_oracle_question_llm(client=client, probe=probe, model_name=args.panel_b_question_model)
                with open(cache_path, "w") as f:
                    json.dump(judged, f)
                print(f"[panel B question] cache save item={probe['item']['item_id']} path={cache_path.name}")
            probe_row = dict(probe)
            probe_row["oracle_question"] = judged["question"]
            probe_row["oracle_question_reason"] = judged["reason"]
            out.append(probe_row)
            logs.append(
                {
                    "item_id": probe["item"]["item_id"],
                    "source_question": probe["item"]["prompt"],
                    "answer_token": probe["answer_token"],
                    "candidate_type": probe["candidate_type"],
                    "target_context": probe["target_context"],
                    "cot_context": probe["cot_context"],
                    "oracle_question": judged["question"],
                    "reason": judged["reason"],
                    "raw": judged["raw"],
                }
            )
    _write_jsonl(run_dir / "panel_b_oracle_questions.jsonl", logs)
    return out


def _rerank_panel_b_probe_candidates_llm(
    item: dict,
    candidates: list[dict],
    args: argparse.Namespace,
    checkpoint: str,
    client: httpx.Client,
) -> tuple[dict, dict]:
    rerank_candidates = candidates[:args.panel_b_rerank_top_k]
    cache_path = _panel_b_probe_rerank_cache_path(args=args, checkpoint=checkpoint, item=item, candidates=rerank_candidates)
    if cache_path.exists():
        with open(cache_path) as f:
            judged = json.load(f)
        print(f"[panel B rerank] cache hit item={item['item_id']} path={cache_path.name}")
    else:
        judged = _judge_panel_b_probe_candidate_llm(client=client, item=item, candidates=rerank_candidates, model_name=args.panel_b_rerank_model)
        with open(cache_path, "w") as f:
            json.dump(judged, f)
        print(f"[panel B rerank] cache save item={item['item_id']} path={cache_path.name}")
    candidate_id_raw = int(judged["candidate_id_raw"]) if "candidate_id_raw" in judged else int(judged["candidate_id"])
    candidate_token_raw = judged["candidate_token"] if "candidate_token" in judged else ""
    selected_source = "topk_id"
    if 1 <= candidate_id_raw <= len(rerank_candidates):
        candidate_id = candidate_id_raw - 1
        selected = dict(rerank_candidates[candidate_id])
    elif 0 <= candidate_id_raw < len(rerank_candidates):
        candidate_id = candidate_id_raw
        selected = dict(rerank_candidates[candidate_id])
    else:
        matches = [idx for idx, row in enumerate(candidates) if row["answer_token"] == candidate_token_raw]
        if len(matches) != 1:
            normalized = _normalize_single_token(candidate_token_raw)
            matches = [idx for idx, row in enumerate(candidates) if _normalize_single_token(row["answer_token"]) == normalized]
        if len(matches) != 1:
            raise AssertionError(f"cached candidate_id={candidate_id_raw} invalid for n_topk={len(rerank_candidates)} token={candidate_token_raw!r}")
        candidate_id = matches[0]
        selected = dict(candidates[candidate_id])
        selected_source = "full_token_match"
    selected["rerank_reason"] = judged["reason"]
    selected["rerank_raw"] = judged["raw"]
    log_row = {
        "item_id": item["item_id"],
        "question": item["prompt"],
        "target_response": item["target_response"],
        "model": args.panel_b_rerank_model,
        "selected_candidate_id": candidate_id,
        "selected_candidate_id_raw": candidate_id_raw,
        "selected_candidate_token_raw": candidate_token_raw,
        "selected_candidate_source": selected_source,
        "selected_answer_token": selected["answer_token"],
        "selected_candidate_type": selected["candidate_type"],
        "selected_focus_position_rel": selected["focus_position_rel"],
        "selected_focus_position_frac": selected["focus_position_frac"],
        "reason": judged["reason"],
        "raw": judged["raw"],
        "candidates": [
            {
                "candidate_id": idx,
                "answer_token": row["answer_token"],
                "candidate_type": row["candidate_type"],
                "is_named_entity": row["is_named_entity"],
                "in_question": row["in_question"],
                "focus_position_rel": row["focus_position_rel"],
                "focus_position_frac": row["focus_position_frac"],
                "priority": row["priority"],
                "target_context": row["target_context"],
                "cot_context": row["cot_context"],
            }
            for idx, row in enumerate(rerank_candidates)
        ],
    }
    return selected, log_row


def _find_unique_candidate_span(cot_text: str, candidate: str) -> tuple[int, int] | None:
    exact_count = cot_text.count(candidate)
    if exact_count == 1:
        rel_start = cot_text.index(candidate)
        return rel_start, rel_start + len(candidate)
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", flags=re.IGNORECASE)
    matches = list(pattern.finditer(cot_text))
    if len(matches) != 1:
        return None
    return matches[0].start(), matches[0].end()


def _locate_candidate_position(item: dict, tokenizer, alignment: dict, candidate: str, allow_multi_token: bool = False, choose_last_token: bool = False) -> dict | None:
    span = _find_unique_candidate_span(item["cot_text"], candidate)
    if span is None:
        return None
    rel_start, rel_end = span
    abs_start = alignment["cot_start"] + rel_start
    abs_end = alignment["cot_start"] + rel_end
    token_hits = [i for i, (s, e) in enumerate(alignment["offsets"]) if s < abs_end and e > abs_start]
    if len(token_hits) == 0:
        return None
    if allow_multi_token is False and len(token_hits) != 1:
        return None
    token_position = token_hits[-1] if choose_last_token else token_hits[0]
    if token_position < alignment["prompt_len"]:
        return None
    if token_position not in alignment["base_position_set"]:
        return None
    focus_rel = alignment["base_positions"].index(token_position)
    denom = max(1, len(alignment["base_positions"]) - 1)
    return {
        "focus_position": token_position,
        "focus_position_rel": focus_rel,
        "focus_position_frac": focus_rel / denom,
        "focus_token_decoded": tokenizer.decode([alignment["input_ids"][token_position]], skip_special_tokens=False),
    }


def _build_informative_candidates(item: dict) -> list[tuple[str, bool, bool]]:
    question_lower = item["prompt"].lower()
    candidates: list[tuple[str, bool, bool]] = []
    seen = set()

    for match in SINGLE_TOKEN_CANDIDATE_RE.findall(item["target_response"]):
        candidate = match
        normalized = candidate.replace(",", "")
        if candidate not in seen:
            seen.add(candidate)
            in_question = _token_in_text(candidate, question_lower) or _token_in_text(normalized, question_lower)
            candidates.append((candidate, True, in_question))
        if normalized != candidate and normalized not in seen:
            seen.add(normalized)
            in_question = _token_in_text(normalized, question_lower)
            candidates.append((normalized, True, in_question))

    for match in WORD_TOKEN_CANDIDATE_RE.findall(item["target_response"]):
        candidate = match
        lowered = candidate.lower()
        is_named_entity = _candidate_is_named_entity(candidate)
        if lowered in COMMON_STOPWORDS:
            continue
        if lowered in LOW_INFO_WORDS:
            continue
        if lowered.endswith("ly") and is_named_entity is False:
            continue
        if len(lowered) < 6 and is_named_entity is False:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        in_question = _token_in_text(lowered, question_lower)
        candidates.append((candidate, False, in_question))

    return candidates


def _collect_informative_token_candidates_for_item(
    item: dict,
    tokenizer,
    target_rel_fraction: float,
    target_rel_tolerance: float,
    min_rel_fraction: float,
    context_chars: int,
) -> list[dict]:
    alignment = _prepare_item_token_alignment(item, tokenizer)
    candidates = _build_informative_candidates(item)
    question_lower = item["prompt"].lower()
    rows = []
    for candidate, is_numeric, in_question in candidates:
        located = _locate_candidate_position(item, tokenizer, alignment, candidate, allow_multi_token=is_numeric, choose_last_token=is_numeric)
        if located is None:
            continue
        priority = _candidate_priority(
            item=item,
            question_lower=question_lower,
            candidate=candidate,
            is_numeric=is_numeric,
            in_question=in_question,
            focus_position_frac=located["focus_position_frac"],
            target_rel_fraction=target_rel_fraction,
            target_rel_tolerance=target_rel_tolerance,
            min_rel_fraction=min_rel_fraction,
        )
        rows.append(
            {
                "item": item,
                "answer_token": candidate,
                "candidate_type": "numeric" if is_numeric else "word",
                "is_named_entity": _candidate_is_named_entity(candidate),
                "in_question": in_question,
                "focus_position": located["focus_position"],
                "focus_position_rel": located["focus_position_rel"],
                "focus_position_frac": located["focus_position_frac"],
                "focus_token_decoded": located["focus_token_decoded"],
                "priority": priority,
                "target_context": _extract_candidate_context(item["target_response"], candidate, window_chars=context_chars),
                "cot_context": _extract_candidate_context(item["cot_text"], candidate, window_chars=context_chars),
            }
        )
    return sorted(rows, key=lambda row: row["priority"], reverse=True)


def _pick_informative_token_for_item(
    item: dict,
    tokenizer,
    target_rel_fraction: float,
    target_rel_tolerance: float,
    min_rel_fraction: float,
    args: argparse.Namespace | None = None,
    checkpoint: str | None = None,
    client: httpx.Client | None = None,
    rerank_logs: list[dict] | None = None,
) -> dict | None:
    candidates = _collect_informative_token_candidates_for_item(
        item=item,
        tokenizer=tokenizer,
        target_rel_fraction=target_rel_fraction,
        target_rel_tolerance=target_rel_tolerance,
        min_rel_fraction=min_rel_fraction,
        context_chars=args.panel_b_rerank_context_chars if args is not None else 64,
    )
    if len(candidates) == 0:
        return None
    if args is None or args.panel_b_rerank == "none":
        return candidates[0]
    assert checkpoint is not None
    assert client is not None
    assert rerank_logs is not None
    selected, log_row = _rerank_panel_b_probe_candidates_llm(item=item, candidates=candidates, args=args, checkpoint=checkpoint, client=client)
    rerank_logs.append(log_row)
    return selected


def _probe_has_required_window(probe: dict, window: int) -> bool:
    n_positions = len(probe["item"]["_base_positions"])
    rel = probe["focus_position_rel"]
    return (rel - window) >= 0 and (rel + window) < n_positions


def _load_selected_probes(path: str | Path, item_pool: list[dict]) -> list[dict]:
    item_by_id = {item["item_id"]: item for item in item_pool}
    rows = [json.loads(line) for line in open(path)]
    probes = []
    for row in rows:
        item = item_by_id[row["item_id"]]
        probe = {
            "question_index": int(row["question_index"]) if row["question_index"] is not None else len(probes),
            "item": item,
            "oracle_question": row["oracle_question"],
            "answer_token": row["answer_token"],
            "candidate_type": row["candidate_type"],
            "is_named_entity": row["is_named_entity"],
            "in_question": row["in_question"],
            "focus_position": row["focus_position_abs"],
            "focus_position_rel": row["focus_position_rel"],
            "focus_position_frac": row["focus_position_frac"],
            "focus_token_decoded": row["focus_token_decoded"],
            "priority": row["priority"],
            "target_context": row["target_context"],
            "cot_context": row["cot_context"],
        }
        if "oracle_question_reason" in row and row["oracle_question_reason"] is not None:
            probe["oracle_question_reason"] = row["oracle_question_reason"]
        if row["rerank_reason"] is not None:
            probe["rerank_reason"] = row["rerank_reason"]
        if row["hardness_score"] is not None:
            probe["hardness_score"] = row["hardness_score"]
        if row["hard_guess"] is not None:
            probe["hard_guess"] = row["hard_guess"]
        if row["hard_confidence"] is not None:
            probe["hard_confidence"] = row["hard_confidence"]
        if row["hard_is_correct"] is not None:
            probe["hard_is_correct"] = row["hard_is_correct"]
        probes.append(probe)
    return sorted(probes, key=lambda probe: probe["question_index"])


def _select_diverse_probes(probes: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(probes)
    rng.shuffle(shuffled)
    ranked = sorted(
        shuffled,
        key=lambda row: (
            row["hardness_score"] if "hardness_score" in row else -1.0,
            row["priority"],
        ),
        reverse=True,
    )
    selected = []
    selected_ids = set()
    used_domains = set()
    for probe in ranked:
        domain = probe["item"]["domain"]
        item_id = probe["item"]["item_id"]
        if domain in used_domains:
            continue
        selected.append(probe)
        selected_ids.add(item_id)
        used_domains.add(domain)
        if len(selected) >= n:
            return selected
    for probe in ranked:
        item_id = probe["item"]["item_id"]
        if item_id in selected_ids:
            continue
        selected.append(probe)
        selected_ids.add(item_id)
        if len(selected) >= n:
            return selected
    return selected


def _pick_informative_token_probes(
    item_pool: list[dict],
    tokenizer,
    num_questions: int,
    relative_window: int,
    target_rel_fraction: float,
    target_rel_tolerance: float,
    min_rel_fraction: float,
    require_numeric: bool,
    word_fallback: bool,
    seed: int,
    args: argparse.Namespace,
    checkpoint: str,
    run_dir: Path,
) -> list[dict]:
    answerable_items = [item for item in item_pool if item["answerable"] is True]
    ranked_probes = []
    rerank_logs = []
    with (httpx.Client(timeout=90.0) if args.panel_b_rerank == "llm" else contextlib.nullcontext()) as client:
        for item in tqdm(answerable_items, desc="panel B: scan informative tokens", leave=False):
            probe = _pick_informative_token_for_item(
                item=item,
                tokenizer=tokenizer,
                target_rel_fraction=target_rel_fraction,
                target_rel_tolerance=target_rel_tolerance,
                min_rel_fraction=min_rel_fraction,
                args=args,
                checkpoint=checkpoint,
                client=client,
                rerank_logs=rerank_logs,
            )
            if probe is not None:
                if not _probe_has_required_window(probe, window=relative_window):
                    continue
                ranked_probes.append(probe)
    if len(rerank_logs) > 0:
        _write_jsonl(run_dir / "panel_b_probe_reranker.jsonl", rerank_logs)
    numeric_probes = [probe for probe in ranked_probes if probe["candidate_type"] == "numeric"]
    word_probes = [probe for probe in ranked_probes if probe["candidate_type"] != "numeric"]
    selected = []

    if require_numeric is True:
        selected = _select_diverse_probes(numeric_probes, n=num_questions, seed=seed + 701)
        if len(selected) < num_questions and word_fallback is True:
            chosen_ids = set(p["item"]["item_id"] for p in selected)
            extra_pool = [p for p in word_probes if p["item"]["item_id"] not in chosen_ids]
            extra = _select_diverse_probes(extra_pool, n=num_questions - len(selected), seed=seed + 907)
            selected = selected + extra
    else:
        selected = _select_diverse_probes(ranked_probes, n=num_questions, seed=seed + 701)

    return selected[:num_questions]


def _panel_a(
    model,
    tokenizer,
    items: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    n_layers = len(args.layers)
    prediction_rows: list[dict] = []
    judge_rows: list[dict] = []
    summary_rows: list[dict] = []

    full_activation_items = [
        {
            "context_input_ids": item["context_input_ids"],
            "context_positions": item["_base_positions"] * n_layers,
        }
        for item in items
    ]
    full_activations = _extract_activations_batched(
        model=model,
        tokenizer=tokenizer,
        items=full_activation_items,
        layers=args.layers,
        activation_extract_batch_size=args.activation_extract_batch_size,
        device=args.device,
    )

    with (httpx.Client(timeout=90.0) if args.variance_metric == "llm" else contextlib.nullcontext()) as client:
        for temperature in tqdm(args.temperatures, desc="panel A: temperature sweep"):
            rollout_activations = []
            rollout_prompts = []
            meta_rows = []
            for item_idx, item in enumerate(items):
                for resample_idx in range(args.resamples_per_item):
                    rollout_activations.append(full_activations[item_idx])
                    rollout_prompts.append(item["prompt"])
                    meta_rows.append({
                        "item_idx": item_idx,
                        "temperature": temperature,
                        "resample_idx": resample_idx,
                    })

            predictions = _oracle_generate(
                model=model,
                tokenizer=tokenizer,
                activations=rollout_activations,
                prompts=rollout_prompts,
                layers=args.layers,
                args=args,
                generation_temperature=temperature,
            )
            assert len(predictions) == len(meta_rows)

            by_item_answers: dict[int, list[str]] = {}
            for meta, prediction in zip(meta_rows, predictions):
                item = items[meta["item_idx"]]
                if meta["item_idx"] not in by_item_answers:
                    by_item_answers[meta["item_idx"]] = []
                by_item_answers[meta["item_idx"]].append(prediction)
                prediction_rows.append({
                    "panel": "A",
                    "item_id": item["item_id"],
                    "answerable": item["answerable"],
                    "temperature": temperature,
                    "resample_idx": meta["resample_idx"],
                    "prediction": prediction,
                })

            t_judge_rows = []
            for item_idx, answers in tqdm(by_item_answers.items(), desc=f"judge T={temperature}", leave=False):
                item = items[item_idx]
                judged = _judge_variance(
                    question=item["prompt"],
                    answers=answers,
                    args=args,
                    client=client,
                )
                row = {
                    "panel": "A",
                    "item_id": item["item_id"],
                    "answerable": item["answerable"],
                    "temperature": temperature,
                    "question": item["prompt"],
                    "target_response": item["target_response"],
                    "n_answers": len(answers),
                    "variance": judged["variance"],
                    "reason": judged["reason"],
                    "canonical_answer": judged["canonical_answer"],
                    "judge_raw": judged["raw"],
                    "answers": answers,
                }
                t_judge_rows.append(row)
                judge_rows.append(row)

            answerable_variances = [row["variance"] for row in t_judge_rows if row["answerable"] is True]
            unanswerable_variances = [row["variance"] for row in t_judge_rows if row["answerable"] is False]
            summary_rows.append({
                "temperature": temperature,
                "answerable_mean_variance": float(np.mean(answerable_variances)),
                "answerable_std_variance": float(np.std(answerable_variances)),
                "unanswerable_mean_variance": float(np.mean(unanswerable_variances)),
                "unanswerable_std_variance": float(np.std(unanswerable_variances)),
                "answerable_n": len(answerable_variances),
                "unanswerable_n": len(unanswerable_variances),
            })

            gc.collect()
            torch.cuda.empty_cache()

    return prediction_rows, judge_rows, summary_rows


def _panel_b(
    model,
    tokenizer,
    probe_picks: list[dict],
    checkpoint: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    panel_b_resamples = args.panel_b_resamples if args.panel_b_resamples > 0 else args.resamples_per_item
    assert panel_b_resamples > 0
    prediction_rows: list[dict] = []
    judge_rows: list[dict] = []
    question_meta: list[dict] = []

    with (httpx.Client(timeout=90.0) if args.variance_metric == "llm" else contextlib.nullcontext()) as client:
        for local_question_idx, probe in enumerate(tqdm(probe_picks, desc="panel B: questions")):
            question_idx = probe["question_index"] if "question_index" in probe else local_question_idx
            probe_item = probe["item"]
            base_positions = list(probe_item["_base_positions"])
            focus_position = probe["focus_position"]
            focus_rel = base_positions.index(focus_position)

            start_rel = max(0, focus_rel - args.panel_b_relative_window)
            end_rel = min(len(base_positions) - 1, focus_rel + args.panel_b_relative_window)
            stride = args.panel_b_position_stride
            assert stride > 0
            rel_indices = [rel_idx for rel_idx in range(start_rel, end_rel + 1) if ((rel_idx - focus_rel) % stride) == 0]
            if focus_rel not in rel_indices:
                rel_indices.append(focus_rel)
                rel_indices = sorted(rel_indices)
            if args.panel_b_max_positions > 0 and len(rel_indices) > args.panel_b_max_positions:
                rel_indices = sorted(sorted(rel_indices, key=lambda rel_idx: (abs(rel_idx - focus_rel), rel_idx))[:args.panel_b_max_positions])
            assert focus_rel in rel_indices

            cache_path = _panel_b_question_cache_path(args=args, checkpoint=checkpoint, probe=probe, rel_indices=rel_indices)
            if cache_path.exists():
                with open(cache_path) as f:
                    cached = json.load(f)
                prediction_rows.extend(cached["prediction_rows"])
                judge_rows.extend(cached["judge_rows"])
                question_meta.append(cached["question_meta"])
                print(f"[panel B] cache hit q{question_idx} item={probe_item['item_id']} path={cache_path.name}")
                continue

            panel_items = []
            meta_rows = []
            for rel_idx in rel_indices:
                abs_pos = base_positions[rel_idx]
                rel_offset = rel_idx - focus_rel
                for resample_idx in range(panel_b_resamples):
                    panel_items.append(
                        {
                            "context_input_ids": probe_item["context_input_ids"],
                            "context_positions": [abs_pos],
                        }
                    )
                    meta_rows.append(
                        {
                            "cot_position_abs": abs_pos,
                            "cot_position_rel": rel_idx,
                            "rel_offset": rel_offset,
                            "resample_idx": resample_idx,
                        }
                    )

            activations = _extract_activations_batched(
                model=model,
                tokenizer=tokenizer,
                items=panel_items,
                layers=[args.panel_b_layer],
                activation_extract_batch_size=args.activation_extract_batch_size,
                device=args.device,
            )
            prompts = [probe["oracle_question"]] * len(panel_items)
            predictions = _oracle_generate(
                model=model,
                tokenizer=tokenizer,
                activations=activations,
                prompts=prompts,
                layers=[args.panel_b_layer],
                args=args,
                generation_temperature=args.panel_b_temperature,
            )
            assert len(predictions) == len(meta_rows)

            answers_by_position: dict[int, list[str]] = {}
            question_prediction_rows = []
            for meta, pred in zip(meta_rows, predictions):
                rel_idx = meta["cot_position_rel"]
                if rel_idx not in answers_by_position:
                    answers_by_position[rel_idx] = []
                answers_by_position[rel_idx].append(pred)
                question_prediction_rows.append(
                    {
                        "panel": "B",
                        "question_index": question_idx,
                        "item_id": probe_item["item_id"],
                        "source_question": probe_item["prompt"],
                        "question": probe["oracle_question"],
                        "answer_token": probe["answer_token"],
                        "layer": args.panel_b_layer,
                        "cot_position_abs": meta["cot_position_abs"],
                        "cot_position_rel": rel_idx,
                        "rel_offset": meta["rel_offset"],
                        "resample_idx": meta["resample_idx"],
                        "temperature": args.panel_b_temperature,
                        "prediction": pred,
                    }
                )
            prediction_rows.extend(question_prediction_rows)

            question_judge_rows = []
            for rel_idx in tqdm(rel_indices, desc=f"panel B: judge q{question_idx}", leave=False):
                abs_pos = base_positions[rel_idx]
                rel_offset = rel_idx - focus_rel
                answers = answers_by_position[rel_idx]
                judged = _judge_variance(
                    question=probe["oracle_question"],
                    answers=answers,
                    args=args,
                    client=client,
                )
                question_judge_rows.append(
                    {
                        "panel": "B",
                        "question_index": question_idx,
                        "item_id": probe_item["item_id"],
                        "source_question": probe_item["prompt"],
                        "question": probe["oracle_question"],
                        "target_response": probe_item["target_response"],
                        "answer_token": probe["answer_token"],
                        "candidate_type": probe["candidate_type"],
                        "focus_token_decoded": probe["focus_token_decoded"],
                        "focus_position_abs": focus_position,
                        "focus_position_rel": focus_rel,
                        "cot_position_abs": abs_pos,
                        "cot_position_rel": rel_idx,
                        "rel_offset": rel_offset,
                        "answers": answers,
                        "variance": judged["variance"],
                        "reason": judged["reason"],
                        "canonical_answer": judged["canonical_answer"],
                        "judge_raw": judged["raw"],
                    }
                )
            judge_rows.extend(question_judge_rows)

            question_meta_row = {
                "question_index": question_idx,
                "item_id": probe_item["item_id"],
                "domain": probe_item["domain"],
                "tier": probe_item["tier"],
                "source_question": probe_item["prompt"],
                "question": probe["oracle_question"],
                "target_response": probe_item["target_response"],
                "answer_token": probe["answer_token"],
                "candidate_type": probe["candidate_type"],
                "in_question": probe["in_question"],
                "focus_token_decoded": probe["focus_token_decoded"],
                "focus_position_abs": focus_position,
                "focus_position_rel": focus_rel,
                "focus_position_frac": probe["focus_position_frac"],
                "num_positions_total": len(base_positions),
                "num_positions_used": len(rel_indices),
            }
            question_meta.append(question_meta_row)

            cached_payload = {
                "prediction_rows": question_prediction_rows,
                "judge_rows": question_judge_rows,
                "question_meta": question_meta_row,
            }
            with open(cache_path, "w") as f:
                json.dump(cached_payload, f)
            print(f"[panel B] cache save q{question_idx} item={probe_item['item_id']} path={cache_path.name}")

            gc.collect()
            torch.cuda.empty_cache()

    aligned_groups: dict[int, list[float]] = {}
    for row in judge_rows:
        offset = row["rel_offset"]
        if offset not in aligned_groups:
            aligned_groups[offset] = []
        aligned_groups[offset].append(row["variance"])
    aligned_summary = []
    for offset in sorted(aligned_groups):
        values = aligned_groups[offset]
        aligned_summary.append(
            {
                "rel_offset": offset,
                "mean_variance": float(np.mean(values)),
                "std_variance": float(np.std(values)),
                "n_questions": len(values),
            }
        )

    probe_meta = {
        "num_questions": len(probe_picks),
        "questions": question_meta,
        "panel_b_layer": args.panel_b_layer,
        "panel_b_temperature": args.panel_b_temperature,
        "panel_b_resamples": panel_b_resamples,
        "panel_b_relative_window": args.panel_b_relative_window,
        "panel_b_position_stride": args.panel_b_position_stride,
        "panel_b_target_rel_fraction": args.panel_b_target_rel_fraction,
        "panel_b_target_rel_tolerance": args.panel_b_target_rel_tolerance,
        "panel_b_require_numeric": bool(args.panel_b_require_numeric),
        "panel_b_word_fallback": bool(args.panel_b_word_fallback),
        "panel_b_question_mode": args.panel_b_question_mode,
        "panel_b_question_model": args.panel_b_question_model,
        "panel_b_rerank": args.panel_b_rerank,
        "panel_b_rerank_model": args.panel_b_rerank_model,
        "panel_b_rerank_top_k": args.panel_b_rerank_top_k,
        "panel_b_rerank_context_chars": args.panel_b_rerank_context_chars,
        "panel_b_hard_token_filter": args.panel_b_hard_token_filter,
        "panel_b_hard_token_model": args.panel_b_hard_token_model,
        "panel_b_hard_token_max_confidence": args.panel_b_hard_token_max_confidence,
        "panel_b_hard_token_context_tokens": args.panel_b_hard_token_context_tokens,
        "panel_b_cache_dir": str(_get_panel_b_cache_dir(args)),
        "panel_b_cache_version": args.panel_b_cache_version,
    }
    return prediction_rows, judge_rows, aligned_summary, probe_meta


def _plot(
    panel_a_summary: list[dict],
    panel_b_aligned_summary: list[dict],
    panel_b_meta: dict,
    output_path: Path,
    model_name: str,
    checkpoint_name: str,
    variance_label: str,
    panel_a_redraws: int,
) -> None:
    panel_a_summary = sorted(panel_a_summary, key=lambda row: row["temperature"])
    temperatures = [row["temperature"] for row in panel_a_summary]
    answerable_var = [row["answerable_mean_variance"] for row in panel_a_summary]
    answerable_sd = [row["answerable_std_variance"] for row in panel_a_summary]
    unanswerable_var = [row["unanswerable_mean_variance"] for row in panel_a_summary]
    unanswerable_sd = [row["unanswerable_std_variance"] for row in panel_a_summary]

    panel_b_aligned_summary = sorted(panel_b_aligned_summary, key=lambda row: row["rel_offset"])
    xs = [row["rel_offset"] for row in panel_b_aligned_summary]
    ys = [row["mean_variance"] for row in panel_b_aligned_summary]
    ys_sd = [row["std_variance"] for row in panel_b_aligned_summary]
    y_lower = np.clip(np.array(ys) - np.array(ys_sd), 0.0, 1.0)
    y_upper = np.clip(np.array(ys) + np.array(ys_sd), 0.0, 1.0)
    n_per_offset = [row["n_questions"] for row in panel_b_aligned_summary]
    n_min = min(n_per_offset)
    n_max = max(n_per_offset)
    n_offset_label = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_a.set_facecolor("white")
    ax_b.set_facecolor("white")

    ans_lower = np.clip(np.array(answerable_var) - np.array(answerable_sd), 0.0, 1.0)
    ans_upper = np.clip(np.array(answerable_var) + np.array(answerable_sd), 0.0, 1.0)
    unans_lower = np.clip(np.array(unanswerable_var) - np.array(unanswerable_sd), 0.0, 1.0)
    unans_upper = np.clip(np.array(unanswerable_var) + np.array(unanswerable_sd), 0.0, 1.0)
    ax_a.fill_between(temperatures, ans_lower, ans_upper, color="#1f77b4", alpha=0.18, linewidth=0.0)
    ax_a.fill_between(temperatures, unans_lower, unans_upper, color="#d62728", alpha=0.18, linewidth=0.0)
    ax_a.plot(temperatures, answerable_var, marker="o", linewidth=2.0, color="#1f77b4", label="Answerable (mean ±1 SD)")
    ax_a.plot(temperatures, unanswerable_var, marker="s", linewidth=2.0, color="#d62728", label="Unanswerable (mean ±1 SD)")
    ax_a.set_xlabel("T (temperature)")
    ax_a.set_ylabel(f"{variance_label} variance")
    ax_a.set_title("A. Variance vs temperature")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.grid(True, alpha=0.3)
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend([handles[1], handles[0]], [labels[1], labels[0]])
    ax_a.text(
        0.02,
        0.98,
        f"samples/class={panel_a_summary[0]['answerable_n']}\nredraws/item={panel_a_redraws}",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    ax_b.fill_between(xs, y_lower, y_upper, color="#2ca02c", alpha=0.18, linewidth=0.0)
    ax_b.plot(xs, ys, linewidth=1.8, color="#2ca02c", label="Mean aligned variance (±1 SD)")
    ax_b.axvline(0, linestyle="--", linewidth=1.4, color="#ff7f0e", label="Informative token position")
    ax_b.set_xlabel("Relative CoT offset from informative token")
    ax_b.set_ylabel(f"{variance_label} variance")
    ax_b.set_title("B. Aligned informative-token probing")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend()
    ax_b.text(
        0.02,
        0.98,
        (
            f"questions={panel_b_meta['num_questions']}  n/offset={n_offset_label}\n"
            f"redraws/pos={panel_b_meta['panel_b_resamples']}\n"
            f"window=±{panel_b_meta['panel_b_relative_window']} stride={panel_b_meta['panel_b_position_stride']}  layer={panel_b_meta['panel_b_layer']}  T={panel_b_meta['panel_b_temperature']}"
        ),
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    ckpt_label = Path(checkpoint_name).name
    fig.suptitle(f"Activation Oracle Uncertainty Resampling ({model_name}, {ckpt_label}, metric={variance_label})", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[load] model={args.model} checkpoint={checkpoint}")
    model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint, args.device)
    n_layers = len(args.layers)

    print(f"[data] loading cot_description split={args.split}")
    raw_data = load_task_data("cot_description", split=args.split, n=None, shuffle=True)
    prepare_context_ids(raw_data, tokenizer, layers=args.layers)
    for idx, item in enumerate(raw_data):
        item["item_id"] = f"cot_desc_{idx:05d}"
        item["_base_positions"] = _extract_base_positions_for_item(item, n_layers)

    panel_a_items = []
    panel_a_predictions = []
    panel_a_judges = []
    panel_a_summary = []
    if not bool(args.skip_panel_a):
        selected_items = _balanced_diverse_subset(
            raw_data,
            n_per_class=args.n_per_class,
            seed=args.seed,
            diversity_fields=args.diversity_fields,
        )
        selected_ids = set(item["item_id"] for item in selected_items)
        panel_a_items = [item for item in raw_data if item["item_id"] in selected_ids]
        print(f"[panel A] items={len(panel_a_items)} (answerable={args.n_per_class}, unanswerable={args.n_per_class})")
        panel_a_predictions, panel_a_judges, panel_a_summary = _panel_a(model, tokenizer, panel_a_items, args)
    else:
        print("[panel A] skipped")

    if args.panel_b_selected_probes_in:
        print(f"[panel B] loading selected probes from {args.panel_b_selected_probes_in}")
        probe_picks = _load_selected_probes(args.panel_b_selected_probes_in, raw_data)
    else:
        print("[panel B] selecting informative-token probe items")
        panel_b_answerable_count = sum(1 for item in raw_data if item["answerable"] is True)
        panel_b_probe_pool_target = min(panel_b_answerable_count, args.panel_b_num_questions * 12)
        assert panel_b_probe_pool_target >= args.panel_b_num_questions
        probe_pool = _pick_informative_token_probes(
            raw_data,
            tokenizer=tokenizer,
            num_questions=panel_b_probe_pool_target,
            relative_window=args.panel_b_relative_window,
            target_rel_fraction=args.panel_b_target_rel_fraction,
            target_rel_tolerance=args.panel_b_target_rel_tolerance,
            min_rel_fraction=args.panel_b_min_rel_fraction,
            require_numeric=bool(args.panel_b_require_numeric),
            word_fallback=bool(args.panel_b_word_fallback),
            seed=args.seed,
            args=args,
            checkpoint=checkpoint,
            run_dir=run_dir,
        )
        probe_pool = _attach_panel_b_oracle_questions(
            probes=probe_pool,
            mode=args.panel_b_question_mode,
            args=args,
            checkpoint=checkpoint,
            run_dir=run_dir,
        )
        probe_pool = _apply_panel_b_hard_token_filter(
            probes=probe_pool,
            tokenizer=tokenizer,
            args=args,
            checkpoint=checkpoint,
            run_dir=run_dir,
        )
        probe_picks = _select_diverse_probes(probe_pool, n=args.panel_b_num_questions, seed=args.seed + 1701)
        assert len(probe_picks) >= args.panel_b_num_questions
        probe_picks = probe_picks[:args.panel_b_num_questions]
        for question_idx, probe in enumerate(probe_picks):
            probe["question_index"] = question_idx
    if args.panel_b_num_shards > 1:
        probe_picks = [probe for probe in probe_picks if (probe["question_index"] % args.panel_b_num_shards) == args.panel_b_shard_index]
        assert len(probe_picks) > 0
        print(f"[panel B] shard {args.panel_b_shard_index + 1}/{args.panel_b_num_shards} handling {len(probe_picks)} questions")

    probe_rows = []
    for probe in probe_picks:
        probe_item = probe["item"]
        left_ctx, right_ctx = _build_masked_token_context(
            item=probe_item,
            token_position=probe["focus_position"],
            tokenizer=tokenizer,
            context_tokens=args.panel_b_hard_token_context_tokens,
        )
        probe_rows.append({
            "question_index": probe["question_index"] if "question_index" in probe else None,
            "item_id": probe_item["item_id"],
            "domain": probe_item["domain"],
            "tier": probe_item["tier"],
            "source_question": probe_item["prompt"],
            "oracle_question": probe["oracle_question"],
            "oracle_question_reason": probe["oracle_question_reason"] if "oracle_question_reason" in probe else None,
            "answer_token": probe["answer_token"],
            "candidate_type": probe["candidate_type"],
            "is_named_entity": probe["is_named_entity"],
            "in_question": probe["in_question"],
            "focus_position_abs": probe["focus_position"],
            "focus_position_rel": probe["focus_position_rel"],
            "focus_position_frac": probe["focus_position_frac"],
            "focus_token_decoded": probe["focus_token_decoded"],
            "priority": probe["priority"],
            "target_context": probe["target_context"],
            "cot_context": probe["cot_context"],
            "rerank_reason": probe["rerank_reason"] if "rerank_reason" in probe else None,
            "hardness_score": probe["hardness_score"] if "hardness_score" in probe else None,
            "hard_guess": probe["hard_guess"] if "hard_guess" in probe else None,
            "hard_confidence": probe["hard_confidence"] if "hard_confidence" in probe else None,
            "hard_is_correct": probe["hard_is_correct"] if "hard_is_correct" in probe else None,
            "left_ctx": left_ctx,
            "right_ctx": right_ctx,
        })
        print(
            f"[panel B] item={probe_item['item_id']} token={probe['answer_token']} type={probe['candidate_type']} "
            f"focus_pos={probe['focus_position']} rel={probe['focus_position_rel']} "
            f"rel_frac={probe['focus_position_frac']:.3f} decoded={probe['focus_token_decoded']!r} "
            f"oracle_q={probe['oracle_question']!r}"
        )
    _write_jsonl(run_dir / "panel_b_selected_probes.jsonl", probe_rows)
    if bool(args.panel_b_stop_after_selection):
        _write_jsonl(run_dir / "panel_a_predictions.jsonl", panel_a_predictions)
        _write_jsonl(run_dir / "panel_a_judgments.jsonl", panel_a_judges)
        _write_jsonl(run_dir / "panel_a_summary.jsonl", panel_a_summary)
        print("[panel B] stopping after selection as requested")
        return

    panel_b_predictions, panel_b_judges, panel_b_aligned_summary, panel_b_meta = _panel_b(
        model=model,
        tokenizer=tokenizer,
        probe_picks=probe_picks,
        checkpoint=checkpoint,
        args=args,
    )

    figure_path = None
    if len(panel_a_summary) > 0:
        figure_path = run_dir / "oracle_uncertainty_resampling.png"
        _plot(
            panel_a_summary=panel_a_summary,
            panel_b_aligned_summary=panel_b_aligned_summary,
            panel_b_meta=panel_b_meta,
            output_path=figure_path,
            model_name=args.model,
            checkpoint_name=checkpoint,
            variance_label=args.variance_metric,
            panel_a_redraws=args.resamples_per_item,
        )

    _write_jsonl(run_dir / "panel_a_predictions.jsonl", panel_a_predictions)
    _write_jsonl(run_dir / "panel_a_judgments.jsonl", panel_a_judges)
    _write_jsonl(run_dir / "panel_a_summary.jsonl", panel_a_summary)
    _write_jsonl(run_dir / "panel_b_predictions.jsonl", panel_b_predictions)
    _write_jsonl(run_dir / "panel_b_judgments.jsonl", panel_b_judges)
    _write_jsonl(run_dir / "panel_b_aligned_summary.jsonl", panel_b_aligned_summary)

    summary = {
        "created_utc": run_ts,
        "checkpoint": checkpoint,
        "model": args.model,
        "split": args.split,
        "layers": args.layers,
        "n_per_class": args.n_per_class,
        "diversity_fields": args.diversity_fields,
        "temperatures": args.temperatures,
        "resamples_per_item": args.resamples_per_item,
        "panel_b_resamples": args.panel_b_resamples,
        "panel_b_layer": args.panel_b_layer,
        "panel_b_temperature": args.panel_b_temperature,
        "panel_b_num_questions": len(panel_b_meta["questions"]),
        "panel_b_requested_num_questions": args.panel_b_num_questions,
        "panel_b_relative_window": args.panel_b_relative_window,
        "panel_b_position_stride": args.panel_b_position_stride,
        "panel_b_target_rel_fraction": args.panel_b_target_rel_fraction,
        "panel_b_target_rel_tolerance": args.panel_b_target_rel_tolerance,
        "panel_b_min_rel_fraction": args.panel_b_min_rel_fraction,
        "panel_b_require_numeric": bool(args.panel_b_require_numeric),
        "panel_b_word_fallback": bool(args.panel_b_word_fallback),
        "panel_b_question_mode": args.panel_b_question_mode,
        "panel_b_question_model": args.panel_b_question_model,
        "panel_b_rerank": args.panel_b_rerank,
        "panel_b_rerank_model": args.panel_b_rerank_model,
        "panel_b_rerank_top_k": args.panel_b_rerank_top_k,
        "panel_b_rerank_context_chars": args.panel_b_rerank_context_chars,
        "panel_b_hard_token_filter": args.panel_b_hard_token_filter,
        "panel_b_hard_token_model": args.panel_b_hard_token_model,
        "panel_b_hard_token_max_confidence": args.panel_b_hard_token_max_confidence,
        "panel_b_hard_token_context_tokens": args.panel_b_hard_token_context_tokens,
        "panel_b_num_shards": args.panel_b_num_shards,
        "panel_b_shard_index": args.panel_b_shard_index,
        "panel_b_cache_dir": str(_get_panel_b_cache_dir(args)),
        "panel_b_cache_version": args.panel_b_cache_version,
        "variance_metric": args.variance_metric,
        "min_variance_floor": args.min_variance_floor,
        "panel_a_summary": panel_a_summary,
        "panel_b_meta": panel_b_meta,
        "panel_b_aligned_summary": panel_b_aligned_summary,
        "figure_path": str(figure_path) if figure_path is not None else None,
        "panel_a_items": [item["item_id"] for item in panel_a_items],
        "elapsed_seconds": time.time() - t_start,
    }
    _write_json(run_dir / "summary.json", summary)

    print(f"\nSaved run artifacts to: {run_dir}")
    if figure_path is not None:
        print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
