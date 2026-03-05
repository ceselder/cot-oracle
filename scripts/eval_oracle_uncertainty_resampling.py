#!/usr/bin/env python3
"""Evaluate activation-oracle uncertainty via activation-position resampling on cot_description."""

from __future__ import annotations

import argparse
import contextlib
import gc
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

SINGLE_TOKEN_CANDIDATE_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "was", "is", "are", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "at", "by", "from", "into",
    "over", "under", "within", "during", "after", "before", "between",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-oracle uncertainty via resampling")
    parser.add_argument("--checkpoint", default="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n-per-class", type=int, default=40, help="Answerable + unanswerable items each")
    parser.add_argument("--t-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--resamples-per-item", type=int, default=8)
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
    return parser.parse_args()


def _resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    return str(path.resolve()) if path.exists() else checkpoint


def _load_model_and_tokenizer(model_name: str, checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
        oracle_adapter_name="default",
    )
    return predictions


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
    response = client.post(OPENROUTER_CHAT_COMPLETIONS_URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    parsed = extract_judge_json(raw)
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


def _pick_single_token_probe(item_pool: list[dict], tokenizer, n_layers: int) -> dict:
    for item in item_pool:
        if item["answerable"] is False:
            continue
        candidates = []
        seen = set()
        for match in SINGLE_TOKEN_CANDIDATE_RE.findall(item["target_response"]):
            candidate = match
            if candidate not in item["cot_text"] and candidate.replace(",", "") in item["cot_text"]:
                candidate = candidate.replace(",", "")
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        if len(candidates) == 0:
            continue

        user_msg = item["hinted_prompt"] if "hinted_prompt" in item else item["question"] if "question" in item else ""
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

        base_positions = _extract_base_positions_for_item(item, n_layers)
        base_position_set = set(base_positions)
        for candidate in candidates:
            if item["cot_text"].count(candidate) != 1:
                continue
            rel_start = item["cot_text"].index(candidate)
            abs_start = cot_start + rel_start
            abs_end = abs_start + len(candidate)
            token_hits = [i for i, (s, e) in enumerate(offsets) if s < abs_end and e > abs_start]
            if len(token_hits) != 1:
                continue
            token_position = token_hits[0]
            if token_position < prompt_len:
                continue
            if token_position not in base_position_set:
                continue
            return {
                "item": item,
                "answer_token": candidate,
                "focus_position": token_position,
                "focus_token_decoded": tokenizer.decode([input_ids[token_position]], skip_special_tokens=False),
            }

        target_lower = item["target_response"].lower()
        token_id_counts = {}
        for pos in base_positions:
            tid = input_ids[pos]
            token_id_counts[tid] = token_id_counts[tid] + 1 if tid in token_id_counts else 1
        for pos in base_positions:
            tid = input_ids[pos]
            if token_id_counts[tid] != 1:
                continue
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            cleaned = decoded.strip().lower()
            if len(cleaned) < 4:
                continue
            if re.search(r"[a-z0-9]", cleaned) is None:
                continue
            if cleaned in COMMON_STOPWORDS:
                continue
            if re.search(rf"\b{re.escape(cleaned)}\b", target_lower) is None:
                continue
            return {
                "item": item,
                "answer_token": cleaned,
                "focus_position": pos,
                "focus_token_decoded": decoded,
            }
    raise ValueError("Could not find an answerable cot_description item with a unique single-token answer feature")


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

    with (httpx.Client(timeout=90.0) if args.variance_metric == "llm" else contextlib.nullcontext()) as client:
        for t_value in tqdm(args.t_values, desc="panel A: T sweep"):
            resample_items = []
            meta_rows = []
            for item_idx, item in enumerate(items):
                base_positions = item["_base_positions"]
                k = min(t_value, len(base_positions))
                for resample_idx in range(args.resamples_per_item):
                    rng = random.Random(f"{args.seed}:panel_a:{item_idx}:{t_value}:{resample_idx}")
                    sampled_positions = sorted(rng.sample(base_positions, k)) if k < len(base_positions) else list(base_positions)
                    resample_items.append({
                        "context_input_ids": item["context_input_ids"],
                        "context_positions": sampled_positions * n_layers,
                    })
                    meta_rows.append({
                        "item_idx": item_idx,
                        "t_value": t_value,
                        "resample_idx": resample_idx,
                        "sampled_positions": sampled_positions,
                    })

            activations = _extract_activations_batched(
                model=model,
                tokenizer=tokenizer,
                items=resample_items,
                layers=args.layers,
                activation_extract_batch_size=args.activation_extract_batch_size,
                device=args.device,
            )
            prompts = [items[meta["item_idx"]]["prompt"] for meta in meta_rows]
            predictions = _oracle_generate(
                model=model,
                tokenizer=tokenizer,
                activations=activations,
                prompts=prompts,
                layers=args.layers,
                args=args,
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
                    "t_value": t_value,
                    "resample_idx": meta["resample_idx"],
                    "prediction": prediction,
                    "sampled_positions": meta["sampled_positions"],
                })

            t_judge_rows = []
            for item_idx, answers in tqdm(by_item_answers.items(), desc=f"judge T={t_value}", leave=False):
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
                    "t_value": t_value,
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
                "t_value": t_value,
                "answerable_mean_variance": float(np.mean(answerable_variances)),
                "unanswerable_mean_variance": float(np.mean(unanswerable_variances)),
                "answerable_n": len(answerable_variances),
                "unanswerable_n": len(unanswerable_variances),
            })

            gc.collect()
            torch.cuda.empty_cache()

    return prediction_rows, judge_rows, summary_rows


def _panel_b(
    model,
    tokenizer,
    probe_item: dict,
    focus_position: int,
    answer_token: str,
    focus_token_decoded: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    base_positions = list(probe_item["_base_positions"])
    if args.panel_b_max_positions > 0:
        base_positions = base_positions[: args.panel_b_max_positions]
    n_positions = len(base_positions)
    assert n_positions > 0
    layer_predictions: dict[int, list[str]] = {}
    prediction_rows: list[dict] = []

    for layer in tqdm(args.layers, desc="panel B: layers"):
        layer_items = [
            {
                "context_input_ids": probe_item["context_input_ids"],
                "context_positions": [position],
            }
            for position in base_positions
        ]
        activations = _extract_activations_batched(
            model=model,
            tokenizer=tokenizer,
            items=layer_items,
            layers=[layer],
            activation_extract_batch_size=args.activation_extract_batch_size,
            device=args.device,
        )
        prompts = [probe_item["prompt"]] * n_positions
        predictions = _oracle_generate(
            model=model,
            tokenizer=tokenizer,
            activations=activations,
            prompts=prompts,
            layers=[layer],
            args=args,
        )
        assert len(predictions) == n_positions
        layer_predictions[layer] = predictions
        for rel_idx, (abs_pos, pred) in enumerate(zip(base_positions, predictions)):
            prediction_rows.append({
                "panel": "B",
                "item_id": probe_item["item_id"],
                "layer": layer,
                "cot_position_abs": abs_pos,
                "cot_position_rel": rel_idx,
                "prediction": pred,
            })

        gc.collect()
        torch.cuda.empty_cache()

    judge_rows: list[dict] = []
    eps = 1e-6
    with (httpx.Client(timeout=90.0) if args.variance_metric == "llm" else contextlib.nullcontext()) as client:
        for rel_idx, abs_pos in tqdm(list(enumerate(base_positions)), desc="panel B: judge"):
            answers = [layer_predictions[layer][rel_idx] for layer in args.layers]
            judged = _judge_variance(
                question=probe_item["prompt"],
                answers=answers,
                args=args,
                client=client,
            )
            variance = judged["variance"]
            inv_variance = 1.0 / max(variance, eps)
            judge_rows.append({
                "panel": "B",
                "item_id": probe_item["item_id"],
                "question": probe_item["prompt"],
                "target_response": probe_item["target_response"],
                "answer_token": answer_token,
                "focus_token_decoded": focus_token_decoded,
                "focus_position_abs": focus_position,
                "cot_position_abs": abs_pos,
                "cot_position_rel": rel_idx,
                "answers": answers,
                "variance": variance,
                "inv_variance": inv_variance,
                "reason": judged["reason"],
                "canonical_answer": judged["canonical_answer"],
                "judge_raw": judged["raw"],
            })

    probe_meta = {
        "item_id": probe_item["item_id"],
        "question": probe_item["prompt"],
        "target_response": probe_item["target_response"],
        "answer_token": answer_token,
        "focus_token_decoded": focus_token_decoded,
        "focus_position_abs": focus_position,
        "focus_position_rel": base_positions.index(focus_position),
        "num_positions": len(base_positions),
    }
    return prediction_rows, judge_rows, probe_meta


def _plot(panel_a_summary: list[dict], panel_b_judges: list[dict], probe_meta: dict, output_path: Path, model_name: str, checkpoint_name: str, variance_label: str) -> None:
    panel_a_summary = sorted(panel_a_summary, key=lambda row: row["t_value"])
    t_values = [row["t_value"] for row in panel_a_summary]
    answerable_var = [row["answerable_mean_variance"] for row in panel_a_summary]
    unanswerable_var = [row["unanswerable_mean_variance"] for row in panel_a_summary]

    panel_b_judges = sorted(panel_b_judges, key=lambda row: row["cot_position_rel"])
    xs = [row["cot_position_rel"] for row in panel_b_judges]
    ys = [row["inv_variance"] for row in panel_b_judges]
    focus_x = probe_meta["focus_position_rel"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_a.set_facecolor("white")
    ax_b.set_facecolor("white")

    ax_a.plot(t_values, answerable_var, marker="o", linewidth=2.0, color="#1f77b4", label="Answerable")
    ax_a.plot(t_values, unanswerable_var, marker="s", linewidth=2.0, color="#d62728", label="Unanswerable")
    ax_a.set_xlabel("T sampled activation positions")
    ax_a.set_ylabel(f"{variance_label} variance")
    ax_a.set_title("A. Variance vs T")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()

    ax_b.plot(xs, ys, linewidth=1.8, color="#2ca02c")
    ax_b.axvline(focus_x, linestyle="--", linewidth=1.4, color="#ff7f0e", label=f"focus token: {probe_meta['answer_token']}")
    ax_b.set_xlabel("CoT position (relative index)")
    ax_b.set_ylabel(f"1 / {variance_label} variance")
    ax_b.set_title("B. Single-position probing")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend()

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
    model, tokenizer = _load_model_and_tokenizer(args.model, checkpoint)
    n_layers = len(args.layers)

    print(f"[data] loading cot_description split={args.split}")
    raw_data = load_task_data("cot_description", split=args.split, n=None, shuffle=True)
    prepare_context_ids(raw_data, tokenizer, layers=args.layers)
    for idx, item in enumerate(raw_data):
        item["item_id"] = f"cot_desc_{idx:05d}"
        item["_base_positions"] = _extract_base_positions_for_item(item, n_layers)

    selected_items = _balanced_subset(raw_data, n_per_class=args.n_per_class, seed=args.seed)
    selected_ids = set(item["item_id"] for item in selected_items)
    panel_a_items = [item for item in raw_data if item["item_id"] in selected_ids]

    print(f"[panel A] items={len(panel_a_items)} (answerable={args.n_per_class}, unanswerable={args.n_per_class})")
    panel_a_predictions, panel_a_judges, panel_a_summary = _panel_a(model, tokenizer, panel_a_items, args)

    print("[panel B] selecting single-token probe item")
    probe_pick = _pick_single_token_probe(raw_data, tokenizer=tokenizer, n_layers=n_layers)
    probe_item = probe_pick["item"]
    print(
        f"[panel B] item={probe_item['item_id']} token={probe_pick['answer_token']} "
        f"focus_pos={probe_pick['focus_position']} decoded={probe_pick['focus_token_decoded']!r}"
    )
    panel_b_predictions, panel_b_judges, probe_meta = _panel_b(
        model=model,
        tokenizer=tokenizer,
        probe_item=probe_item,
        focus_position=probe_pick["focus_position"],
        answer_token=probe_pick["answer_token"],
        focus_token_decoded=probe_pick["focus_token_decoded"],
        args=args,
    )

    figure_path = run_dir / "oracle_uncertainty_resampling.png"
    _plot(
        panel_a_summary=panel_a_summary,
        panel_b_judges=panel_b_judges,
        probe_meta=probe_meta,
        output_path=figure_path,
        model_name=args.model,
        checkpoint_name=checkpoint,
        variance_label=args.variance_metric,
    )

    _write_jsonl(run_dir / "panel_a_predictions.jsonl", panel_a_predictions)
    _write_jsonl(run_dir / "panel_a_judgments.jsonl", panel_a_judges)
    _write_jsonl(run_dir / "panel_b_predictions.jsonl", panel_b_predictions)
    _write_jsonl(run_dir / "panel_b_judgments.jsonl", panel_b_judges)

    summary = {
        "created_utc": run_ts,
        "checkpoint": checkpoint,
        "model": args.model,
        "split": args.split,
        "layers": args.layers,
        "n_per_class": args.n_per_class,
        "t_values": args.t_values,
        "resamples_per_item": args.resamples_per_item,
        "variance_metric": args.variance_metric,
        "min_variance_floor": args.min_variance_floor,
        "panel_a_summary": panel_a_summary,
        "panel_b_probe": probe_meta,
        "figure_path": str(figure_path),
        "panel_a_items": [item["item_id"] for item in panel_a_items],
        "elapsed_seconds": time.time() - t_start,
    }
    _write_json(run_dir / "summary.json", summary)

    print(f"\nSaved run artifacts to: {run_dir}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
