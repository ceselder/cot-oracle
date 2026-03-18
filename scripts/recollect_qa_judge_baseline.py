#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(Path.home() / ".env")

from data_loading import load_task_data
from qa_scorer import (
    OPENROUTER_CHAT_COMPLETIONS_URL,
    QA_SCORE_SYSTEM,
    QA_SCORE_MAX_TOKENS,
    build_qa_score_prompt,
    compute_token_f1_scores,
    extract_judge_json,
    get_score_model,
)

SCORE_MODEL = get_score_model()

QA_TASKS = ("sqa", "chunked_convqa", "chunked_compqa")
BASELINE_SYSTEM = (
    "You are analyzing a language model's chain-of-thought reasoning. "
    "Answer the question about the reasoning using only the provided reasoning trace. "
    "Be concise and directly answer the question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recollect QA judge baselines on the same eval slices as run_eval")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--tasks", nargs="+", default=list(QA_TASKS))
    parser.add_argument("--max-items", type=int, default=None, help="Override eval.max_items_per_eval from config")
    parser.add_argument("--baseline-model", default=None, help="Override baselines.llm_monitor.model from config")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override baselines.llm_monitor.max_tokens from config")
    parser.add_argument("--temperature", type=float, default=None, help="Override baselines.llm_monitor.temperature from config")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output-dir", default="logs/qa_scorer_baseline")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_slice(task_name: str, max_items: int) -> list[dict]:
    try:
        items = load_task_data(task_name, split="test", n=max_items, shuffle=False)
    except Exception:
        items = []
    if not items:
        items = load_task_data(task_name, split="train", n=max_items, shuffle=False)
    if not items:
        raise RuntimeError(f"No eval slice available for {task_name}")
    for item in items:
        if "prompt" not in item:
            raise ValueError(f"{task_name} item missing prompt")
        if "target_response" not in item:
            raise ValueError(f"{task_name} item missing target_response")
        _get_reasoning_text(item)
    return items


def _get_reasoning_text(item: dict) -> str:
    if "cot_text" in item:
        return item["cot_text"]
    if "raw_read_prompt" in item and item["raw_read_prompt"]:
        return item["raw_read_prompt"][0]["content"]
    raise ValueError(f"Item missing reasoning text fields: {sorted(item.keys())}")


def _build_baseline_user_prompt(item: dict) -> str:
    return (
        f"Reasoning trace:\n{_get_reasoning_text(item)}\n\n"
        f"Question about the reasoning:\n{item['prompt']}\n\n"
        "Answer the question directly."
    )


def _messages_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


async def _post_chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    pbar: tqdm,
) -> str:
    async with semaphore:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
        response = await client.post(OPENROUTER_CHAT_COMPLETIONS_URL, json=body, headers=headers)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        pbar.update(1)
        return text


async def _gather_text_responses(
    message_batches: list[list[dict]],
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    desc: str,
) -> list[str]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=90.0) as client:
        with tqdm(total=len(message_batches), desc=desc) as pbar:
            coros = [
                _post_chat(client, semaphore, messages, model, max_tokens, temperature, pbar)
                for messages in message_batches
            ]
            return await asyncio.gather(*coros)


async def _run_task(
    task_name: str,
    items: list[dict],
    baseline_model: str,
    baseline_max_tokens: int,
    baseline_temperature: float,
    concurrency: int,
) -> tuple[dict, list[dict]]:
    baseline_messages = [
        [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": _build_baseline_user_prompt(item)}]
        for item in items
    ]
    baseline_responses = await _gather_text_responses(
        baseline_messages,
        model=baseline_model,
        max_tokens=baseline_max_tokens,
        temperature=baseline_temperature,
        concurrency=concurrency,
        desc=f"Baseline ({task_name})",
    )

    judge_messages = [
        [
            {"role": "system", "content": QA_SCORE_SYSTEM},
            {"role": "user", "content": build_qa_score_prompt(task_name, item["prompt"], item["target_response"], baseline_response)},
        ]
        for item, baseline_response in zip(items, baseline_responses)
    ]
    judge_responses = await _gather_text_responses(
        judge_messages,
        model=SCORE_MODEL,
        max_tokens=QA_SCORE_MAX_TOKENS,
        temperature=0.0,
        concurrency=concurrency,
        desc=f"Judge ({task_name})",
    )

    traces = []
    scores = []
    token_f1_scores = compute_token_f1_scores(baseline_responses, [item["target_response"] for item in items])
    for idx, (item, baseline_messages_i, baseline_response, judge_response) in enumerate(zip(items, baseline_messages, baseline_responses, judge_responses)):
        judged = extract_judge_json(judge_response)
        score = float(judged["score"])
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Judge score out of range for {task_name}: {score}")
        scores.append(score)
        traces.append({
            "task": task_name,
            "eval_index": idx,
            "prompt_hash": _messages_hash(baseline_messages_i),
            "judge_hash": _messages_hash(judge_messages[idx]),
            "question": item.get("question", ""),
            "oracle_prompt": item["prompt"],
            "cot_text": _get_reasoning_text(item),
            "target_response": item["target_response"],
            "baseline_model": baseline_model,
            "baseline_response": baseline_response,
            "judge_model": SCORE_MODEL,
            "judge_score": score,
            "token_f1": token_f1_scores[idx],
            "judge_reason": str(judged["reason"]).strip(),
            "judge_raw": judge_response,
        })

    summary = {
        "task": task_name,
        "n_items": len(items),
        "baseline_model": baseline_model,
        "judge_model": SCORE_MODEL,
        "mean_judge_score": sum(scores) / len(scores),
        "per_item_judge_score": scores,
        "mean_token_f1": sum(token_f1_scores) / len(token_f1_scores),
        "per_item_token_f1": token_f1_scores,
    }
    return summary, traces


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    max_items = args.max_items if args.max_items is not None else int(cfg["eval"]["max_items_per_eval"])
    llm_cfg = cfg["baselines"]["llm_monitor"]
    baseline_model = args.baseline_model or llm_cfg["model"]
    baseline_max_tokens = args.max_tokens if args.max_tokens is not None else int(llm_cfg["max_tokens"])
    baseline_temperature = args.temperature if args.temperature is not None else float(llm_cfg["temperature"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for task_name in args.tasks:
        if task_name not in QA_TASKS:
            raise ValueError(f"Unsupported QA task: {task_name}")
        items = _load_eval_slice(task_name, max_items=max_items)
        print(f"\n[{task_name}] Recollecting {len(items)} items from the unified eval slice")
        summary, traces = asyncio.run(_run_task(
            task_name=task_name,
            items=items,
            baseline_model=baseline_model,
            baseline_max_tokens=baseline_max_tokens,
            baseline_temperature=baseline_temperature,
            concurrency=args.concurrency,
        ))
        with open(output_dir / f"{task_name}_traces.jsonl", "w") as f:
            for row in traces:
                f.write(json.dumps(row) + "\n")
        with open(output_dir / f"{task_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        combined[task_name] = summary
        print(f"[{task_name}] mean_judge_score={summary['mean_judge_score']:.3f} token_f1={summary['mean_token_f1']:.3f} over {summary['n_items']} items")

    with open(output_dir / "results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
