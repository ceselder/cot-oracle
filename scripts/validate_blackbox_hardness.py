#!/usr/bin/env python3
"""
Validate that infogap tasks are genuinely hard for text-only monitors.

Samples 200 examples per task, sends partial CoT text + question (no activations)
to Claude Sonnet via OpenRouter, scores responses against ground truth.

Acceptance criteria:
  - Binary tasks: < 65% accuracy
  - MCQ tasks: < 65% accuracy
  - Open-ended (remaining_strategy): token F1 < 0.3

Usage:
    python scripts/validate_blackbox_hardness.py
    python scripts/validate_blackbox_hardness.py --tasks early_answer_pred backtrack_pred
    python scripts/validate_blackbox_hardness.py --n-samples 50  # quick test
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv(os.path.expanduser("~/.env"))

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemini-2.5-flash"
API_BASE = "https://openrouter.ai/api/v1/chat/completions"

BINARY_TASKS = {"early_answer_pred", "backtrack_pred", "error_pred", "self_correction", "verification"}
MCQ_TASKS = {"branch_pred", "completion_pred"}
OPENENDED_TASKS = {"remaining_strategy"}

ALL_TASKS = sorted(BINARY_TASKS | MCQ_TASKS | OPENENDED_TASKS)


def call_llm(prompt: str, max_tokens: int = 200) -> str:
    resp = requests.post(
        API_BASE,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def token_f1(pred: str, target: str) -> float:
    pred_tokens = set(pred.lower().split())
    target_tokens = set(target.lower().split())
    if not pred_tokens or not target_tokens:
        return 0.0
    common = pred_tokens & target_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def build_text_prompt(dp: dict, tokenizer) -> str:
    """Reconstruct the partial CoT text from context_input_ids for text-only evaluation."""
    context_ids = dp["context_input_ids"]
    # Cap at last 2000 tokens to stay within context limits
    if len(context_ids) > 2000:
        context_ids = context_ids[-2000:]
    partial_text = tokenizer.decode(context_ids, skip_special_tokens=True)

    # Extract the task question from the oracle prompt (after the "layers X, Y, Z. " prefix)
    oracle_prompt = dp["prompt"]
    match = re.search(r"layers \d+(?:, \d+)*\.\s*(.+)", oracle_prompt, re.DOTALL)
    question = match.group(1) if match else oracle_prompt

    return (
        f"Below is a partial chain-of-thought from a language model. "
        f"The reasoning is truncated — you cannot see the rest.\n\n"
        f"--- Partial CoT ---\n{partial_text}\n--- End ---\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely (just the answer label or short text):"
    )


def _pred_is_affirmative(pred: str) -> bool | None:
    """Classify a free-form prediction as affirmative, negative, or ambiguous."""
    p = pred.lower().strip().rstrip(".")
    if p in ("yes", "true", "correct", "likely", "will"):
        return True
    if p.startswith("yes") or p.startswith("it will") or p.startswith("the model will "):
        return True
    if p in ("no", "false", "unlikely", "won't", "will not"):
        return False
    if p.startswith("no") or p.startswith("it will not") or p.startswith("the model will not"):
        return False
    return None


# Map each binary task's labels to affirmative/negative
_BINARY_POSITIVE = {
    "will_backtrack": True, "will_continue_forward": False,
    "will_error": True, "correct_step": False,
    "will_self_correct": True, "will_not_correct": False,
    "will_verify": True, "will_not_verify": False,
}


def score_binary(prediction: str, target: str) -> bool:
    """Check if prediction matches target label, handling yes/no answers."""
    pred_lower = prediction.lower().strip()
    # Direct label match
    if target.lower() in pred_lower:
        return True
    # Yes/No → label mapping
    affirm = _pred_is_affirmative(prediction)
    if affirm is not None and target in _BINARY_POSITIVE:
        target_is_positive = _BINARY_POSITIVE[target]
        return affirm == target_is_positive
    return False


def score_mcq(prediction: str, target: str) -> bool:
    """Check if prediction matches A or B."""
    pred_clean = prediction.strip().upper()
    if target == "A":
        return pred_clean.startswith("A") or "option a" in prediction.lower()
    return pred_clean.startswith("B") or "option b" in prediction.lower()


def main():
    parser = argparse.ArgumentParser(description="Validate blackbox hardness of infogap tasks")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument("--input-dir", default="data/precomputed")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="eval_logs/infogap_validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    import random
    random.seed(args.seed)

    results = {}

    for task_name in args.tasks:
        jsonl_path = Path(args.input_dir) / f"{task_name}.jsonl"
        if not jsonl_path.exists():
            print(f"\n  Skipping {task_name}: {jsonl_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Validating: {task_name}")
        print(f"{'=' * 60}")

        # Load and sample
        data = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        random.shuffle(data)
        samples = data[:args.n_samples]
        print(f"  Sampled {len(samples)} examples")

        # Score
        log_entries = []
        scores = []

        for dp in tqdm(samples, desc=task_name):
            text_prompt = build_text_prompt(dp, tokenizer)
            target = dp["target_response"]

            try:
                prediction = call_llm(text_prompt)
            except Exception as e:
                print(f"  API error: {e}", flush=True)
                time.sleep(2)
                continue

            if task_name in BINARY_TASKS:
                correct = score_binary(prediction, target)
            elif task_name in MCQ_TASKS:
                correct = score_mcq(prediction, target)
            else:
                correct = token_f1(prediction, target)

            scores.append(correct)
            log_entries.append({
                "task": task_name,
                "target": target,
                "prediction": prediction,
                "score": correct,
                "prompt_preview": text_prompt[:500],
            })

            # Rate limiting
            time.sleep(0.3)

        # Compute metrics
        if task_name in OPENENDED_TASKS:
            avg_f1 = sum(scores) / len(scores) if scores else 0.0
            threshold = 0.3
            passed = avg_f1 < threshold
            metric_str = f"avg F1 = {avg_f1:.3f} (threshold < {threshold})"
        else:
            accuracy = sum(scores) / len(scores) if scores else 0.0
            threshold = 0.65
            passed = accuracy < threshold
            metric_str = f"accuracy = {accuracy:.3f} (threshold < {threshold})"

        status = "PASS" if passed else "FAIL"
        print(f"  {task_name}: {metric_str} [{status}]")
        results[task_name] = {"metric": metric_str, "status": status, "n": len(scores)}

        # Save log
        log_path = output_dir / f"{task_name}_log.jsonl"
        with open(log_path, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  Log saved to {log_path}")

        # Print target distribution
        if task_name in BINARY_TASKS | MCQ_TASKS:
            target_dist = Counter(dp["target_response"] for dp in samples)
            print(f"  Target distribution: {dict(target_dist)}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for task_name, info in results.items():
        print(f"  {task_name:25s} {info['metric']:40s} [{info['status']}] (n={info['n']})")


if __name__ == "__main__":
    main()
