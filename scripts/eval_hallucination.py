#!/usr/bin/env python3
"""
Standalone eval for the hallucination detection task.

Loads a CoT Oracle checkpoint and evaluates on the hallucination_detection
test split, printing accuracy, precision/recall, confusion matrix, and
per-category breakdown.

Usage:
    python scripts/eval_hallucination.py --checkpoint ceselder/cot-oracle-v8
    python scripts/eval_hallucination.py --checkpoint ceselder/cot-oracle-v8 --max-items 20
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval_loop import run_eval
from data_loading import load_task_data
from tasks import TASKS


def load_oracle(checkpoint: str, device: str = "cuda"):
    """Load base Qwen3-8B + LoRA adapter."""
    base_model_name = "Qwen/Qwen3-8B"

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    return model, tokenizer


def print_confusion_matrix(tp: int, fp: int, fn: int, tn: int):
    """Pretty-print a 2x2 confusion matrix."""
    print("\n  Confusion Matrix:")
    print("                    Predicted")
    print("                 halluc.  factual")
    print(f"  Actual halluc.  {tp:5d}    {fn:5d}")
    print(f"  Actual factual  {fp:5d}    {tn:5d}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection task")
    parser.add_argument("--checkpoint", required=True, help="HF repo or local path to oracle checkpoint")
    parser.add_argument("--max-items", type=int, default=50, help="Max test items to evaluate")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size for eval")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27], help="Activation layers")
    args = parser.parse_args()

    model, tokenizer = load_oracle(args.checkpoint, args.device)

    # Run eval via unified eval loop
    print(f"\nRunning hallucination_detection eval (max_items={args.max_items})...")
    metrics, traces = run_eval(
        model=model,
        tokenizer=tokenizer,
        task_names=["hallucination_detection"],
        max_items=args.max_items,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        layers=args.layers,
    )

    # Print overall metrics from run_eval
    print("\n" + "=" * 60)
    print("HALLUCINATION DETECTION EVAL RESULTS")
    print("=" * 60)

    accuracy = metrics.get("eval/hallucination_detection", 0.0)
    n_eval = int(metrics.get("eval_n/hallucination_detection", 0))
    n_unparsed = int(metrics.get("eval_unparsed/hallucination_detection", 0))
    print(f"\n  Overall accuracy: {accuracy:.1%}")
    print(f"  Items evaluated:  {n_eval}")
    if n_unparsed:
        print(f"  Unparsed:         {n_unparsed}")

    # Per-category breakdown using traces
    task_traces = traces.get("hallucination_detection", [])
    if not task_traces:
        print("\n  No traces available for per-category breakdown.")
        return

    # Load test data to get prompt_category
    test_data = load_task_data("hallucination_detection", split="test", n=args.max_items)

    # Build per-category stats from traces + test data
    # Traces are in same order as test_data (both limited to max_items)
    task_def = TASKS["hallucination_detection"]
    pos_label = task_def.positive_label  # "hallucinated"
    neg_label = task_def.negative_label  # "factual"

    # Confusion matrix counts
    tp = fp = fn = tn = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for i, trace in enumerate(task_traces):
        if i >= len(test_data):
            break

        item = test_data[i]
        category = item.get("prompt_category", "unknown")
        expected = trace.get("expected", "").strip().lower()
        predicted = trace.get("predicted", "").strip().lower()

        # Classify prediction
        pred_is_pos = any(kw.lower() in predicted for kw in task_def.positive_keywords)
        pred_is_neg = any(kw.lower() in predicted for kw in task_def.negative_keywords)
        # Negative keywords checked first (longer matches) to avoid "factual" matching "factually inaccurate"
        if pred_is_neg and not any(kw.lower() in predicted for kw in ("factually inaccurate",)):
            pred_label = neg_label
        elif pred_is_pos:
            pred_label = pos_label
        else:
            pred_label = "unknown"

        actual_is_pos = pos_label.lower() in expected
        correct = trace.get("correct", "")
        is_correct = correct == "True" or correct is True

        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        # Confusion matrix (positive = hallucinated)
        if actual_is_pos and pred_label == pos_label:
            tp += 1
            category_stats[category]["tp"] += 1
        elif not actual_is_pos and pred_label == pos_label:
            fp += 1
            category_stats[category]["fp"] += 1
        elif actual_is_pos and pred_label != pos_label:
            fn += 1
            category_stats[category]["fn"] += 1
        else:
            tn += 1
            category_stats[category]["tn"] += 1

    # Print confusion matrix
    print_confusion_matrix(tp, fp, fn, tn)

    # Precision / recall / F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n  Hallucinated class:")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1:        {f1:.1%}")

    factual_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    factual_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    factual_f1 = 2 * factual_precision * factual_recall / (factual_precision + factual_recall) if (factual_precision + factual_recall) > 0 else 0.0

    print(f"\n  Factual class:")
    print(f"    Precision: {factual_precision:.1%}")
    print(f"    Recall:    {factual_recall:.1%}")
    print(f"    F1:        {factual_f1:.1%}")

    # Per-category table
    print(f"\n  Per-category breakdown:")
    print(f"  {'Category':<25s}  {'Acc':>6s}  {'N':>4s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'TN':>3s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*3}")

    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {cat:<25s}  {acc:5.1%}  {s['total']:4d}  {s['tp']:3d}  {s['fp']:3d}  {s['fn']:3d}  {s['tn']:3d}")

    # Example predictions
    print(f"\n  Sample predictions (first 5):")
    for i, trace in enumerate(task_traces[:5]):
        q = trace.get("question", "")[:60]
        print(f"    [{i}] Q: {q}...")
        print(f"        Expected:  {trace.get('expected', '')[:40]}")
        print(f"        Predicted: {trace.get('predicted', '')[:40]}")
        print(f"        Correct:   {trace.get('correct', '')}")


if __name__ == "__main__":
    main()
