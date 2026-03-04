#!/usr/bin/env python3
"""
Hierarchical OOD Generalization: Linear Probes vs Pretrained Oracle (Zero-Shot).

Single script that:
  A) Loads Qwen3-8B, extracts activations from all 3 taxonomy splits
  B) Trains & evaluates linear probes (per-layer + concat)
  C) Evaluates pretrained oracle zero-shot on all splits (no task-specific training)
  D) Text baseline: LLM answers directly from raw text (upper bound)
  E) Prints comparison table and saves results

Usage:
  python scripts/taxonomy_ood/run_experiment.py --oracle-checkpoint PATH
  python scripts/taxonomy_ood/run_experiment.py --skip-oracle --skip-text-baseline  # probes only
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add src/ and baselines/ to path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "baselines"))

import torch
from tqdm.auto import tqdm

from core.ao import load_model_with_ao, using_adapter, run_oracle_on_activations, generate_cot, TRAINED_PLACEHOLDER
from data_loading import load_task_data
from tasks import TASKS
from cot_utils import get_cot_stride_positions
from evals.activation_cache import build_full_text_from_prompt_and_cot

# Baselines
from shared import BaselineInput, extract_multilayer_activations
from linear_probe import run_linear_probe


LAYERS = [9, 18, 27]
STRIDE = 5
DEVICE = "cuda"
CACHE_DIR = Path(os.environ["CACHE_DIR"])
RESULTS_DIR = CACHE_DIR / "taxonomy_ood_results"


def load_split_data(split: str) -> list[dict]:
    """Load taxonomy_ood data for a given split from HF."""
    return load_task_data("taxonomy_ood", split=split, n=None, shuffle=False)


def extract_activations_for_split(
    model, tokenizer, items: list[dict], split_name: str,
) -> list[BaselineInput]:
    """Extract activations for all items in a split, return BaselineInputs."""
    model.eval()
    results = []

    for i, item in enumerate(tqdm(items, desc=f"Extracting activations ({split_name})")):
        cot_text = item["cot_text"].strip()
        prompt = item["prompt"]

        full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=STRIDE)

        if len(positions) < 1:
            continue

        tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_tensor = tok_out["input_ids"].to(DEVICE)
        attn_mask = tok_out["attention_mask"].to(DEVICE)
        seq_len = input_tensor.shape[1]
        positions = [p for p in positions if p < seq_len]
        if len(positions) < 1:
            continue

        with using_adapter(model, None):
            acts_by_layer = extract_multilayer_activations(
                model, input_tensor, attn_mask, positions, LAYERS,
            )

        results.append(BaselineInput(
            eval_name="taxonomy_ood",
            example_id=f"{split_name}_{i}",
            clean_prompt=prompt,
            test_prompt=prompt,
            correct_answer=item["target_response"],
            nudge_answer=None,
            ground_truth_label=item["label"],
            clean_response=cot_text,
            test_response=cot_text,
            activations_by_layer=acts_by_layer,
            metadata={
                "positions": positions,
                "split": split_name,
                "item_name": item.get("item_name", ""),
                "medium": item.get("medium", ""),
                "coarse": item.get("coarse", ""),
            },
        ))

    print(f"  {split_name}: {len(results)} items with activations")
    return results


def run_linear_probe_experiment(
    train_inputs: list[BaselineInput],
    narrow_ood_inputs: list[BaselineInput],
    broad_ood_inputs: list[BaselineInput],
) -> dict:
    """Train linear probe on train split, evaluate on all splits."""
    results = {}

    for eval_name, test_inputs in [
        ("train", train_inputs),
        ("narrow_ood", narrow_ood_inputs),
        ("broad_ood", broad_ood_inputs),
    ]:
        print(f"\n  Linear probe eval on {eval_name} ({len(test_inputs)} items)...")
        probe_results = run_linear_probe(
            train_inputs,
            layers=LAYERS,
            lr=0.01,
            epochs=100,
            weight_decay=0.0001,
            pooling="mean",
            device=DEVICE,
            test_inputs=test_inputs if eval_name != "train" else None,
            k_folds=5,
        )
        results[eval_name] = probe_results["per_layer"]

    return results


def evaluate_oracle_on_split(
    model, tokenizer, items: list[dict], split_name: str,
    oracle_adapter_name: str = "default",
) -> dict:
    """Evaluate pretrained oracle zero-shot on a split."""
    task_def = TASKS["taxonomy_ood"]
    model.eval()

    predictions = []
    ground_truths = []
    traces = []

    for i, item in enumerate(tqdm(items, desc=f"Oracle eval ({split_name})")):
        cot_text = item["cot_text"].strip()
        prompt = item["prompt"]

        full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_stride_positions(len(prompt_ids), len(all_ids), stride=STRIDE)

        if len(positions) < 1:
            predictions.append("")
            ground_truths.append(item["label"])
            continue

        tok_out = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_tensor = tok_out["input_ids"].to(DEVICE)
        attn_mask = tok_out["attention_mask"].to(DEVICE)
        seq_len = input_tensor.shape[1]
        positions = [p for p in positions if p < seq_len]
        if len(positions) < 1:
            predictions.append("")
            ground_truths.append(item["label"])
            continue

        # Extract activations (base model, no adapter)
        with using_adapter(model, None):
            acts_by_layer = extract_multilayer_activations(
                model, input_tensor, attn_mask, positions, LAYERS,
            )

        # Flatten multi-layer activations: [K_per_layer * n_layers, D]
        flat_acts = torch.cat([acts_by_layer[l] for l in LAYERS], dim=0)

        # Generate with oracle adapter + activation injection
        response = run_oracle_on_activations(
            model=model,
            tokenizer=tokenizer,
            activations=flat_acts,
            oracle_prompt=prompt,
            model_name="Qwen/Qwen3-8B",
            act_layer=LAYERS,
            max_new_tokens=task_def.max_new_tokens,
            device=DEVICE,
            placeholder_token=TRAINED_PLACEHOLDER,
            oracle_adapter_name=oracle_adapter_name,
        )

        # Parse binary prediction
        resp_lower = response.lower().strip()
        if any(kw in resp_lower for kw in task_def.positive_keywords):
            pred = task_def.positive_label
        elif any(kw in resp_lower for kw in task_def.negative_keywords):
            pred = task_def.negative_label
        else:
            pred = resp_lower

        predictions.append(pred)
        ground_truths.append(item["label"])

        traces.append({
            "idx": i,
            "item_name": item.get("item_name", ""),
            "fine": item.get("fine", ""),
            "medium": item.get("medium", ""),
            "coarse": item.get("coarse", ""),
            "prediction": pred,
            "raw_response": response,
            "ground_truth": item["label"],
            "correct": pred == item["label"],
        })

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "traces": traces,
    }


def evaluate_text_baseline(
    model, tokenizer, items: list[dict], split_name: str,
) -> dict:
    """Text baseline: LLM answers from raw text directly (no activations). Upper bound."""
    task_def = TASKS["taxonomy_ood"]
    model.eval()

    predictions = []
    ground_truths = []
    traces = []

    for i, item in enumerate(tqdm(items, desc=f"Text baseline ({split_name})")):
        cot_text = item["cot_text"].strip()
        question = f"Text: {cot_text}\n\nIs this animal a predator? Answer Yes or No."

        # Generate with base model (no adapter)
        response = generate_cot(
            model, tokenizer, question,
            max_new_tokens=task_def.max_new_tokens,
            device=DEVICE,
            adapter_name=None,
        )

        # Parse binary prediction
        resp_lower = response.lower().strip()
        if any(kw in resp_lower for kw in task_def.positive_keywords):
            pred = task_def.positive_label
        elif any(kw in resp_lower for kw in task_def.negative_keywords):
            pred = task_def.negative_label
        else:
            pred = resp_lower

        predictions.append(pred)
        ground_truths.append(item["label"])

        traces.append({
            "idx": i,
            "item_name": item.get("item_name", ""),
            "fine": item.get("fine", ""),
            "medium": item.get("medium", ""),
            "coarse": item.get("coarse", ""),
            "prediction": pred,
            "raw_response": response,
            "ground_truth": item["label"],
            "correct": pred == item["label"],
        })

    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        "traces": traces,
    }


def print_results_table(probe_results: dict, oracle_results: dict, text_results: dict):
    """Print formatted comparison table."""
    header = f"{'Method':<20} | {'Train':>8} | {'Narrow OOD':>10} | {'Broad OOD':>10}"
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    # Probe results
    for layer_name in ["layer_9", "layer_18", "layer_27", "concat_all"]:
        row = f"Probe ({layer_name})"
        row = f"{row:<20}"
        for split in ["train", "narrow_ood", "broad_ood"]:
            acc = probe_results.get(split, {}).get(layer_name, {}).get("accuracy", 0)
            row += f" | {acc:>8.1%}"
        print(row)

    # Oracle results
    row = f"{'Oracle (zero-shot)':<20}"
    for split in ["train", "narrow_ood", "broad_ood"]:
        acc = oracle_results.get(split, {}).get("accuracy", 0)
        row += f" | {acc:>8.1%}"
    print(row)

    # Text baseline
    row = f"{'Text baseline':<20}"
    for split in ["train", "narrow_ood", "broad_ood"]:
        acc = text_results.get(split, {}).get("accuracy", 0)
        row += f" | {acc:>8.1%}"
    print(row)

    print(separator)


def save_results(probe_results: dict, oracle_results: dict, text_results: dict, log_dir: Path):
    """Save results JSON and per-example logs."""
    log_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe": {},
        "oracle": {},
        "text_baseline": {},
    }

    for split in ["train", "narrow_ood", "broad_ood"]:
        combined["probe"][split] = {
            layer: {k: v for k, v in metrics.items() if k != "per_item"}
            for layer, metrics in probe_results.get(split, {}).items()
        }
        oracle_split = oracle_results.get(split, {})
        combined["oracle"][split] = {
            "accuracy": oracle_split.get("accuracy", 0),
            "correct": oracle_split.get("correct", 0),
            "total": oracle_split.get("total", 0),
        }
        text_split = text_results.get(split, {})
        combined["text_baseline"][split] = {
            "accuracy": text_split.get("accuracy", 0),
            "correct": text_split.get("correct", 0),
            "total": text_split.get("total", 0),
        }

    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Per-example traces
    for method, method_results in [("oracle", oracle_results), ("text_baseline", text_results)]:
        for split in ["train", "narrow_ood", "broad_ood"]:
            traces = method_results.get(split, {}).get("traces", [])
            if traces:
                trace_path = log_dir / f"{method}_traces_{split}.jsonl"
                with open(trace_path, "w") as f:
                    for t in traces:
                        f.write(json.dumps(t) + "\n")
                print(f"  {method} traces ({split}): {trace_path}")


def main():
    parser = argparse.ArgumentParser(description="Taxonomy OOD: Probe vs Oracle (zero-shot) vs Text baseline")
    parser.add_argument("--oracle-checkpoint", type=str, default=None,
                        help="Path to pretrained oracle checkpoint for zero-shot eval")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip oracle evaluation")
    parser.add_argument("--skip-probes", action="store_true",
                        help="Skip linear probe experiments")
    parser.add_argument("--skip-text-baseline", action="store_true",
                        help="Skip text baseline (direct LLM QA)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max items per split (for debugging)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    log_dir = RESULTS_DIR / timestamp

    # ── Phase A: Load model & extract activations ──
    print("=" * 60)
    print("Phase A: Loading model & extracting activations")
    print("=" * 60)

    model, tokenizer = load_model_with_ao("Qwen/Qwen3-8B")

    splits_data = {}
    splits_inputs = {}
    for split_name in ["train", "narrow_ood", "broad_ood"]:
        items = load_split_data(split_name)
        if args.max_items:
            items = items[:args.max_items]
        splits_data[split_name] = items
        print(f"\n  Loaded {len(items)} items for {split_name}")

        inputs = extract_activations_for_split(model, tokenizer, items, split_name)
        splits_inputs[split_name] = inputs

    # ── Phase B: Linear probes ──
    probe_results = {}
    if not args.skip_probes:
        print("\n" + "=" * 60)
        print("Phase B: Linear probe experiments")
        print("=" * 60)

        probe_results = run_linear_probe_experiment(
            splits_inputs["train"],
            splits_inputs["narrow_ood"],
            splits_inputs["broad_ood"],
        )

    # ── Phase C: Oracle (zero-shot, no training) ──
    oracle_results = {}
    if not args.skip_oracle:
        assert args.oracle_checkpoint, "--oracle-checkpoint required for oracle eval (no training, zero-shot only)"
        print("\n" + "=" * 60)
        print(f"Phase C: Oracle zero-shot eval (checkpoint: {args.oracle_checkpoint})")
        print("=" * 60)

        model.load_adapter(args.oracle_checkpoint, adapter_name="taxonomy_oracle", is_trainable=False)

        for split_name in ["train", "narrow_ood", "broad_ood"]:
            items = splits_data[split_name]
            result = evaluate_oracle_on_split(
                model, tokenizer, items, split_name,
                oracle_adapter_name="taxonomy_oracle",
            )
            oracle_results[split_name] = result
            print(f"  {split_name}: accuracy={result['accuracy']:.1%} "
                  f"({result['correct']}/{result['total']})")

    # ── Phase D: Text baseline (upper bound) ──
    text_results = {}
    if not args.skip_text_baseline:
        print("\n" + "=" * 60)
        print("Phase D: Text baseline (direct LLM QA, upper bound)")
        print("=" * 60)

        for split_name in ["train", "narrow_ood", "broad_ood"]:
            items = splits_data[split_name]
            result = evaluate_text_baseline(model, tokenizer, items, split_name)
            text_results[split_name] = result
            print(f"  {split_name}: accuracy={result['accuracy']:.1%} "
                  f"({result['correct']}/{result['total']})")

    # ── Phase E: Results ──
    print("\n" + "=" * 60)
    print("Phase E: Results")
    print("=" * 60)

    if probe_results or oracle_results or text_results:
        print_results_table(probe_results, oracle_results, text_results)
        save_results(probe_results, oracle_results, text_results, log_dir)


if __name__ == "__main__":
    main()
