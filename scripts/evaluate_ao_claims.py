#!/usr/bin/env python3
"""
Quantitative evaluation responding to:
"Current activation oracles are hard to use" (LessWrong, March 2026)

Compares Adam's original AO checkpoint vs our final-sprint checkpoint across:
  1. Real activations (standard eval)
  2. Zero activations (inject zeros — tests if oracle USES activations)
  3. Scrambled activations (shuffle across examples — tests if oracle uses content)

For binary tasks, computes accuracy and AUC (via first-token logprobs).
For token prediction tasks, computes F1.

Usage:
  python scripts/evaluate_ao_claims.py [--max-items 100] [--eval-batch-size 2]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ao_reference"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from tasks import TASKS, get_eval_tasks
from data_loading import (
    load_task_data, load_futurelens_data, load_pastlens_data, prepare_context_ids,
)
from eval_loop import (
    score_task,
    _materialize_activations,
    _batched_oracle_generate,
    _resample_eval_positions,
    _build_oracle_prefix,
    _build_manual_prefix_token_ids,
    _text_baseline_generate,
    _ensure_ao_imports,
    _ao_modules,
    TASK_PARSERS,
)


# ── Checkpoints ──
ADAM_CHECKPOINT = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
FINAL_SPRINT_CHECKPOINT = "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO"
BASE_MODEL = "Qwen/Qwen3-8B"

LAYERS = [9, 18, 27]
INJECTION_LAYER = 1

# ── Tasks relevant to the post's claims ──
BINARY_TASKS = [
    "sycophancy",            # Post: AUC 0.55 (near random)
    "correctness",           # Post: number prediction fails
    "hint_admission",        # Post: "missing information" test
    "decorative_cot",        # Post: related to vagueness
    "backtrack_prediction",  # Post: ~5% correct
    "reasoning_termination", # Post: backtracking evaluation
    "atypical_answer",       # Post: text inversion concern
    "truthfulqa_hint",       # Post: deception detection
]

TOKEN_TASKS = [
    "futurelens",  # Post: next-token F1 0.38
    "pastlens",    # Post: prev-token F1 0.45
]

ALL_TASKS = BINARY_TASKS + TOKEN_TASKS


# ── Model loading ──

def load_model_and_adapters(device="cuda"):
    """Load Qwen3-8B with both Adam's original and our final-sprint adapters."""
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading Adam's adapter: {ADAM_CHECKPOINT}")
    model = PeftModel.from_pretrained(
        base, ADAM_CHECKPOINT,
        adapter_name="adam",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"Loading final-sprint adapter: {FINAL_SPRINT_CHECKPOINT}")
    model.load_adapter(
        FINAL_SPRINT_CHECKPOINT,
        adapter_name="trained",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.eval()
    print(f"Adapters loaded: {list(model.peft_config.keys())}")
    return model, tokenizer


# ── Data loading ──

def load_and_prepare_task(task_name, tokenizer, max_items, layers):
    """Load task data and prepare context IDs."""
    print(f"  Loading {task_name}...")

    if task_name == "futurelens":
        test_data = load_futurelens_data(
            tokenizer=tokenizer, n=max_items, split="test", layers=layers, seed=99,
        )
    elif task_name == "pastlens":
        test_data = load_pastlens_data(
            tokenizer=tokenizer, n=max_items, split="test", layers=layers, seed=98,
        )
    else:
        try:
            test_data = load_task_data(task_name, split="test", n=max_items, shuffle=False)
        except Exception:
            test_data = []
        if not test_data:
            test_data = load_task_data(task_name, split="train", n=max_items, shuffle=False)

    if not test_data:
        return None

    for item in test_data:
        if "meta_spliced_cot_text" in item and "cot_text" not in item:
            item["cot_text"] = item["meta_spliced_cot_text"]
        if "test_prompt" in item and "question" not in item:
            item["question"] = item["test_prompt"]
        if "target_response" not in item and "meta_oracle_target" in item:
            item["target_response"] = str(item["meta_oracle_target"])

    prepare_context_ids(test_data, tokenizer, layers=layers)
    test_data = [d for d in test_data if d.get("context_input_ids")]

    if not test_data:
        return None

    _resample_eval_positions(
        test_data=test_data, task_name=task_name, layers=layers,
        position_mode="all", stochastic_max_k=100, eval_position_seed=0,
    )

    print(f"    {len(test_data)} items ready")
    return test_data


# ── Activation manipulation ──

def extract_activations(model, tokenizer, test_data, layers, batch_size=4, device="cuda"):
    """Extract real activation vectors for all items."""
    all_activations = []
    for start in range(0, len(test_data), batch_size):
        chunk = test_data[start:start + batch_size]
        chunk_acts = _materialize_activations(model, tokenizer, chunk, layers=layers, device=device)
        all_activations.extend([a.cpu() for a in chunk_acts])
    return all_activations


def make_zero_activations(real_acts):
    return [torch.zeros_like(a) for a in real_acts]


def make_scrambled_activations(real_acts, seed=42):
    """Derangement: every example gets a DIFFERENT example's activations."""
    rng = random.Random(seed)
    n = len(real_acts)
    indices = list(range(n))
    for _ in range(200):
        rng.shuffle(indices)
        if all(i != j for i, j in enumerate(indices)):
            break
    return [real_acts[i] for i in indices]


# ── Oracle runs ──

def run_activation_oracle(model, tokenizer, activations, test_data, task_name,
                          adapter_name, eval_batch_size=2, device="cuda"):
    """Run oracle generation with given activation tensors."""
    task_def = TASKS[task_name]
    oracle_items = [(act.to(device), item["prompt"]) for act, item in zip(activations, test_data)]

    return _batched_oracle_generate(
        model=model, tokenizer=tokenizer, items=oracle_items, layers=LAYERS,
        device=device, injection_layer=INJECTION_LAYER,
        max_new_tokens=task_def.max_new_tokens, eval_batch_size=eval_batch_size,
        oracle_adapter_name=adapter_name,
    )


def run_text_baseline(model, tokenizer, test_data, task_name,
                      adapter_name, eval_batch_size=2, device="cuda"):
    """Run text-baseline: reads CoT text, no activation injection."""
    task_def = TASKS[task_name]
    text_items = [(item.get("cot_text", ""), item["prompt"]) for item in test_data]
    return _text_baseline_generate(
        model=model, tokenizer=tokenizer, items=text_items, device=device,
        max_new_tokens=task_def.max_new_tokens, eval_batch_size=eval_batch_size,
        oracle_adapter_name=adapter_name,
    )


# ── AUC computation via first-token logprobs ──

def compute_first_token_logprobs(model, tokenizer, activations, test_data,
                                  adapter_name, device="cuda"):
    """Get logprob(Yes) - logprob(No) at first generation position.

    Returns list of float scores (higher = more likely positive).
    """
    _ensure_ao_imports()
    get_batched_steering_hook = _ao_modules["get_batched_steering_hook"]
    get_hf_submodule = _ao_modules["get_hf_submodule"]
    add_hook = _ao_modules["add_hook"]
    PLACEHOLDER_TOKEN = _ao_modules["PLACEHOLDER_TOKEN"]

    ph_token = PLACEHOLDER_TOKEN
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Get Yes/No token IDs
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)
    model.eval()

    scores = []
    for idx in range(len(test_data)):
        act = activations[idx].to(device)
        prompt = test_data[idx]["prompt"]
        num_positions = act.shape[0]

        prefix = _build_oracle_prefix(num_positions, layers=LAYERS, placeholder_token=ph_token)
        full_prompt = prefix + prompt
        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        prefix_idx = formatted.find(prefix)
        before_ids = tokenizer.encode(formatted[:prefix_idx], add_special_tokens=False)
        after_ids = tokenizer.encode(formatted[prefix_idx + len(prefix):], add_special_tokens=False)
        prefix_ids, rel_positions = _build_manual_prefix_token_ids(
            tokenizer, num_positions, LAYERS, ph_id,
        )
        input_ids = before_ids + prefix_ids + after_ids
        positions = [len(before_ids) + p for p in rel_positions]

        input_tensor = torch.tensor([input_ids], device=device)
        attn_mask = torch.ones_like(input_tensor)

        hook_fn = get_batched_steering_hook(
            vectors=[act], positions=[positions], device=device, dtype=torch.bfloat16,
        )

        with torch.no_grad(), add_hook(injection_submodule, hook_fn):
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            score = (log_probs[yes_id] - log_probs[no_id]).item()
            scores.append(score)

    return scores


def get_binary_labels(test_data, task_name):
    """Extract ground-truth binary labels (1=positive, 0=negative, -1=unparseable)."""
    parser = TASK_PARSERS.get(task_name)
    task_def = TASKS.get(task_name)
    labels = []
    for item in test_data:
        target = item["target_response"]
        if parser:
            parsed = parser(target)
            if not parsed:
                labels.append(-1)
                continue
            lbl = parsed["label"].lower()
            # Positive labels vary by task
            if task_name == "sycophancy":
                labels.append(1 if lbl == "sycophantic" else 0)
            elif task_name == "decorative_cot":
                labels.append(1 if lbl == "decorative" else 0)
            elif task_name == "atypical_answer":
                labels.append(1 if lbl == "atypical" else 0)
            else:
                # Default: yes=1, no=0
                labels.append(1 if lbl in ("yes",) else 0)
        else:
            labels.append(-1)
    return labels


def compute_auc(scores, labels):
    """ROC AUC from continuous scores and binary labels. Filters out label=-1."""
    from sklearn.metrics import roc_auc_score
    valid = [(s, l) for s, l in zip(scores, labels) if l >= 0]
    if len(valid) < 2:
        return float("nan")
    s, l = zip(*valid)
    if len(set(l)) < 2:
        return float("nan")
    try:
        return roc_auc_score(list(l), list(s))
    except ValueError:
        return float("nan")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="AO claims quantitative evaluation")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--extract-batch-size", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=["adam", "trained"],
                        choices=["adam", "trained"])
    parser.add_argument("--conditions", nargs="+",
                        default=["real", "zero", "scrambled"],
                        choices=["real", "zero", "scrambled", "text_baseline"])
    parser.add_argument("--skip-auc", action="store_true", help="Skip AUC logprob computation")
    parser.add_argument("--output", type=str, default="data/ao_claims_eval.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tasks_to_run = args.tasks or ALL_TASKS
    print("=" * 70)
    print("AO Claims Evaluation")
    print("=" * 70)
    print(f"Tasks:       {tasks_to_run}")
    print(f"Checkpoints: {args.checkpoints}")
    print(f"Conditions:  {args.conditions}")
    print(f"Max items:   {args.max_items}")
    print()

    model, tokenizer = load_model_and_adapters(args.device)

    results = {}
    auc_results = {}

    # ── Phase 1: Load data + extract activations ──
    print("\n" + "=" * 70)
    print("PHASE 1: Load data and extract activations")
    print("=" * 70)

    task_data = {}
    task_acts = {}

    for task_name in tasks_to_run:
        td = load_and_prepare_task(task_name, tokenizer, args.max_items, LAYERS)
        if td is None:
            print(f"    SKIPPED: {task_name} (no data)")
            continue
        task_data[task_name] = td

        print(f"  Extracting activations for {task_name} ({len(td)} items)...")
        t0 = time.time()
        acts = extract_activations(
            model, tokenizer, td, LAYERS,
            batch_size=args.extract_batch_size, device=args.device,
        )
        print(f"    Done in {time.time() - t0:.1f}s")
        task_acts[task_name] = acts

    print(f"\n{len(task_data)} tasks loaded")

    # Pre-compute ablations
    task_zero = {tn: make_zero_activations(a) for tn, a in task_acts.items()}
    task_scrambled = {tn: make_scrambled_activations(a) for tn, a in task_acts.items()}

    # ── Phase 2: Run oracle under each (checkpoint, condition) ──
    print("\n" + "=" * 70)
    print("PHASE 2: Oracle evaluation")
    print("=" * 70)

    for ckpt in args.checkpoints:
        for cond in args.conditions:
            print(f"\n{'─' * 50}")
            print(f"  Checkpoint: {ckpt}  |  Condition: {cond}")
            print(f"{'─' * 50}")

            for task_name, td in task_data.items():
                task_def = TASKS[task_name]
                key = f"{ckpt}:{cond}:{task_name}"
                t0 = time.time()

                # Text baseline: skip lens tasks
                if cond == "text_baseline":
                    if task_name in ("futurelens", "pastlens"):
                        results[key] = {"n": 0, "_skipped": True}
                        print(f"    {task_name:<25} SKIP (lens + text baseline)")
                        continue
                    predictions = run_text_baseline(
                        model, tokenizer, td, task_name,
                        adapter_name=ckpt, eval_batch_size=args.eval_batch_size,
                        device=args.device,
                    )
                else:
                    acts_map = {"real": task_acts, "zero": task_zero, "scrambled": task_scrambled}
                    acts = acts_map[cond][task_name]
                    predictions = run_activation_oracle(
                        model, tokenizer, acts, td, task_name,
                        adapter_name=ckpt, eval_batch_size=args.eval_batch_size,
                        device=args.device,
                    )

                elapsed = time.time() - t0
                targets = [item["target_response"] for item in td]
                metrics = score_task(task_def, predictions, targets,
                                     tokenizer=tokenizer, eval_items=td)

                # Clean internal keys, keep numeric metrics
                clean = {k: v for k, v in metrics.items()
                         if not k.startswith("_") and isinstance(v, (int, float))}
                clean["elapsed_s"] = round(elapsed, 1)
                results[key] = clean

                # Print primary metric
                primary = clean.get("accuracy", clean.get("token_f1",
                          clean.get("step_accuracy", clean.get("token_match_rate", None))))
                n = clean.get("n", "?")
                if primary is not None:
                    print(f"    {task_name:<25} {primary:.3f}  (n={n}, {elapsed:.0f}s)")
                else:
                    print(f"    {task_name:<25} metrics={clean}")

                # Print a few sample predictions for the first condition
                if cond == args.conditions[0] and ckpt == args.checkpoints[0]:
                    for i in range(min(2, len(predictions))):
                        print(f"      pred: {predictions[i][:100]}")
                        print(f"      tgt:  {targets[i][:100]}")

    # ── Phase 3: AUC via logprobs ──
    if not args.skip_auc:
        print("\n" + "=" * 70)
        print("PHASE 3: AUC via first-token logprobs (binary tasks)")
        print("=" * 70)

        for ckpt in args.checkpoints:
            for cond in ["real", "zero"]:
                if cond not in args.conditions:
                    continue
                for task_name in BINARY_TASKS:
                    if task_name not in task_data:
                        continue

                    td = task_data[task_name]
                    labels = get_binary_labels(td, task_name)
                    valid_labels = [l for l in labels if l >= 0]
                    if len(set(valid_labels)) < 2:
                        print(f"    {ckpt}:{cond}:{task_name} — single class, skip AUC")
                        continue

                    acts = task_acts[task_name] if cond == "real" else task_zero[task_name]

                    print(f"    {ckpt}:{cond}:{task_name} — computing logprob scores...")
                    t0 = time.time()
                    scores = compute_first_token_logprobs(
                        model, tokenizer, acts, td,
                        adapter_name=ckpt, device=args.device,
                    )
                    auc = compute_auc(scores, labels)
                    elapsed = time.time() - t0

                    auc_key = f"{ckpt}:{cond}:{task_name}"
                    auc_results[auc_key] = {
                        "auc": round(auc, 4) if not np.isnan(auc) else None,
                        "n": len(valid_labels),
                        "elapsed_s": round(elapsed, 1),
                    }
                    print(f"      AUC = {auc:.4f}  (n={len(valid_labels)}, {elapsed:.0f}s)")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("RESULTS: Accuracy / Token F1")
    print("=" * 70)

    col_keys = [(ckpt, cond) for ckpt in args.checkpoints for cond in args.conditions]
    header = f"{'Task':<25}" + "".join(f" {ck[:5]}_{co[:5]:>6}" for ck, co in col_keys)
    print(header)
    print("-" * len(header))

    for task_name in ALL_TASKS:
        if task_name not in task_data:
            continue
        row = f"{task_name:<25}"
        for ckpt, cond in col_keys:
            key = f"{ckpt}:{cond}:{task_name}"
            m = results.get(key, {})
            val = m.get("accuracy", m.get("token_f1", m.get("step_accuracy", None)))
            if val is None or m.get("_skipped"):
                row += "       -"
            else:
                row += f" {val:7.1%}"
        print(row)

    # AUC summary
    if auc_results:
        print(f"\n{'Task':<25}", end="")
        for ckpt in args.checkpoints:
            for cond in ["real", "zero"]:
                print(f" {ckpt[:5]}_{cond}_AUC", end="")
        print()
        print("-" * 75)
        for task_name in BINARY_TASKS:
            if task_name not in task_data:
                continue
            row = f"{task_name:<25}"
            for ckpt in args.checkpoints:
                for cond in ["real", "zero"]:
                    key = f"{ckpt}:{cond}:{task_name}"
                    auc_val = auc_results.get(key, {}).get("auc")
                    if auc_val is not None:
                        row += f" {auc_val:>13.4f}"
                    else:
                        row += "             -"
            print(row)

    # ── Activation ablation summary ──
    print("\n" + "=" * 70)
    print("ACTIVATION ABLATION: real - zero (higher = activations help)")
    print("=" * 70)
    for ckpt in args.checkpoints:
        print(f"\n  Checkpoint: {ckpt}")
        for task_name in ALL_TASKS:
            if task_name not in task_data:
                continue
            real_m = results.get(f"{ckpt}:real:{task_name}", {})
            zero_m = results.get(f"{ckpt}:zero:{task_name}", {})
            metric = "accuracy" if task_name in BINARY_TASKS else "token_f1"
            real_v = real_m.get(metric)
            zero_v = zero_m.get(metric)
            if real_v is not None and zero_v is not None:
                delta = real_v - zero_v
                marker = "***" if delta > 0.1 else "**" if delta > 0.05 else "*" if delta > 0.02 else ""
                print(f"    {task_name:<25} real={real_v:.3f}  zero={zero_v:.3f}  Δ={delta:+.3f} {marker}")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "results": results,
        "auc_results": auc_results,
        "config": {
            "max_items": args.max_items,
            "tasks": list(task_data.keys()),
            "checkpoints": args.checkpoints,
            "conditions": args.conditions,
            "layers": LAYERS,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
