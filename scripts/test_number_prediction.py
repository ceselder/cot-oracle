#!/usr/bin/env python3
"""
Test: Can the oracle predict what number the model is computing?

From the LessWrong post: Adam's AO confabulates "10" and "12" regardless of problem.

Two conditions:
  A) "pre-answer" — activations from the CoT BEFORE the final answer appears
  B) "no-cot" — model given problem with no CoT (thinking disabled), activations
     from control tokens only

n=100 random arithmetic problems, temperature=0.5 for variety.
Computes token F1 on extracted numbers.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ao_reference"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAM_CHECKPOINT = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
FINAL_SPRINT_CHECKPOINT = "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO"
BASE_MODEL = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
INJECTION_LAYER = 1
N_PROBLEMS = 100
TEMPERATURE = 0.5


def generate_random_problems(n=100, seed=42):
    """Generate n random arithmetic problems with known answers."""
    rng = random.Random(seed)
    problems = []
    ops = ['+', '-', '*']
    for _ in range(n):
        op = rng.choice(ops)
        if op == '*':
            a, b = rng.randint(2, 50), rng.randint(2, 50)
        elif op == '+':
            a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        else:
            a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        expr = f"{a} {op} {b}"
        answer = eval(expr)
        problems.append({"problem": f"What is {expr}?", "expression": expr, "answer": answer})
    return problems


def load_model_and_adapters():
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading Adam's adapter...")
    model = PeftModel.from_pretrained(
        base, ADAM_CHECKPOINT, adapter_name="adam",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"Loading final-sprint adapter...")
    model.load_adapter(
        FINAL_SPRINT_CHECKPOINT, adapter_name="trained",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, problem_text, use_cot=True, temperature=0.0):
    """Generate a response. Returns (response_text, full_ids, prompt_len)."""
    if use_cot:
        content = problem_text + " Think step by step."
    else:
        content = problem_text + " Answer with just the number, nothing else."

    messages = [{"role": "user", "content": content}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=512 if use_cot else 32,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with model.disable_adapter(), torch.no_grad():
        model.eval()
        outputs = model.generate(input_ids, **gen_kwargs)

    full_ids = outputs[0].tolist()
    response = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
    return response, full_ids, prompt_len


def find_answer_boundary(tokenizer, full_ids, prompt_len, answer):
    """Find the FIRST token position where the final answer starts appearing.

    We scan from the end backwards to find the final answer mention, then
    return the position just BEFORE it (so activations don't see the answer).
    """
    answer_str = str(answer)
    # Build cumulative decoded text from each position
    response_ids = full_ids[prompt_len:]

    # Find last occurrence of the answer in the response
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    last_idx = response_text.rfind(answer_str)

    if last_idx < 0:
        return None

    # Find which token position corresponds to this character position
    char_count = 0
    for i, tid in enumerate(response_ids):
        token_text = tokenizer.decode([tid])
        char_count += len(token_text)
        if char_count > last_idx:
            # This token contains the start of the answer
            return prompt_len + i
    return None


def extract_activations_at_positions(model, tokenizer, full_ids, positions, layers):
    """Extract activations at specific positions across multiple layers."""
    from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

    input_tensor = torch.tensor([full_ids], device="cuda")
    attn_mask = torch.ones_like(input_tensor)
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers}

    with model.disable_adapter(), torch.no_grad():
        model.eval()
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules=submodules,
            inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
            min_offset=None, max_offset=None,
        )

    vectors = []
    for layer in layers:
        layer_acts = acts_by_layer[layer]
        for pos in positions:
            vectors.append(layer_acts[0, pos, :].detach())

    if not vectors:
        return None
    return torch.stack(vectors, dim=0)


def query_oracle(model, tokenizer, activations, question, adapter_name, layers):
    """Query the oracle with given activations and question."""
    from core.ao import get_batched_steering_hook, TRAINED_PLACEHOLDER
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.steering_hooks import add_hook
    from eval_loop import _build_manual_prefix_token_ids

    ph_token = TRAINED_PLACEHOLDER
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    num_positions = activations.shape[0]
    prefix_ids, rel_positions = _build_manual_prefix_token_ids(
        tokenizer, num_positions, layers, ph_id,
    )

    messages = [{"role": "user", "content": "PLACEHOLDER"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ph_idx = formatted.find("PLACEHOLDER")
    before_ids = tokenizer.encode(formatted[:ph_idx], add_special_tokens=False)
    after_ids = tokenizer.encode(formatted[ph_idx + len("PLACEHOLDER"):], add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    input_ids = before_ids + prefix_ids + question_ids + after_ids
    positions = [len(before_ids) + p for p in rel_positions]

    input_tensor = torch.tensor([input_ids], device="cuda")
    attn_mask = torch.ones_like(input_tensor)

    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)

    hook_fn = get_batched_steering_hook(
        vectors=[activations.to("cuda")],
        positions=[positions],
        device="cuda", dtype=torch.bfloat16,
    )

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        model.eval()
        outputs = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=32, do_sample=False, pad_token_id=pad_id,
        )

    response_ids = outputs[0][len(input_ids):]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def extract_number(text):
    """Extract the first number from oracle response text."""
    # Match negative numbers, decimals, integers
    m = re.search(r'-?\d+\.?\d*', text)
    if m:
        try:
            val = float(m.group())
            if val == int(val):
                return int(val)
            return val
        except ValueError:
            pass
    return None


def number_token_f1(predicted, target):
    """Token-level F1 between predicted and target number strings."""
    pred_str = str(predicted) if predicted is not None else ""
    tgt_str = str(target)
    pred_chars = list(pred_str)
    tgt_chars = list(tgt_str)
    if not pred_chars or not tgt_chars:
        return 0.0
    # Character-level F1
    common = 0
    tgt_remaining = list(tgt_chars)
    for c in pred_chars:
        if c in tgt_remaining:
            common += 1
            tgt_remaining.remove(c)
    if common == 0:
        return 0.0
    precision = common / len(pred_chars)
    recall = common / len(tgt_chars)
    return 2 * precision * recall / (precision + recall)


def main():
    model, tokenizer = load_model_and_adapters()

    problems = generate_random_problems(N_PROBLEMS)
    question = "What number is the model thinking about?"

    results = []

    print(f"\n{'=' * 80}")
    print(f"NUMBER PREDICTION TEST (n={N_PROBLEMS}, temperature={TEMPERATURE})")
    print(f"{'=' * 80}")

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        true_answer = prob["answer"]

        # Generate CoT response
        response, full_ids, prompt_len = generate_response(
            model, tokenizer, problem_text, use_cot=True, temperature=TEMPERATURE,
        )

        # Find where the final answer appears, use activations BEFORE it
        answer_boundary = find_answer_boundary(tokenizer, full_ids, prompt_len, true_answer)

        if answer_boundary is not None:
            # Use 5 positions ending 2 tokens before the answer
            end_pos = max(prompt_len + 1, answer_boundary - 2)
            start_pos = max(prompt_len, end_pos - 5)
            positions = list(range(start_pos, end_pos))
            condition = "pre-answer"
        else:
            # Answer not found in response — use last 5 tokens
            resp_end = len(full_ids)
            positions = list(range(max(prompt_len, resp_end - 5), resp_end))
            condition = "end-of-response"

        if not positions:
            positions = [prompt_len]

        # Extract activations
        activations = extract_activations_at_positions(model, tokenizer, full_ids, positions, LAYERS)
        if activations is None:
            continue

        # Query both oracles
        entry = {
            "i": i, "problem": problem_text, "expression": prob["expression"],
            "true_answer": true_answer, "condition": condition,
            "n_positions": len(positions),
            "response_snippet": response[:200],
        }

        for adapter in ["adam", "trained"]:
            resp = query_oracle(model, tokenizer, activations, question, adapter, LAYERS)
            predicted_num = extract_number(resp)
            f1 = number_token_f1(predicted_num, true_answer)
            exact = 1 if predicted_num == true_answer else 0

            entry[f"{adapter}_response"] = resp
            entry[f"{adapter}_predicted"] = predicted_num
            entry[f"{adapter}_f1"] = f1
            entry[f"{adapter}_exact"] = exact

        results.append(entry)

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            adam_f1s = [r["adam_f1"] for r in results]
            trained_f1s = [r["trained_f1"] for r in results]
            adam_exact = [r["adam_exact"] for r in results]
            trained_exact = [r["trained_exact"] for r in results]
            print(f"  [{i+1:>3}/{N_PROBLEMS}]  "
                  f"Adam: F1={np.mean(adam_f1s):.3f} exact={np.mean(adam_exact):.3f}  |  "
                  f"Ours: F1={np.mean(trained_f1s):.3f} exact={np.mean(trained_exact):.3f}  |  "
                  f"last: {prob['expression']}={true_answer}, "
                  f"adam={entry['adam_predicted']}, ours={entry['trained_predicted']}")

    # ── Final metrics ──
    adam_f1s = [r["adam_f1"] for r in results]
    trained_f1s = [r["trained_f1"] for r in results]
    adam_exact = [r["adam_exact"] for r in results]
    trained_exact = [r["trained_exact"] for r in results]

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS (n={len(results)})")
    print(f"{'=' * 80}")
    print(f"  Adam:    Token F1 = {np.mean(adam_f1s):.3f} ± {np.std(adam_f1s):.3f}   Exact match = {np.mean(adam_exact):.3f}")
    print(f"  Ours:    Token F1 = {np.mean(trained_f1s):.3f} ± {np.std(trained_f1s):.3f}   Exact match = {np.mean(trained_exact):.3f}")
    print(f"  Δ F1:    {np.mean(trained_f1s) - np.mean(adam_f1s):+.3f}")
    print(f"  Δ Exact: {np.mean(trained_exact) - np.mean(adam_exact):+.3f}")

    # ── Bar chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Token F1 bar chart
        ax = axes[0]
        labels = ["Adam's AO", "Ours (final sprint)"]
        f1_means = [np.mean(adam_f1s), np.mean(trained_f1s)]
        f1_stds = [np.std(adam_f1s) / np.sqrt(len(adam_f1s)),
                    np.std(trained_f1s) / np.sqrt(len(trained_f1s))]
        bars = ax.bar(labels, f1_means, yerr=f1_stds, capsize=5,
                      color=["#ff6b6b", "#51cf66"], alpha=0.85, edgecolor="black")
        ax.set_ylabel("Token F1")
        ax.set_title("Number Prediction: Token F1\n(pre-answer activations)")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, f1_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

        # Exact match bar chart
        ax = axes[1]
        exact_means = [np.mean(adam_exact), np.mean(trained_exact)]
        exact_stds = [np.std(adam_exact) / np.sqrt(len(adam_exact)),
                      np.std(trained_exact) / np.sqrt(len(trained_exact))]
        bars = ax.bar(labels, exact_means, yerr=exact_stds, capsize=5,
                      color=["#ff6b6b", "#51cf66"], alpha=0.85, edgecolor="black")
        ax.set_ylabel("Exact Match Rate")
        ax.set_title("Number Prediction: Exact Match\n(pre-answer activations)")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, exact_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        chart_path = "data/number_prediction_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {chart_path}")
        plt.close()
    except ImportError:
        print("\n(matplotlib not available, skipping chart)")

    # Save full results
    output_path = "data/number_prediction_test.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {"n": N_PROBLEMS, "temperature": TEMPERATURE, "layers": LAYERS},
            "summary": {
                "adam_f1_mean": round(float(np.mean(adam_f1s)), 4),
                "adam_f1_std": round(float(np.std(adam_f1s)), 4),
                "adam_exact_mean": round(float(np.mean(adam_exact)), 4),
                "trained_f1_mean": round(float(np.mean(trained_f1s)), 4),
                "trained_f1_std": round(float(np.std(trained_f1s)), 4),
                "trained_exact_mean": round(float(np.mean(trained_exact)), 4),
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
