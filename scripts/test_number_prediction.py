#!/usr/bin/env python3
"""
Test: Can the oracle predict what number the model is computing?

From the LessWrong post: Adam's AO confabulates "10" and "12" regardless of problem.

Uses SPARSE positions across the full CoT (stride-20), not dense windows.
This avoids text inversion of nearby tokens.

n=100 random arithmetic problems, temperature=0.5 for variety.
Computes token F1 on extracted numbers.
"""

from __future__ import annotations

import json
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
STRIDE = 20  # sparse positions across CoT


def generate_random_problems(n=100, seed=42):
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


def generate_response(model, tokenizer, problem_text, temperature=0.0):
    content = problem_text + " Think step by step."
    messages = [{"role": "user", "content": content}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = input_ids.shape[1]
    gen_kwargs = dict(max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
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


def get_sparse_positions(prompt_len, total_len, stride=20):
    """Evenly-spaced positions across the CoT region."""
    cot_region = list(range(prompt_len, total_len))
    if not cot_region:
        return [prompt_len]
    positions = cot_region[::stride]
    # Always include the last position
    if cot_region[-1] not in positions:
        positions.append(cot_region[-1])
    return positions


def extract_activations_at_positions(model, tokenizer, full_ids, positions, layers):
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
    m = re.search(r'-?\d+\.?\d*', text)
    if m:
        try:
            val = float(m.group())
            return int(val) if val == int(val) else val
        except ValueError:
            pass
    return None


def number_token_f1(predicted, target):
    """Char-level F1 between the predicted and target number strings."""
    pred_str = str(predicted) if predicted is not None else ""
    tgt_str = str(target)
    if not pred_str or not tgt_str:
        return 0.0
    pred_chars = list(pred_str)
    tgt_chars = list(tgt_str)
    tgt_remaining = list(tgt_chars)
    common = 0
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

    # Two questions: generic + our trained prompt style
    questions = {
        "generic": "What number is the model thinking about?",
        "answer_trajectory": "What answer is the model converging toward? State the numerical answer.",
    }

    results = []

    print(f"\n{'=' * 80}")
    print(f"NUMBER PREDICTION TEST (n={N_PROBLEMS}, temp={TEMPERATURE}, stride={STRIDE})")
    print(f"Sparse positions across full CoT to avoid text inversion")
    print(f"{'=' * 80}")

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        true_answer = prob["answer"]

        # Generate CoT response
        response, full_ids, prompt_len = generate_response(
            model, tokenizer, problem_text, temperature=TEMPERATURE,
        )

        # Sparse positions across CoT
        positions = get_sparse_positions(prompt_len, len(full_ids), stride=STRIDE)
        n_pos = len(positions)

        # Show what tokens we're actually feeding
        pos_tokens = [tokenizer.decode([full_ids[p]]) for p in positions[:5]]

        # Extract activations
        activations = extract_activations_at_positions(model, tokenizer, full_ids, positions, LAYERS)
        if activations is None:
            continue

        entry = {
            "i": i, "problem": problem_text, "expression": prob["expression"],
            "true_answer": true_answer, "n_positions": n_pos,
            "response_snippet": response[:300],
            "sample_tokens": pos_tokens,
        }

        for q_key, question in questions.items():
            for adapter in ["adam", "trained"]:
                resp = query_oracle(model, tokenizer, activations, question, adapter, LAYERS)
                predicted_num = extract_number(resp)
                f1 = number_token_f1(predicted_num, true_answer)
                exact = 1 if predicted_num == true_answer else 0

                col = f"{adapter}_{q_key}"
                entry[f"{col}_response"] = resp
                entry[f"{col}_predicted"] = predicted_num
                entry[f"{col}_f1"] = f1
                entry[f"{col}_exact"] = exact

        results.append(entry)

        if (i + 1) % 10 == 0 or i == 0:
            for q_key in questions:
                adam_f1s = [r[f"adam_{q_key}_f1"] for r in results]
                trained_f1s = [r[f"trained_{q_key}_f1"] for r in results]
                adam_exact = [r[f"adam_{q_key}_exact"] for r in results]
                trained_exact = [r[f"trained_{q_key}_exact"] for r in results]
                print(f"  [{i+1:>3}/{N_PROBLEMS}] q={q_key[:15]:>15}  "
                      f"Adam: F1={np.mean(adam_f1s):.3f} ex={np.mean(adam_exact):.3f}  |  "
                      f"Ours: F1={np.mean(trained_f1s):.3f} ex={np.mean(trained_exact):.3f}")

    # ── Final metrics ──
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS (n={len(results)}, stride={STRIDE})")
    print(f"{'=' * 80}")

    for q_key, question in questions.items():
        adam_f1s = [r[f"adam_{q_key}_f1"] for r in results]
        trained_f1s = [r[f"trained_{q_key}_f1"] for r in results]
        adam_exact = [r[f"adam_{q_key}_exact"] for r in results]
        trained_exact = [r[f"trained_{q_key}_exact"] for r in results]

        print(f"\n  Question: \"{question}\"")
        print(f"  Adam:    F1 = {np.mean(adam_f1s):.3f} ± {np.std(adam_f1s):.3f}   Exact = {np.mean(adam_exact):.3f}")
        print(f"  Ours:    F1 = {np.mean(trained_f1s):.3f} ± {np.std(trained_f1s):.3f}   Exact = {np.mean(trained_exact):.3f}")
        print(f"  Δ F1 = {np.mean(trained_f1s) - np.mean(adam_f1s):+.3f}   Δ Exact = {np.mean(trained_exact) - np.mean(adam_exact):+.3f}")

    # ── Confabulation analysis ──
    print(f"\n{'=' * 80}")
    print(f"CONFABULATION ANALYSIS (what numbers does each oracle default to?)")
    print(f"{'=' * 80}")
    for adapter in ["adam", "trained"]:
        predictions = [r[f"{adapter}_generic_predicted"] for r in results if r[f"{adapter}_generic_predicted"] is not None]
        from collections import Counter
        counter = Counter(predictions)
        top10 = counter.most_common(10)
        print(f"\n  {adapter} top-10 predicted numbers:")
        for num, count in top10:
            print(f"    {num:>8}: {count}x ({count/len(predictions)*100:.0f}%)")

    # ── Bar chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_idx, q_key in enumerate(questions):
            ax = axes[ax_idx]
            adam_f1s = [r[f"adam_{q_key}_f1"] for r in results]
            trained_f1s = [r[f"trained_{q_key}_f1"] for r in results]
            adam_exact = [r[f"adam_{q_key}_exact"] for r in results]
            trained_exact = [r[f"trained_{q_key}_exact"] for r in results]

            x = np.arange(2)
            width = 0.35
            f1_vals = [np.mean(adam_f1s), np.mean(trained_f1s)]
            exact_vals = [np.mean(adam_exact), np.mean(trained_exact)]
            f1_errs = [np.std(adam_f1s)/np.sqrt(len(adam_f1s)),
                       np.std(trained_f1s)/np.sqrt(len(trained_f1s))]
            exact_errs = [np.std(adam_exact)/np.sqrt(len(adam_exact)),
                          np.std(trained_exact)/np.sqrt(len(trained_exact))]

            b1 = ax.bar(x - width/2, f1_vals, width, yerr=f1_errs, label="Token F1",
                        color=["#ff6b6b", "#51cf66"], alpha=0.85, capsize=4, edgecolor="black")
            b2 = ax.bar(x + width/2, exact_vals, width, yerr=exact_errs, label="Exact Match",
                        color=["#ffa8a8", "#8ce99a"], alpha=0.85, capsize=4, edgecolor="black")

            ax.set_xticks(x)
            ax.set_xticklabels(["Adam's AO", "Ours"])
            ax.set_ylabel("Score")
            ax.set_title(f"Q: \"{q_key}\"")
            ax.set_ylim(0, 1)
            ax.legend()

            for bar in b1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
            for bar in b2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"Number Prediction from Sparse CoT Activations (stride={STRIDE}, n={N_PROBLEMS})", fontweight="bold")
        plt.tight_layout()
        chart_path = "data/number_prediction_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {chart_path}")
        plt.close()
    except ImportError:
        print("\n(matplotlib not available, skipping chart)")

    # Save
    output_path = "data/number_prediction_test.json"
    os.makedirs("data", exist_ok=True)
    summary = {}
    for q_key in questions:
        for adapter in ["adam", "trained"]:
            f1s = [r[f"{adapter}_{q_key}_f1"] for r in results]
            exacts = [r[f"{adapter}_{q_key}_exact"] for r in results]
            summary[f"{adapter}_{q_key}_f1"] = round(float(np.mean(f1s)), 4)
            summary[f"{adapter}_{q_key}_exact"] = round(float(np.mean(exacts)), 4)

    with open(output_path, "w") as f:
        json.dump({
            "config": {"n": N_PROBLEMS, "temperature": TEMPERATURE, "stride": STRIDE, "layers": LAYERS},
            "summary": summary,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
