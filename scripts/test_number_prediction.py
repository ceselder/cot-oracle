#!/usr/bin/env python3
"""
Replicate the number prediction test from:
"Current activation oracles are hard to use" (LessWrong, March 2026)

Their setup:
  - Arithmetic problems with NO chain of thought (think tags closed)
  - Activations from control tokens only (<|im_start|>assistant)
  - OR from the full user prompt
  - Questions: "What number is the model about to produce?"
  - Result: AO confabulates "10" and "12" regardless of problem

We replicate this exactly, comparing Adam's AO vs our final sprint.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter
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


def generate_random_problems(n=100, seed=42):
    rng = random.Random(seed)

    # Start with examples from/inspired by the blog post
    problems = [
        {"problem": "What is -93 + (-42 % 162) + -8369?", "expression": "-93 + (-42 % 162) + -8369", "answer": -93 + (-42 % 162) + -8369},
        {"problem": "What is (44 // -49) % ((15 - 51) * 25)?", "expression": "(44 // -49) % ((15 - 51) * 25)", "answer": (44 // -49) % ((15 - 51) * 25)},
        {"problem": "What is (-60 * -44) + (-73 + -76)?", "expression": "(-60 * -44) + (-73 + -76)", "answer": (-60 * -44) + (-73 + -76)},
        {"problem": "What is 87 - 38 + 68 - (-79 + 42)?", "expression": "87 - 38 + 68 - (-79 + 42)", "answer": 87 - 38 + 68 - (-79 + 42)},
        {"problem": "What is 1 + 1?", "expression": "1 + 1", "answer": 2},
    ]

    # Fill rest with random arithmetic
    ops = ['+', '-', '*']
    while len(problems) < n:
        op = rng.choice(ops)
        if op == '*':
            a, b = rng.randint(2, 50), rng.randint(2, 50)
        else:
            a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        expr = f"{a} {op} {b}"
        answer = eval(expr)
        problems.append({"problem": f"What is {expr}?", "expression": expr, "answer": answer})
    return problems[:n]


def load_model_and_adapters():
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("Loading Adam's adapter...")
    model = PeftModel.from_pretrained(
        base, ADAM_CHECKPOINT, adapter_name="adam",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print("Loading final-sprint adapter...")
    model.load_adapter(
        FINAL_SPRINT_CHECKPOINT, adapter_name="trained",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def build_no_cot_input(tokenizer, problem_text):
    """Build input with no CoT — thinking disabled, model outputs answer directly.

    Returns (full_ids_list, prompt_len, control_token_positions, user_prompt_positions).
    """
    # Format: <|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n
    messages = [{"role": "user", "content": problem_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tokenizer.encode(formatted, add_special_tokens=False)

    # Find control tokens: <|im_start|>assistant\n at the end
    # These are the last few tokens before generation starts
    assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    # Find where assistant marker starts
    control_start = None
    for i in range(len(full_ids) - len(assistant_marker), -1, -1):
        if full_ids[i:i+len(assistant_marker)] == assistant_marker:
            control_start = i
            break

    if control_start is None:
        # Fallback: last 3 tokens
        control_positions = list(range(max(0, len(full_ids) - 3), len(full_ids)))
    else:
        control_positions = list(range(control_start, len(full_ids)))

    # User prompt positions: tokens between <|im_start|>user and <|im_end|>
    user_start = len(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
    user_end = control_start if control_start else len(full_ids) - 3
    user_positions = list(range(user_start, user_end))

    return full_ids, len(full_ids), control_positions, user_positions


def generate_no_cot_answer(model, tokenizer, problem_text):
    """Generate answer with no CoT (direct answer)."""
    messages = [{"role": "user", "content": problem_text + " Answer with just the number."}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    with model.disable_adapter(), torch.no_grad():
        model.eval()
        outputs = model.generate(
            input_ids, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def extract_activations_at_positions(model, full_ids, positions, layers):
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
        for pos in positions:
            vectors.append(acts_by_layer[layer][0, pos, :].detach())
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
        vectors=[activations.to("cuda")], positions=[positions],
        device="cuda", dtype=torch.bfloat16,
    )
    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        model.eval()
        outputs = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=32, do_sample=False, pad_token_id=pad_id,
        )
    return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)


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
    pred_str = str(predicted) if predicted is not None else ""
    tgt_str = str(target)
    if not pred_str or not tgt_str:
        return 0.0
    tgt_remaining = list(tgt_str)
    common = 0
    for c in pred_str:
        if c in tgt_remaining:
            common += 1
            tgt_remaining.remove(c)
    if common == 0:
        return 0.0
    precision = common / len(pred_str)
    recall = common / len(tgt_str)
    return 2 * precision * recall / (precision + recall)


def main():
    model, tokenizer = load_model_and_adapters()
    problems = generate_random_problems(N_PROBLEMS)

    # Exact questions from Arya's post (Table in "Number prediction" section)
    questions = [
        "About to produce a number?",
        "Planning to answer?",
    ]

    # Two conditions from the post: control tokens only, full user prompt
    conditions = ["control_tokens", "full_prompt"]

    results = []

    print(f"\n{'=' * 80}")
    print(f"NUMBER PREDICTION — REPLICATING BLOGPOST (n={N_PROBLEMS})")
    print(f"No chain of thought. Control tokens or full prompt activations.")
    print(f"{'=' * 80}\n")

    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        true_answer = prob["answer"]

        # Build input (no CoT)
        full_ids, prompt_len, control_positions, user_positions = \
            build_no_cot_input(tokenizer, problem_text)

        # Also get the model's actual answer for reference
        model_answer = generate_no_cot_answer(model, tokenizer, problem_text)

        entry = {
            "i": i, "expression": prob["expression"], "true_answer": true_answer,
            "model_answer": model_answer,
            "n_control_tokens": len(control_positions),
            "n_user_tokens": len(user_positions),
        }

        for cond in conditions:
            positions = control_positions if cond == "control_tokens" else user_positions
            if not positions:
                continue

            activations = extract_activations_at_positions(model, full_ids, positions, LAYERS)
            if activations is None:
                continue

            for q in questions:
                for adapter in ["adam", "trained"]:
                    resp = query_oracle(model, tokenizer, activations, q, adapter, LAYERS)
                    predicted = extract_number(resp)
                    f1 = number_token_f1(predicted, true_answer)
                    exact = 1 if predicted == true_answer else 0

                    col = f"{adapter}_{cond}_{questions.index(q)}"
                    entry[f"{col}_response"] = resp
                    entry[f"{col}_predicted"] = predicted
                    entry[f"{col}_f1"] = f1
                    entry[f"{col}_exact"] = exact

        results.append(entry)

        if (i + 1) % 10 == 0 or i == 0:
            # Print progress for control_tokens, first question
            for cond in conditions:
                adam_f1 = np.mean([r.get(f"adam_{cond}_0_f1", 0) for r in results])
                ours_f1 = np.mean([r.get(f"trained_{cond}_0_f1", 0) for r in results])
                adam_ex = np.mean([r.get(f"adam_{cond}_0_exact", 0) for r in results])
                ours_ex = np.mean([r.get(f"trained_{cond}_0_exact", 0) for r in results])
                print(f"  [{i+1:>3}/{N_PROBLEMS}] {cond:>15}  "
                      f"Adam: F1={adam_f1:.3f} ex={adam_ex:.3f}  |  "
                      f"Ours: F1={ours_f1:.3f} ex={ours_ex:.3f}")

    # ── Final metrics ──
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS (n={len(results)})")
    print(f"{'=' * 80}")

    summary = {}
    for cond in conditions:
        for qi, q in enumerate(questions):
            print(f"\n  Condition: {cond}")
            print(f"  Question:  \"{q}\"")
            for adapter in ["adam", "trained"]:
                col = f"{adapter}_{cond}_{qi}"
                f1s = [r.get(f"{col}_f1", 0) for r in results]
                exacts = [r.get(f"{col}_exact", 0) for r in results]
                print(f"    {adapter:>7}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  Exact={np.mean(exacts):.3f}")
                summary[f"{col}_f1"] = round(float(np.mean(f1s)), 4)
                summary[f"{col}_exact"] = round(float(np.mean(exacts)), 4)

    # ── Confabulation analysis ──
    print(f"\n{'=' * 80}")
    print(f"CONFABULATION ANALYSIS")
    print(f"{'=' * 80}")
    for adapter in ["adam", "trained"]:
        preds = [r.get(f"{adapter}_control_tokens_0_predicted") for r in results
                 if r.get(f"{adapter}_control_tokens_0_predicted") is not None]
        counter = Counter(preds)
        top = counter.most_common(10)
        print(f"\n  {adapter} (control tokens) top-10 predictions:")
        for num, count in top:
            print(f"    {num:>8}: {count}x ({count/len(preds)*100:.0f}%)")

    # ── Sample outputs ──
    print(f"\n{'=' * 80}")
    print(f"SAMPLE OUTPUTS (first 10, control tokens)")
    print(f"{'=' * 80}")
    print(f"{'Expression':<20} {'True':>6} {'Model':>8} {'Adam':>8} {'Ours':>8}")
    print("-" * 55)
    for r in results[:10]:
        adam_p = r.get("adam_control_tokens_0_predicted", "?")
        ours_p = r.get("trained_control_tokens_0_predicted", "?")
        model_a = extract_number(r.get("model_answer", ""))
        print(f"{r['expression']:<20} {r['true_answer']:>6} {str(model_a):>8} {str(adam_p):>8} {str(ours_p):>8}")

    # ── Bar chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, cond in enumerate(conditions):
            ax = axes[ax_idx]
            cond_label = "Control Tokens Only" if cond == "control_tokens" else "Full User Prompt"

            # Use first question
            adam_f1s = [r.get(f"adam_{cond}_0_f1", 0) for r in results]
            ours_f1s = [r.get(f"trained_{cond}_0_f1", 0) for r in results]
            adam_exacts = [r.get(f"adam_{cond}_0_exact", 0) for r in results]
            ours_exacts = [r.get(f"trained_{cond}_0_exact", 0) for r in results]

            x = np.arange(2)
            width = 0.35
            f1_vals = [np.mean(adam_f1s), np.mean(ours_f1s)]
            exact_vals = [np.mean(adam_exacts), np.mean(ours_exacts)]
            f1_errs = [np.std(adam_f1s)/np.sqrt(len(adam_f1s)),
                       np.std(ours_f1s)/np.sqrt(len(ours_f1s))]

            b1 = ax.bar(x - width/2, f1_vals, width, yerr=f1_errs, label="Char F1",
                        color=["#ff6b6b", "#51cf66"], alpha=0.85, capsize=4, edgecolor="black")
            b2 = ax.bar(x + width/2, exact_vals, width, label="Exact Match",
                        color=["#ffa8a8", "#8ce99a"], alpha=0.85, edgecolor="black")

            ax.set_xticks(x)
            ax.set_xticklabels(["Adam's AO\n(original)", "Ours\n(final sprint)"])
            ax.set_ylabel("Score")
            ax.set_title(f"{cond_label}")
            ax.set_ylim(0, max(0.6, max(f1_vals) + 0.1))
            ax.legend(loc="upper right")

            for bar in b1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            for bar in b2:
                if bar.get_height() > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.suptitle("Number Prediction from Pre-Answer Activations (No CoT)\n"
                     "Q: \"About to produce a number?\" (from Arya et al.)",
                     fontweight="bold", fontsize=13)
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
    with open(output_path, "w") as f:
        json.dump({"config": {"n": N_PROBLEMS, "layers": LAYERS, "conditions": conditions},
                   "summary": summary, "results": results}, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
