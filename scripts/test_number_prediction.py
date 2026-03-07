#!/usr/bin/env python3
"""
Replicate the number prediction test from:
"Current activation oracles are hard to use" (LessWrong, March 2026)

Key fix: Adam's AO was trained on single-layer (50% = layer 16 for 8B),
ours was trained on multi-layer [9, 18, 27]. Each gets its native format.

Tests with CoT: model solves arithmetic step by step,
we extract activations from the full CoT at stride-5 positions.
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

# Adam: single layer at 50% (layer 16 for 8B)
ADAM_LAYERS = [16]
# Ours: multi-layer [9, 18, 27]
OUR_LAYERS = [9, 18, 27]

N_PROBLEMS = 100
STRIDE = 5

QUESTIONS = [
    "About to produce a number?",
    "Planning to answer?",
]


def generate_random_problems(n=100, seed=42):
    rng = random.Random(seed)
    problems = [
        {"problem": "What is -93 + (-42 % 162) + -8369?", "expression": "-93+(-42%162)+-8369",
         "answer": -93 + (-42 % 162) + -8369},
        {"problem": "What is (-60 * -44) + (-73 + -76)?", "expression": "(-60*-44)+(-73+-76)",
         "answer": (-60 * -44) + (-73 + -76)},
        {"problem": "What is 87 - 38 + 68 - (-79 + 42)?", "expression": "87-38+68-(-79+42)",
         "answer": 87 - 38 + 68 - (-79 + 42)},
        {"problem": "What is 1 + 1?", "expression": "1+1", "answer": 2},
    ]
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


def generate_cot(model, tokenizer, problem_text):
    messages = [{"role": "user", "content": problem_text + " Think step by step."}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = input_ids.shape[1]
    with model.disable_adapter(), torch.no_grad():
        model.eval()
        outputs = model.generate(input_ids, max_new_tokens=512, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    full_ids = outputs[0].tolist()
    response = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
    return response, full_ids, prompt_len


def get_stride_positions(prompt_len, total_len, stride=5):
    positions = list(range(prompt_len, total_len, stride))
    if not positions:
        positions = [prompt_len]
    return positions


def extract_activations(model, full_ids, positions, layers):
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
    return torch.stack(vectors, dim=0) if vectors else None


def query_oracle(model, tokenizer, activations, question, adapter_name, layers):
    from core.ao import get_batched_steering_hook, SPECIAL_TOKEN
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.steering_hooks import add_hook
    from eval_loop import _build_manual_prefix_token_ids

    ph_token = SPECIAL_TOKEN
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
    injection_submodule = get_hf_submodule(model, 1, use_lora=True)

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
    results = []

    print(f"\n{'=' * 80}")
    print(f"NUMBER PREDICTION (n={N_PROBLEMS}, with CoT, stride={STRIDE})")
    print(f"Adam: single layer 16 (50%). Ours: layers [9,18,27].")
    print(f"{'=' * 80}\n")

    for i, prob in enumerate(problems):
        response, full_ids, prompt_len = generate_cot(model, tokenizer, prob["problem"])
        positions = get_stride_positions(prompt_len, len(full_ids), stride=STRIDE)

        entry = {"i": i, "expression": prob["expression"], "true_answer": prob["answer"],
                 "n_positions": len(positions), "response_len": len(full_ids) - prompt_len}

        adam_acts = extract_activations(model, full_ids, positions, ADAM_LAYERS)
        our_acts = extract_activations(model, full_ids, positions, OUR_LAYERS)

        if adam_acts is None or our_acts is None:
            continue

        for qi, q in enumerate(QUESTIONS):
            adam_resp = query_oracle(model, tokenizer, adam_acts, q, "adam", ADAM_LAYERS)
            adam_num = extract_number(adam_resp)
            entry[f"adam_{qi}_response"] = adam_resp
            entry[f"adam_{qi}_predicted"] = adam_num
            entry[f"adam_{qi}_f1"] = number_token_f1(adam_num, prob["answer"])
            entry[f"adam_{qi}_exact"] = 1 if adam_num == prob["answer"] else 0

            our_resp = query_oracle(model, tokenizer, our_acts, q, "trained", OUR_LAYERS)
            our_num = extract_number(our_resp)
            entry[f"trained_{qi}_response"] = our_resp
            entry[f"trained_{qi}_predicted"] = our_num
            entry[f"trained_{qi}_f1"] = number_token_f1(our_num, prob["answer"])
            entry[f"trained_{qi}_exact"] = 1 if our_num == prob["answer"] else 0

        results.append(entry)

        if (i + 1) % 10 == 0 or i == 0:
            for qi, q in enumerate(QUESTIONS):
                af = np.mean([r[f"adam_{qi}_f1"] for r in results])
                of = np.mean([r[f"trained_{qi}_f1"] for r in results])
                ae = np.mean([r[f"adam_{qi}_exact"] for r in results])
                oe = np.mean([r[f"trained_{qi}_exact"] for r in results])
                print(f"  [{i+1:>3}/{N_PROBLEMS}] q{qi}  "
                      f"Adam: F1={af:.3f} ex={ae:.3f}  |  Ours: F1={of:.3f} ex={oe:.3f}")

    # Final
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS (n={len(results)})")
    print(f"{'=' * 80}")
    summary = {}
    for qi, q in enumerate(QUESTIONS):
        af = [r[f"adam_{qi}_f1"] for r in results]
        of = [r[f"trained_{qi}_f1"] for r in results]
        ae = [r[f"adam_{qi}_exact"] for r in results]
        oe = [r[f"trained_{qi}_exact"] for r in results]
        print(f"\n  Q: \"{q}\"")
        print(f"    Adam (L16):       F1={np.mean(af):.3f}±{np.std(af):.3f}  Exact={np.mean(ae):.3f}")
        print(f"    Ours (L9,18,27):  F1={np.mean(of):.3f}±{np.std(of):.3f}  Exact={np.mean(oe):.3f}")
        summary[f"adam_q{qi}_f1"] = round(float(np.mean(af)), 4)
        summary[f"adam_q{qi}_exact"] = round(float(np.mean(ae)), 4)
        summary[f"trained_q{qi}_f1"] = round(float(np.mean(of)), 4)
        summary[f"trained_q{qi}_exact"] = round(float(np.mean(oe)), 4)

    # Confabulation
    print(f"\n{'=' * 80}")
    print(f"CONFABULATION ANALYSIS (Q: \"{QUESTIONS[0]}\")")
    print(f"{'=' * 80}")
    for adapter, layers_str in [("adam", "L16"), ("trained", "L9,18,27")]:
        preds = [r.get(f"{adapter}_0_predicted") for r in results
                 if r.get(f"{adapter}_0_predicted") is not None]
        top = Counter(preds).most_common(10)
        print(f"\n  {adapter} ({layers_str}) top-10:")
        for num, count in top:
            print(f"    {num:>8}: {count}x ({count/len(preds)*100:.0f}%)")

    # Samples
    print(f"\n{'=' * 80}")
    print(f"SAMPLES (first 15)")
    print(f"{'=' * 80}")
    print(f"{'Expression':<25} {'True':>6} {'Adam(L16)':>12} {'Ours(L9,18,27)':>15}")
    print("-" * 62)
    for r in results[:15]:
        print(f"{r['expression']:<25} {r['true_answer']:>6} "
              f"{str(r.get('adam_0_predicted','?')):>12} "
              f"{str(r.get('trained_0_predicted','?')):>15}")

    # Chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm as scipy_norm

        def wilson_ci(k, n, z=1.96):
            """Wilson score 95% confidence interval for binomial proportion."""
            if n == 0:
                return 0, 0
            p = k / n
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
            return max(0, centre - margin), min(1, centre + margin)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

        for qi, q in enumerate(QUESTIONS):
            ax = axes[qi]
            af = [r[f"adam_{qi}_f1"] for r in results]
            of = [r[f"trained_{qi}_f1"] for r in results]
            ae = [r[f"adam_{qi}_exact"] for r in results]
            oe = [r[f"trained_{qi}_exact"] for r in results]

            n = len(results)
            x = np.arange(2)
            width = 0.35
            f1_vals = [np.mean(af), np.mean(of)]
            f1_stds = [np.std(af), np.std(of)]
            exact_vals = [np.mean(ae), np.mean(oe)]
            f1_errs = [s / np.sqrt(n) for s in f1_stds]

            # Wilson CI for exact match (binomial)
            adam_k = int(sum(ae))
            ours_k = int(sum(oe))
            adam_lo, adam_hi = wilson_ci(adam_k, n)
            ours_lo, ours_hi = wilson_ci(ours_k, n)
            exact_lo = [exact_vals[0] - adam_lo, exact_vals[1] - ours_lo]
            exact_hi = [adam_hi - exact_vals[0], ours_hi - exact_vals[1]]

            b1 = ax.bar(x - width/2, f1_vals, width, yerr=f1_errs, label="Char F1 (±SEM)",
                        color=["#e74c3c", "#2ecc71"], alpha=0.85, capsize=5, edgecolor="black", linewidth=0.8)
            b2 = ax.bar(x + width/2, exact_vals, width, yerr=[exact_lo, exact_hi],
                        label="Exact Match (Wilson 95% CI)",
                        color=["#fadbd8", "#abebc6"], alpha=0.85, capsize=5, edgecolor="black", linewidth=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels([
                f"Adam's AO\n(single layer 16, 50% depth)",
                f"Ours\n(layers 9, 18, 27 — 25/50/75%)"
            ], fontsize=10)
            ax.set_ylabel("Score", fontsize=11)
            ax.set_title(f"Q: \"{q}\"", fontsize=12, fontweight="bold")
            ymax = max(max(f1_vals), max(exact_vals))
            ax.set_ylim(0, max(0.15, ymax + 0.18))
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)

            # Annotate F1 bars with mean±std
            for i, bar in enumerate(b1):
                h = bar.get_height()
                if h > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2, h + f1_errs[i] + 0.012,
                            f"{h:.3f}±{f1_stds[i]:.3f}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")
            # Annotate exact match bars with k/n
            exact_ks = [adam_k, ours_k]
            for i, bar in enumerate(b2):
                h = bar.get_height()
                top = h + [exact_hi][0][i]
                ax.text(bar.get_x() + bar.get_width()/2, top + 0.012,
                        f"{exact_ks[i]}/{n}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        methodology = (
            f"Number Prediction from CoT Activations  (n={len(results)})\n"
            f"Task: Qwen3-8B (base, no adapter) solves random arithmetic (+-*) with greedy CoT (max 512 tokens, enable_thinking=False).\n"
            f"Activations extracted from residual stream at stride-{STRIDE} over full CoT region, then injected via norm-matched addition at layer 1.\n"
            f"Adam's AO (adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B): single layer 16 (50% depth), prefix 'L16:¶¶¶...'\n"
            f"Ours (ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO): layers 9,18,27 (25/50/75%), prefix 'L9:¶¶¶ L18:¶¶¶ L27:¶¶¶.'\n"
            f"Each oracle queried with its native layer format. Oracle generates up to 32 tokens (greedy). First number extracted via regex.\n"
            f"Char F1 = character-level F1 between predicted and target number strings. Error bars: SEM (F1), Wilson 95% CI (exact match)."
        )
        fig.suptitle(methodology, fontsize=8.5, fontweight="bold", family="monospace",
                     linespacing=1.5, ha="center")
        plt.tight_layout(rect=[0, 0, 1, 0.78])
        chart_path = "data/number_prediction_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"\nChart saved to {chart_path}")
        plt.close()
    except ImportError:
        print("\n(matplotlib not available, skipping chart)")

    output_path = "data/number_prediction_test.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"config": {"n": N_PROBLEMS, "stride": STRIDE,
                               "adam_layers": ADAM_LAYERS, "our_layers": OUR_LAYERS,
                               "questions": QUESTIONS},
                   "summary": summary, "results": results}, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
