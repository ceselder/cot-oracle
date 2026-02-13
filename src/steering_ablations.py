"""
Ablation experiments to understand what Y-X vectors actually encode.

Experiments:
1. Math vector on math questions (baseline - already done)
2. Math vector on IRRELEVANT questions (trivia, general knowledge)
3. Random vector with same norm (noise control)
4. Same-question vector (upper bound - using vector from same Q)
5. Shuffled vectors (break correspondence)
"""

import torch
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_utils import get_steering_hook, add_hook, get_layer_module


# Math questions (same as before)
MATH_QUESTIONS = [
    {"question": "What is 17 * 24?", "choices": {"A": "408", "B": "400", "C": "418", "D": "398"}, "answer": "A"},
    {"question": "What is 156 / 12?", "choices": {"A": "12", "B": "13", "C": "14", "D": "15"}, "answer": "B"},
    {"question": "What is 23 + 47 + 89?", "choices": {"A": "159", "B": "149", "C": "169", "D": "139"}, "answer": "A"},
    {"question": "What is 15% of 240?", "choices": {"A": "32", "B": "34", "C": "36", "D": "38"}, "answer": "C"},
    {"question": "If x + 5 = 12, what is x * 3?", "choices": {"A": "18", "B": "21", "C": "24", "D": "15"}, "answer": "B"},
    {"question": "What is 2^8?", "choices": {"A": "128", "B": "256", "C": "512", "D": "64"}, "answer": "B"},
    {"question": "What is the sum of the first 10 positive integers?", "choices": {"A": "45", "B": "50", "C": "55", "D": "60"}, "answer": "C"},
    {"question": "What is 7! / 5!?", "choices": {"A": "42", "B": "56", "C": "30", "D": "21"}, "answer": "A"},
    {"question": "What is 45 * 22?", "choices": {"A": "990", "B": "980", "C": "1000", "D": "970"}, "answer": "A"},
    {"question": "What is 1000 - 387?", "choices": {"A": "613", "B": "623", "C": "603", "D": "633"}, "answer": "A"},
]

# Irrelevant questions (trivia, general knowledge - NOT math)
IRRELEVANT_QUESTIONS = [
    {"question": "What is the capital of France?", "choices": {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}, "answer": "A"},
    {"question": "Which planet is known as the Red Planet?", "choices": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"}, "answer": "B"},
    {"question": "Who wrote Romeo and Juliet?", "choices": {"A": "Dickens", "B": "Shakespeare", "C": "Austen", "D": "Hemingway"}, "answer": "B"},
    {"question": "What is the chemical symbol for gold?", "choices": {"A": "Ag", "B": "Au", "C": "Fe", "D": "Cu"}, "answer": "B"},
    {"question": "Which ocean is the largest?", "choices": {"A": "Atlantic", "B": "Indian", "C": "Pacific", "D": "Arctic"}, "answer": "C"},
    {"question": "What year did World War II end?", "choices": {"A": "1943", "B": "1944", "C": "1945", "D": "1946"}, "answer": "C"},
    {"question": "What is the hardest natural substance?", "choices": {"A": "Gold", "B": "Iron", "C": "Diamond", "D": "Platinum"}, "answer": "C"},
    {"question": "Which country has the largest population?", "choices": {"A": "USA", "B": "India", "C": "China", "D": "Russia"}, "answer": "C"},
    {"question": "What is the speed of light in km/s (approx)?", "choices": {"A": "300,000", "B": "150,000", "C": "500,000", "D": "100,000"}, "answer": "A"},
    {"question": "Who painted the Mona Lisa?", "choices": {"A": "Picasso", "B": "Van Gogh", "C": "Da Vinci", "D": "Monet"}, "answer": "C"},
]


def format_mcq_direct(q: dict) -> str:
    choices_str = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"""{q["question"]}

{choices_str}

Answer with just the letter (A, B, C, or D):"""


def format_mcq_cot(q: dict) -> str:
    choices_str = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"""{q["question"]}

{choices_str}

Think step by step, then give your final answer as a single letter."""


def extract_answer_letter(text: str) -> str:
    text = text.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if text == letter:
            return letter
        if text.startswith(f"{letter}.") or text.startswith(f"{letter}:") or text.startswith(f"{letter})"):
            return letter
    import re
    match = re.search(r'answer[:\s]+([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter
    return "?"


def get_activations(model, tokenizer, prompt: str, layer_idx: int, device: str = "cuda") -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layer_module = get_layer_module(model, layer_idx)
    activations = None

    def hook(module, inp, out):
        nonlocal activations
        activations = out[0] if isinstance(out, tuple) else out

    handle = layer_module.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations[0, -1, :].clone()


def generate_response(model, tokenizer, prompt: str, steering_vector=None,
                      layer_idx=1, steering_coeff=1.0, max_new_tokens=10, device="cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is not None:
        layer_module = get_layer_module(model, layer_idx)
        last_pos = inputs["input_ids"].shape[1] - 1
        hook_fn = get_steering_hook(
            vectors=[[steering_vector]],
            positions=[[last_pos]],
            steering_coefficient=steering_coeff,
            device=device,
            dtype=model.dtype,
        )
        with add_hook(layer_module, hook_fn):
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    else:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_output[len(prompt_text):].strip()


def generate_cot(model, tokenizer, prompt: str, max_new_tokens=256, device="cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_output[len(prompt_text):].strip()


def compute_steering_vectors(model, tokenizer, questions, extraction_layer, device):
    """Compute Y-X vectors for a set of questions."""
    vectors = []
    cots = []

    for q in tqdm(questions, desc="Computing steering vectors"):
        # X: direct answer activations
        prompt_direct = format_mcq_direct(q)
        x = get_activations(model, tokenizer, prompt_direct, extraction_layer, device)

        # Generate CoT
        prompt_cot = format_mcq_cot(q)
        cot = generate_cot(model, tokenizer, prompt_cot, device=device)

        # Y: post-CoT activations
        full_cot_prompt = prompt_cot + "\n" + cot
        y = get_activations(model, tokenizer, full_cot_prompt, extraction_layer, device)

        vectors.append(y - x)
        cots.append(cot)

    return vectors, cots


def evaluate_with_steering(model, tokenizer, questions, steering_vector, device):
    """Evaluate questions with a steering vector."""
    results = []

    for q in questions:
        prompt = format_mcq_direct(q)

        # Baseline
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        baseline_answer = extract_answer_letter(baseline_raw)

        # Steered
        steered_raw = generate_response(model, tokenizer, prompt, steering_vector, device=device)
        steered_answer = extract_answer_letter(steered_raw)

        results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline_answer": baseline_answer,
            "baseline_correct": baseline_answer == q["answer"],
            "steered_answer": steered_answer,
            "steered_correct": steered_answer == q["answer"],
        })

    return results


def run_ablations(model, tokenizer, extraction_layer, device):
    """Run all ablation experiments."""
    results = {}

    print("\n" + "="*60)
    print("COMPUTING MATH STEERING VECTORS")
    print("="*60)
    math_vectors, math_cots = compute_steering_vectors(
        model, tokenizer, MATH_QUESTIONS, extraction_layer, device
    )

    # Average math vector
    avg_math_vector = torch.stack(math_vectors).mean(dim=0)
    avg_norm = avg_math_vector.norm().item()

    print(f"\nAverage math vector norm: {avg_norm:.2f}")

    # ===== EXPERIMENT 1: Math vector on Math (cross-question) =====
    print("\n" + "="*60)
    print("EXP 1: Math vector on MATH questions (cross-question)")
    print("="*60)

    exp1_results = []
    for i, q in enumerate(tqdm(MATH_QUESTIONS, desc="Testing")):
        # Use vector from a DIFFERENT question
        other_idx = (i + 5) % len(MATH_QUESTIONS)
        steering_vec = math_vectors[other_idx]

        prompt = format_mcq_direct(q)
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        steered_raw = generate_response(model, tokenizer, prompt, steering_vec, device=device)

        exp1_results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline": extract_answer_letter(baseline_raw),
            "steered": extract_answer_letter(steered_raw),
        })

    exp1_baseline_acc = sum(1 for r in exp1_results if r["baseline"] == r["ground_truth"]) / len(exp1_results)
    exp1_steered_acc = sum(1 for r in exp1_results if r["steered"] == r["ground_truth"]) / len(exp1_results)
    results["math_on_math_cross"] = {"baseline": exp1_baseline_acc, "steered": exp1_steered_acc}
    print(f"Baseline: {exp1_baseline_acc:.1%}, Steered: {exp1_steered_acc:.1%}, Delta: {exp1_steered_acc - exp1_baseline_acc:+.1%}")

    # ===== EXPERIMENT 2: Math vector on IRRELEVANT questions =====
    print("\n" + "="*60)
    print("EXP 2: Math vector on IRRELEVANT questions (trivia)")
    print("="*60)

    exp2_results = []
    for i, q in enumerate(tqdm(IRRELEVANT_QUESTIONS, desc="Testing")):
        steering_vec = math_vectors[i % len(math_vectors)]

        prompt = format_mcq_direct(q)
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        steered_raw = generate_response(model, tokenizer, prompt, steering_vec, device=device)

        exp2_results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline": extract_answer_letter(baseline_raw),
            "steered": extract_answer_letter(steered_raw),
        })

    exp2_baseline_acc = sum(1 for r in exp2_results if r["baseline"] == r["ground_truth"]) / len(exp2_results)
    exp2_steered_acc = sum(1 for r in exp2_results if r["steered"] == r["ground_truth"]) / len(exp2_results)
    results["math_on_irrelevant"] = {"baseline": exp2_baseline_acc, "steered": exp2_steered_acc}
    print(f"Baseline: {exp2_baseline_acc:.1%}, Steered: {exp2_steered_acc:.1%}, Delta: {exp2_steered_acc - exp2_baseline_acc:+.1%}")

    # ===== EXPERIMENT 3: RANDOM vector on Math =====
    print("\n" + "="*60)
    print("EXP 3: RANDOM vector (same norm) on Math questions")
    print("="*60)

    random_vector = torch.randn_like(avg_math_vector)
    random_vector = random_vector / random_vector.norm() * avg_norm

    exp3_results = []
    for q in tqdm(MATH_QUESTIONS, desc="Testing"):
        prompt = format_mcq_direct(q)
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        steered_raw = generate_response(model, tokenizer, prompt, random_vector, device=device)

        exp3_results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline": extract_answer_letter(baseline_raw),
            "steered": extract_answer_letter(steered_raw),
        })

    exp3_baseline_acc = sum(1 for r in exp3_results if r["baseline"] == r["ground_truth"]) / len(exp3_results)
    exp3_steered_acc = sum(1 for r in exp3_results if r["steered"] == r["ground_truth"]) / len(exp3_results)
    results["random_on_math"] = {"baseline": exp3_baseline_acc, "steered": exp3_steered_acc}
    print(f"Baseline: {exp3_baseline_acc:.1%}, Steered: {exp3_steered_acc:.1%}, Delta: {exp3_steered_acc - exp3_baseline_acc:+.1%}")

    # ===== EXPERIMENT 4: Same-question vector (upper bound) =====
    print("\n" + "="*60)
    print("EXP 4: SAME-question vector on Math (upper bound)")
    print("="*60)

    exp4_results = []
    for i, q in enumerate(tqdm(MATH_QUESTIONS, desc="Testing")):
        steering_vec = math_vectors[i]  # Same question!

        prompt = format_mcq_direct(q)
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        steered_raw = generate_response(model, tokenizer, prompt, steering_vec, device=device)

        exp4_results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline": extract_answer_letter(baseline_raw),
            "steered": extract_answer_letter(steered_raw),
        })

    exp4_baseline_acc = sum(1 for r in exp4_results if r["baseline"] == r["ground_truth"]) / len(exp4_results)
    exp4_steered_acc = sum(1 for r in exp4_results if r["steered"] == r["ground_truth"]) / len(exp4_results)
    results["same_question"] = {"baseline": exp4_baseline_acc, "steered": exp4_steered_acc}
    print(f"Baseline: {exp4_baseline_acc:.1%}, Steered: {exp4_steered_acc:.1%}, Delta: {exp4_steered_acc - exp4_baseline_acc:+.1%}")

    # ===== EXPERIMENT 5: Average vector on Math =====
    print("\n" + "="*60)
    print("EXP 5: AVERAGE math vector on Math questions")
    print("="*60)

    exp5_results = []
    for q in tqdm(MATH_QUESTIONS, desc="Testing"):
        prompt = format_mcq_direct(q)
        baseline_raw = generate_response(model, tokenizer, prompt, None, device=device)
        steered_raw = generate_response(model, tokenizer, prompt, avg_math_vector, device=device)

        exp5_results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "baseline": extract_answer_letter(baseline_raw),
            "steered": extract_answer_letter(steered_raw),
        })

    exp5_baseline_acc = sum(1 for r in exp5_results if r["baseline"] == r["ground_truth"]) / len(exp5_results)
    exp5_steered_acc = sum(1 for r in exp5_results if r["steered"] == r["ground_truth"]) / len(exp5_results)
    results["avg_vector_on_math"] = {"baseline": exp5_baseline_acc, "steered": exp5_steered_acc}
    print(f"Baseline: {exp5_baseline_acc:.1%}, Steered: {exp5_steered_acc:.1%}, Delta: {exp5_steered_acc - exp5_baseline_acc:+.1%}")

    return results, {
        "exp1": exp1_results,
        "exp2": exp2_results,
        "exp3": exp3_results,
        "exp4": exp4_results,
        "exp5": exp5_results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--extraction-layer", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/steering_ablations.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    if args.extraction_layer is None:
        extraction_layer = model.config.num_hidden_layers // 2
    else:
        extraction_layer = args.extraction_layer

    print(f"Extraction layer: {extraction_layer}")

    summary, details = run_ablations(model, tokenizer, extraction_layer, args.device)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Experiment':<30} {'Baseline':<10} {'Steered':<10} {'Delta':<10}")
    print("-"*60)
    for name, res in summary.items():
        delta = res['steered'] - res['baseline']
        print(f"{name:<30} {res['baseline']:<10.1%} {res['steered']:<10.1%} {delta:+.1%}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
