"""
Steering Token Length Analysis: Test hypothesis that steering just triggers more test-time compute.

Hypothesis: The Y-X vector doesn't encode reasoning - it just puts the model in "reasoning mode"
causing it to generate longer responses with more computation.

This script:
1. Runs steering experiments on more data
2. Tracks token lengths of baseline vs steered responses
3. Saves all actual responses
4. Plots token length distributions and correlations with accuracy
"""

import torch
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict

from steering_utils import get_steering_hook, add_hook, get_layer_module


# Extended MCQ Questions - more data points
MCQ_QUESTIONS = [
    # Original questions
    {"question": "What is 17 * 24?", "choices": {"A": "408", "B": "400", "C": "418", "D": "398"}, "answer": "A"},
    {"question": "What is 156 / 12?", "choices": {"A": "12", "B": "13", "C": "14", "D": "15"}, "answer": "B"},
    {"question": "What is 23 + 47 + 89?", "choices": {"A": "159", "B": "149", "C": "169", "D": "139"}, "answer": "A"},
    {"question": "What is 15% of 240?", "choices": {"A": "32", "B": "34", "C": "36", "D": "38"}, "answer": "C"},
    {"question": "If x + 5 = 12, what is x * 3?", "choices": {"A": "18", "B": "21", "C": "24", "D": "15"}, "answer": "B"},
    {"question": "What is the area of a rectangle with length 8 and width 6?", "choices": {"A": "42", "B": "44", "C": "46", "D": "48"}, "answer": "D"},
    {"question": "What is 2^8?", "choices": {"A": "128", "B": "256", "C": "512", "D": "64"}, "answer": "B"},
    {"question": "What is the sum of the first 10 positive integers?", "choices": {"A": "45", "B": "50", "C": "55", "D": "60"}, "answer": "C"},
    {"question": "If a train travels at 60 mph for 2.5 hours, how far does it go?", "choices": {"A": "120 miles", "B": "140 miles", "C": "150 miles", "D": "160 miles"}, "answer": "C"},
    {"question": "What is 7! / 5!?", "choices": {"A": "42", "B": "56", "C": "30", "D": "21"}, "answer": "A"},
    {"question": "What is the square root of 169?", "choices": {"A": "11", "B": "12", "C": "13", "D": "14"}, "answer": "C"},
    {"question": "If 3x - 7 = 14, what is x?", "choices": {"A": "5", "B": "6", "C": "7", "D": "8"}, "answer": "C"},
    {"question": "What is 45 * 22?", "choices": {"A": "990", "B": "980", "C": "1000", "D": "970"}, "answer": "A"},
    {"question": "What is 1000 - 387?", "choices": {"A": "613", "B": "623", "C": "603", "D": "633"}, "answer": "A"},
    {"question": "What is the perimeter of a square with side 15?", "choices": {"A": "55", "B": "60", "C": "65", "D": "70"}, "answer": "B"},
    {"question": "If y = 2x + 3 and x = 4, what is y?", "choices": {"A": "9", "B": "10", "C": "11", "D": "12"}, "answer": "C"},
    {"question": "What is 144 / 16?", "choices": {"A": "8", "B": "9", "C": "10", "D": "11"}, "answer": "B"},
    {"question": "What is 25% of 84?", "choices": {"A": "19", "B": "20", "C": "21", "D": "22"}, "answer": "C"},
    {"question": "What is 13^2?", "choices": {"A": "156", "B": "163", "C": "169", "D": "176"}, "answer": "C"},
    {"question": "How many minutes are in 3.5 hours?", "choices": {"A": "200", "B": "210", "C": "220", "D": "230"}, "answer": "B"},
    # Additional harder questions
    {"question": "What is 37 * 43?", "choices": {"A": "1591", "B": "1581", "C": "1601", "D": "1571"}, "answer": "A"},
    {"question": "What is 2048 / 64?", "choices": {"A": "30", "B": "32", "C": "34", "D": "36"}, "answer": "B"},
    {"question": "What is 18^2 - 12^2?", "choices": {"A": "180", "B": "170", "C": "190", "D": "160"}, "answer": "A"},
    {"question": "What is 3/8 + 5/8?", "choices": {"A": "3/4", "B": "7/8", "C": "1", "D": "15/16"}, "answer": "C"},
    {"question": "What is the value of 5! (5 factorial)?", "choices": {"A": "100", "B": "120", "C": "125", "D": "150"}, "answer": "B"},
    {"question": "If 2^n = 64, what is n?", "choices": {"A": "5", "B": "6", "C": "7", "D": "8"}, "answer": "B"},
    {"question": "What is 15% of 340?", "choices": {"A": "49", "B": "51", "C": "53", "D": "55"}, "answer": "B"},
    {"question": "What is the LCM of 12 and 18?", "choices": {"A": "24", "B": "36", "C": "54", "D": "72"}, "answer": "B"},
    {"question": "What is sqrt(256) + sqrt(144)?", "choices": {"A": "26", "B": "28", "C": "30", "D": "32"}, "answer": "B"},
    {"question": "What is 999 + 999 + 999?", "choices": {"A": "2997", "B": "2987", "C": "3007", "D": "2977"}, "answer": "A"},
    {"question": "What is 72 / 0.8?", "choices": {"A": "80", "B": "85", "C": "90", "D": "95"}, "answer": "C"},
    {"question": "What is 4^3 + 3^4?", "choices": {"A": "145", "B": "137", "C": "153", "D": "129"}, "answer": "A"},
    {"question": "If a = 3 and b = 4, what is a^2 + b^2?", "choices": {"A": "20", "B": "25", "C": "30", "D": "35"}, "answer": "B"},
    {"question": "What is 1/4 of 1/2?", "choices": {"A": "1/4", "B": "1/6", "C": "1/8", "D": "1/12"}, "answer": "C"},
    {"question": "What is 11 * 11 * 11?", "choices": {"A": "1221", "B": "1331", "C": "1441", "D": "1111"}, "answer": "B"},
    {"question": "What is 50% of 50% of 100?", "choices": {"A": "20", "B": "25", "C": "30", "D": "35"}, "answer": "B"},
    {"question": "If x/4 = 7, what is x?", "choices": {"A": "24", "B": "26", "C": "28", "D": "30"}, "answer": "C"},
    {"question": "What is the GCD of 48 and 64?", "choices": {"A": "8", "B": "12", "C": "16", "D": "24"}, "answer": "C"},
    {"question": "What is 2.5 * 4.4?", "choices": {"A": "10", "B": "11", "C": "12", "D": "13"}, "answer": "B"},
    {"question": "How many seconds in 2 hours?", "choices": {"A": "7000", "B": "7100", "C": "7200", "D": "7300"}, "answer": "C"},
]


@dataclass
class DetailedResult:
    question_idx: int
    question: str
    ground_truth: str
    source_question_idx: int

    # Baseline
    baseline_answer: str
    baseline_correct: bool
    baseline_raw: str
    baseline_token_count: int

    # Steered
    steered_answer: str
    steered_correct: bool
    steered_raw: str
    steered_token_count: int

    # Token length diff
    token_length_diff: int  # steered - baseline

    # Vector stats
    vector_norm: float


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
        if f"answer is {letter}" in text.lower() or f"answer: {letter}" in text.lower():
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


def generate_with_steering(
    model, tokenizer, prompt: str,
    steering_vector: torch.Tensor | None,
    layer_idx: int = 1,
    steering_coeff: float = 1.0,
    max_new_tokens: int = 100,  # Increased to see if steering causes longer outputs
    device: str = "cuda",
) -> tuple[str, int]:
    """Generate response and return (text, token_count)."""
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Count new tokens generated
    new_token_count = outputs.shape[1] - inputs["input_ids"].shape[1]

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response = full_output[len(prompt_text):].strip()

    return response, new_token_count


def generate_cot(model, tokenizer, prompt: str, max_new_tokens: int = 256, device: str = "cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_output[len(prompt_text):].strip()


def run_token_length_experiment(
    model, tokenizer, questions: list[dict],
    extraction_layer: int,
    injection_layer: int = 1,
    steering_coeff: float = 1.0,
    device: str = "cuda",
    n_experiments: int = 40,
) -> list[DetailedResult]:
    results = []
    indices = list(range(len(questions)))
    random.shuffle(indices)

    for i in tqdm(range(min(n_experiments, len(questions))), desc="Running experiments"):
        test_idx = indices[i]
        source_idx = indices[(i + 1) % len(questions)]

        test_q = questions[test_idx]
        source_q = questions[source_idx]

        # Compute steering vector from source question
        prompt_direct = format_mcq_direct(source_q)
        x = get_activations(model, tokenizer, prompt_direct, extraction_layer, device)

        prompt_cot = format_mcq_cot(source_q)
        cot_response = generate_cot(model, tokenizer, prompt_cot, device=device)

        full_cot_prompt = prompt_cot + "\n" + cot_response
        y = get_activations(model, tokenizer, full_cot_prompt, extraction_layer, device)

        v = y - x

        # Test on target question
        test_prompt = format_mcq_direct(test_q)

        # Baseline
        baseline_raw, baseline_tokens = generate_with_steering(
            model, tokenizer, test_prompt,
            steering_vector=None,
            max_new_tokens=100,
            device=device,
        )
        baseline_answer = extract_answer_letter(baseline_raw)

        # Steered
        steered_raw, steered_tokens = generate_with_steering(
            model, tokenizer, test_prompt,
            steering_vector=v,
            layer_idx=injection_layer,
            steering_coeff=steering_coeff,
            max_new_tokens=100,
            device=device,
        )
        steered_answer = extract_answer_letter(steered_raw)

        result = DetailedResult(
            question_idx=test_idx,
            question=test_q["question"],
            ground_truth=test_q["answer"],
            source_question_idx=source_idx,
            baseline_answer=baseline_answer,
            baseline_correct=baseline_answer == test_q["answer"],
            baseline_raw=baseline_raw,
            baseline_token_count=baseline_tokens,
            steered_answer=steered_answer,
            steered_correct=steered_answer == test_q["answer"],
            steered_raw=steered_raw,
            steered_token_count=steered_tokens,
            token_length_diff=steered_tokens - baseline_tokens,
            vector_norm=v.norm().item(),
        )
        results.append(result)

        print(f"\n  Q{test_idx}: {test_q['question'][:40]}...")
        print(f"  Baseline: {baseline_answer} ({baseline_tokens} tokens) | Steered: {steered_answer} ({steered_tokens} tokens)")
        print(f"  Token diff: {result.token_length_diff:+d}")

    return results


def create_analysis_plots(results: list[DetailedResult], output_dir: Path):
    """Create plots analyzing token length relationship to accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    baseline_tokens = [r.baseline_token_count for r in results]
    steered_tokens = [r.steered_token_count for r in results]
    token_diffs = [r.token_length_diff for r in results]

    baseline_correct = [r.baseline_correct for r in results]
    steered_correct = [r.steered_correct for r in results]

    # Figure 1: Token length distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of baseline vs steered token counts
    axes[0].hist(baseline_tokens, alpha=0.7, label='Baseline', bins=20, color='blue')
    axes[0].hist(steered_tokens, alpha=0.7, label='Steered', bins=20, color='orange')
    axes[0].set_xlabel('Token Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Token Length Distribution')
    axes[0].legend()

    # Histogram of token length differences
    axes[1].hist(token_diffs, bins=20, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', label='No change')
    axes[1].axvline(x=np.mean(token_diffs), color='purple', linestyle='-', label=f'Mean: {np.mean(token_diffs):.1f}')
    axes[1].set_xlabel('Token Length Diff (Steered - Baseline)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Token Length Change from Steering')
    axes[1].legend()

    # Scatter: token diff vs accuracy change
    accuracy_change = [int(r.steered_correct) - int(r.baseline_correct) for r in results]
    axes[2].scatter(token_diffs, accuracy_change, alpha=0.6)
    axes[2].set_xlabel('Token Length Diff')
    axes[2].set_ylabel('Accuracy Change (-1, 0, 1)')
    axes[2].set_title('Token Length vs Accuracy Change')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_analysis.png', dpi=150)
    plt.close()

    # Figure 2: Breakdown by outcome
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Categorize outcomes
    categories = {
        'steering_helped': [],
        'steering_hurt': [],
        'both_correct': [],
        'both_wrong': [],
    }

    for r in results:
        if not r.baseline_correct and r.steered_correct:
            categories['steering_helped'].append(r.token_length_diff)
        elif r.baseline_correct and not r.steered_correct:
            categories['steering_hurt'].append(r.token_length_diff)
        elif r.baseline_correct and r.steered_correct:
            categories['both_correct'].append(r.token_length_diff)
        else:
            categories['both_wrong'].append(r.token_length_diff)

    # Box plot of token diff by outcome
    data_to_plot = [categories[k] for k in categories.keys() if len(categories[k]) > 0]
    labels = [k.replace('_', '\n') for k in categories.keys() if len(categories[k]) > 0]

    if data_to_plot:
        bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['green', 'red', 'blue', 'gray'][:len(data_to_plot)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_ylabel('Token Length Diff')
        axes[0].set_title('Token Length Change by Outcome')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Bar chart: mean token lengths by outcome
    means = {k: np.mean(v) if v else 0 for k, v in categories.items()}
    counts = {k: len(v) for k, v in categories.items()}

    x = np.arange(len(means))
    bars = axes[1].bar(x, list(means.values()), color=['green', 'red', 'blue', 'gray'], alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([k.replace('_', '\n') for k in means.keys()])
    axes[1].set_ylabel('Mean Token Length Diff')
    axes[1].set_title('Mean Token Length Change by Outcome')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add count annotations
    for bar, count in zip(bars, counts.values()):
        axes[1].annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'token_length_by_outcome.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def save_responses(results: list[DetailedResult], output_path: Path):
    """Save all responses to a file for inspection."""
    output = []
    for r in results:
        output.append({
            "question_idx": r.question_idx,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "baseline": {
                "answer": r.baseline_answer,
                "correct": r.baseline_correct,
                "tokens": r.baseline_token_count,
                "raw_response": r.baseline_raw,
            },
            "steered": {
                "answer": r.steered_answer,
                "correct": r.steered_correct,
                "tokens": r.steered_token_count,
                "raw_response": r.steered_raw,
            },
            "token_diff": r.token_length_diff,
            "vector_norm": r.vector_norm,
        })

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Responses saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Steering token length analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--extraction-layer", type=int, default=None)
    parser.add_argument("--injection-layer", type=int, default=1)
    parser.add_argument("--steering-coeff", type=float, default=1.0)
    parser.add_argument("--n-experiments", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="results/token_analysis")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model}...")

    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    if args.extraction_layer is None:
        n_layers = model.config.num_hidden_layers
        extraction_layer = n_layers // 2
    else:
        extraction_layer = args.extraction_layer

    print(f"Extraction layer: {extraction_layer}")
    print(f"Injection layer: {args.injection_layer}")
    print(f"Steering coefficient: {args.steering_coeff}")
    print(f"Running {args.n_experiments} experiments...")

    results = run_token_length_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=MCQ_QUESTIONS,
        extraction_layer=extraction_layer,
        injection_layer=args.injection_layer,
        steering_coeff=args.steering_coeff,
        device=args.device,
        n_experiments=args.n_experiments,
    )

    # Summary statistics
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS RESULTS")
    print("="*60)

    baseline_acc = sum(r.baseline_correct for r in results) / len(results)
    steered_acc = sum(r.steered_correct for r in results) / len(results)

    baseline_tokens_mean = np.mean([r.baseline_token_count for r in results])
    steered_tokens_mean = np.mean([r.steered_token_count for r in results])
    token_diff_mean = np.mean([r.token_length_diff for r in results])

    print(f"\nAccuracy:")
    print(f"  Baseline: {baseline_acc:.1%}")
    print(f"  Steered:  {steered_acc:.1%}")
    print(f"  Delta:    {steered_acc - baseline_acc:+.1%}")

    print(f"\nToken Lengths:")
    print(f"  Baseline mean: {baseline_tokens_mean:.1f}")
    print(f"  Steered mean:  {steered_tokens_mean:.1f}")
    print(f"  Mean diff:     {token_diff_mean:+.1f}")

    # Correlation analysis
    token_diffs = np.array([r.token_length_diff for r in results])
    accuracy_changes = np.array([int(r.steered_correct) - int(r.baseline_correct) for r in results])

    if np.std(token_diffs) > 0 and np.std(accuracy_changes) > 0:
        correlation = np.corrcoef(token_diffs, accuracy_changes)[0, 1]
        print(f"\nCorrelation (token_diff vs accuracy_change): {correlation:.3f}")

    # Cases breakdown
    steering_helped = sum(1 for r in results if not r.baseline_correct and r.steered_correct)
    steering_hurt = sum(1 for r in results if r.baseline_correct and not r.steered_correct)

    print(f"\nOutcome breakdown:")
    print(f"  Steering helped: {steering_helped}")
    print(f"  Steering hurt:   {steering_hurt}")

    # When steering helped, did it produce longer responses?
    helped_diffs = [r.token_length_diff for r in results if not r.baseline_correct and r.steered_correct]
    hurt_diffs = [r.token_length_diff for r in results if r.baseline_correct and not r.steered_correct]

    if helped_diffs:
        print(f"\nWhen steering HELPED, token diff: {np.mean(helped_diffs):+.1f} (n={len(helped_diffs)})")
    if hurt_diffs:
        print(f"When steering HURT,   token diff: {np.mean(hurt_diffs):+.1f} (n={len(hurt_diffs)})")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_responses(results, output_dir / "responses.json")
    create_analysis_plots(results, output_dir)

    # Save summary
    summary = {
        "config": {
            "model": args.model,
            "extraction_layer": extraction_layer,
            "injection_layer": args.injection_layer,
            "steering_coeff": args.steering_coeff,
            "n_experiments": len(results),
        },
        "accuracy": {
            "baseline": baseline_acc,
            "steered": steered_acc,
            "delta": steered_acc - baseline_acc,
        },
        "token_lengths": {
            "baseline_mean": baseline_tokens_mean,
            "steered_mean": steered_tokens_mean,
            "diff_mean": token_diff_mean,
        },
        "outcome_counts": {
            "steering_helped": steering_helped,
            "steering_hurt": steering_hurt,
            "both_correct": sum(1 for r in results if r.baseline_correct and r.steered_correct),
            "both_wrong": sum(1 for r in results if not r.baseline_correct and not r.steered_correct),
        },
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")

    # Hypothesis verdict
    print("\n" + "="*60)
    print("HYPOTHESIS CHECK: Is this just more test-time compute?")
    print("="*60)

    if token_diff_mean > 5:
        print(f"Steered responses ARE longer on average (+{token_diff_mean:.1f} tokens)")
        if helped_diffs and np.mean(helped_diffs) > 5:
            print("AND cases where steering helped had longer responses.")
            print("=> CONSISTENT with 'just more compute' hypothesis")
        else:
            print("BUT cases where steering helped did NOT have meaningfully longer responses.")
            print("=> INCONSISTENT with 'just more compute' hypothesis")
    else:
        print(f"Steered responses are NOT significantly longer ({token_diff_mean:+.1f} tokens)")
        print("=> INCONSISTENT with 'just more compute' hypothesis")


if __name__ == "__main__":
    main()
