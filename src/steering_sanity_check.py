"""
Steering Sanity Check: Does Y-X actually encode CoT reasoning?

The core experiment:
1. Get activations X when model answers directly (no CoT)
2. Get activations Y when model answers after CoT
3. Compute steering vector v = Y - X
4. Inject v when answering a DIFFERENT question
5. Check if this improves accuracy

If it works: The vector encodes something meaningful about reasoning
If it doesn't: Need to rethink the approach
"""

import torch
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from steering_utils import get_steering_hook, add_hook, get_layer_module


# --- MCQ Math Questions ---
# Questions where models often fail without CoT but succeed with it

MCQ_QUESTIONS = [
    {
        "question": "What is 17 * 24?",
        "choices": {"A": "408", "B": "400", "C": "418", "D": "398"},
        "answer": "A"
    },
    {
        "question": "What is 156 / 12?",
        "choices": {"A": "12", "B": "13", "C": "14", "D": "15"},
        "answer": "B"
    },
    {
        "question": "What is 23 + 47 + 89?",
        "choices": {"A": "159", "B": "149", "C": "169", "D": "139"},
        "answer": "A"
    },
    {
        "question": "What is 15% of 240?",
        "choices": {"A": "32", "B": "34", "C": "36", "D": "38"},
        "answer": "C"
    },
    {
        "question": "If x + 5 = 12, what is x * 3?",
        "choices": {"A": "18", "B": "21", "C": "24", "D": "15"},
        "answer": "B"
    },
    {
        "question": "What is the area of a rectangle with length 8 and width 6?",
        "choices": {"A": "42", "B": "44", "C": "46", "D": "48"},
        "answer": "D"
    },
    {
        "question": "What is 2^8?",
        "choices": {"A": "128", "B": "256", "C": "512", "D": "64"},
        "answer": "B"
    },
    {
        "question": "What is the sum of the first 10 positive integers?",
        "choices": {"A": "45", "B": "50", "C": "55", "D": "60"},
        "answer": "C"
    },
    {
        "question": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "choices": {"A": "120 miles", "B": "140 miles", "C": "150 miles", "D": "160 miles"},
        "answer": "C"
    },
    {
        "question": "What is 7! / 5!?",
        "choices": {"A": "42", "B": "56", "C": "30", "D": "21"},
        "answer": "A"
    },
    {
        "question": "What is the square root of 169?",
        "choices": {"A": "11", "B": "12", "C": "13", "D": "14"},
        "answer": "C"
    },
    {
        "question": "If 3x - 7 = 14, what is x?",
        "choices": {"A": "5", "B": "6", "C": "7", "D": "8"},
        "answer": "C"
    },
    {
        "question": "What is 45 * 22?",
        "choices": {"A": "990", "B": "980", "C": "1000", "D": "970"},
        "answer": "A"
    },
    {
        "question": "What is 1000 - 387?",
        "choices": {"A": "613", "B": "623", "C": "603", "D": "633"},
        "answer": "A"
    },
    {
        "question": "What is the perimeter of a square with side 15?",
        "choices": {"A": "55", "B": "60", "C": "65", "D": "70"},
        "answer": "B"
    },
    {
        "question": "If y = 2x + 3 and x = 4, what is y?",
        "choices": {"A": "9", "B": "10", "C": "11", "D": "12"},
        "answer": "C"
    },
    {
        "question": "What is 144 / 16?",
        "choices": {"A": "8", "B": "9", "C": "10", "D": "11"},
        "answer": "B"
    },
    {
        "question": "What is 25% of 84?",
        "choices": {"A": "19", "B": "20", "C": "21", "D": "22"},
        "answer": "C"
    },
    {
        "question": "What is 13^2?",
        "choices": {"A": "156", "B": "163", "C": "169", "D": "176"},
        "answer": "C"
    },
    {
        "question": "How many minutes are in 3.5 hours?",
        "choices": {"A": "200", "B": "210", "C": "220", "D": "230"},
        "answer": "B"
    },
]


@dataclass
class ExperimentResult:
    question_idx: int
    question: str
    ground_truth: str

    # Source question (used to compute steering vector)
    source_question_idx: int
    source_cot: str

    # Results
    baseline_answer: str
    baseline_correct: bool
    steered_answer: str
    steered_correct: bool

    # Vector stats
    vector_norm: float

    # Raw outputs for inspection
    baseline_raw: str
    steered_raw: str


def format_mcq_direct(q: dict) -> str:
    """Format question for direct answering (no CoT)."""
    choices_str = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"""{q["question"]}

{choices_str}

Answer with just the letter (A, B, C, or D):"""


def format_mcq_cot(q: dict) -> str:
    """Format question for CoT answering."""
    choices_str = "\n".join(f"{k}: {v}" for k, v in q["choices"].items())
    return f"""{q["question"]}

{choices_str}

Think step by step, then give your final answer as a single letter."""


def extract_answer_letter(text: str) -> str:
    """Extract the answer letter from model output."""
    text = text.strip().upper()

    # Look for standalone letter
    for letter in ["A", "B", "C", "D"]:
        if text == letter:
            return letter
        if text.startswith(f"{letter}.") or text.startswith(f"{letter}:") or text.startswith(f"{letter})"):
            return letter
        if f"answer is {letter}" in text.lower() or f"answer: {letter}" in text.lower():
            return letter

    # Look for letter in "The answer is X" pattern
    import re
    match = re.search(r'answer[:\s]+([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Just look for any A-D
    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter

    return "?"


def get_activations(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Get activations at the last token position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layer_module = get_layer_module(model, layer_idx)

    activations = None
    def hook(module, inp, out):
        nonlocal activations
        if isinstance(out, tuple):
            activations = out[0]
        else:
            activations = out

    handle = layer_module.register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Return activation at last token
    return activations[0, -1, :].clone()


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor | None,
    layer_idx: int = 1,  # Inject at layer 1 per AO paper
    steering_coeff: float = 1.0,
    max_new_tokens: int = 10,
    device: str = "cuda",
) -> str:
    """Generate response, optionally with steering vector injection."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is not None:
        # Set up steering hook
        layer_module = get_layer_module(model, layer_idx)

        # Inject at the last prompt token (position -1 in the prompt)
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

    # Decode and strip prompt
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response = full_output[len(prompt_text):].strip()

    return response


def generate_cot(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """Generate chain of thought response."""
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
    response = full_output[len(prompt_text):].strip()

    return response


def run_steering_experiment(
    model,
    tokenizer,
    questions: list[dict],
    extraction_layer: int,  # Layer to extract Y and X from
    injection_layer: int = 1,  # Layer to inject steering vector (default: 1)
    steering_coeff: float = 1.0,
    device: str = "cuda",
    n_experiments: int = 20,
) -> list[ExperimentResult]:
    """
    Run the steering sanity check experiment.

    For each test question:
    1. Pick a different question as "source"
    2. Generate CoT for source question
    3. Compute Y-X vector from source question
    4. Use that vector to steer the model on test question
    5. Compare steered vs baseline accuracy
    """
    results = []

    # Shuffle questions
    indices = list(range(len(questions)))
    random.shuffle(indices)

    for i in tqdm(range(min(n_experiments, len(questions))), desc="Running experiments"):
        test_idx = indices[i]
        # Pick a different question as source
        source_idx = indices[(i + 1) % len(questions)]

        test_q = questions[test_idx]
        source_q = questions[source_idx]

        # --- Step 1: Compute steering vector from source question ---

        # X: activations when answering source directly
        prompt_direct = format_mcq_direct(source_q)
        x = get_activations(model, tokenizer, prompt_direct, extraction_layer, device)

        # Generate CoT for source
        prompt_cot = format_mcq_cot(source_q)
        cot_response = generate_cot(model, tokenizer, prompt_cot, device=device)

        # Y: activations after CoT (at the answer position)
        # We append the CoT and then measure activations
        full_cot_prompt = prompt_cot + "\n" + cot_response
        y = get_activations(model, tokenizer, full_cot_prompt, extraction_layer, device)

        # Steering vector
        v = y - x

        # --- Step 2: Test on target question ---

        test_prompt = format_mcq_direct(test_q)

        # Baseline: no steering
        baseline_raw = generate_with_steering(
            model, tokenizer, test_prompt,
            steering_vector=None,
            max_new_tokens=10,
            device=device,
        )
        baseline_answer = extract_answer_letter(baseline_raw)

        # Steered
        steered_raw = generate_with_steering(
            model, tokenizer, test_prompt,
            steering_vector=v,
            layer_idx=injection_layer,
            steering_coeff=steering_coeff,
            max_new_tokens=10,
            device=device,
        )
        steered_answer = extract_answer_letter(steered_raw)

        result = ExperimentResult(
            question_idx=test_idx,
            question=test_q["question"],
            ground_truth=test_q["answer"],
            source_question_idx=source_idx,
            source_cot=cot_response[:200] + "..." if len(cot_response) > 200 else cot_response,
            baseline_answer=baseline_answer,
            baseline_correct=baseline_answer == test_q["answer"],
            steered_answer=steered_answer,
            steered_correct=steered_answer == test_q["answer"],
            vector_norm=v.norm().item(),
            baseline_raw=baseline_raw,
            steered_raw=steered_raw,
        )
        results.append(result)

        # Print progress
        status = ""
        if result.baseline_correct and result.steered_correct:
            status = "both correct"
        elif not result.baseline_correct and result.steered_correct:
            status = "STEERING HELPED!"
        elif result.baseline_correct and not result.steered_correct:
            status = "steering hurt"
        else:
            status = "both wrong"

        print(f"\n  Q{test_idx}: {test_q['question'][:40]}...")
        print(f"  Baseline: {baseline_answer} (correct: {test_q['answer']}) | Steered: {steered_answer}")
        print(f"  -> {status}")

    return results


def summarize_results(results: list[ExperimentResult]) -> dict:
    """Compute summary statistics."""
    n = len(results)

    baseline_correct = sum(1 for r in results if r.baseline_correct)
    steered_correct = sum(1 for r in results if r.steered_correct)

    # Cases where steering changed the outcome
    steering_helped = sum(1 for r in results if not r.baseline_correct and r.steered_correct)
    steering_hurt = sum(1 for r in results if r.baseline_correct and not r.steered_correct)
    both_correct = sum(1 for r in results if r.baseline_correct and r.steered_correct)
    both_wrong = sum(1 for r in results if not r.baseline_correct and not r.steered_correct)

    avg_vector_norm = sum(r.vector_norm for r in results) / n if n > 0 else 0

    return {
        "n_experiments": n,
        "baseline_accuracy": baseline_correct / n if n > 0 else 0,
        "steered_accuracy": steered_correct / n if n > 0 else 0,
        "accuracy_delta": (steered_correct - baseline_correct) / n if n > 0 else 0,
        "steering_helped": steering_helped,
        "steering_hurt": steering_hurt,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "avg_vector_norm": avg_vector_norm,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Steering sanity check experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to use")
    parser.add_argument("--extraction-layer", type=int, default=None, help="Layer to extract Y-X from (default: 50% depth)")
    parser.add_argument("--injection-layer", type=int, default=1, help="Layer to inject steering (default: 1)")
    parser.add_argument("--steering-coeff", type=float, default=1.0, help="Steering coefficient")
    parser.add_argument("--n-experiments", type=int, default=20, help="Number of experiments")
    parser.add_argument("--output", type=str, default="results/steering_sanity_check.json")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
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

    # Determine extraction layer (50% depth if not specified)
    if args.extraction_layer is None:
        n_layers = model.config.num_hidden_layers
        extraction_layer = n_layers // 2
    else:
        extraction_layer = args.extraction_layer

    print(f"Extraction layer: {extraction_layer}")
    print(f"Injection layer: {args.injection_layer}")
    print(f"Steering coefficient: {args.steering_coeff}")

    # Run experiment
    results = run_steering_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=MCQ_QUESTIONS,
        extraction_layer=extraction_layer,
        injection_layer=args.injection_layer,
        steering_coeff=args.steering_coeff,
        device=args.device,
        n_experiments=args.n_experiments,
    )

    # Summarize
    summary = summarize_results(results)

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Experiments: {summary['n_experiments']}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.1%}")
    print(f"Steered accuracy: {summary['steered_accuracy']:.1%}")
    print(f"Accuracy delta: {summary['accuracy_delta']:+.1%}")
    print(f"Steering helped: {summary['steering_helped']}")
    print(f"Steering hurt: {summary['steering_hurt']}")
    print(f"Both correct: {summary['both_correct']}")
    print(f"Both wrong: {summary['both_wrong']}")
    print(f"Avg vector norm: {summary['avg_vector_norm']:.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model": args.model,
            "extraction_layer": extraction_layer,
            "injection_layer": args.injection_layer,
            "steering_coeff": args.steering_coeff,
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)

    delta = summary['accuracy_delta']
    if delta > 0.1:
        print("Steering improved accuracy by >10%.")
        print("The Y-X vector encodes meaningful reasoning information.")
        print("Proceed with Phase 1 (probes) and Phase 2 (oracle training).")
    elif delta > 0:
        print("Steering had a small positive effect.")
        print("There's some signal in the vector, but it's noisy.")
        print("Consider: averaging multiple CoT rollouts, different layers, attention patterns.")
    elif delta > -0.1:
        print("Steering had minimal effect (within noise).")
        print("The vector might not encode generalizable reasoning.")
        print("Try: different extraction layers, different steering coefficients, SAE features.")
    else:
        print("Steering hurt accuracy significantly.")
        print("The vector might encode question-specific info, not general reasoning.")
        print("Reconsider the approach: maybe Y-X encodes content, not computation.")


if __name__ == "__main__":
    main()
