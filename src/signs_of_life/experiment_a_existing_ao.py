"""
Experiment A: Does the existing AO read anything from CoT activations?

Load Qwen3-1.7B + AO LoRA. Generate CoTs on math problems.
Extract activations at sentence boundaries. Feed to oracle.
Ask questions about the reasoning.

Pass: AO says anything coherent about math content.
Fail: gibberish / completely unrelated.
"""

import json
import argparse
from pathlib import Path

import torch

from ao_lib import (
    load_model_with_ao,
    generate_cot,
    split_cot_into_sentences,
    collect_activations_at_positions,
    find_sentence_boundary_positions,
    run_oracle_on_activations,
    layer_percent_to_layer,
    MATH_PROBLEMS,
    SignsOfLifeResult,
)


def run_experiment_a(
    model_name: str = "Qwen/Qwen3-1.7B",
    n_problems: int = 10,
    output_path: str = "results/signs_of_life/experiment_a.json",
    device: str = "cuda",
):
    print("=" * 60)
    print("EXPERIMENT A: Does existing AO read CoT activations?")
    print("=" * 60)

    model, tokenizer = load_model_with_ao(model_name, use_8bit=True, device=device)
    act_layer = layer_percent_to_layer(model_name, 50)
    print(f"Activation extraction layer: {act_layer} (50% of {model_name})")

    oracle_prompts = [
        "What topic is this text about?",
        "Can you predict the next 20 tokens that come after this?",
        "What is the model currently thinking about?",
        "Is this mathematical reasoning?",
    ]

    results = []
    problems = MATH_PROBLEMS[:n_problems]

    for i, question in enumerate(problems):
        print(f"\n--- Problem {i+1}/{n_problems}: {question} ---")

        # Generate CoT
        cot_text = generate_cot(model, tokenizer, question, max_new_tokens=512, device=device)
        sentences = split_cot_into_sentences(cot_text)
        print(f"CoT length: {len(cot_text)} chars, {len(sentences)} sentences")

        if len(sentences) < 2:
            print("  Skipping: too few sentences")
            continue

        # Format the full text as the model would see it
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        # Append the CoT (as if the model generated it)
        full_text = formatted + cot_text

        # Find sentence boundary positions
        boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
        if len(boundary_positions) < 2:
            print("  Skipping: couldn't find enough boundary positions")
            continue

        # Limit to 10 positions to keep prompt reasonable
        positions_to_use = boundary_positions[:10]
        print(f"  Using {len(positions_to_use)} sentence boundary positions")

        # Collect activations at those positions
        activations = collect_activations_at_positions(
            model, tokenizer, full_text, act_layer, positions_to_use, device=device,
        )
        print(f"  Activations shape: {activations.shape}")

        # Run oracle with each prompt
        oracle_responses = {}
        for prompt in oracle_prompts:
            response = run_oracle_on_activations(
                model, tokenizer, activations, prompt,
                model_name=model_name, act_layer=act_layer,
                max_new_tokens=100, device=device,
            )
            oracle_responses[prompt] = response
            print(f"  Oracle [{prompt[:40]}...]: {response[:100]}")

        # Also try with just a single token (last sentence boundary)
        single_act = activations[-1:, :]  # [1, d_model]
        single_response = run_oracle_on_activations(
            model, tokenizer, single_act,
            "What is the model thinking about?",
            model_name=model_name, act_layer=act_layer,
            max_new_tokens=50, device=device,
        )
        oracle_responses["single_token_last_sentence"] = single_response
        print(f"  Single token oracle: {single_response[:100]}")

        result = SignsOfLifeResult(
            question=question,
            cot_text=cot_text,
            sentences=sentences,
            boundary_positions=positions_to_use,
            oracle_responses=oracle_responses,
        )
        results.append(result)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        serializable.append({
            "question": r.question,
            "cot_text": r.cot_text,
            "sentences": r.sentences,
            "boundary_positions": r.boundary_positions,
            "oracle_responses": r.oracle_responses,
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed {len(results)}/{n_problems} problems")
    print("\nSample oracle responses:")
    if results:
        r = results[0]
        for prompt, response in r.oracle_responses.items():
            print(f"\n  Q: {prompt}")
            print(f"  A: {response[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--output", default="results/signs_of_life/experiment_a.json")
    args = parser.parse_args()

    run_experiment_a(
        model_name=args.model,
        n_problems=args.n_problems,
        output_path=args.output,
    )
