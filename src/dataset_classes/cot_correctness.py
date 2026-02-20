"""
Task 5: Correctness Prediction (~15K examples)

Binary classification: is the model's final answer correct?
Uses sentence-structured format with 3 acts per sentence boundary.

Ground truth: compare extracted answer to ground truth from corpus.
Labels: "correct" or "incorrect"
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_correctness_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    seed: int = 42,
) -> list[dict]:
    """
    Generate correctness prediction training data.

    Each example: sentence-boundary activations -> correct / incorrect.
    Ground truth from corpus (cot_correct field). Balanced 50/50.
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    # Split into correct/incorrect pools
    correct_pool = [e for e in corpus if e.get("cot_correct")]
    incorrect_pool = [e for e in corpus if not e.get("cot_correct")]

    if not correct_pool:
        raise ValueError("No correct entries in corpus")
    if not incorrect_pool:
        raise ValueError("No incorrect entries in corpus")

    print(f"  correct: {len(correct_pool)}, incorrect: {len(incorrect_pool)}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    datapoints = []

    while len(datapoints) < num_examples:
        # Alternate 50/50
        if len(datapoints) % 2 == 0:
            entry = random.choice(correct_pool)
            target = "correct"
        else:
            entry = random.choice(incorrect_pool)
            target = "incorrect"

        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        full_text = formatted + cot_text
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(context_ids)]
        if len(positions) < 2:
            continue

        N = len(positions)

        context_slice = context_ids[:positions[-1] + 1]

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"Is the model's final answer correct? Answer: correct or incorrect."
        )

        datapoints.append({
            "datapoint_type": "cot_correctness",
            "prompt": prompt,
            "target_response": target,
            "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
            "num_positions": N,
            "context_input_ids": context_slice,
            "context_positions": list(positions),
        })

    return datapoints[:num_examples]
