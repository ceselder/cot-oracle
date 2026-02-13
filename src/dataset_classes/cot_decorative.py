"""
Task 3: Decorative CoT Detection (~10K examples, self-supervised)

Binary classification: is this CoT load_bearing or decorative?
Uses sentence-structured format with 3 acts per sentence boundary (L25%, L50%, L75%).

Ground truth comes from the corpus itself:
- load_bearing: CoT correct, direct answer wrong -> reasoning was necessary
- both_correct (decorative): both CoT and direct correct -> model already knew

Balanced 50/50 sampling.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_decorative_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 10000,
    max_sentences: int = 15,
    seed: int = 42,
) -> list[dict]:
    """
    Generate decorative CoT detection training data with sentence-structured format.

    Each example: activations at ALL sentence boundaries (3 per boundary, one per layer)
    -> load_bearing / decorative.

    Requires corpus generated with --keep-all.
    """
    from signs_of_life.ao_lib import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    # Split into load-bearing and decorative pools
    load_bearing = [e for e in corpus if e.get("category") == "load_bearing"]
    decorative = [e for e in corpus if e.get("category") == "both_correct"]

    if not load_bearing:
        raise ValueError("No load_bearing entries in corpus. Was --keep-all used during generation?")
    if not decorative:
        raise ValueError("No both_correct entries in corpus. Was --keep-all used during generation?")

    print(f"  load_bearing: {len(load_bearing)}, decorative (both_correct): {len(decorative)}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    datapoints = []

    while len(datapoints) < num_examples:
        # Alternate 50/50
        if len(datapoints) % 2 == 0:
            entry = random.choice(load_bearing)
            target = "load_bearing"
        else:
            entry = random.choice(decorative)
            target = "decorative"

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

        # Cap and validate positions
        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(context_ids)]
        if len(positions) < 2:
            continue

        N = len(positions)

        context_slice = context_ids[:positions[-1] + 1]

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"Is this chain-of-thought reasoning load-bearing or decorative? "
            f"Answer: load_bearing or decorative."
        )

        datapoints.append({
            "datapoint_type": "cot_decorative",
            "prompt": prompt,
            "target_response": target,
            "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
            "num_positions": N,
            "context_input_ids": context_slice,
            "context_positions": list(positions),
        })

    return datapoints[:num_examples]
