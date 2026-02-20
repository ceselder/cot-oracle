"""
Task 2: Context Prediction — 25-Token Stride (~30K examples)

Feed the ENTIRE CoT trajectory at fixed 25-token stride positions, reconstruct chunks.
Uses 3 layers per position [25%, 50%, 75%].

Question: "Activations from {N} strided CoT positions (every 25 tokens).
           Predict the next {K} tokens after position {T}."
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_sentence_prediction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 30000,
    max_sentences: int = 15,
    min_k_tokens: int = 1,
    max_k_tokens: int = 15,
    seed: int = 42,
) -> list[dict]:
    """
    Generate strided-position context prediction training data.

    Each example injects activations from all CoT positions on a fixed stride
    (every 25 tokens, capped at `max_sentences` positions), 3 per position
    (one per layer). The oracle predicts tokens near a randomly chosen position.

    Returns list of dicts for create_training_datapoint().
    """
    from cot_utils import layer_percent_to_layer, get_cot_stride_positions

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 10

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        entry = random.choice(corpus)

        # Build full text for tokenization
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
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])

        positions = get_cot_stride_positions(
            prompt_token_count=prompt_len,
            total_token_count=len(full_ids),
            stride=25,
            max_positions=max_sentences,
        )
        if len(positions) < 2:
            continue

        N = len(positions)

        # Pick target position and direction
        target_idx = random.randint(0, N - 1)
        direction = random.choice(["future", "past"])
        k_tokens = random.randint(min_k_tokens, max_k_tokens)

        pivot_pos = positions[target_idx]

        if direction == "future":
            start = pivot_pos + 1
            end = start + k_tokens
            if end > len(full_ids):
                continue
            target_ids = full_ids[start:end]
            question = (
                f"Activations from {N} strided CoT positions (every 25 tokens). "
                f"Predict the next {k_tokens} tokens after position {target_idx + 1}."
            )
        else:
            end = pivot_pos
            start = end - k_tokens
            if start < 0:
                continue
            target_ids = full_ids[start:end]
            question = (
                f"Activations from {N} strided CoT positions (every 25 tokens). "
                f"Predict the {k_tokens} tokens before position {target_idx + 1}."
            )

        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue

        # Context for on-the-fly collection
        context_cutoff = max(positions)
        context_input_ids_slice = full_ids[:context_cutoff + 1]

        datapoints.append({
            "datapoint_type": "cot_sentence_prediction",
            "prompt": question,
            "target_response": target_text,
            "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
            "num_positions": len(positions),
            "context_input_ids": context_input_ids_slice,
            "context_positions": list(positions),
        })

    return datapoints[:num_examples]
