"""
CoT Decorative Detection — Is this CoT load-bearing or decorative?

Binary classification from corpus metadata (cot_correct + direct_correct).
Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_decorative_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate decorative CoT detection data with multi-layer stride.

    load_bearing: CoT correct, direct wrong -> reasoning was necessary
    decorative: both correct -> model already knew the answer
    """
    from cot_utils import get_cot_stride_positions, layer_percent_to_layer

    random.seed(seed)

    LAYERS = [
        layer_percent_to_layer(model_name, 25),
        layer_percent_to_layer(model_name, 50),
        layer_percent_to_layer(model_name, 75),
    ]

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip():
                    corpus.append(entry)

    load_bearing = [e for e in corpus if e.get("category") == "load_bearing"]
    decorative = [e for e in corpus if e.get("category") == "both_correct"]

    # Fallback: use cot_correct + direct_correct fields
    if not load_bearing or not decorative:
        load_bearing = [e for e in corpus
                        if e.get("cot_correct") and not e.get("direct_correct")]
        decorative = [e for e in corpus
                      if e.get("cot_correct") and e.get("direct_correct")]

    if not load_bearing:
        raise ValueError("No load_bearing entries in corpus")
    if not decorative:
        raise ValueError("No decorative entries in corpus")

    print(f"  load_bearing: {len(load_bearing)}, decorative: {len(decorative)}")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []
    attempts = 0

    while len(datapoints) < num_examples and attempts < num_examples * 3:
        attempts += 1

        if len(datapoints) % 2 == 0:
            entry = random.choice(load_bearing)
            target = "load_bearing"
        else:
            entry = random.choice(decorative)
            target = "decorative"

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

        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        positions = get_cot_stride_positions(
            prompt_len, len(full_ids),
            stride=stride,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * 3
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Is this chain-of-thought load-bearing or decorative? "
            f"Answer: load_bearing or decorative."
        )

        datapoints.append({
            "datapoint_type": "cot_decorative",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} decorative detection examples")
    return datapoints[:num_examples]
