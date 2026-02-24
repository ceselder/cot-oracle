"""
Causal CoT Prediction — Stage-wise (~30K examples)

For sentence i, feed activations from sentences 1..i ONLY (causal masking).
Predict the next sentence or next N tokens after sentence i.

Unlike sentence_prediction which feeds ALL boundaries and picks a target,
this is autoregressive: the oracle only sees what came before.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_causal_prediction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 30000,
    max_sentences: int = 15,
    min_k_tokens: int = 5,
    max_k_tokens: int = 25,
    seed: int = 42,
) -> list[dict]:
    """
    Generate causal (stage-wise) CoT prediction data.

    Each example: activations at boundaries 1..i -> predict next chunk.
    The oracle can only see past activations, mimicking causal attention.
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if (entry.get("boundary_positions")
                        and len(entry["boundary_positions"]) >= 3):
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No entries with >=3 boundary_positions in {corpus_path}")

    layer = layer_percent_to_layer(model_name, 50)

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 5

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        entry = random.choice(corpus)

        boundary_positions = entry["boundary_positions"]
        sentences = entry.get("sentences", [])
        if len(boundary_positions) < 3 or len(sentences) < 3:
            continue

        # Build tokenized context
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

        # Cap and validate positions
        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(full_ids)]
        if len(positions) < 3:
            continue

        N = len(positions)

        # Pick a causal cutoff: feed 1..i, predict after i
        # i must be at least 1 and leave room for target
        i = random.randint(1, N - 1)
        causal_positions = positions[:i]  # only past activations

        # Target: tokens after boundary i
        boundary_pos = positions[i - 1]  # last included boundary
        start = boundary_pos + 1
        k_tokens = random.randint(min_k_tokens, max_k_tokens)
        end = min(start + k_tokens, len(full_ids))

        if end <= start:
            continue

        target_ids = full_ids[start:end]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue

        # Context: tokens up through the last causal boundary
        context_slice = full_ids[:causal_positions[-1] + 1]

        prompt = (
            f"Activations from {i} of {N} reasoning steps so far. "
            f"Predict what comes next in the reasoning."
        )

        datapoints.append({
            "datapoint_type": "cot_causal_prediction",
            "prompt": prompt,
            "target_response": target_text,
            "layer": layer,
            "num_positions": len(causal_positions),
            "context_input_ids": context_slice,
            "context_positions": list(causal_positions),
        })

    print(f"  Generated {len(datapoints)} causal prediction examples")
    return datapoints[:num_examples]
