"""
Full CoT Reconstruction (~15K examples)

Feed ALL sentence boundary activations, predict the entire CoT text.
Teaches rich semantic reconstruction from the full activation trajectory.

Target is truncated to max_target_tokens to keep training manageable.
"""

import json
import random
import re

from transformers import AutoTokenizer


def load_cot_full_reconstruction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    max_target_tokens: int = 8192,
    seed: int = 42,
) -> list[dict]:
    """
    Generate full CoT reconstruction training data.

    Each example: all sentence boundary activations -> predict full CoT text.
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if (entry.get("boundary_positions")
                        and len(entry["boundary_positions"]) >= 2):
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No entries with boundary_positions in {corpus_path}")

    layer = layer_percent_to_layer(model_name, 50)

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 5

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        entry = random.choice(corpus)

        boundary_positions = entry["boundary_positions"]
        if len(boundary_positions) < 2:
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
        if len(positions) < 2:
            continue

        N = len(positions)
        context_slice = full_ids[:positions[-1] + 1]

        # Target: the full CoT text, cleaned and truncated
        clean_cot = re.sub(r"<think>|</think>", "", entry["cot_response"]).strip()
        if not clean_cot:
            continue

        # Truncate to max_target_tokens
        target_ids = tokenizer.encode(clean_cot, add_special_tokens=False)
        if len(target_ids) > max_target_tokens:
            target_text = tokenizer.decode(target_ids[:max_target_tokens], skip_special_tokens=True)
        else:
            target_text = clean_cot

        if not target_text.strip():
            continue

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"Reconstruct the full chain of thought."
        )

        datapoints.append({
            "datapoint_type": "cot_full_reconstruction",
            "prompt": prompt,
            "target_response": target_text,
            "layer": layer,
            "num_positions": N,
            "context_input_ids": context_slice,
            "context_positions": list(positions),
        })

    print(f"  Generated {len(datapoints)} full reconstruction examples")
    return datapoints[:num_examples]
