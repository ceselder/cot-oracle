"""
CoT Next-Step Prediction — Single activation at cutoff.

Pick a random cutoff position k in the CoT, feed only the activation at
that position, and predict the next 50 tokens after it.
"""

import json
import random
import re

from transformers import AutoTokenizer


def load_cot_next_step_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 60000,
    stride: int | str = None,
    predict_tokens: int = 50,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate next-step prediction training data.

    For each example:
      1. Tokenize question + CoT
      2. Get stride positions over the CoT
      3. Pick a random cutoff k (where we stop feeding activations)
      4. Sample n_feed ~ Uniform(1, k+1) with 1 double-weighted,
         then randomly pick n_feed positions from [0..k]
      5. Target = next `predict_tokens` tokens after the k-th stride position

    Returns list of dicts compatible with dicts_to_training_data().
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip():
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No valid entries in {corpus_path}")

    print(f"  Loaded {len(corpus)} corpus entries for next-step prediction")

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 5

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        entry = random.choice(corpus)

        # Tokenize question + CoT (thinking part only)
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

        # Get all stride positions
        all_positions = get_cot_positions(
            prompt_len, len(full_ids),
            stride=stride, tokenizer=tokenizer, input_ids=full_ids,
        )
        if len(all_positions) < 3:
            continue

        # Pick random cutoff positions (up to 3 per entry)
        max_k = len(all_positions) - 1
        for _ in range(3):
            k = random.randint(0, max_k)
            cutoff_pos = all_positions[k]

            # Target: next predict_tokens tokens after cutoff
            target_start = cutoff_pos + 1
            target_end = min(target_start + predict_tokens, len(full_ids))
            if target_end - target_start < 5:
                continue

            target_ids = full_ids[target_start:target_end]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            if not target_text.strip():
                continue

            # Only feed the single activation at the cutoff position
            context_positions = [cutoff_pos] * len(LAYERS)
            num_positions = len(context_positions)

            # Context: tokens up to cutoff position
            context_slice = full_ids[:cutoff_pos + 1]

            layers_str = ", ".join(str(l) for l in LAYERS)
            prompt = (
                f"Activations from {num_positions} positions across layers {layers_str}. "
                f"Predict the next {target_end - target_start} tokens of reasoning."
            )

            datapoints.append({
                "datapoint_type": "cot_next_step",
                "prompt": prompt,
                "target_response": target_text,
                "layer": LAYERS[0],
                "layers": LAYERS,
                "num_positions": num_positions,
                "context_input_ids": context_slice,
                "context_positions": context_positions,
            })

            if len(datapoints) >= num_examples:
                break

    print(f"  Generated {len(datapoints)} next-step prediction examples")
    return datapoints[:num_examples]
