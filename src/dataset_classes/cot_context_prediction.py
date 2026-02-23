"""
Task 1: CoT Context Prediction — Random Positions (~100K examples)

High-volume backbone. Random positions in the CoT, 1 random layer per example
(randomly chosen from layer_percents). AO's materialize_missing_steering_vectors
handles heterogeneous batches where different examples request different layers.

v3 change: Always include ~5 prompt activation positions alongside CoT
positions, so the oracle always knows what question is being answered.

v4 change: Random layer per example from [25%, 50%, 75%] depth.
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def _get_prompt_positions(formatted_len: int, n_positions: int = 5) -> list[int]:
    """Get evenly spaced positions from the prompt region."""
    if formatted_len < n_positions:
        return list(range(formatted_len))
    step = formatted_len / (n_positions + 1)
    return [int(step * (i + 1)) for i in range(n_positions)]


def load_cot_context_prediction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 100000,
    min_k_tokens: int = 1,
    max_k_tokens: int = 20,
    min_k_activations: int = 1,
    max_k_activations: int = 20,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate PastLens-style training data from CoT corpus.

    Each example randomly picks one layer from layer_percents.
    AO handles heterogeneous batches (different layers per example).

    Returns list of dicts with keys needed for create_training_datapoint():
        datapoint_type, prompt, target_response, layer, num_positions,
        context_input_ids, context_positions
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    # Load corpus
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

    while len(datapoints) < num_examples:
        attempts += 1
        if attempts % 10000 == 0:
            print(f"  context_pred: {len(datapoints)}/{num_examples} after {attempts} attempts")
        if attempts > num_examples * 20:
            print(f"  WARNING: Too many skips ({attempts} attempts for {len(datapoints)} examples), stopping")
            break
        entry = random.choice(corpus)

        # Tokenize only the CoT thinking portion (NOT the final answer)
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry["cot_response"]
        # Truncate at </think> to exclude final answer
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        full_text = formatted + cot_text
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        L = len(context_ids)

        # Compute where the prompt ends and CoT begins
        formatted_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(formatted_ids)

        # Random parameters
        k_tokens = random.randint(min_k_tokens, max_k_tokens)
        k_acts = random.randint(min_k_activations, max_k_activations)
        direction = random.choice(["past", "future"])

        # Get CoT activation positions
        if direction == "past":
            act_begin_min = max(k_tokens, prompt_len)
            act_begin_max = L - k_acts - 1
            if act_begin_max < act_begin_min:
                continue
            act_begin = random.randint(act_begin_min, act_begin_max)
            cot_act_positions = list(range(act_begin, act_begin + k_acts))
            token_positions = list(range(act_begin - k_tokens, act_begin))
            target_tokens = [context_ids[p] for p in token_positions]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            prompt_text = f"Can you predict the previous {k_tokens} tokens that came before this?"
        else:
            act_begin_min = max(1, prompt_len)
            act_begin_max = L - k_acts - k_tokens
            if act_begin_max < act_begin_min:
                continue
            act_begin = random.randint(act_begin_min, act_begin_max)
            cot_act_positions = list(range(act_begin, act_begin + k_acts))
            last_act = cot_act_positions[-1]
            token_positions = list(range(last_act + 1, last_act + 1 + k_tokens))
            target_tokens = [context_ids[p] for p in token_positions]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            prompt_text = f"Can you predict the next {k_tokens} tokens that come after this?"

        # Get prompt positions (evenly spaced in the prompt region)
        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)

        # Combine: prompt positions first, then CoT positions
        all_positions = prompt_positions + cot_act_positions

        # Pick one random layer for this example
        layer = random.choice(layers)

        # Context for on-the-fly collection: tokens up through last activation position
        context_cutoff = max(all_positions)
        context_input_ids_slice = context_ids[:context_cutoff + 1]

        datapoints.append({
            "datapoint_type": "cot_context_prediction",
            "prompt": prompt_text,
            "target_response": target_text,
            "layer": layer,
            "num_positions": len(all_positions),
            "context_input_ids": context_input_ids_slice,
            "context_positions": all_positions,
        })

    return datapoints[:num_examples]
