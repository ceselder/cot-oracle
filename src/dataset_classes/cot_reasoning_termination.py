"""
CoT Reasoning Termination Prediction — Will the model stop thinking soon?

Binary classification: given CoT activations up to a truncation point,
predict whether the model will emit </think> within the next 100 tokens.

Positive ("will_terminate"): prefix cut 25-55 tokens from actual </think>
Negative ("will_continue"):  prefix cut 300+ tokens from actual </think>
Ambiguous range (56-299 tokens) is skipped.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_reasoning_termination_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int = 5,
    max_positions_per_layer: int = 20,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate reasoning termination prediction training data.

    For each example:
      1. Get full CoT from corpus (must contain </think>)
      2. Pick a random truncation point in the CoT token sequence
      3. Compute remaining_tokens = distance from truncation to </think>
      4. Label: 25-55 remaining -> "will_terminate", 300+ -> "will_continue"
      5. Get stride positions up to truncation, triple for 3 layers

    Balanced 50/50 between will_terminate and will_continue.
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
                cot = entry.get("cot_response", "")
                if cot.strip():
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No entries with cot_response in corpus at {corpus_path}")

    print(f"  Corpus: {len(corpus)} entries with CoT")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Pre-tokenize all entries to find which can produce positive/negative examples
    tokenized = []
    for entry in corpus:
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        # cot_response is the thinking text (</think> tag already stripped)
        cot_text = entry["cot_response"]

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        cot_len = len(full_ids) - prompt_len

        if cot_len < 60:  # need at least 60 CoT tokens for positive examples
            continue

        tokenized.append({
            "entry": entry,
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "cot_len": cot_len,
        })

    # Split into pools based on what examples they can produce
    # Positive: needs cot_len >= 25 (truncate 25-55 from end)
    # Negative: needs cot_len >= 300 (truncate 300+ from end)
    positive_pool = [t for t in tokenized if t["cot_len"] >= 25]
    negative_pool = [t for t in tokenized if t["cot_len"] > 300]

    if not positive_pool:
        raise ValueError("No entries long enough for positive examples (need >=25 CoT tokens)")
    if not negative_pool:
        raise ValueError("No entries long enough for negative examples (need >300 CoT tokens)")

    print(f"  Positive pool (>=25 CoT tokens): {len(positive_pool)}")
    print(f"  Negative pool (>300 CoT tokens): {len(negative_pool)}")

    datapoints = []
    attempts = 0

    while len(datapoints) < num_examples and attempts < num_examples * 5:
        attempts += 1

        # Alternate between positive and negative
        if len(datapoints) % 2 == 0:
            # Positive: truncate 25-55 tokens from end
            t = random.choice(positive_pool)
            remaining = random.randint(25, min(55, t["cot_len"] - 1))
            trunc_pos = t["prompt_len"] + t["cot_len"] - remaining
            target = "will_terminate"
        else:
            # Negative: truncate 300+ tokens from end
            t = random.choice(negative_pool)
            remaining = random.randint(300, t["cot_len"] - 1)
            trunc_pos = t["prompt_len"] + t["cot_len"] - remaining
            target = "will_continue"

        # Sanity: truncation must be after prompt
        if trunc_pos <= t["prompt_len"] + 5:
            continue

        # Get stride positions up to truncation point
        positions = get_cot_stride_positions(
            t["prompt_len"], trunc_pos,
            stride=stride,
            max_positions=max_positions_per_layer,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(t["prompt_len"], n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * 3
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = t["full_ids"][:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Will the model terminate reasoning (emit </think>) within the next 100 tokens? "
            f"Answer: will_terminate or will_continue."
        )

        datapoints.append({
            "datapoint_type": "cot_reasoning_termination",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} reasoning termination examples "
          f"({sum(1 for d in datapoints if d['target_response'] == 'will_terminate')} pos, "
          f"{sum(1 for d in datapoints if d['target_response'] == 'will_continue')} neg)")
    return datapoints[:num_examples]
