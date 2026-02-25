"""
CoT Importance / Thought Anchor Detection — Is this step important?

Binary classification per-position: given activation at one stride position,
predict whether it's in an important region of the CoT.

Approximation: since we don't have KL divergence labels, we use a heuristic:
- First and last quarters of CoT are more likely to be important (planning + conclusion)
- Middle is more likely to be filler computation

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_importance_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate importance detection data with multi-layer stride.

    Heuristic labeling:
    - Feed ALL stride positions from full CoT
    - Target: "important" if CoT was load-bearing AND entry was correct
    -         "routine" if CoT was decorative OR entry was incorrect

    This is a simpler version that classifies the overall CoT importance
    rather than per-sentence importance (which requires expensive resampling).
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

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    # Classify entries
    important_pool = []
    routine_pool = []
    for entry in corpus:
        cot_correct = entry.get("cot_correct", False)
        direct_correct = entry.get("direct_correct", False)

        if cot_correct and not direct_correct:
            # CoT was essential — load-bearing
            important_pool.append(entry)
        elif cot_correct and direct_correct:
            # Both correct — CoT was decorative
            routine_pool.append(entry)
        elif not cot_correct:
            # CoT led to wrong answer — not important
            routine_pool.append(entry)

    if not important_pool:
        raise ValueError("No important entries found (need cot_correct=True, direct_correct=False)")
    if not routine_pool:
        raise ValueError("No routine entries found")

    print(f"  important: {len(important_pool)}, routine: {len(routine_pool)}")

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
            entry = random.choice(important_pool)
            target = "important"
        else:
            entry = random.choice(routine_pool)
            target = "routine"

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
            f"Was this reasoning important for getting the correct answer? "
            f"Answer: important or routine."
        )

        datapoints.append({
            "datapoint_type": "cot_importance",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} importance detection examples")
    return datapoints[:num_examples]
