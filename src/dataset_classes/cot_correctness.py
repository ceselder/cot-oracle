"""
CoT Correctness Prediction — Is the model's final answer correct?

Binary classification: given CoT activations, predict whether the
model's final answer is correct or incorrect.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_correctness_data(
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
    Generate correctness prediction training data with multi-layer stride.

    Each example: stride activations from CoT -> correct / incorrect.
    Ground truth from corpus (cot_correct field). Balanced 50/50.
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip():
                    corpus.append(entry)

    correct_pool = [e for e in corpus if e.get("cot_correct")]
    incorrect_pool = [e for e in corpus if not e.get("cot_correct")]

    if not correct_pool:
        raise ValueError("No correct entries in corpus")
    if not incorrect_pool:
        raise ValueError("No incorrect entries in corpus")

    print(f"  correct: {len(correct_pool)}, incorrect: {len(incorrect_pool)}")

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
            entry = random.choice(correct_pool)
            target = "correct"
        else:
            entry = random.choice(incorrect_pool)
            target = "incorrect"

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
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Is the model's final answer correct? Answer: correct or incorrect."
        )

        datapoints.append({
            "datapoint_type": "cot_correctness",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} correctness examples")
    return datapoints[:num_examples]
