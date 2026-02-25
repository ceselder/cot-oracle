"""
CoT Answer Prediction — Predict the final answer from CoT activations.

Feed all stride activations from the full CoT trace (3 layers) and have
the oracle predict what the model's final answer will be.

This teaches the oracle to understand how chain-of-thought drives the
final answer — a prerequisite for detecting when the CoT is decorative
or when hidden factors override stated reasoning.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random
import re

from transformers import AutoTokenizer


def _extract_answer_text(entry: dict) -> str | None:
    """Extract a clean answer string from a corpus entry.

    Tries multiple fields: direct_response, answer, correct_answer.
    For math problems, tries to extract just the numerical answer.
    """
    # Try direct_response first (the model's actual answer after </think>)
    direct = entry.get("direct_response", "").strip()
    if direct:
        direct = re.sub(r"<think>.*?</think>", "", direct, flags=re.DOTALL).strip()
        boxed = re.search(r"\\boxed\{([^}]+)\}", direct)
        if boxed:
            return boxed.group(1).strip()
        if len(direct) < 200:
            return direct
        return direct[:200]

    # Try answer field
    answer = entry.get("answer") or entry.get("correct_answer") or ""
    answer = str(answer).strip()
    if answer:
        return str(answer)[:200]

    return None


def load_cot_answer_prediction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 40000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate answer prediction training data.

    For each example:
      1. Tokenize question + full CoT
      2. Get all stride positions over the CoT
      3. Feed all positions from all 3 layers + prompt positions
      4. Target = the model's final answer

    Returns list of dicts compatible with dicts_to_training_data().
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
        raise ValueError(f"No valid entries in {corpus_path}")

    print(f"  Loaded {len(corpus)} corpus entries for answer prediction")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []
    indices = list(range(len(corpus)))
    passes_needed = (num_examples // max(1, len(corpus))) + 2

    for _ in range(passes_needed):
        if len(datapoints) >= num_examples:
            break
        random.shuffle(indices)

        for idx in indices:
            if len(datapoints) >= num_examples:
                break

            entry = corpus[idx]
            answer = _extract_answer_text(entry)
            if not answer:
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
                f"What is the model's final answer?"
            )

            datapoints.append({
                "datapoint_type": "cot_answer_prediction",
                "prompt": prompt,
                "target_response": answer,
                "layer": LAYERS[0],
                "layers": LAYERS,
                "num_positions": num_positions,
                "context_input_ids": context_slice,
                "context_positions": context_positions,
            })

    print(f"  Generated {len(datapoints)} answer prediction examples")
    return datapoints[:num_examples]
