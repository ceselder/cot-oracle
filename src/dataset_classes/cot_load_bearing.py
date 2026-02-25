"""
CoT Load-Bearing Detection — Is this CoT necessary for the correct answer?

Binary classification: given CoT activations, predict whether the
chain-of-thought is load-bearing (model needs it to answer correctly)
or decorative (model would get it right without CoT too).

Ground truth comes from comparing the model's answer with vs without CoT.
Corpus entries with direct_response let us check this without regeneration.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random
import re

from transformers import AutoTokenizer


def _normalize_answer(text: str) -> str:
    """Normalize an answer for comparison."""
    text = text.strip().lower()
    text = re.sub(r"[,\s]+", "", text)
    # Extract boxed
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    # Try to extract a number
    num = re.search(r"-?\d+\.?\d*", text)
    if num:
        return num.group()
    return text[:50]


def load_cot_load_bearing_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 30000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate load-bearing detection training data.

    For each corpus entry that has both cot_response and direct_response:
      1. Compare answers: does CoT change the answer?
      2. If correct_answer is available, check if CoT helped get it right
      3. Label: "load_bearing" if CoT needed, "decorative" if not
      4. Feed all stride activations from CoT, predict the label

    Returns list of dicts compatible with dicts_to_training_data().
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Need both CoT and direct response to determine load-bearing
                if (entry.get("cot_response", "").strip()
                        and entry.get("direct_response", "").strip()):
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No entries with both cot_response and direct_response in {corpus_path}")

    print(f"  Loaded {len(corpus)} corpus entries with both CoT and direct responses")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Determine labels
    labeled = []
    for entry in corpus:
        cot_answer = _normalize_answer(entry["cot_response"])
        direct_answer = _normalize_answer(entry["direct_response"])
        correct = entry.get("correct_answer", entry.get("answer", ""))

        if correct:
            correct_norm = _normalize_answer(str(correct))
            cot_correct = cot_answer == correct_norm
            direct_correct = direct_answer == correct_norm

            if cot_correct and not direct_correct:
                label = "load_bearing"
            elif cot_correct and direct_correct:
                label = "decorative"
            elif not cot_correct and direct_correct:
                label = "decorative"  # CoT actually hurt
            else:
                # Both wrong — skip, ambiguous
                continue
        else:
            # No ground truth — compare if answers differ
            if cot_answer != direct_answer:
                label = "load_bearing"  # CoT changed the answer
            else:
                label = "decorative"

        labeled.append((entry, label))

    # Balance classes
    load_bearing = [(e, l) for e, l in labeled if l == "load_bearing"]
    decorative = [(e, l) for e, l in labeled if l == "decorative"]
    print(f"  Labels: {len(load_bearing)} load_bearing, {len(decorative)} decorative")

    # Oversample the minority class to balance
    if load_bearing and decorative:
        minority = min(len(load_bearing), len(decorative))
        max_per_class = num_examples // 2
        target_per_class = min(max_per_class, max(len(load_bearing), len(decorative)))

        def oversample(items, target):
            result = []
            while len(result) < target:
                result.extend(items)
            random.shuffle(result)
            return result[:target]

        balanced = (
            oversample(load_bearing, target_per_class) +
            oversample(decorative, target_per_class)
        )
        random.shuffle(balanced)
    else:
        balanced = labeled

    datapoints = []
    for entry, label in balanced:
        if len(datapoints) >= num_examples:
            break

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
            f"Is this chain of thought load-bearing or decorative? "
            f"Answer: load_bearing or decorative."
        )

        datapoints.append({
            "datapoint_type": "cot_load_bearing",
            "prompt": prompt,
            "target_response": label,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} load-bearing detection examples")
    return datapoints[:num_examples]
