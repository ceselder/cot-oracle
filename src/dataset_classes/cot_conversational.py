"""
Conversational CoT QA — open-ended questions about CoT activations.

Loads pre-generated conversational QA from JSONL (built by generate_conv_qa.py
or build_cot_qa_dataset.py). Diverse question types about CoT, answered by
DeepSeek v3.2 or Gemini Flash.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random
import re

from transformers import AutoTokenizer


def load_cot_conversational_data(
    corpus_path: str,
    qa_jsonl_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversational QA training data with multi-layer stride activations.

    Matches QA pairs (by corpus_id) to corpus entries for activation extraction.
    Uses 3 layers and stride-based positions.

    Returns list of dicts compatible with dicts_to_training_data().
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    # Load corpus into lookup by id
    corpus_by_id = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus_by_id[entry["id"]] = entry

    # Load QA pairs
    qa_pairs = []
    with open(qa_jsonl_path) as f:
        for line in f:
            if line.strip():
                qa = json.loads(line)
                if qa.get("corpus_id") in corpus_by_id:
                    qa_pairs.append(qa)

    if not qa_pairs:
        raise ValueError(f"No matching QA pairs between {qa_jsonl_path} and {corpus_path}")

    random.shuffle(qa_pairs)
    print(f"  Loaded {len(qa_pairs)} matching QA pairs")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []

    for qa in qa_pairs:
        if len(datapoints) >= num_examples:
            break

        entry = corpus_by_id[qa["corpus_id"]]

        # Build tokenized context
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry.get("cot_response", "")
        if not cot_text.strip():
            continue
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

        # Extract the question text from the QA pair prompt
        # Old format: "Activations from N sentence boundaries. <question>"
        # New format: use question directly with stride-based prompt
        original_prompt = qa.get("prompt", "")
        # Try to extract just the question part
        q_match = re.search(r"(?:sentence boundaries|positions)\.\s*(.*)", original_prompt)
        question_text = q_match.group(1) if q_match else original_prompt

        target = qa["target_response"]
        if not target or not target.strip():
            continue

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"{question_text}"
        )

        datapoints.append({
            "datapoint_type": "cot_conversational",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} conversational QA examples "
          f"(from {len(qa_pairs)} available pairs)")
    return datapoints[:num_examples]
