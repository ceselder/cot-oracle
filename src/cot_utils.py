"""
Lightweight CoT utilities (no peft/torch dependency).

Extracted from ao_lib.py for use in CPU-only scripts like corpus generation.
"""

import re


def split_cot_into_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences. Removes <think> tags first."""
    text = re.sub(r'<think>|</think>', '', cot_text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def find_sentence_boundary_positions(
    tokenizer,
    formatted_text: str,
    sentences: list[str],
) -> list[int]:
    """
    Find the token positions of the last token of each sentence in the formatted text.
    Returns a list of token positions (indices into the tokenized sequence).
    """
    tokens = tokenizer(formatted_text, add_special_tokens=False)
    token_ids = tokens["input_ids"]
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids[0].tolist() if len(token_ids.shape) > 1 else token_ids.tolist()
    elif isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    full_text_decoded = tokenizer.decode(token_ids)

    positions = []
    search_start = 0

    for sentence in sentences:
        anchor = sentence[-20:] if len(sentence) > 20 else sentence
        idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            anchor = sentence[-10:] if len(sentence) > 10 else sentence
            idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            continue

        char_end = idx + len(anchor)
        search_start = char_end

        char_count = 0
        token_pos = -1
        for t_idx, t_id in enumerate(token_ids):
            decoded = tokenizer.decode([t_id])
            char_count += len(decoded)
            if char_count >= char_end:
                token_pos = t_idx
                break

        if token_pos >= 0:
            positions.append(token_pos)

    return positions


# Layer count lookup (no torch dependency)
LAYER_COUNTS = {
    "Qwen/Qwen3-0.6B": 28,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-4B": 36,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-14B": 40,
    "Qwen/Qwen3-32B": 64,
}


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))
