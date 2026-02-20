"""
Lightweight CoT utilities (no peft/torch dependency).

Extracted from ao_lib.py for use in CPU-only scripts like corpus generation.
"""

import bisect
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

    Uses a single pass to build char→token mapping instead of per-token decode.
    """
    tokens = tokenizer(formatted_text, add_special_tokens=False)
    token_ids = tokens["input_ids"]
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids[0].tolist() if len(token_ids.shape) > 1 else token_ids.tolist()
    elif isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    full_text_decoded = tokenizer.decode(token_ids)

    # Build cumulative char offsets in ONE pass (the old code re-scanned per sentence)
    cum_chars = []
    char_count = 0
    for t_id in token_ids:
        char_count += len(tokenizer.decode([t_id]))
        cum_chars.append(char_count)

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

        # Binary search for the token whose cumulative chars >= char_end
        token_pos = bisect.bisect_left(cum_chars, char_end)
        if token_pos < len(token_ids):
            positions.append(token_pos)

    return positions


def get_cot_stride_positions(
    prompt_token_count: int,
    total_token_count: int,
    stride: int = 25,
    max_positions: int | None = None,
    include_last: bool = True,
) -> list[int]:
    """Get fixed-stride positions over the CoT token region.

    Positions start right after the prompt tokens and proceed every `stride` tokens.
    Optionally includes the last token to keep late-CoT signal.
    """
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1

    if cot_end - cot_start + 1 < 2:
        return []

    positions = list(range(cot_start, cot_end + 1, max(1, stride)))
    if include_last and positions and positions[-1] != cot_end:
        positions.append(cot_end)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    positions = deduped

    if max_positions is not None and len(positions) > max_positions:
        # Keep strict fixed-stride spacing; do not subsample/warp spacing.
        positions = positions[:max_positions]

    if len(positions) < 2:
        return [cot_start, cot_end]
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
