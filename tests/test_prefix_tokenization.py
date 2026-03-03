"""Test oracle prefix tokenization with the real Qwen3-8B tokenizer.

Validates that manual prefix tokenization (split text, insert placeholder IDs
directly) is necessary because naive tokenization merges the last " ?" with
the trailing "\\n" into a single " ?\\n" token (17607), losing one placeholder.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ao_reference"))

import pytest
from transformers import AutoTokenizer

PLACEHOLDER_TOKEN = " ?"
MODEL_NAME = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    if not tok.pad_token_id:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _build_prefix(num_positions: int, layers: list[int], ph: str = PLACEHOLDER_TOKEN) -> str:
    if len(layers) == 1:
        return f"L{layers[0]}:" + ph * num_positions + "\n"
    k = num_positions // len(layers)
    return " ".join(f"L{layer}:" + ph * k for layer in layers) + "\n"


def _manual_prefix_token_ids(tokenizer, num_positions, layers, ph_id):
    """Manual approach: tokenize labels individually, insert placeholder IDs."""
    prefix_layers = list(layers)
    block_sizes = [num_positions]
    if len(prefix_layers) > 1:
        k = num_positions // len(prefix_layers)
        block_sizes = [k] * len(prefix_layers)

    prefix_ids = []
    positions = []
    for i, (layer_idx, block_size) in enumerate(zip(prefix_layers, block_sizes)):
        label = f"L{layer_idx}:"
        if i > 0:
            label = " " + label
        prefix_ids.extend(tokenizer.encode(label, add_special_tokens=False))
        positions.extend(range(len(prefix_ids), len(prefix_ids) + block_size))
        prefix_ids.extend([ph_id] * block_size)
    prefix_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
    return prefix_ids, positions


def _tokenize_manual(tokenizer, formatted, prefix, ph_id, num_positions, layers):
    """Manual approach: split around prefix, insert placeholder IDs directly."""
    idx = formatted.find(prefix)
    assert idx >= 0
    before_ids = tokenizer.encode(formatted[:idx], add_special_tokens=False)
    after_ids = tokenizer.encode(formatted[idx + len(prefix):], add_special_tokens=False)
    prefix_ids, rel_positions = _manual_prefix_token_ids(tokenizer, num_positions, layers, ph_id)
    input_ids = before_ids + prefix_ids + after_ids
    positions = [len(before_ids) + p for p in rel_positions]
    return input_ids, positions


# ── Parameterized configs ──

LAYER_CONFIGS = [
    [9],
    [9, 18, 27],
    [0, 17],
    [9, 18, 27, 35],
]

PROMPTS = [
    "Is the stated answer typical or atypical for the question?",
    "Predict the next 50 tokens of reasoning.",
    "Does the chain of thought admit to using a hint? Answer yes or no.",
    "What is 2+2? Is it 4? Or maybe 5?",
]

POSITION_COUNTS = [3, 12, 30, 60]


# ── Core tests ──

def test_placeholder_is_single_token(tokenizer):
    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert len(ph_id) == 1


def test_last_placeholder_merges_with_newline(tokenizer):
    """The tokenizer merges ' ?' + '\\n' into ' ?\\n' (token 17607).
    This is the fundamental reason why naive tokenize-and-scan fails."""
    ids = tokenizer.encode(" ?\n", add_special_tokens=False)
    assert len(ids) == 1, f"Expected ' ?\\n' to be a single token, got {ids}"
    assert ids[0] != 937, "Should NOT be the plain ' ?' token"

    # Verify the merged token is different from the placeholder
    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]
    merged_id = ids[0]
    assert merged_id != ph_id


def test_naive_scan_loses_last_placeholder(tokenizer):
    """Naive tokenize-and-scan always finds num_positions-1 placeholders."""
    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]

    for layers in LAYER_CONFIGS:
        for num_pos in POSITION_COUNTS:
            num_positions = num_pos - (num_pos % len(layers)) or len(layers)
            prefix = _build_prefix(num_positions, layers)
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            found = sum(1 for t in ids if t == ph_id)
            assert found == num_positions - 1, (
                f"layers={layers}, K={num_positions}: "
                f"naive found {found}, expected {num_positions - 1} (last merges with \\n)"
            )


@pytest.mark.parametrize("layers", LAYER_CONFIGS, ids=lambda l: f"L{l}")
@pytest.mark.parametrize("num_pos", POSITION_COUNTS, ids=lambda n: f"K{n}")
@pytest.mark.parametrize("prompt", PROMPTS[:3], ids=lambda p: p[:30])
def test_manual_tokenization_finds_all_positions(tokenizer, layers, num_pos, prompt):
    """Manual prefix tokenization correctly finds all num_positions placeholders."""
    num_positions = num_pos - (num_pos % len(layers)) or len(layers)

    prefix = _build_prefix(num_positions, layers)
    full_prompt = prefix + prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]
    input_ids, positions = _tokenize_manual(tokenizer, formatted, prefix, ph_id, num_positions, layers)

    assert len(positions) == num_positions
    # Every position must point to the placeholder token ID
    for p in positions:
        assert input_ids[p] == ph_id, f"Position {p} has token {input_ids[p]}, expected {ph_id}"


@pytest.mark.parametrize("layers", LAYER_CONFIGS, ids=lambda l: f"L{l}")
def test_manual_positions_respect_layer_blocks(tokenizer, layers):
    """Placeholder positions are grouped by layer in the correct order."""
    num_positions = 12 - (12 % len(layers)) or len(layers)
    k = num_positions // len(layers)

    prefix = _build_prefix(num_positions, layers)
    prompt = "test"
    messages = [{"role": "user", "content": prefix + prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]
    input_ids, positions = _tokenize_manual(tokenizer, formatted, prefix, ph_id, num_positions, layers)

    # Positions should be monotonically increasing
    assert positions == sorted(positions)
    # Each layer block should be contiguous
    for li in range(len(layers)):
        block = positions[li * k : (li + 1) * k]
        assert block == list(range(block[0], block[0] + k)), (
            f"Layer {layers[li]} block not contiguous: {block}"
        )


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:30])
def test_prompt_question_marks_dont_interfere(tokenizer, prompt):
    """Question marks in the prompt text don't appear as placeholder tokens
    at prefix positions (manual approach computes positions structurally)."""
    layers = [9, 18, 27]
    num_positions = 12
    prefix = _build_prefix(num_positions, layers)
    messages = [{"role": "user", "content": prefix + prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]
    input_ids, positions = _tokenize_manual(tokenizer, formatted, prefix, ph_id, num_positions, layers)

    # All placeholder positions must be in the prefix region, not the prompt
    prefix_start = formatted.find(prefix)
    before_len = len(tokenizer.encode(formatted[:prefix_start], add_special_tokens=False))
    prefix_ids, _ = _manual_prefix_token_ids(tokenizer, num_positions, layers, ph_id)
    prefix_end = before_len + len(prefix_ids)

    for p in positions:
        assert before_len <= p < prefix_end, (
            f"Position {p} outside prefix region [{before_len}, {prefix_end})"
        )


def test_eval_and_train_produce_same_ids(tokenizer):
    """The eval-loop _build_manual_prefix_token_ids matches the training-path
    _build_manual_prefix_tokens from dataset_utils.py."""
    from eval_loop import _build_manual_prefix_token_ids
    from nl_probes.utils.dataset_utils import _build_manual_prefix_tokens, SPECIAL_TOKEN

    ph_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]

    for layers in LAYER_CONFIGS:
        for num_pos in POSITION_COUNTS:
            num_positions = num_pos - (num_pos % len(layers)) or len(layers)

            # eval path
            eval_ids, eval_pos = _build_manual_prefix_token_ids(
                tokenizer, num_positions, layers, ph_id,
            )

            # training path
            _, train_ids, train_pos = _build_manual_prefix_tokens(
                tokenizer, layers[0], num_positions, layers,
            )

            assert eval_ids == train_ids, (
                f"layers={layers}, K={num_positions}: token ID mismatch"
            )
            assert eval_pos == train_pos, (
                f"layers={layers}, K={num_positions}: position mismatch"
            )
