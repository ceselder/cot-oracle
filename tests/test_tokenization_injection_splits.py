"""Tests for CoT tokenization, activation injection hooks, and OOD split integrity.

Group 1 (tests 1-4): prepare_context_ids produces positions pointing at CoT tokens.
Group 2 (tests 5-8): Steering hooks modify exactly the target positions.
Group 3 (tests 9-10): OOD tasks have disjoint train/test sources and prompts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ao_reference"))

import pytest
import torch
from transformers import AutoTokenizer

from data_loading import prepare_context_ids, load_task_data
from core.ao import get_steering_hook, get_batched_steering_hook

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    if not tok.pad_token_id:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _make_item(question: str, cot_text: str, task: str = "test_task") -> dict:
    return {"task": task, "question": question, "cot_text": cot_text, "prompt": "Dummy prompt", "target_response": "yes", "datapoint_type": task}


# ─── Group 1: CoT Tokenization (prepare_context_ids) ───


def test_positions_point_to_cot_tokens_not_prompt(tokenizer):
    """Positions from prepare_context_ids must decode to CoT content, not template tokens."""
    question = "What is the capital of France?"
    cot_text = "The capital of France is Paris because it has been the seat of government for centuries."
    item = _make_item(question, cot_text)

    result = prepare_context_ids([item], tokenizer, layers=LAYERS)[0]
    full_ids = result["context_input_ids"]
    positions = result["context_positions"]

    n_layers = len(LAYERS)
    base_positions = positions[:len(positions) // n_layers]

    # Decode all tokens at base positions — they should be CoT content
    cot_token_ids = [full_ids[p] for p in base_positions]
    decoded_cot = tokenizer.decode(cot_token_ids)
    # The decoded text must overlap with the cot_text (not template/prompt)
    assert "Paris" in decoded_cot or "capital" in decoded_cot or "France" in decoded_cot, (
        f"Decoded CoT region doesn't match cot_text. Got: {decoded_cot!r}"
    )

    # Tokens *before* the CoT region should be prompt/template
    prompt_end = base_positions[0]
    prompt_ids = full_ids[:prompt_end]
    decoded_prompt = tokenizer.decode(prompt_ids)
    # Template tokens like <|im_start|> should appear in the prompt region
    assert "<|im_start|>" in decoded_prompt, (
        f"Expected template tokens before CoT region. Got: {decoded_prompt!r}"
    )
    # The question should appear in the prompt region
    assert "capital" in decoded_prompt.lower() or "france" in decoded_prompt.lower(), (
        f"Expected question content in prompt region. Got: {decoded_prompt!r}"
    )


def test_position_range_covers_full_cot(tokenizer):
    """Positions should span from prompt_len to len(full_ids)-1, contiguous within each layer."""
    question = "Explain gravity."
    cot_text = "Gravity is a fundamental force of nature that attracts objects with mass toward each other."
    item = _make_item(question, cot_text)

    result = prepare_context_ids([item], tokenizer, layers=LAYERS)[0]
    full_ids = result["context_input_ids"]
    positions = result["context_positions"]

    n_layers = len(LAYERS)
    K = len(positions) // n_layers
    base_positions = positions[:K]

    # Must be contiguous
    assert base_positions == list(range(base_positions[0], base_positions[0] + K)), (
        f"Base positions not contiguous: {base_positions[:10]}..."
    )

    # Compute expected prompt_len to verify start
    prompt_msgs = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    expected_start = len(prompt_ids)

    assert base_positions[0] == expected_start, (
        f"CoT start {base_positions[0]} != expected prompt_len {expected_start}"
    )

    # End should be at the last token of full_ids
    assert base_positions[-1] == len(full_ids) - 1, (
        f"CoT end {base_positions[-1]} != last token idx {len(full_ids) - 1}"
    )


def test_layer_tiling_is_correct(tokenizer):
    """For 3 layers, context_positions should be base_positions repeated 3 times."""
    item = _make_item("Why is the sky blue?", "Light scatters in the atmosphere due to Rayleigh scattering.")

    result = prepare_context_ids([item], tokenizer, layers=LAYERS)[0]
    positions = result["context_positions"]

    n_layers = len(LAYERS)
    K = len(positions) // n_layers
    assert len(positions) == K * n_layers

    base = positions[:K]
    for layer_idx in range(1, n_layers):
        chunk = positions[layer_idx * K : (layer_idx + 1) * K]
        assert chunk == base, f"Layer {layer_idx} chunk differs from base: {chunk[:5]}... vs {base[:5]}..."


def test_different_cot_lengths_produce_different_position_counts(tokenizer):
    """Short CoT → fewer positions than long CoT."""
    short_cot = "The answer is yes."
    long_cot = "Let me think about this carefully. " * 50 + "The answer is yes."

    item_short = _make_item("Q?", short_cot)
    item_long = _make_item("Q?", long_cot)

    result_short = prepare_context_ids([item_short], tokenizer, layers=LAYERS)[0]
    result_long = prepare_context_ids([item_long], tokenizer, layers=LAYERS)[0]

    n_short = result_short["num_positions"]
    n_long = result_long["num_positions"]

    assert n_long > n_short, f"Long CoT ({n_long} positions) should have more positions than short ({n_short})"


# ─── Group 2: Activation Injection (Steering Hook) ───


def test_hook_modifies_exactly_target_positions():
    """Steering hook should modify target positions and leave others untouched."""
    L, D = 20, 64
    K = 5
    target_positions = [3, 7, 11, 14, 18]

    # Use non-zero residuals (hook norm-matches, so zero orig → zero output)
    resid = torch.randn(1, L, D)
    orig = resid.clone()
    vectors = torch.randn(K, D)

    hook_fn = get_steering_hook(vectors, target_positions, device="cpu", dtype=torch.float32)
    output = hook_fn(None, None, resid)

    for pos in range(L):
        diff = (output[0, pos] - orig[0, pos]).abs().sum()
        if pos in target_positions:
            assert diff > 0, f"Position {pos} should be modified"
        else:
            assert diff == 0, f"Position {pos} should be unchanged"


def test_steering_vector_count_must_match_positions():
    """Mismatched vector count and position count should cause an indexing error."""
    D = 64
    vectors = torch.randn(5, D)
    positions = [1, 2, 3, 4, 5, 6]  # 6 positions but only 5 vectors

    hook_fn = get_steering_hook(vectors, positions, device="cpu", dtype=torch.float32)

    resid = torch.zeros(1, 20, D)
    with pytest.raises((IndexError, RuntimeError)):
        hook_fn(None, None, resid)


def test_norm_matching_scales_by_original_norm():
    """Injected component should be scaled by the original residual norm."""
    D = 64
    K = 3
    positions = [2, 5, 8]

    # Create residual with known norms at target positions
    resid = torch.zeros(1, 10, D)
    for i, pos in enumerate(positions):
        resid[0, pos] = torch.randn(D)
        resid[0, pos] = resid[0, pos] / resid[0, pos].norm() * (10.0 + i * 5)  # norms: 10, 15, 20

    orig_norms = [resid[0, pos].norm().item() for pos in positions]
    vectors = torch.randn(K, D)
    normed_vectors = torch.nn.functional.normalize(vectors, dim=-1)

    hook_fn = get_steering_hook(vectors, positions, device="cpu", dtype=torch.float32, steering_coefficient=1.0)
    output = hook_fn(None, None, resid.clone())

    # The hook does: resid[pos] = normed_vec * orig_norm * coeff + orig
    # So the injected component is normed_vec * orig_norm (unit vector scaled by orig norm)
    for i, pos in enumerate(positions):
        modified = output[0, pos]
        orig = resid[0, pos]
        injected = modified - orig
        # The injected component should have norm ≈ orig_norm (since normed_vec has norm 1)
        assert abs(injected.norm().item() - orig_norms[i]) < 0.1, (
            f"Injected norm {injected.norm().item():.2f} != original norm {orig_norms[i]:.2f}"
        )


def test_batched_hook_handles_variable_k_per_item():
    """Batched hook with different K per item should inject correctly without crosstalk."""
    D = 64
    B = 2
    L = 30

    positions_0 = [1, 3, 5, 7, 9]           # 5 positions
    positions_1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # 10 positions

    vectors_0 = torch.randn(5, D)
    vectors_1 = torch.randn(10, D)

    resid = torch.randn(B, L, D)
    orig = resid.clone()

    hook_fn = get_batched_steering_hook(
        vectors=[vectors_0, vectors_1],
        positions=[positions_0, positions_1],
        device="cpu",
        dtype=torch.float32,
    )

    output = hook_fn(None, None, resid)

    # Item 0: only positions_0 should be modified
    for pos in range(L):
        diff = (output[0, pos] - orig[0, pos]).abs().sum()
        if pos in positions_0:
            assert diff > 0, f"Item 0, pos {pos} should be modified"
        else:
            assert diff == 0, f"Item 0, pos {pos} should be unchanged"

    # Item 1: only positions_1 should be modified
    for pos in range(L):
        diff = (output[1, pos] - orig[1, pos]).abs().sum()
        if pos in positions_1:
            assert diff > 0, f"Item 1, pos {pos} should be modified"
        else:
            assert diff == 0, f"Item 1, pos {pos} should be unchanged"


# ─── Group 3: Data Leakage ───
#
# Ensure that everything we evaluate on (in run_comprehensive_eval.py and
# eval_loop.py) never appears during training. Covers:
#   - Source-based OOD disjointness (3 republished tasks with source column)
#   - Question/prompt overlap between train/test for ALL trainable tasks
#   - Every evaluated trainable task must have a test split (no fallback to train)

from tasks import TASKS

# Tasks skipped by comprehensive eval (no standard data loading path)
_COMPREHENSIVE_EVAL_SKIP = {
    "futurelens", "pastlens",
    "futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb",
    "probe_sycophancy", "deception_detection",
}


def _get_evaluated_trainable_tasks() -> list[str]:
    """Tasks that are trainable AND evaluated in comprehensive eval."""
    return [
        name for name, tdef in TASKS.items()
        if tdef.trainable
        and tdef.hf_repo
        and not tdef.needs_rot13_adapter
        and name not in _COMPREHENSIVE_EVAL_SKIP
    ]


def _get_question(item: dict) -> str:
    """Best-effort unique question fingerprint for an item."""
    return item.get("hinted_prompt") or item.get("question") or item.get("prompt", "")


# ── Source-based OOD disjointness (tasks with explicit source column) ──

OOD_TASKS = ["answer_trajectory", "atypical_answer", "reasoning_termination"]


@pytest.mark.parametrize("task_name", OOD_TASKS)
def test_ood_split_source_disjointness(task_name):
    """Train and test splits must have completely disjoint source labels."""
    train_data = load_task_data(task_name, split="train", n=None, shuffle=False)
    test_data = load_task_data(task_name, split="test", n=None, shuffle=False)

    train_sources = {item["source"] for item in train_data}
    test_sources = {item["source"] for item in test_data}

    overlap = train_sources & test_sources
    assert overlap == set(), (
        f"{task_name}: source overlap between train/test: {overlap}"
    )


# ── No CoT text overlap between train/test (exact activation leakage) ──

_EVALUATED_TRAINABLE = _get_evaluated_trainable_tasks()


@pytest.mark.parametrize("task_name", _EVALUATED_TRAINABLE)
def test_no_eval_cot_in_training(task_name):
    """No cot_text from the test split should appear in the train split.

    cot_text is what produces the activations the oracle reads, so any overlap
    means we evaluate on the exact same activations used during training.
    """
    try:
        train_data = load_task_data(task_name, split="train", n=None, shuffle=False)
    except Exception as exc:
        pytest.skip(f"No train split available: {exc}")
    try:
        test_data = load_task_data(task_name, split="test", n=None, shuffle=False)
    except Exception as exc:
        pytest.skip(f"No test split available: {exc}")

    train_cots = {item["cot_text"] for item in train_data if item.get("cot_text")}
    test_cots = {item["cot_text"] for item in test_data if item.get("cot_text")}

    overlap = train_cots & test_cots
    assert overlap == set(), (
        f"{task_name}: {len(overlap)}/{len(test_cots)} test cot_texts appear in train split. "
        f"Example: {list(overlap)[0][:120]}..."
    )


# ── No question overlap for tasks with per-example questions ──

@pytest.mark.parametrize("task_name", _EVALUATED_TRAINABLE)
def test_no_eval_question_in_training(task_name):
    """No question/prompt text from test split should appear in train split.

    Skips tasks where the 'question' field is actually a shared oracle template
    (< 50 unique values), since those aren't per-example questions.
    """
    try:
        train_data = load_task_data(task_name, split="train", n=None, shuffle=False)
    except Exception as exc:
        pytest.skip(f"No train split available: {exc}")
    try:
        test_data = load_task_data(task_name, split="test", n=None, shuffle=False)
    except Exception as exc:
        pytest.skip(f"No test split available: {exc}")

    train_questions = {_get_question(item) for item in train_data}
    test_questions = {_get_question(item) for item in test_data}

    # Skip if the "question" field is really a small set of oracle templates
    if len(train_questions) < 50 or len(test_questions) < 50:
        pytest.skip(
            f"question field appears to be a shared template "
            f"({len(train_questions)} unique train, {len(test_questions)} unique test)"
        )

    overlap = train_questions & test_questions
    assert overlap == set(), (
        f"{task_name}: {len(overlap)} questions from test split appear in train split. "
        f"Examples: {list(overlap)[:3]}"
    )


# ── Every evaluated trainable task must have a test split ──

def test_all_evaluated_trainable_tasks_have_test_split():
    """All trainable tasks evaluated in comprehensive eval must have a test split.

    If a task falls back to train split during eval, that's data leakage.
    """
    missing_test = []
    for task_name in _EVALUATED_TRAINABLE:
        try:
            data = load_task_data(task_name, split="test", n=1, shuffle=False)
            assert len(data) > 0
        except Exception:
            missing_test.append(task_name)

    assert missing_test == [], (
        f"These trainable tasks are evaluated but lack a test split (eval falls back to train = leakage): {missing_test}"
    )
