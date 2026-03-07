import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [ROOT / "baselines", ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm_monitor import _build_prompt, build_reasoning_monitor_user_message
from shared import BaselineInput


def _make_input(*, eval_name: str, clean_prompt: str, test_prompt: str, metadata: dict | None = None) -> BaselineInput:
    return BaselineInput(
        eval_name=eval_name,
        example_id="example_0",
        clean_prompt=clean_prompt,
        test_prompt=test_prompt,
        correct_answer="reference",
        nudge_answer=None,
        ground_truth_label="reference",
        clean_response="Prefix sentence. Another sentence.",
        test_response="Prefix sentence. Another sentence.",
        activations_by_layer={},
        metadata=metadata or {},
    )


def test_build_reasoning_monitor_user_message_preserves_legacy_format_by_default():
    msg = build_reasoning_monitor_user_message(
        "Prefix sentence.",
        "What happens next?",
        original_question="Solve 2 + 2.",
        include_question=False,
    )
    assert msg == "Chain of thought:\nPrefix sentence.\n\nWhat happens next?"


def test_build_reasoning_monitor_user_message_includes_original_question_when_enabled():
    msg = build_reasoning_monitor_user_message(
        "Prefix sentence.",
        "What happens next?",
        original_question="Solve 2 + 2.",
        include_question=True,
    )
    assert msg.startswith("Original question the model was answering:\nSolve 2 + 2.\n\n")
    assert "Chain of thought:\nPrefix sentence." in msg
    assert msg.endswith("What happens next?")


def test_chunked_generation_prompt_includes_original_question_when_requested():
    inp = _make_input(
        eval_name="chunked_convqa",
        clean_prompt="Solve 2 + 2.",
        test_prompt="What answer is the model leaning toward?",
        metadata={"question": "Solve 2 + 2."},
    )
    prompt = _build_prompt(inp, "chunked_convqa", "generation", include_question=True)
    assert "Original question the model was answering:\nSolve 2 + 2." in prompt
    assert "Question about the reasoning:\nWhat answer is the model leaning toward?" in prompt


def test_chunked_generation_prompt_avoids_duplicate_question_block():
    inp = _make_input(
        eval_name="chunked_convqa",
        clean_prompt="What answer is the model leaning toward?",
        test_prompt="What answer is the model leaning toward?",
        metadata={"question": "What answer is the model leaning toward?"},
    )
    prompt = _build_prompt(inp, "chunked_convqa", "generation", include_question=True)
    assert "Original question the model was answering:" not in prompt
