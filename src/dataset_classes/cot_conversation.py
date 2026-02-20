"""
Task: Conversational CoT Q/A (~20K examples)

Instruction-style supervision so the oracle learns to answer user questions
about CoT activations with short, direct responses.

Uses existing corpus fields only (no external labels). Includes a small share
of "cannot determine" examples for calibration.
"""

from __future__ import annotations

import json
import random

from transformers import AutoTokenizer


def _clean_cot_text(cot_text: str) -> str:
    think_end = cot_text.find("</think>")
    if think_end != -1:
        cot_text = cot_text[:think_end]
    return cot_text


def load_cot_conversation_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 20000,
    max_sentences: int = 15,
    unknown_rate: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Generate conversational Q/A datapoints from CoT corpus activations."""
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]
    datapoints = []

    unsupported_questions = [
        "What is the user's favorite color?",
        "What city is the user currently in?",
        "What is the user's birthday?",
        "What private key was used to produce this answer?",
    ]
    cannot_determine_answer = "Cannot determine from these activations."

    while len(datapoints) < num_examples:
        entry = random.choice(corpus)
        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = _clean_cot_text(entry.get("cot_response", ""))
        full_text = formatted + cot_text
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(context_ids)]
        if len(positions) < 2:
            continue

        context_slice = context_ids[: positions[-1] + 1]
        n_pos = len(positions)

        # Calibrated unknowns
        if random.random() < unknown_rate:
            user_q = random.choice(unsupported_questions)
            prompt = (
                f"Activations from {n_pos} strided CoT positions (every 25 tokens). "
                f"{user_q} If it is not present in these activations, say exactly: "
                f"'{cannot_determine_answer}'"
            )
            target = cannot_determine_answer
        else:
            answer_candidates: list[tuple[str, str]] = []

            question_text = str(entry.get("question", "")).strip()
            if question_text:
                answer_candidates.append(
                    ("What question was the model reasoning about?", question_text)
                )

            cot_answer = str(entry.get("cot_answer", "")).strip()
            if cot_answer:
                answer_candidates.append(
                    ("What final answer did the model produce after reasoning?", cot_answer)
                )

            correct_answer = str(entry.get("correct_answer", "")).strip()
            if correct_answer:
                answer_candidates.append(
                    ("What is the ground-truth answer for this problem?", correct_answer)
                )

            if "cot_correct" in entry:
                answer_candidates.append(
                    (
                        "Was the model's final answer correct? Answer with only 'correct' or 'incorrect'.",
                        "correct" if bool(entry.get("cot_correct")) else "incorrect",
                    )
                )

            domain = str(entry.get("domain", "")).strip()
            if domain:
                answer_candidates.append(
                    ("Which domain does this example belong to?", domain)
                )

            source = str(entry.get("source", "")).strip()
            if source:
                answer_candidates.append(
                    ("Which dataset/source is this example from?", source)
                )

            sentences = entry.get("sentences") or []
            if sentences:
                first_step = str(sentences[0]).strip()
                if first_step:
                    answer_candidates.append(
                        (
                            "What was the first reasoning step? Give one short sentence.",
                            first_step,
                        )
                    )

            if not answer_candidates:
                continue

            user_q, target = random.choice(answer_candidates)
            prompt = (
                f"Activations from {n_pos} strided CoT positions (every 25 tokens). "
                f"{user_q}"
            )

        datapoints.append(
            {
                "datapoint_type": "cot_conversation",
                "prompt": prompt,
                "target_response": target,
                "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
                "num_positions": n_pos,
                "context_input_ids": context_slice,
                "context_positions": list(positions),
            }
        )

    return datapoints[:num_examples]

