"""
Task 4: Answer Tracking (10K examples)

Given activation at a sentence boundary, predict the model's current best-guess answer
(as measured by logit lens at 75% depth).

Tracks answer evolution — decorative CoT shows flat answer,
load-bearing shows gradual emergence.
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def load_cot_answer_tracking_data(
    corpus_path: str,
    labels_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate answer tracking training data.

    Each example: single sentence boundary activation →
    current top token and its probability.
    """
    from signs_of_life.ao_lib import layer_percent_to_layer

    random.seed(seed)

    # Load corpus
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus[entry["id"]] = entry

    # Load answer tracking labels
    labels_by_id = {}
    with open(labels_path) as f:
        for line in f:
            if line.strip():
                label = json.loads(line)
                key = (label["id"], label["sentence_idx"])
                labels_by_id[key] = label

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    candidates = []
    for key, label in labels_by_id.items():
        entry_id, s_idx = key
        if entry_id not in corpus:
            continue
        entry = corpus[entry_id]
        if s_idx >= len(entry.get("boundary_positions", [])):
            continue
        candidates.append((entry, label, s_idx))

    if not candidates:
        raise ValueError("No valid candidates found")

    datapoints = []

    while len(datapoints) < num_examples:
        entry, label, s_idx = random.choice(candidates)
        layer = random.choice(layers)

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        pos = entry["boundary_positions"][s_idx]
        if pos >= len(context_ids):
            continue

        context_slice = context_ids[:pos + 1]

        top_token = label["top_token"]
        top_prob = label["top_prob"]
        answer_prob = label["answer_prob"]

        # Target response includes both the current top prediction and answer probability
        target = (
            f"The model's current best guess is \"{top_token}\" "
            f"(probability: {top_prob:.3f}). "
            f"The correct answer token has probability {answer_prob:.3f}."
        )

        prompt = "What is the model's current answer at this point in the reasoning?"

        datapoints.append({
            "datapoint_type": "cot_answer_tracking",
            "prompt": prompt,
            "target_response": target,
            "layer": layer,
            "num_positions": 1,
            "context_input_ids": context_slice,
            "context_positions": [pos],
        })

    return datapoints[:num_examples]
