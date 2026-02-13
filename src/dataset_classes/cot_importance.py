"""
Task 2: Thought Anchor Detection (30K examples)

Binary classification: thought_anchor vs not_anchor.
Top 30% of sentences by KL divergence (within each problem) are thought anchors.

Balanced 50/50 sampling ensures oracle can't just predict the majority class.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def load_cot_importance_data(
    corpus_path: str,
    labels_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 30000,
    anchor_percentile: float = 0.70,
    seed: int = 42,
) -> list[dict]:
    """
    Generate thought anchor detection training data.

    Each example: single sentence boundary activation -> thought_anchor / not_anchor.
    Uses within-problem percentile ranking: top 30% by KL = thought_anchor.

    Args:
        anchor_percentile: Sentences at or above this percentile are thought anchors.
            Default 0.70 means top 30% are anchors.
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

    # Load importance labels
    raw_labels = []
    with open(labels_path) as f:
        for line in f:
            if line.strip():
                raw_labels.append(json.loads(line))

    # Group by problem and assign within-problem percentile ranks
    by_problem = defaultdict(list)
    for label in raw_labels:
        by_problem[label["id"]].append(label)

    labels_by_id = {}
    for prob_id, prob_labels in by_problem.items():
        sorted_labels = sorted(prob_labels, key=lambda x: x["kl_divergence"])
        n = len(sorted_labels)
        for rank, label in enumerate(sorted_labels):
            percentile = rank / max(n - 1, 1)  # 0.0 = lowest KL, 1.0 = highest
            is_anchor = percentile >= anchor_percentile
            label["is_anchor"] = is_anchor
            label["importance_percentile"] = percentile
            key = (label["id"], label["sentence_idx"])
            labels_by_id[key] = label

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    # Build candidate pools for balanced sampling
    anchors = []
    non_anchors = []
    for key, label in labels_by_id.items():
        entry_id, s_idx = key
        if entry_id not in corpus:
            continue
        entry = corpus[entry_id]
        if s_idx >= len(entry.get("boundary_positions", [])):
            continue
        if label["is_anchor"]:
            anchors.append((entry, label, s_idx))
        else:
            non_anchors.append((entry, label, s_idx))

    if not anchors or not non_anchors:
        raise ValueError(f"Unbalanced data: {len(anchors)} anchors, {len(non_anchors)} non-anchors")

    print(f"  thought_anchor: {len(anchors)}, not_anchor: {len(non_anchors)}")

    datapoints = []

    while len(datapoints) < num_examples:
        # Alternate 50/50 between anchor and non-anchor
        if len(datapoints) % 2 == 0:
            entry, label, s_idx = random.choice(anchors)
            target = "thought_anchor"
        else:
            entry, label, s_idx = random.choice(non_anchors)
            target = "not_anchor"

        layer = random.choice(layers)

        # Build context for activation at this sentence boundary
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

        prompt = "Is this reasoning step a thought anchor (causally important for the final answer) or not? Answer: thought_anchor or not_anchor."

        datapoints.append({
            "datapoint_type": "cot_importance",
            "prompt": prompt,
            "target_response": target,
            "layer": layer,
            "num_positions": 1,
            "context_input_ids": context_slice,
            "context_positions": [pos],
        })

    return datapoints[:num_examples]
