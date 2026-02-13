"""
Task 3: Sentence Taxonomy Classification (15K examples)

Given activation at a sentence boundary, classify the sentence type:
problem_setup, plan_generation, fact_retrieval, active_computation,
uncertainty_management, result_consolidation, self_checking, final_answer.
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def load_cot_taxonomy_data(
    corpus_path: str,
    labels_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate taxonomy classification training data.

    Each example: single sentence boundary activation → category name.
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

    # Load taxonomy labels
    labels_by_id = {}
    with open(labels_path) as f:
        for line in f:
            if line.strip():
                label = json.loads(line)
                key = (label["id"], label["sentence_idx"])
                labels_by_id[key] = label

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    # Group by category for balanced sampling
    by_category = {}
    for key, label in labels_by_id.items():
        entry_id, s_idx = key
        if entry_id not in corpus:
            continue
        entry = corpus[entry_id]
        if s_idx >= len(entry.get("boundary_positions", [])):
            continue
        cat = label["category"]
        by_category.setdefault(cat, []).append((entry, label, s_idx))

    print("  Category distribution:")
    for cat, items in sorted(by_category.items()):
        print(f"    {cat}: {len(items)}")

    all_candidates = [(entry, label, s_idx) for items in by_category.values() for entry, label, s_idx in items]
    if not all_candidates:
        raise ValueError("No valid candidates found")

    # Category descriptions for richer targets
    category_descriptions = {
        "problem_setup": "This step parses or rephrases the problem.",
        "plan_generation": "This step states a plan or decides on an approach.",
        "fact_retrieval": "This step recalls a fact, formula, or definition.",
        "active_computation": "This step performs a calculation or algebraic manipulation.",
        "uncertainty_management": "This step expresses confusion or re-evaluates the approach.",
        "result_consolidation": "This step aggregates intermediate results.",
        "self_checking": "This step verifies or confirms previous work.",
        "final_answer": "This step states the final answer.",
    }

    datapoints = []

    while len(datapoints) < num_examples:
        # Sample roughly balanced across categories
        cat = random.choice(list(by_category.keys()))
        if not by_category[cat]:
            continue
        entry, label, s_idx = random.choice(by_category[cat])

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
        category = label["category"]
        description = category_descriptions.get(category, f"This is a {category} step.")

        target = f"{category}. {description}"
        prompt = "What type of reasoning step is this? Classify as: problem_setup, plan_generation, fact_retrieval, active_computation, uncertainty_management, result_consolidation, self_checking, or final_answer."

        datapoints.append({
            "datapoint_type": "cot_taxonomy",
            "prompt": prompt,
            "target_response": target,
            "layer": layer,
            "num_positions": 1,
            "context_input_ids": context_slice,
            "context_positions": [pos],
        })

    return datapoints[:num_examples]
