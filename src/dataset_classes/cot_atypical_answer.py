"""
CoT Atypical Answer Prediction — Is this a majority or minority answer?

Binary classification: given CoT activations from a rollout, predict whether
the model's answer matches the majority answer (across 25 rollouts) or is
a minority/atypical answer.

Data source: HuggingFace dataset or local JSONL from precompute_atypical_training.py.
Each item already has question, cot_text, and label (majority/minority).

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


# HF dataset repo for fallback download
HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-atypical-answer-rollouts"


def load_cot_atypical_answer_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 20000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate atypical answer prediction training data with multi-layer stride.

    Each example: stride activations from a rollout CoT -> majority / minority.

    corpus_path is used as the path to the atypical answer JSONL file. If it
    doesn't exist or doesn't contain atypical data, falls back to HF download.
    The actual path is overridden at the call sites (train.py, precompute_training_data.py)
    via the atypical_data_path kwarg or the special corpus type.
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    # Load atypical answer data
    data_path = _kwargs.get("atypical_data_path", corpus_path)
    items = _load_atypical_data(data_path)

    if not items:
        raise ValueError(
            f"No atypical answer data found at {data_path}. "
            f"Run scripts/precompute_atypical_training.py first or download from {HF_REPO}."
        )

    # Split into pools for balanced sampling
    majority_pool = [e for e in items if e["label"] == "majority"]
    minority_pool = [e for e in items if e["label"] == "minority"]

    if not majority_pool:
        raise ValueError("No majority-labeled items in atypical data")
    if not minority_pool:
        raise ValueError("No minority-labeled items in atypical data")

    print(f"  majority: {len(majority_pool)}, minority: {len(minority_pool)}")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []
    attempts = 0

    while len(datapoints) < num_examples and attempts < num_examples * 3:
        attempts += 1

        # Balanced 50/50 majority/minority
        if len(datapoints) % 2 == 0:
            entry = random.choice(majority_pool)
            target = "majority"
        else:
            entry = random.choice(minority_pool)
            target = "minority"

        # Reconstruct the prompt: question + choices formatted as chat
        question_content = entry["question"]
        if entry.get("choices"):
            question_content += "\n\n" + entry["choices"]

        messages = [{"role": "user", "content": question_content}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # The cot_text is the full model response (generated with enable_thinking=False,
        # so it's plain reasoning + answer, no <think> wrapper)
        cot_text = entry["cot_text"]

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        positions = get_cot_positions(
            prompt_len, len(full_ids),
            stride=stride, tokenizer=tokenizer, input_ids=full_ids,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Is this a majority or minority answer for this question? "
            f"Answer: majority or minority."
        )

        datapoints.append({
            "datapoint_type": "cot_atypical_answer",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} atypical answer examples")
    return datapoints[:num_examples]


def _load_atypical_data(path: str) -> list[dict]:
    """Load atypical answer data — always tries HuggingFace first, falls back to local JSONL."""
    from pathlib import Path

    # Always try HuggingFace first
    print(f"  Downloading atypical answer data from {HF_REPO}...")
    try:
        import importlib
        hf_datasets = importlib.import_module("datasets")
        ds = hf_datasets.load_dataset(HF_REPO, split="train")
        items = []
        for row in ds:
            if row.get("cot_text") and row.get("label") in ("majority", "minority"):
                items.append(dict(row))
        if items:
            print(f"  Downloaded {len(items)} items from HuggingFace")
            return items
    except Exception as e:
        print(f"  Warning: HF download failed: {e}")

    # Fallback: local JSONL
    local = Path(path)
    if local.exists() and local.suffix == ".jsonl":
        items = []
        with open(local) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("cot_text") and item.get("label") in ("majority", "minority"):
                        items.append(item)
        if items:
            print(f"  Loaded {len(items)} items from local fallback {local}")
            return items

    return []
