"""
Open-ended ConvQA — Questions about CoT reasoning traces.

Each row has a CoT trace, a Gemini-generated question about the reasoning,
and a Gemini-generated answer. The oracle sees activations from the full CoT
and answers the question.

Data source: mats-10-sprint-cs-jb/cot-oracle-convqa (HuggingFace).
"""

import random
import re

from transformers import AutoTokenizer


HF_DATASET = "mats-10-sprint-cs-jb/cot-oracle-convqa"


def load_cot_openended_convqa_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """Load open-ended ConvQA training data from HuggingFace."""
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)

    rows = _load_from_hf(split="train")
    random.shuffle(rows)
    print(f"  Loaded {len(rows)} open-ended ConvQA rows from HF (train split)")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []
    skipped = 0

    for row in rows:
        if len(datapoints) >= num_examples:
            break

        cot_text = row["cot_text"]
        question = row["original_question"]
        query = row["prompt"]
        target_response = row["target_response"]

        if not target_response or not target_response.strip():
            skipped += 1
            continue

        # Strip </think> if present
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        # Tokenize question + CoT
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Get stride positions over the CoT region
        positions = get_cot_positions(
            prompt_len, len(full_ids),
            stride=stride, tokenizer=tokenizer, input_ids=full_ids,
        )
        if len(positions) < 2:
            skipped += 1
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        oracle_prompt = f"Activations from {num_positions} positions across layers {layers_str}. {query}"

        datapoints.append({
            "datapoint_type": "cot_openended_convqa",
            "prompt": oracle_prompt,
            "target_response": target_response,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} openended_convqa examples ({skipped} skipped)")
    return datapoints[:num_examples]


def _load_from_hf(split: str = "train") -> list[dict]:
    """Load open-ended ConvQA dataset from HuggingFace."""
    import importlib
    hf_datasets = importlib.import_module("datasets")

    print(f"  Downloading open-ended ConvQA from {HF_DATASET} (split={split})...")
    ds = hf_datasets.load_dataset(HF_DATASET, split=split)
    rows = [dict(row) for row in ds]
    print(f"  Downloaded {len(rows)} rows")
    return rows
