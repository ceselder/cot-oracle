"""
FineWeb ConvQA — Questions about web document excerpts.

Each row has a FineWeb excerpt, a Gemini-generated question about the content,
and a Gemini-generated answer. The oracle sees activations from the excerpt
and answers the question.

Data source: mats-10-sprint-cs-jb/fineweb-convqa (HuggingFace).
"""

import random

from transformers import AutoTokenizer


HF_DATASET = "mats-10-sprint-cs-jb/fineweb-convqa"


def load_cot_fineweb_convqa_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """Load FineWeb ConvQA training data from HuggingFace.

    Extracts activations from the raw excerpt text (no chat template —
    these are web documents, not CoT reasoning traces).
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)
    LAYERS = get_injection_layers(model_name)

    rows = _load_from_hf(split="train")
    random.shuffle(rows)
    print(f"  Loaded {len(rows)} FineWeb ConvQA rows from HF (train split)")

    datapoints = []
    skipped = 0

    for row in rows:
        if len(datapoints) >= num_examples:
            break

        excerpt = row["excerpt"]
        query = row["prompt"]
        target_response = row["target_response"]

        if not target_response or not target_response.strip():
            skipped += 1
            continue

        # Tokenize excerpt directly (no chat template — raw web text)
        excerpt_ids = tokenizer(excerpt, add_special_tokens=False)["input_ids"]
        if len(excerpt_ids) < 10:
            skipped += 1
            continue

        # Stride positions over the entire excerpt
        positions = get_cot_stride_positions(0, len(excerpt_ids), stride=stride or 5)
        if len(positions) < 2:
            skipped += 1
            continue

        context_positions = positions * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = excerpt_ids[:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        oracle_prompt = f"Activations from {num_positions} positions across layers {layers_str}. {query}"

        datapoints.append({
            "datapoint_type": "cot_fineweb_convqa",
            "prompt": oracle_prompt,
            "target_response": target_response,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} fineweb_convqa examples ({skipped} skipped)")
    return datapoints[:num_examples]


def _load_from_hf(split: str = "train") -> list[dict]:
    """Load FineWeb ConvQA dataset from HuggingFace."""
    import importlib
    hf_datasets = importlib.import_module("datasets")

    print(f"  Downloading FineWeb ConvQA from {HF_DATASET} (split={split})...")
    ds = hf_datasets.load_dataset(HF_DATASET, split=split)
    rows = [dict(row) for row in ds]
    print(f"  Downloaded {len(rows)} rows")
    return rows
