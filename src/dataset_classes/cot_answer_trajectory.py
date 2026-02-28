"""
Answer trajectory training data loader.

Loads precomputed answer_trajectory data from HuggingFace, swapping
target_response with the cleaned response_filtered field.

The precomputed data already has context_positions, context_input_ids,
layers, etc — this loader just performs the field swap and filters
out rows with empty/missing response_filtered.
"""

import json
import random

from huggingface_hub import hf_hub_download


HF_REPO = "ceselder/cot-oracle-answer-trajectory"
HF_FILENAME = "answer_trajectory_cleaned.jsonl"


def load_cot_answer_trajectory_data(
    corpus_path: str,  # unused, kept for loader API compatibility
    tokenizer,         # unused
    model_name: str,   # unused
    num_examples: int = 60000,
    stride: int | str = None,  # unused
    seed: int = 42,
    **kwargs,
) -> list[dict]:
    """Load cleaned answer trajectory data from HuggingFace.

    Downloads the cleaned JSONL, swaps target_response with response_filtered,
    and filters out rows where response_filtered is empty.
    """
    random.seed(seed)

    # Download from HF
    print(f"  [answer_trajectory] Downloading from {HF_REPO}...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
    )

    # Load and filter
    data = []
    skipped = 0
    with open(local_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned = row.get("response_filtered", "").strip()
            if not cleaned:
                skipped += 1
                continue
            # Swap target_response with the cleaned version
            row["target_response"] = cleaned
            data.append(row)

    print(f"  [answer_trajectory] Loaded {len(data)} examples ({skipped} skipped, empty response_filtered)")

    # Shuffle and truncate
    random.shuffle(data)
    if len(data) > num_examples:
        data = data[:num_examples]

    print(f"  [answer_trajectory] Using {len(data)} examples")
    return data
