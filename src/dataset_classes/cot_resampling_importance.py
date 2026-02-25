"""
Resampling Importance — predict which CoT steps are causally important.

Uses per-sentence importance scores from the resampling methodology
(thought-anchors, Bogdan et al. 2025). Each entry has sentences with
KL and accuracy delta scores from 20-rollout resampling at truncation points.

Two sub-tasks:
  1. importance_ranking: predict top-3 important step indices (generative)
  2. importance_binary: does this CoT contain any critically important steps? (binary)

Data source: HuggingFace dataset mats-10-sprint-cs-jb/cot-oracle-resampling-importancepp
"""

import json
import random
import re

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_resampling_importance_data(
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 5000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
    ranking_fraction: float = 0.6,
    hf_dataset: str = "mats-10-sprint-cs-jb/cot-oracle-resampling-importancepp",
    **_kwargs,
) -> list[dict]:
    """
    Generate resampling importance training data.

    Args:
        ranking_fraction: fraction of examples that are ranking tasks (rest are binary)
        hf_dataset: HuggingFace dataset ID to load
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers
    from datasets import load_dataset

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    ds = load_dataset(hf_dataset, split="train")
    print(f"  Loaded {len(ds)} entries from {hf_dataset}")

    # Split into entries with and without important sentences
    has_important = []
    no_important = []

    for row in ds:
        si = json.loads(row["sentence_importance"])
        texts = [s["sentence_text"] for s in si]
        cot_text = " ".join(texts)
        if not cot_text.strip():
            continue

        entry = {
            "question": row["question"],
            "cot_text": cot_text,
            "n_sentences": row["n_sentences"],
            "n_important": row["n_important"],
            "top_k_indices": json.loads(row["top_k_indices"]) if isinstance(row["top_k_indices"], str) else row["top_k_indices"],
        }

        if row["n_important"] > 0:
            has_important.append(entry)
        else:
            no_important.append(entry)

    print(f"  Entries with important steps: {len(has_important)}, without: {len(no_important)}")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    def _tokenize_entry(entry):
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        full_text = formatted + entry["cot_text"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        return full_ids, len(prompt_ids)

    # Pre-tokenize
    print("  Tokenizing entries with important steps...")
    tokenized_important = []
    for entry in tqdm(has_important, desc="  importance/has_important", leave=False):
        full_ids, prompt_len = _tokenize_entry(entry)
        if len(full_ids) - prompt_len < 10:
            continue
        tokenized_important.append((full_ids, prompt_len, entry))

    print("  Tokenizing entries without important steps...")
    tokenized_none = []
    for entry in tqdm(no_important, desc="  importance/no_important", leave=False):
        full_ids, prompt_len = _tokenize_entry(entry)
        if len(full_ids) - prompt_len < 10:
            continue
        tokenized_none.append((full_ids, prompt_len, entry))

    print(f"  Tokenized: {len(tokenized_important)} with, {len(tokenized_none)} without")

    layers_str = ", ".join(str(l) for l in LAYERS)
    n_ranking = int(num_examples * ranking_fraction)
    n_binary = num_examples - n_ranking

    datapoints = []

    # --- Ranking task: predict top-k indices ---
    pbar = tqdm(total=n_ranking, desc="  importance_ranking: sampling", leave=False)
    idx = 0
    while len(datapoints) < n_ranking and idx < len(tokenized_important) * 20:
        full_ids, prompt_len, entry = tokenized_important[idx % len(tokenized_important)]
        idx += 1

        positions = get_cot_stride_positions(
            prompt_len, len(full_ids), stride=stride,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)
        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        # Target: top-k indices (1-indexed for human readability)
        top_k = entry["top_k_indices"][:3]
        target = ", ".join(str(i + 1) for i in top_k)

        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"This chain of thought has {entry['n_sentences']} steps. "
            f"Which step numbers are most causally important for the final answer? "
            f"List the top 3 step numbers, comma-separated."
        )

        datapoints.append({
            "datapoint_type": "resampling_importance",
            "prompt": prompt,
            "target_response": target,
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": list(context_slice),
            "context_positions": list(context_positions),
        })
        pbar.update(1)
    pbar.close()

    # --- Binary task: does this CoT have important steps? ---
    pbar = tqdm(total=n_binary, desc="  importance_binary: sampling", leave=False)
    idx = 0
    while len(datapoints) < n_ranking + n_binary and idx < max(len(tokenized_important), len(tokenized_none)) * 20:
        # Alternate for balance
        current_count = len(datapoints) - n_ranking
        if current_count % 2 == 0 and tokenized_important:
            full_ids, prompt_len, entry = random.choice(tokenized_important)
            target = "yes"
        elif tokenized_none:
            full_ids, prompt_len, entry = random.choice(tokenized_none)
            target = "no"
        else:
            full_ids, prompt_len, entry = random.choice(tokenized_important)
            target = "yes"
        idx += 1

        positions = get_cot_stride_positions(
            prompt_len, len(full_ids), stride=stride,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)
        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Does this chain of thought contain any critically important reasoning steps "
            f"that significantly change the final answer? Answer: yes or no."
        )

        datapoints.append({
            "datapoint_type": "resampling_importance",
            "prompt": prompt,
            "target_response": target,
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": list(context_slice),
            "context_positions": list(context_positions),
        })
        pbar.update(1)
    pbar.close()

    random.shuffle(datapoints)
    print(f"  Generated {len(datapoints)} resampling importance examples ({n_ranking} ranking + {n_binary} binary)")
    return datapoints[:num_examples]
