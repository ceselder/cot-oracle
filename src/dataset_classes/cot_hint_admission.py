"""
CoT Hint Admission — Did the model use an external hint?

Generation task: given CoT activations from a hinted rollout, describe whether
the model used the hint, what the hint was, and how it influenced reasoning.

Four cases:
  - hint_used_wrong: model followed an incorrect hint
  - hint_used_correct: model followed a correct hint (was uncertain without it)
  - hint_resisted: model ignored the hint (wrong or correct)

Data source: HuggingFace dataset or local JSONL from precompute_hint_admission.py.
Each item has hinted_prompt, cot_text, label, and target_response.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


HF_REPO = "mats-10-sprint-cs-jb/qwen3-8b-hint-admission-rollouts"

# Target balance: 35% hint_used_wrong, 15% hint_used_correct,
# 35% hint_resisted (wrong hint), 15% hint_resisted (correct hint)
LABEL_WEIGHTS = {
    "hint_used_wrong": 0.35,
    "hint_used_correct": 0.15,
    "hint_resisted": 0.50,  # combined wrong+correct hint resisted
}


def load_cot_hint_admission_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate hint admission training data with multi-layer stride.

    Each example: stride activations from a hinted rollout CoT -> description
    of whether/how the model used the hint.

    corpus_path is overridden at call sites via hint_admission_data_path.
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    data_path = _kwargs.get("hint_admission_data_path", corpus_path)
    items = _load_hint_admission_data(data_path)

    if not items:
        raise ValueError(
            f"No hint admission data found at {data_path}. "
            f"Run scripts/precompute_hint_admission.py first or download from {HF_REPO}."
        )

    # Split into pools by label
    pools = {}
    for item in items:
        label = item["label"]
        pools.setdefault(label, []).append(item)

    for label, pool in pools.items():
        print(f"  {label}: {len(pool)}")

    if not pools:
        raise ValueError("No labeled items in hint admission data")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Compute per-label sample counts for balanced sampling
    label_counts = {}
    for label, weight in LABEL_WEIGHTS.items():
        label_counts[label] = int(num_examples * weight)
    # Distribute remainder
    remainder = num_examples - sum(label_counts.values())
    for label in list(label_counts.keys())[:remainder]:
        label_counts[label] += 1

    datapoints = []
    attempts = 0

    # Sample from each label pool proportionally
    for label, target_n in label_counts.items():
        pool = pools.get(label, [])
        if not pool:
            print(f"  WARNING: no items for label '{label}', skipping")
            continue

        count = 0
        local_attempts = 0
        while count < target_n and local_attempts < target_n * 3:
            local_attempts += 1
            entry = random.choice(pool)

            # Reconstruct the hinted prompt as chat
            hinted_prompt = entry.get("hinted_prompt", "")
            if not hinted_prompt:
                continue

            messages = [{"role": "user", "content": hinted_prompt}]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            cot_text = entry["cot_text"]
            full_text = formatted + cot_text
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            positions = get_cot_stride_positions(
                prompt_len, len(full_ids),
                stride=stride,
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
                f"Did the model use an external hint in its reasoning? If so, describe "
                f"what hint was used and how it influenced the answer."
            )

            target = entry.get("target_response", label)

            datapoints.append({
                "datapoint_type": "cot_hint_admission",
                "prompt": prompt,
                "target_response": target,
                "layer": LAYERS[0],
                "layers": LAYERS,
                "num_positions": num_positions,
                "context_input_ids": context_slice,
                "context_positions": context_positions,
            })
            count += 1

    random.shuffle(datapoints)
    print(f"  Generated {len(datapoints)} hint admission examples")
    return datapoints[:num_examples]


def _load_hint_admission_data(path: str) -> list[dict]:
    """Load hint admission data — tries HuggingFace first, falls back to local JSONL."""
    from pathlib import Path

    VALID_LABELS = {"hint_used_wrong", "hint_used_correct", "hint_resisted"}

    # Always try HuggingFace first
    print(f"  Downloading hint admission data from {HF_REPO}...")
    try:
        import importlib
        hf_datasets = importlib.import_module("datasets")
        ds = hf_datasets.load_dataset(HF_REPO, split="train")
        items = []
        for row in ds:
            if row.get("cot_text") and row.get("label") in VALID_LABELS:
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
                    if item.get("cot_text") and item.get("label") in VALID_LABELS:
                        items.append(item)
        if items:
            print(f"  Loaded {len(items)} items from local fallback {local}")
            return items

    return []
