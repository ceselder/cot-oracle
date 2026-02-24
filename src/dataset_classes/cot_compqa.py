"""
Computational QA — Answer questions about CoT reasoning quality.

10 types of questions (non-sequitur, verification, self-correction,
reasoning direction, conclusion validity, first error, error type,
load-bearing steps, soundness, redundant steps).

Data source: HuggingFace mats-10-sprint-cs-jb/cot-oracle-eval-compqa
Ground truth: Gemini 3 Flash annotations.

Each example:
  - context_positions: CoT stride positions * 3 (3 layers)
  - prompt: per-item question about the CoT reasoning quality
  - target: Gemini's ground truth answer
"""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-eval-compqa"


def _load_compqa_items(data_path: str, split: str = "train", seed: int = 42) -> list[dict]:
    """Load compqa items from local cache or HuggingFace.

    Does an 80/20 train/test split with deterministic seed.
    Returns items for the requested split.
    """
    local = Path(data_path)

    if local.exists():
        with open(local) as f:
            all_items = json.load(f)
        if isinstance(all_items, dict) and split in all_items:
            return all_items[split]
        if isinstance(all_items, list):
            pass  # will split below
    else:
        # Download from HuggingFace
        print(f"  [compqa] Downloading from {HF_REPO}...")
        from datasets import load_dataset as _hf_load
        ds = _hf_load(HF_REPO, split="train")

        # Unflatten meta_ fields
        all_items = []
        for row in ds:
            item = {}
            meta = {}
            for k, v in row.items():
                if k.startswith("meta_"):
                    if isinstance(v, str):
                        try:
                            v = json.loads(v)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    meta[k[5:]] = v
                else:
                    item[k] = v
            item["metadata"] = meta
            all_items.append(item)

        # Cache locally
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "w") as f:
            json.dump(all_items, f)
        print(f"  [compqa] Cached {len(all_items)} items to {local}")

    # Deterministic 80/20 split
    rng = random.Random(seed)
    indices = list(range(len(all_items)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * 0.8)

    if split == "train":
        return [all_items[i] for i in indices[:split_idx]]
    else:
        return [all_items[i] for i in indices[split_idx:]]


def load_cot_compqa_data(
    data_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 8000,
    stride: int = 5,
    max_positions_per_layer: int = 20,
    n_prompt_positions: int = 5,
    max_target_tokens: int = 8192,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate CompQA training data from HuggingFace dataset.

    For each item:
      1. Get CoT text from metadata
      2. Tokenize original question + CoT
      3. Get stride positions over CoT region
      4. Oracle prompt = per-item question about reasoning quality
      5. Target = Gemini's ground truth answer

    Uses 80% train split. 20% held out for eval.
    """
    from cot_utils import get_cot_stride_positions, layer_percent_to_layer

    random.seed(seed)

    LAYERS = [
        layer_percent_to_layer(model_name, 25),
        layer_percent_to_layer(model_name, 50),
        layer_percent_to_layer(model_name, 75),
    ]

    items = _load_compqa_items(data_path, split="train", seed=seed)
    print(f"  Loaded {len(items)} compqa train items")
    print(f"  Layers: {LAYERS}")

    if len(items) > num_examples:
        random.shuffle(items)
        items = items[:num_examples]

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []

    for item in items:
        meta = item.get("metadata", {})
        cot_text = (meta.get("cot_text") or "").strip()
        question = (item.get("clean_prompt") or "").strip()
        test_prompt = (item.get("test_prompt") or "").strip()
        correct_answer = (item.get("correct_answer") or "").strip()

        if not cot_text or not question or not test_prompt or not correct_answer:
            continue

        # Tokenize question + CoT
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        # Strip think tags from CoT
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Get stride positions over CoT
        positions = get_cot_stride_positions(
            prompt_len, len(full_ids),
            stride=stride,
            max_positions=max_positions_per_layer,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * 3
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        # Truncate target if needed
        target_ids = tokenizer.encode(correct_answer, add_special_tokens=False)
        if len(target_ids) > max_target_tokens:
            target_text = tokenizer.decode(
                target_ids[:max_target_tokens], skip_special_tokens=True
            )
        else:
            target_text = correct_answer

        if not target_text.strip():
            continue

        layers_str = ", ".join(str(l) for l in LAYERS)
        oracle_prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"{test_prompt}"
        )

        datapoints.append({
            "datapoint_type": "cot_compqa",
            "prompt": oracle_prompt,
            "target_response": target_text,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} compqa training examples")
    return datapoints
