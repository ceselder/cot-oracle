"""CoT sampling from HuggingFace corpus with stride-5 position selection."""

from __future__ import annotations

import random
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_corpus(corpus_name: str) -> list[dict]:
    """Load CoT corpus from HuggingFace."""
    ds = load_dataset(corpus_name, split="train")
    return list(ds)


def sample_positions(
    cot_start: int,
    cot_end: int,
    stride: int,
    min_positions: int,
    max_positions: int,
    rng: random.Random,
) -> list[int]:
    """Sample positions from the CoT region.

    Three modes (randomly chosen):
      - 50%: local — 1-3 adjacent stride positions from a random spot
      - 30%: spread — 5-15 positions spread across the entire CoT
      - 20%: all — every stride position (full CoT view)

    Returns position indices into the full tokenized sequence.
    """
    all_stride_positions = list(range(cot_start, cot_end, stride))
    if not all_stride_positions:
        return []

    roll = rng.random()

    if roll < 0.50:
        # Local: 1-3 adjacent positions
        k = rng.randint(min_positions, min(max_positions, len(all_stride_positions)))
        max_start = len(all_stride_positions) - k
        window_start = rng.randint(0, max_start)
        return all_stride_positions[window_start : window_start + k]

    elif roll < 0.80:
        # Spread: 5-15 positions sampled uniformly across the CoT
        k = rng.randint(5, min(15, len(all_stride_positions)))
        return sorted(rng.sample(all_stride_positions, k))

    else:
        # All: every stride position
        return all_stride_positions


def prepare_example(
    item: dict,
    tokenizer: AutoTokenizer,
    layers: list[int],
    stride: int,
    min_positions: int,
    max_positions: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Tokenize a corpus item and select positions.

    Returns dict with:
        question, cot_response, context_input_ids, selected_positions,
        cot_start, cot_end
    Or None if the item has no usable CoT region.
    """
    question = item.get("question", item.get("prompt", ""))
    cot_response = item.get("cot_response", item.get("cot_text", ""))
    if not cot_response:
        return None

    # Tokenize prompt part
    prompt_msgs = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # Tokenize full prompt + CoT
    full_msgs = prompt_msgs + [{"role": "assistant", "content": cot_response}]
    full_text = tokenizer.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    cot_start = prompt_len
    cot_end = len(full_ids)
    if cot_end <= cot_start:
        return None

    selected = sample_positions(cot_start, cot_end, stride, min_positions, max_positions, rng)
    if not selected:
        return None

    # Replicate positions for each layer: [pos1_L9, pos2_L9, ..., pos1_L18, ...]
    n_layers = len(layers)
    all_positions = []
    for _ in range(n_layers):
        all_positions.extend(selected)

    return {
        "question": question,
        "cot_response": cot_response,
        "context_input_ids": full_ids,
        "selected_positions": all_positions,
        "base_positions": selected,  # positions before layer replication
        "cot_start": cot_start,
        "cot_end": cot_end,
    }


class CoTSampler:
    """Infinite iterator over CoT examples with position selection."""

    def __init__(
        self,
        corpus_name: str,
        tokenizer: AutoTokenizer,
        layers: list[int],
        stride: int = 5,
        min_positions: int = 1,
        max_positions: int = 3,
        seed: int = 42,
    ):
        self.corpus = load_corpus(corpus_name)
        self.tokenizer = tokenizer
        self.layers = layers
        self.stride = stride
        self.min_positions = min_positions
        self.max_positions = max_positions
        self.rng = random.Random(seed)
        print(f"[data_sampler] Loaded {len(self.corpus)} corpus items")

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        """Sample a random example, retrying on failures."""
        for _ in range(100):
            item = self.rng.choice(self.corpus)
            result = prepare_example(
                item,
                self.tokenizer,
                self.layers,
                self.stride,
                self.min_positions,
                self.max_positions,
                self.rng,
            )
            if result is not None:
                return result
        raise RuntimeError("Failed to sample a valid example after 100 attempts")

    def sample_batch(self, batch_size: int) -> list[dict]:
        """Sample a batch of examples."""
        return [next(self) for _ in range(batch_size)]
