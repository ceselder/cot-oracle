"""CoT sampler with sentence-boundary splitting for GRPO.

Loads pre-computed CoTs from HuggingFace corpus, splits each at a sentence
boundary near the middle, and selects activation positions from the first half
(near the split point). The judge sees both halves; the oracle only gets
activations from the first half.
"""

from __future__ import annotations

import random
import re
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def split_cot_into_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences."""
    text = re.sub(r"<think>|</think>", "", cot_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


class OnlineCoTSampler:
    """Load corpus CoTs, split at sentence boundary, select first-half positions."""

    def __init__(
        self,
        corpus_name: str,
        tokenizer: AutoTokenizer,
        layers: list[int],
        stride: int = 5,
        min_cot_sentences: int = 4,
        seed: int = 42,
        position_mode_probs: dict[str, float] | None = None,
        contiguous_window_fraction: float = 0.4,
        max_contiguous_positions: int = 64,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.layers = layers
        self.stride = stride
        self.min_cot_sentences = min_cot_sentences
        self.rng = random.Random(seed)
        self.contiguous_window_fraction = contiguous_window_fraction
        self.max_contiguous_positions = max_contiguous_positions
        self.position_mode_probs = position_mode_probs or {
            "single": 0.25,
            "last2": 0.125,
            "last3": 0.125,
            "contiguous": 0.5,
        }
        total_prob = sum(self.position_mode_probs.values())
        if total_prob <= 0:
            raise ValueError('position_mode_probs must sum to a positive value')
        self.position_mode_probs = {
            key: value / total_prob for key, value in self.position_mode_probs.items()
        }

        ds = load_dataset(corpus_name, split="train")
        self.corpus = list(ds)
        self.rng.shuffle(self.corpus)
        self._idx = 0
        print(f"[online_sampler] Loaded {len(self.corpus)} corpus items")
        print(
            f"[online_sampler] Position mix: {self.position_mode_probs}, "
            f"window_fraction={self.contiguous_window_fraction}, "
            f"max_contiguous_positions={self.max_contiguous_positions}"
        )

    def sample_batch(self, batch_size: int) -> list[dict[str, Any]]:
        results = []
        attempts = 0
        while len(results) < batch_size and attempts < batch_size * 10:
            item = self.corpus[self._idx % len(self.corpus)]
            self._idx += 1
            attempts += 1

            example = self._prepare_example(item)
            if example is not None:
                results.append(example)

        if len(results) < batch_size:
            print(f"  [sampler] WARNING: only got {len(results)}/{batch_size} valid examples")
        return results

    def _sample_position_mode(self) -> str:
        draw = self.rng.random()
        running = 0.0
        last_mode = "contiguous"
        for mode, prob in self.position_mode_probs.items():
            running += prob
            last_mode = mode
            if draw <= running:
                return mode
        return last_mode

    def _select_base_positions(self, cot_start: int, split_token_pos: int) -> list[int]:
        span = max(split_token_pos - cot_start, 0)
        window_start = cot_start + int((1.0 - self.contiguous_window_fraction) * span)
        candidate_positions = list(range(window_start, split_token_pos + 1))
        if not candidate_positions:
            return [split_token_pos]
        if len(candidate_positions) > self.max_contiguous_positions:
            candidate_positions = candidate_positions[-self.max_contiguous_positions:]

        mode = self._sample_position_mode()
        if mode == "single":
            return candidate_positions[-1:]
        if mode == "last2":
            return candidate_positions[-2:]
        if mode == "last3":
            return candidate_positions[-3:]
        return candidate_positions

    def _prepare_example(self, item: dict) -> dict[str, Any] | None:
        question = item.get("question", item.get("prompt", ""))
        cot_text = item.get("cot_response", item.get("cot_text", ""))
        if not cot_text:
            return None

        sentences = split_cot_into_sentences(cot_text)
        if len(sentences) < self.min_cot_sentences:
            return None

        # Find split: sentence boundary nearest to middle (in 35-65% range)
        char_lens = [len(s) for s in sentences]
        total_chars = sum(char_lens)
        if total_chars < 50:
            return None

        cumulative = []
        running = 0
        for l in char_lens:
            running += l
            cumulative.append(running)

        mid = total_chars / 2
        best_idx = None
        best_dist = float("inf")
        for i, cum in enumerate(cumulative[:-1]):  # never split at very end
            frac = cum / total_chars
            if 0.35 <= frac <= 0.65:
                dist = abs(cum - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        if best_idx is None:
            # fallback: closest to middle
            for i, cum in enumerate(cumulative[:-1]):
                dist = abs(cum / total_chars - 0.5)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        if best_idx is None:
            return None

        split_idx = best_idx + 1
        first_half = " ".join(sentences[:split_idx])
        second_half = " ".join(sentences[split_idx:])
        if not first_half.strip() or not second_half.strip():
            return None

        # Tokenize full prompt + CoT
        prompt_msgs = [{"role": "user", "content": question}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        full_msgs = prompt_msgs + [{"role": "assistant", "content": cot_text}]
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        cot_start = prompt_len
        cot_end = len(full_ids)
        if cot_end <= cot_start + 10:
            return None

        # Approximate token position of split point
        first_half_ids = self.tokenizer.encode(first_half, add_special_tokens=False)
        split_token_pos = min(cot_start + len(first_half_ids), cot_end - 1)

        # Sample a local position pattern near the split point.
        base_positions = self._select_base_positions(cot_start, split_token_pos)

        return {
            "question": question,
            "cot_text": cot_text,
            "first_half": first_half,
            "second_half": second_half,
            "context_input_ids": full_ids,
            "base_positions": base_positions,
            "selected_positions": base_positions * len(self.layers),
            "cot_start": cot_start,
            "cot_end": cot_end,
        }
