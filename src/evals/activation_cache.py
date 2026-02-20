"""Shared activation extraction + caching utilities for evals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from core.ao import (
    collect_activations_at_positions,
    find_sentence_boundary_positions,
    split_cot_into_sentences,
)


@dataclass
class ActivationBundle:
    eval_name: str
    example_id: str
    prompt: str
    cot_text: str
    activations: torch.Tensor | None
    boundary_positions: list[int]
    sentences: list[str]
    clean_response: str | None = None
    test_response: str | None = None
    clean_answer: str | None = None
    test_answer: str | None = None
    metadata: dict | None = None


def build_full_text_from_prompt_and_cot(tokenizer, prompt: str, cot_text: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return formatted + cot_text


def extract_activation_bundle(
    model,
    tokenizer,
    *,
    eval_name: str,
    example_id: str,
    prompt: str,
    cot_text: str,
    act_layer: int,
    device: str = "cuda",
    max_boundaries: int = 10,
    generation_adapter_name: str | None = None,
) -> ActivationBundle | None:
    """Extract activations at sentence boundaries from a CoT trace."""
    cot_text = (cot_text or "").strip()
    if not cot_text:
        return None

    sentences = split_cot_into_sentences(cot_text)
    if len(sentences) < 2:
        return None

    full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)
    boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
    if len(boundary_positions) < 2:
        return None

    if max_boundaries > 0:
        boundary_positions = boundary_positions[:max_boundaries]

    activations = collect_activations_at_positions(
        model,
        tokenizer,
        full_text,
        act_layer,
        boundary_positions,
        device=device,
        adapter_name=generation_adapter_name,
    )

    return ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=boundary_positions,
        sentences=sentences,
    )


def cache_path(base_dir: Path, eval_name: str, example_id: str) -> Path:
    return Path(base_dir) / eval_name / f"{example_id}.pt"


def save_bundle(bundle: ActivationBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_name": bundle.eval_name,
        "example_id": bundle.example_id,
        "prompt": bundle.prompt,
        "cot_text": bundle.cot_text,
        "activations": bundle.activations.cpu() if bundle.activations is not None else None,
        "boundary_positions": bundle.boundary_positions,
        "sentences": bundle.sentences,
        "clean_response": bundle.clean_response,
        "test_response": bundle.test_response,
        "clean_answer": bundle.clean_answer,
        "test_answer": bundle.test_answer,
        "metadata": bundle.metadata or {},
    }
    torch.save(payload, output_path)


def load_bundle(path: Path, map_location: str = "cpu") -> ActivationBundle:
    data = torch.load(path, map_location=map_location)
    return ActivationBundle(
        eval_name=str(data.get("eval_name", "")),
        example_id=str(data.get("example_id", "")),
        prompt=str(data.get("prompt", "")),
        cot_text=str(data.get("cot_text", "")),
        activations=data.get("activations"),
        boundary_positions=list(data.get("boundary_positions", [])),
        sentences=list(data.get("sentences", [])),
        clean_response=data.get("clean_response"),
        test_response=data.get("test_response"),
        clean_answer=data.get("clean_answer"),
        test_answer=data.get("test_answer"),
        metadata=dict(data.get("metadata", {})),
    )


def maybe_load_cached_bundle(
    base_dir: Path | None,
    *,
    eval_name: str,
    example_id: str,
    map_location: str = "cpu",
) -> ActivationBundle | None:
    if base_dir is None:
        return None
    path = cache_path(base_dir, eval_name, example_id)
    if not path.exists():
        return None
    return load_bundle(path, map_location=map_location)

