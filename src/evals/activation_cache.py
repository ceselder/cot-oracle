"""Shared activation extraction + caching utilities for evals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from core.ao import (
    collect_activations_at_positions,
)
from cot_utils import get_cot_positions, get_cot_punctuation_positions


CACHE_FORMAT_VERSION = 2
_CACHE_META_PREFIX = "__cache_"
_NOISE_MODE = False  # set by train.py when --noise-activations is active


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
    generation_adapter_name: str | None = None,
    stride: int,
    **_kwargs,
) -> ActivationBundle | None:
    """Extract activations from a CoT trace using fixed-stride positions."""
    cot_text = (cot_text or "").strip()
    if not cot_text:
        return None

    full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)

    # Tokenize to get prompt/total lengths for stride computation
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    positions = get_cot_positions(len(prompt_ids), len(all_ids), stride=stride, tokenizer=tokenizer, input_ids=all_ids)

    if len(positions) < 2:
        return None

    activations = collect_activations_at_positions(
        model,
        tokenizer,
        full_text,
        act_layer,
        positions,
        device=device,
        adapter_name=generation_adapter_name,
    )

    return ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=positions,
        sentences=[],
    )


def extract_multilayer_activation_bundle(
    model,
    tokenizer,
    *,
    eval_name: str,
    example_id: str,
    prompt: str,
    cot_text: str,
    layers: list[int],
    device: str = "cuda",
    generation_adapter_name: str | None = None,
    stride: int | str,
    **_kwargs,
) -> ActivationBundle | None:
    """Extract activations from multiple layers, concatenated as [K*n_layers, D].

    Matches the training format: for each layer, extract K stride positions,
    then concatenate [K_from_L9, K_from_L18, K_from_L27] -> [3K, D].
    """
    cot_text = (cot_text or "").strip()
    if not cot_text:
        return None

    full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    positions = get_cot_positions(len(prompt_ids), len(all_ids), stride=stride, tokenizer=tokenizer, input_ids=all_ids)

    if len(positions) < 2:
        return None

    layer_acts = []
    for layer in layers:
        acts = collect_activations_at_positions(
            model,
            tokenizer,
            full_text,
            layer,
            positions,
            device=device,
            adapter_name=generation_adapter_name,
        )
        layer_acts.append(acts)

    activations = torch.cat(layer_acts, dim=0)  # [K * n_layers, D]

    return ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=positions,
        sentences=[],
    )


def extract_punctuation_activation_bundle(
    model,
    tokenizer,
    *,
    eval_name: str,
    example_id: str,
    prompt: str,
    cot_text: str,
    layers: list[int],
    device: str = "cuda",
    generation_adapter_name: str | None = None,
    fallback_stride: int,
    **_kwargs,
) -> ActivationBundle | None:
    """Extract activations at punctuation positions from multiple layers.

    Like extract_multilayer_activation_bundle but uses punctuation-based
    positions instead of fixed-stride. Falls back to stride-based if fewer
    than 2 punctuation positions are found.

    Returns activations concatenated as [K*n_layers, D].
    """
    cot_text = (cot_text or "").strip()
    if not cot_text:
        return None

    full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    positions = get_cot_punctuation_positions(
        len(prompt_ids), len(all_ids), tokenizer, all_ids,
        fallback_stride=fallback_stride,
    )

    if len(positions) < 2:
        return None

    layer_acts = []
    for layer in layers:
        acts = collect_activations_at_positions(
            model,
            tokenizer,
            full_text,
            layer,
            positions,
            device=device,
            adapter_name=generation_adapter_name,
        )
        layer_acts.append(acts)

    activations = torch.cat(layer_acts, dim=0)  # [K * n_layers, D]

    return ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=positions,
        sentences=[],
    )


def extract_activations(
    model,
    tokenizer,
    *,
    eval_name: str,
    example_id: str,
    prompt: str,
    cot_text: str,
    stride: int | str,
    layers: list[int] | None = None,
    act_layer: int | None = None,
    device: str = "cuda",
    generation_adapter_name: str | None = None,
    **_kwargs,
) -> ActivationBundle | None:
    """Unified activation extraction entry point.

    Routes internally based on arguments:
    - stride: int for fixed-stride, "punctuation" for punctuation boundaries.
      Passed directly to get_cot_positions() which handles both.
    - layers: list of layer indices for multi-layer extraction (concatenated
      as [K*n_layers, D]). None falls back to single-layer using act_layer.
    """
    cot_text = (cot_text or "").strip()
    if not cot_text:
        return None

    full_text = build_full_text_from_prompt_and_cot(tokenizer, prompt, cot_text)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)
    positions = get_cot_positions(
        len(prompt_ids), len(all_ids), stride=stride,
        tokenizer=tokenizer, input_ids=all_ids,
    )

    if len(positions) < 2:
        return None

    # Determine which layers to extract from
    if layers is not None:
        extract_layers = layers
    elif act_layer is not None:
        extract_layers = [act_layer]
    else:
        raise ValueError(
            "extract_activations() requires either `layers` (multi-layer) "
            "or `act_layer` (single-layer)."
        )

    layer_acts = []
    for layer in extract_layers:
        acts = collect_activations_at_positions(
            model, tokenizer, full_text, layer, positions,
            device=device, adapter_name=generation_adapter_name,
        )
        layer_acts.append(acts)

    activations = torch.cat(layer_acts, dim=0) if len(layer_acts) > 1 else layer_acts[0]

    return ActivationBundle(
        eval_name=eval_name,
        example_id=example_id,
        prompt=prompt,
        cot_text=cot_text,
        activations=activations,
        boundary_positions=positions,
        sentences=[],
    )


def cache_path(base_dir: Path, eval_name: str, example_id: str) -> Path:
    return Path(base_dir) / eval_name / f"{example_id}.pt"


def _sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _invalidate_cache(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _meta_get(meta: dict, key: str, default=None):
    cache_key = f"{_CACHE_META_PREFIX}{key}"
    if cache_key in meta:
        return meta.get(cache_key)
    return meta.get(key, default)


def _normalize_layers(layers: list[int] | None) -> list[int] | None:
    if layers is None:
        return None
    return [int(x) for x in layers]


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
    fd, tmp_path_str = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f"{output_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


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
    stride: int | str | None = None,
    layers: list[int] | None = None,
    model_name: str | None = None,
    placeholder_token: str | None = None,
    oracle_adapter_name: str | None = None,
    generation_adapter_name: str | None = None,
    expected_prompt: str | None = None,
    expected_cot: str | None = None,
    require_cache_v2: bool = False,
) -> ActivationBundle | None:
    """Load a cached activation bundle with optional staleness validation.

    When validation args are provided, stale caches are deleted automatically.
    If require_cache_v2 is True, old cache formats are treated as stale.
    """
    if base_dir is None or _NOISE_MODE:
        return None
    path = cache_path(base_dir, eval_name, example_id)
    if not path.exists():
        return None
    bundle = load_bundle(path, map_location=map_location)
    if bundle.activations is None:
        return None

    meta = dict(bundle.metadata or {})

    if require_cache_v2:
        cached_version = _meta_get(meta, "format_version")
        if cached_version != CACHE_FORMAT_VERSION:
            _invalidate_cache(path)
            return None

    # --- Staleness validation ---
    if stride is not None:
        cached_stride = _meta_get(meta, "stride")
        if cached_stride is None or str(cached_stride) != str(stride):
            _invalidate_cache(path)
            return None

    normalized_layers = _normalize_layers(layers)
    if normalized_layers is not None:
        cached_layers = _meta_get(meta, "layers")
        if cached_layers is None:
            _invalidate_cache(path)
            return None
        try:
            cached_layers = [int(x) for x in cached_layers]
        except Exception:
            _invalidate_cache(path)
            return None
        if cached_layers != normalized_layers:
            _invalidate_cache(path)
            return None
        n_positions = len(bundle.boundary_positions)
        expected_rows = n_positions * len(normalized_layers)
        if bundle.activations.shape[0] != expected_rows:
            _invalidate_cache(path)
            return None

    if model_name is not None:
        cached_model_name = _meta_get(meta, "model_name")
        if cached_model_name != model_name:
            _invalidate_cache(path)
            return None

    if placeholder_token is not None:
        cached_placeholder = _meta_get(meta, "placeholder_token")
        if cached_placeholder != placeholder_token:
            _invalidate_cache(path)
            return None

    if oracle_adapter_name is not None:
        cached_oracle_adapter = _meta_get(meta, "oracle_adapter_name")
        if cached_oracle_adapter != oracle_adapter_name:
            _invalidate_cache(path)
            return None

    if generation_adapter_name is not None:
        cached_generation_adapter = _meta_get(meta, "generation_adapter_name")
        if cached_generation_adapter != generation_adapter_name:
            _invalidate_cache(path)
            return None

    if expected_prompt is not None:
        cached_prompt_sha = _meta_get(meta, "prompt_sha256")
        expected_prompt_sha = _sha256_text(expected_prompt)
        if cached_prompt_sha != expected_prompt_sha:
            _invalidate_cache(path)
            return None

    if expected_cot is not None:
        cached_cot_sha = _meta_get(meta, "cot_sha256")
        expected_cot_sha = _sha256_text(expected_cot)
        if cached_cot_sha != expected_cot_sha:
            _invalidate_cache(path)
            return None

    return bundle


def save_bundle_with_metadata(
    bundle: ActivationBundle,
    base_dir: Path,
    *,
    stride: int | str | None = None,
    layers: list[int] | None = None,
    model_name: str | None = None,
    placeholder_token: str | None = None,
    oracle_adapter_name: str | None = None,
    generation_adapter_name: str | None = None,
    clean_response: str | None = None,
    test_response: str | None = None,
    clean_answer: str | None = None,
    test_answer: str | None = None,
    prompt_for_hash: str | None = None,
    cot_for_hash: str | None = None,
    extra_metadata: dict | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Save an activation bundle with cache metadata for staleness checks.

    Returns the saved path, or None if nothing was saved.
    """
    if bundle is None or bundle.activations is None or _NOISE_MODE:
        return None
    path = cache_path(base_dir, bundle.eval_name, bundle.example_id)
    if path.exists() and not overwrite:
        return None

    bundle.clean_response = clean_response if clean_response is not None else bundle.clean_response
    bundle.test_response = test_response if test_response is not None else bundle.test_response
    bundle.clean_answer = clean_answer if clean_answer is not None else bundle.clean_answer
    bundle.test_answer = test_answer if test_answer is not None else bundle.test_answer
    meta = dict(bundle.metadata or {})
    meta[f"{_CACHE_META_PREFIX}format_version"] = CACHE_FORMAT_VERSION
    if stride is not None:
        meta[f"{_CACHE_META_PREFIX}stride"] = str(stride)
        meta["stride"] = str(stride)
    if layers is not None:
        normalized_layers = _normalize_layers(layers)
        meta[f"{_CACHE_META_PREFIX}layers"] = normalized_layers
        meta["layers"] = normalized_layers
    if model_name is not None:
        meta[f"{_CACHE_META_PREFIX}model_name"] = model_name
    if placeholder_token is not None:
        meta[f"{_CACHE_META_PREFIX}placeholder_token"] = placeholder_token
    if oracle_adapter_name is not None:
        meta[f"{_CACHE_META_PREFIX}oracle_adapter_name"] = oracle_adapter_name
    if generation_adapter_name is not None:
        meta[f"{_CACHE_META_PREFIX}generation_adapter_name"] = generation_adapter_name
    prompt_sha = _sha256_text(prompt_for_hash if prompt_for_hash is not None else bundle.prompt)
    cot_sha = _sha256_text(cot_for_hash if cot_for_hash is not None else bundle.cot_text)
    if prompt_sha is not None:
        meta[f"{_CACHE_META_PREFIX}prompt_sha256"] = prompt_sha
    if cot_sha is not None:
        meta[f"{_CACHE_META_PREFIX}cot_sha256"] = cot_sha
    if extra_metadata:
        meta.update(extra_metadata)
    bundle.metadata = meta

    try:
        save_bundle(bundle, path)
    except Exception as e:
        print(f"    [cache] Failed to save {bundle.eval_name}/{bundle.example_id}: {e}")
        return None
    return path
