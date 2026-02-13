"""
Task: Unverbalized Behavior Detection (30K examples, SAE-supervised)

The key new task for v3. Pipeline:
1. Run CoT through model, extract top-K SAE latents at sentence boundaries
2. Get latent descriptions (from autointerp / max activating examples)
3. Feed to Gemini Flash: "What's happening that's NOT in the CoT text?"
4. Gemini's answer = training label

Oracle sees: activations at sentence boundaries
Oracle predicts: what's unverbalized

This teaches the fundamental skill: detect the gap between text and activations.
Unfaithfulness is a special case of this gap.
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def _load_sae_labels(labels_path: str) -> dict[tuple[str, int], dict]:
    """Load pre-extracted SAE latent labels from JSONL.

    Each line: {"id": str, "sentence_idx": int, "top_features": [...],
                "unverbalized_description": str}
    """
    labels = {}
    with open(labels_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                key = (entry["id"], entry["sentence_idx"])
                labels[key] = entry
    return labels


def extract_sae_labels(
    corpus: list[dict],
    model,
    tokenizer,
    sae,
    feature_descriptions: dict[int, str],
    api_key: str,
    device: str = "cuda",
    top_k: int = 10,
    sae_layer: int = 18,
) -> list[dict]:
    """
    Extract SAE latent descriptions at sentence boundaries, then query
    Gemini Flash for what's unverbalized.

    Args:
        corpus: List of corpus entries with sentences and boundary_positions
        model: The base model (no LoRA)
        tokenizer: Tokenizer
        sae: Loaded SAE (from AO repo's load_sae)
        feature_descriptions: Dict mapping feature index -> description string
        api_key: OpenRouter API key
        device: Device string
        top_k: Number of top SAE features to extract per position
        sae_layer: Layer where SAE was trained

    Returns:
        List of label dicts with unverbalized descriptions
    """
    import torch
    import requests
    from tqdm import tqdm

    labels = []

    for entry in tqdm(corpus, desc="SAE latent extraction"):
        sentences = entry["sentences"]
        boundary_positions = entry.get("boundary_positions", [])

        if len(sentences) < 3 or len(boundary_positions) < 3:
            continue

        # Build full text
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)

        # Hook to capture hidden states at SAE layer
        hidden_states = None
        layer_module = None
        try:
            layer_module = model.model.layers[sae_layer]
        except AttributeError:
            try:
                layer_module = model.base_model.model.model.layers[sae_layer]
            except AttributeError:
                continue

        def hook_fn(module, inp, out):
            nonlocal hidden_states
            hidden_states = out[0] if isinstance(out, tuple) else out

        handle = layer_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if hidden_states is None:
            continue

        # For each sentence boundary, encode through SAE and get top-K features
        for s_idx, pos in enumerate(boundary_positions):
            if pos >= hidden_states.shape[1]:
                continue
            if s_idx >= len(sentences):
                continue

            h = hidden_states[0, pos, :].unsqueeze(0)  # [1, d_model]
            with torch.no_grad():
                encoded = sae.encode(h)  # [1, n_features]

            # Get top-K active features
            values, indices = torch.topk(encoded[0], min(top_k, encoded.shape[1]))
            top_features = []
            for val, idx in zip(values.tolist(), indices.tolist()):
                if val > 0:
                    desc = feature_descriptions.get(idx, f"feature_{idx}")
                    top_features.append({
                        "index": idx,
                        "activation": val,
                        "description": desc,
                    })

            if not top_features:
                continue

            # Query Gemini Flash for unverbalized behavior
            feature_text = "\n".join(
                f"- {f['description']} (activation: {f['activation']:.2f})"
                for f in top_features
            )
            context = " ".join(sentences[max(0, s_idx - 2):s_idx + 1])

            gemini_prompt = (
                f"Below is a sentence from a model's chain-of-thought reasoning, "
                f"along with the top active SAE (sparse autoencoder) features at "
                f"that point in the model's computation.\n\n"
                f"CoT text around this point:\n\"{context}\"\n\n"
                f"Top active internal features:\n{feature_text}\n\n"
                f"Based on the active features, what is happening in the model's "
                f"internal computation that is NOT mentioned in the chain-of-thought "
                f"text? Focus on gaps between stated reasoning and actual computation. "
                f"Answer in 1-2 sentences."
            )

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "google/gemini-2.0-flash-001",
                        "messages": [{"role": "user", "content": gemini_prompt}],
                        "max_tokens": 100,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                result = response.json()
                description = result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                description = "Unable to determine unverbalized behavior."

            labels.append({
                "id": entry["id"],
                "sentence_idx": s_idx,
                "sentence_text": sentences[s_idx],
                "top_features": top_features,
                "unverbalized_description": description,
            })

            time.sleep(0.05)  # Rate limiting

    return labels


def load_cot_unverbalized_data(
    corpus_path: str,
    labels_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 30000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate unverbalized behavior detection training data.

    Each example: single sentence boundary activation -> description of
    what the model is doing internally that isn't stated in the CoT text.

    Requires pre-extracted SAE labels (from extract_sae_labels or
    extract_labels.py --sae-labels).
    """
    from signs_of_life.ao_lib import layer_percent_to_layer

    random.seed(seed)

    # Load corpus
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                corpus[entry["id"]] = entry

    # Load SAE labels
    sae_labels = _load_sae_labels(labels_path)

    if not sae_labels:
        raise ValueError(f"No SAE labels found at {labels_path}")

    print(f"  Loaded {len(sae_labels)} SAE labels")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    # Build candidate pool
    candidates = []
    for key, label in sae_labels.items():
        entry_id, s_idx = key
        if entry_id not in corpus:
            continue
        entry = corpus[entry_id]
        if s_idx >= len(entry.get("boundary_positions", [])):
            continue
        desc = label.get("unverbalized_description", "")
        if not desc or desc == "Unable to determine unverbalized behavior.":
            continue
        candidates.append((entry, label, s_idx))

    if not candidates:
        raise ValueError("No valid candidates with unverbalized descriptions")

    print(f"  Valid candidates: {len(candidates)}")

    datapoints = []

    while len(datapoints) < num_examples:
        entry, label, s_idx = random.choice(candidates)
        layer = random.choice(layers)

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + entry["cot_response"]
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        pos = entry["boundary_positions"][s_idx]
        if pos >= len(context_ids):
            continue

        context_slice = context_ids[:pos + 1]
        target = label["unverbalized_description"]

        prompt = (
            "What is happening in the model's internal computation at this "
            "reasoning step that is NOT stated in the chain-of-thought text?"
        )

        datapoints.append({
            "datapoint_type": "cot_unverbalized",
            "prompt": prompt,
            "target_response": target,
            "layer": layer,
            "num_positions": 1,
            "context_input_ids": context_slice,
            "context_positions": [pos],
        })

    return datapoints[:num_examples]
