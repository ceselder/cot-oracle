"""
CoT Decorative Detection — Is this CoT load-bearing or decorative?

Binary classification from corpus metadata (cot_correct + direct_correct).
Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def _pretokenize_boundary_entries(entries, tokenizer, max_sentences=15, desc=""):
    """Pre-tokenize corpus entries and cache boundary positions + context_ids."""
    cached = []
    for entry in tqdm(entries, desc=f"  {desc}: tokenizing corpus", leave=False):
        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue
        if "_ctx_ids" in entry:
            context_ids = entry["_ctx_ids"]
        else:
            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            cot_text = entry["cot_response"]
            think_end = cot_text.find("</think>")
            if think_end != -1:
                cot_text = cot_text[:think_end]
            context_ids = tokenizer(formatted + cot_text, add_special_tokens=False)["input_ids"]
        positions = [p for p in boundary_positions[:max_sentences] if p < len(context_ids)]
        if len(positions) < 2:
            continue
        cached.append((context_ids, positions))
    return cached


def load_cot_decorative_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 10000,
    stride: int = 5,
    max_positions_per_layer: int = 20,
    n_prompt_positions: int = 5,
    max_sentences: int = 15,
    seed: int = 42,
    corpus_entries: list[dict] | None = None,
    **_kwargs,
) -> list[dict]:
    """
    Generate decorative CoT detection data with multi-layer stride.

    load_bearing: CoT correct, direct wrong -> reasoning was necessary
    decorative: both correct -> model already knew the answer
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    LAYERS = [
        layer_percent_to_layer(model_name, 25),
        layer_percent_to_layer(model_name, 50),
        layer_percent_to_layer(model_name, 75),
    ]

    if corpus_entries is not None:
        corpus = corpus_entries
    else:
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("cot_response", "").strip():
                        corpus.append(entry)

    load_bearing = [e for e in corpus if e.get("category") == "load_bearing"]
    decorative = [e for e in corpus if e.get("category") == "both_correct"]

    # Fallback: use cot_correct + direct_correct fields
    if not load_bearing or not decorative:
        load_bearing = [e for e in corpus
                        if e.get("cot_correct") and not e.get("direct_correct")]
        decorative = [e for e in corpus
                      if e.get("cot_correct") and e.get("direct_correct")]

    if not load_bearing:
        raise ValueError("No load_bearing entries in corpus")
    if not decorative:
        raise ValueError("No decorative entries in corpus")

    print(f"  load_bearing: {len(load_bearing)}, decorative: {len(decorative)}")

    cached_lb = _pretokenize_boundary_entries(load_bearing, tokenizer, max_sentences, "decorative/load_bearing")
    cached_dec = _pretokenize_boundary_entries(decorative, tokenizer, max_sentences, "decorative/decorative")

    layers_str = ", ".join(str(l) for l in LAYERS)

    datapoints = []
    pbar = tqdm(total=num_examples, desc="  decorative: sampling", leave=False)

    while len(datapoints) < num_examples:
        if len(datapoints) % 2 == 0:
            context_ids, positions = random.choice(cached_lb)
            target = "load_bearing"
        else:
            context_ids, positions = random.choice(cached_dec)
            target = "decorative"

        N = len(positions)

        prompt = (
            f"Activations from {N} positions across layers {layers_str}. "
            f"Is this chain-of-thought load-bearing or decorative? "
            f"Answer: load_bearing or decorative."
        )

        datapoints.append({
            "datapoint_type": "cot_decorative",
            "prompt": prompt,
            "target_response": target,
            "layers": LAYERS,
            "num_positions": N,
            "context_input_ids": list(context_ids),
            "context_positions": list(positions),
        })
        pbar.update(1)

    pbar.close()
    return datapoints[:num_examples]
