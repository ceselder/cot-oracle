"""
Task 3: Decorative CoT Detection (~10K examples, self-supervised)

Binary classification: is this CoT load_bearing or decorative?
Uses sentence-structured format with 3 acts per sentence boundary (L25%, L50%, L75%).

Ground truth comes from the corpus itself:
- load_bearing: CoT correct, direct answer wrong -> reasoning was necessary
- both_correct (decorative): both CoT and direct correct -> model already knew

Balanced 50/50 sampling.
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
    layer_percents: list[int],
    num_examples: int = 10000,
    max_sentences: int = 15,
    seed: int = 42,
    corpus_entries: list[dict] | None = None,
) -> list[dict]:
    from signs_of_life.ao_lib import layer_percent_to_layer

    random.seed(seed)

    if corpus_entries is not None:
        corpus = corpus_entries
    else:
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))

    load_bearing = [e for e in corpus if e.get("category") == "load_bearing"]
    decorative = [e for e in corpus if e.get("category") == "both_correct"]

    if not load_bearing:
        raise ValueError("No load_bearing entries in corpus. Was --keep-all used during generation?")
    if not decorative:
        raise ValueError("No both_correct entries in corpus. Was --keep-all used during generation?")

    print(f"  load_bearing: {len(load_bearing)}, decorative (both_correct): {len(decorative)}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    cached_lb = _pretokenize_boundary_entries(load_bearing, tokenizer, max_sentences, "decorative/load_bearing")
    cached_dec = _pretokenize_boundary_entries(decorative, tokenizer, max_sentences, "decorative/decorative")

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
            f"Activations from {N} sentence boundaries. "
            f"Is this chain-of-thought reasoning load-bearing or decorative? "
            f"Answer: load_bearing or decorative."
        )

        datapoints.append({
            "datapoint_type": "cot_decorative",
            "prompt": prompt,
            "target_response": target,
            "layers": layers,
            "num_positions": N,
            "context_input_ids": list(context_ids),
            "context_positions": list(positions),
        })
        pbar.update(1)

    pbar.close()
    return datapoints[:num_examples]
