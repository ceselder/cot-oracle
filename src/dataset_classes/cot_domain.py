"""
Task 4: Domain Classification (~15K examples)

Classify what domain/topic the problem is from. Labels are FREE from corpus metadata.
One random layer per example from [25%, 50%, 75%] depth.

Ground truth: source dataset name -> domain label.
Classes: math, science, logic, commonsense, reading, medical, multi_domain
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_domain_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    seed: int = 42,
) -> list[dict]:
    """
    Generate domain classification training data.

    Each example: sentence-boundary activations -> domain label.
    Domain labels come from SOURCE_TO_DOMAIN mapping in generate_cots.py.
    """
    from signs_of_life.ao_lib import layer_percent_to_layer

    # Inline mapping to avoid import chain that requires peft
    _SOURCE_TO_DOMAIN = {
        "MATH": "math", "GSM8K": "math", "GPQA": "science", "BBH": "logic",
        "ARC": "science", "StrategyQA": "commonsense", "DROP": "reading",
        "LogiQA": "logic", "MMLU-Pro": "multi_domain", "CommonsenseQA": "commonsense",
        "AQUA-RAT": "math", "MedQA": "medical",
    }

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    # Group by domain for balanced sampling
    by_domain = {}
    for entry in corpus:
        domain = entry.get("domain") or _SOURCE_TO_DOMAIN.get(entry["source"], "unknown")
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(entry)

    domains = sorted(by_domain.keys())
    print(f"  Domains: {domains}")
    for d in domains:
        print(f"    {d}: {len(by_domain[d])} entries")

    datapoints = []

    while len(datapoints) < num_examples:
        # Cycle through domains for balance
        domain = domains[len(datapoints) % len(domains)]
        entry = random.choice(by_domain[domain])

        boundary_positions = entry.get("boundary_positions", [])
        if len(boundary_positions) < 2:
            continue

        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        full_text = formatted + cot_text
        context_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        positions = boundary_positions[:max_sentences]
        positions = [p for p in positions if p < len(context_ids)]
        if len(positions) < 2:
            continue

        N = len(positions)

        context_slice = context_ids[:positions[-1] + 1]

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"What domain is this reasoning about? "
            f"Answer with one word: {', '.join(domains)}."
        )

        datapoints.append({
            "datapoint_type": "cot_domain",
            "prompt": prompt,
            "target_response": domain,
            "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
            "num_positions": N,
            "context_input_ids": context_slice,
            "context_positions": list(positions),
        })

    return datapoints[:num_examples]
