"""
CoT Domain Classification — What domain is this reasoning about?

Multi-class classification: given CoT activations, predict the domain/topic.
Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_domain_data(
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
    Generate domain classification data with multi-layer stride.

    Labels from corpus 'domain' or 'source' field.
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    _SOURCE_TO_DOMAIN = {
        "MATH": "math", "GSM8K": "math", "GPQA": "science", "BBH": "logic",
        "ARC": "science", "StrategyQA": "commonsense", "DROP": "reading",
        "LogiQA": "logic", "MMLU-Pro": "multi_domain", "CommonsenseQA": "commonsense",
        "AQUA-RAT": "math", "MedQA": "medical", "ScienceQA": "science",
    }

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip():
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"Empty corpus at {corpus_path}")

    # Group by domain
    by_domain = {}
    for entry in corpus:
        domain = entry.get("domain") or _SOURCE_TO_DOMAIN.get(entry.get("source", ""), "unknown")
        by_domain.setdefault(domain, []).append(entry)

    domains = sorted(by_domain.keys())
    print(f"  Domains: {domains}")
    for d in domains:
        print(f"    {d}: {len(by_domain[d])} entries")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    datapoints = []
    attempts = 0

    while len(datapoints) < num_examples and attempts < num_examples * 3:
        attempts += 1

        domain = domains[len(datapoints) % len(domains)]
        entry = random.choice(by_domain[domain])

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
            f"What domain is this reasoning about? "
            f"Answer with one word: {', '.join(domains)}."
        )

        datapoints.append({
            "datapoint_type": "cot_domain",
            "prompt": prompt,
            "target_response": domain,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} domain classification examples")
    return datapoints[:num_examples]
