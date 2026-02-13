"""
Task 6: Persona Detection (~15K examples)

What persona/system prompt was the model given? Tests reading unverbalized context.
Uses sentence-structured format with 3 acts per sentence boundary.

Requires corpus generated with --personas flag (each entry has a "persona" field).
Ground truth: the system prompt used (known by construction).
"""

import json
import random

from transformers import AutoTokenizer


def load_cot_persona_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    seed: int = 42,
) -> list[dict]:
    """
    Generate persona detection training data.

    Each example: sentence-boundary activations -> persona label.
    Requires persona corpus (generated with --personas).
    Balanced sampling across personas.
    """
    from signs_of_life.ao_lib import layer_percent_to_layer

    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if "persona" in entry:
                    corpus.append(entry)

    if not corpus:
        raise ValueError(
            f"No persona entries in {corpus_path}. "
            f"Generate with: python generate_cots.py --openrouter --personas"
        )

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    # Group by persona for balanced sampling
    by_persona = {}
    for entry in corpus:
        persona = entry["persona"]
        if persona not in by_persona:
            by_persona[persona] = []
        by_persona[persona].append(entry)

    personas = sorted(by_persona.keys())
    print(f"  Personas: {personas}")
    for p in personas:
        print(f"    {p}: {len(by_persona[p])} entries")

    datapoints = []

    while len(datapoints) < num_examples:
        # Cycle through personas for balance
        persona = personas[len(datapoints) % len(personas)]
        entry = random.choice(by_persona[persona])

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
            f"What persona was the model given? "
            f"Answer with one: {', '.join(personas)}."
        )

        datapoints.append({
            "datapoint_type": "cot_persona",
            "prompt": prompt,
            "target_response": persona,
            "layers": layers,  # Multi-layer: [L25%, L50%, L75%]
            "num_positions": N,
            "context_input_ids": context_slice,
            "context_positions": list(positions),
        })

    return datapoints[:num_examples]
