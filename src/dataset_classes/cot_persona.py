"""
Task 6: Persona Detection (~15K examples)

What persona/system prompt was the model given? Tests reading unverbalized context.
Uses sentence-structured format with 3 acts per sentence boundary.

Requires corpus generated with --personas flag (each entry has a "persona" field).
Ground truth: the system prompt used (known by construction).
"""

import json
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_cot_persona_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 15000,
    max_sentences: int = 15,
    seed: int = 42,
    corpus_entries: list[dict] | None = None,
) -> list[dict]:
    """
    Generate persona detection training data.

    Each example: sentence-boundary activations -> persona label.
    Requires persona corpus (generated with --personas).
    Balanced sampling across personas.
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    if corpus_entries is not None:
        corpus = [e for e in corpus_entries if "persona" in e]
    else:
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if "persona" in entry:
                        corpus.append(entry)

    assert corpus, (
        f"No persona entries in {corpus_path}. "
        f"Generate with: python generate_cots.py --openrouter --personas"
    )

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    by_persona: dict[str, list[tuple]] = {}
    for entry in tqdm(corpus, desc="  persona: tokenizing corpus", leave=False):
        persona = entry["persona"]
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

        if persona not in by_persona:
            by_persona[persona] = []
        by_persona[persona].append((context_ids, positions))

    personas = sorted(by_persona.keys())
    print(f"  Personas: {personas}")
    for p in personas:
        print(f"    {p}: {len(by_persona[p])} entries")

    datapoints = []
    pbar = tqdm(total=num_examples, desc="  persona: sampling", leave=False)

    while len(datapoints) < num_examples:
        persona = personas[len(datapoints) % len(personas)]
        context_ids, positions = random.choice(by_persona[persona])

        N = len(positions)

        prompt = (
            f"Activations from {N} sentence boundaries. "
            f"What persona was the model given? "
            f"Answer with one: {', '.join(personas)}."
        )

        datapoints.append({
            "datapoint_type": "cot_persona",
            "prompt": prompt,
            "target_response": persona,
            "layers": layers,
            "num_positions": N,
            "context_input_ids": list(context_ids),
            "context_positions": list(positions),
        })
        pbar.update(1)

    pbar.close()
    return datapoints[:num_examples]
