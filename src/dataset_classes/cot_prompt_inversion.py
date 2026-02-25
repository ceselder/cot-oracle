"""
Prompt Inversion: Reconstruct the original user prompt from CoT activations.

Given activation trajectories from the CoT, predict what question/prompt
generated this reasoning. This is the inverse of full_recon (which
reconstructs the CoT from activations).

Each example:
  - context_positions: CoT stride positions * 3 (same positions, 3 layers)
  - num_positions: 3 * K
  - Target: the original user question/prompt
"""

import json
import random
import re

from transformers import AutoTokenizer


def load_cot_prompt_inversion_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 20000,
    stride: int | str = None,
    max_target_tokens: int = 8192,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate prompt inversion training data.

    For each corpus entry:
      1. Tokenize question + CoT via apply_chat_template(enable_thinking=True)
      2. Get K strided positions over the CoT region
      3. Triple positions for 3 layers: context_positions = positions * 3
      4. Target = the original user question (truncated to max_target_tokens)

    Returns list of dicts compatible with dicts_to_training_data().
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("cot_response", "").strip() and entry.get("question", "").strip():
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No valid entries in {corpus_path}")

    print(f"  Loaded {len(corpus)} corpus entries")
    print(f"  Layers: {LAYERS}")

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 3

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        entry = random.choice(corpus)

        question = entry["question"].strip()

        # Tokenize question + CoT
        messages = [{"role": "user", "content": question}]
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

        # Get prompt length
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        # Get strided positions over CoT region
        positions = get_cot_positions(
            prompt_len, len(full_ids),
            stride=stride, tokenizer=tokenizer, input_ids=full_ids,
        )
        if len(positions) < 2:
            continue

        K = len(positions)

        # Triple positions for 3 layers
        context_positions = positions * len(LAYERS)
        num_positions = len(context_positions)

        # Context slice: include up to the last position
        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        # Target: the original question, truncated if needed
        target_ids = tokenizer.encode(question, add_special_tokens=False)
        if len(target_ids) > max_target_tokens:
            target_text = tokenizer.decode(
                target_ids[:max_target_tokens], skip_special_tokens=True
            )
        else:
            target_text = question

        if not target_text.strip():
            continue

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"What question or prompt generated this reasoning?"
        )

        datapoints.append({
            "datapoint_type": "cot_prompt_inversion",
            "prompt": prompt,
            "target_response": target_text,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    print(f"  Generated {len(datapoints)} prompt inversion examples")
    return datapoints[:num_examples]
