"""
Task 2: Context Prediction — Sentence Boundaries (~30K examples)

Feed the ENTIRE CoT trajectory at sentence boundaries, reconstruct chunks.
One random layer per example from [25%, 50%, 75%] depth.

Question: "Activations from {N} sentence boundaries.
           Predict the next {K} tokens following sentence {T}."
"""

import json
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_cot_sentence_prediction_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 30000,
    max_sentences: int = 15,
    min_k_tokens: int = 1,
    max_k_tokens: int = 15,
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

    assert corpus, f"Empty corpus at {corpus_path}"

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    cached = []
    for entry in tqdm(corpus, desc="  sentence_prediction: tokenizing corpus", leave=False):
        boundary_positions = entry.get("boundary_positions", [])
        sentences = entry.get("sentences", [])
        if len(boundary_positions) < 2 or len(sentences) < 2:
            continue

        if "_ctx_ids" in entry:
            full_ids = entry["_ctx_ids"]
        else:
            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            cot_text = entry["cot_response"]
            think_end = cot_text.find("</think>")
            if think_end != -1:
                cot_text = cot_text[:think_end]
            full_ids = tokenizer(formatted + cot_text, add_special_tokens=False)["input_ids"]

        valid_bp = [p for p in boundary_positions if p < len(full_ids)]
        if len(valid_bp) < 2:
            continue

        cached.append((full_ids, valid_bp, sentences))

    datapoints = []
    attempts = 0
    max_attempts = num_examples * 10
    pbar = tqdm(total=num_examples, desc="  sentence_prediction: sampling", leave=False)

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1
        full_ids, boundary_positions, sentences = random.choice(cached)

        n_sent = min(len(boundary_positions), len(sentences), max_sentences)
        bp = boundary_positions[:n_sent]
        N = len(bp)

        target_sent = random.randint(0, N - 1)
        direction = random.choice(["future", "past"])
        k_tokens = random.randint(min_k_tokens, max_k_tokens)
        boundary_pos = bp[target_sent]

        if direction == "future":
            start = boundary_pos + 1
            end = start + k_tokens
            if end > len(full_ids):
                continue
            target_ids = full_ids[start:end]
            question = f"Activations from {N} sentence boundaries. Predict the next {k_tokens} tokens following sentence {target_sent + 1}."
        else:
            end = boundary_pos
            start = end - k_tokens
            if start < 0:
                continue
            target_ids = full_ids[start:end]
            question = f"Activations from {N} sentence boundaries. Predict the {k_tokens} tokens before sentence {target_sent + 1}."

        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue

        context_cutoff = max(bp)
        context_input_ids_slice = full_ids[:context_cutoff + 1]

        datapoints.append({
            "datapoint_type": "cot_sentence_prediction",
            "prompt": question,
            "target_response": target_text,
            "layers": layers,
            "num_positions": len(bp),
            "context_input_ids": context_input_ids_slice,
            "context_positions": list(bp),
        })
        pbar.update(1)

    pbar.close()
    return datapoints[:num_examples]
