"""
CoT Correctness Prediction — Is the model's final answer correct?

Binary classification: given CoT activations, predict whether the
model's final answer is correct or incorrect.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset_classes.cot_decorative import _pretokenize_boundary_entries


def load_cot_correctness_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int = 5,
    max_positions_per_layer: int = 20,
    n_prompt_positions: int = 5,
    max_sentences: int = 15,
    seed: int = 42,
    corpus_entries: list[dict] | None = None,
    **_kwargs,
) -> list[dict]:
    """
    Generate correctness prediction training data with multi-layer stride.

    Each example: stride activations from CoT -> correct / incorrect.
    Ground truth from corpus (cot_correct field). Balanced 50/50.
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

    assert corpus, f"Empty corpus at {corpus_path}"

    correct_pool = [e for e in corpus if e.get("cot_correct")]
    incorrect_pool = [e for e in corpus if not e.get("cot_correct")]

    if not correct_pool:
        raise ValueError("No correct entries in corpus")
    if not incorrect_pool:
        raise ValueError("No incorrect entries in corpus")

    print(f"  correct: {len(correct_pool)}, incorrect: {len(incorrect_pool)}")

    cached_correct = _pretokenize_boundary_entries(correct_pool, tokenizer, max_sentences, "correctness/correct")
    cached_incorrect = _pretokenize_boundary_entries(incorrect_pool, tokenizer, max_sentences, "correctness/incorrect")

    layers_str = ", ".join(str(l) for l in LAYERS)

    datapoints = []
    pbar = tqdm(total=num_examples, desc="  correctness: sampling", leave=False)

    while len(datapoints) < num_examples:
        if len(datapoints) % 2 == 0:
            context_ids, positions = random.choice(cached_correct)
            target = "correct"
        else:
            context_ids, positions = random.choice(cached_incorrect)
            target = "incorrect"

        N = len(positions)

        prompt = (
            f"Activations from {N} positions across layers {layers_str}. "
            f"Is the model's final answer correct? Answer: correct or incorrect."
        )

        datapoints.append({
            "datapoint_type": "cot_correctness",
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
