"""
Task 5: Correctness Prediction (~15K examples)

Binary classification: is the model's final answer correct?
Uses sentence-structured format with 3 acts per sentence boundary.

Ground truth: compare extracted answer to ground truth from corpus.
Labels: "correct" or "incorrect"
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
    layer_percents: list[int],
    num_examples: int = 15000,
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

    assert corpus, f"Empty corpus at {corpus_path}"

    correct_pool = [e for e in corpus if e.get("cot_correct")]
    incorrect_pool = [e for e in corpus if not e.get("cot_correct")]

    if not correct_pool:
        raise ValueError("No correct entries in corpus")
    if not incorrect_pool:
        raise ValueError("No incorrect entries in corpus")

    print(f"  correct: {len(correct_pool)}, incorrect: {len(incorrect_pool)}")

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    cached_correct = _pretokenize_boundary_entries(correct_pool, tokenizer, max_sentences, "correctness/correct")
    cached_incorrect = _pretokenize_boundary_entries(incorrect_pool, tokenizer, max_sentences, "correctness/incorrect")

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
            f"Activations from {N} sentence boundaries. "
            f"Is the model's final answer correct? Answer: correct or incorrect."
        )

        datapoints.append({
            "datapoint_type": "cot_correctness",
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
