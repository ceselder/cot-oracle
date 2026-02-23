"""
FineWeb Context Prediction — PastLens-style training from web text.

Streams from HuggingFace FineWeb + LMSYS Chat-1M (50/50 mix) and generates
context prediction examples: given K activation positions, predict the
previous/next K tokens.

This matches Adam's original AO training pipeline (PastLensDatasetLoader)
but returns list[dict] compatible with our dicts_to_training_data() converter.

Uses HF streaming so no full dataset download is needed.
"""

import random
from fractions import Fraction
from typing import Generator

from datasets import load_dataset
from transformers import AutoTokenizer


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    pretrain_frac: float = 0.5,
    split: str = "train",
    streaming: bool = True,
) -> Generator:
    """Stream a mix of FineWeb pretrain data and LMSYS chat data.

    Yields raw text strings, alternating between pretrain and chat
    at the specified ratio (default 50/50).
    """
    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain

    bos_token = tokenizer.bos_token or tokenizer.eos_token

    def gen():
        while True:
            for _ in range(n_pretrain):
                sample = bos_token + next(pretrain_ds)["text"]
                yield sample
            for _ in range(n_chat):
                sample = tokenizer.apply_chat_template(
                    next(chat_ds)["conversation"], tokenize=False,
                )
                yield sample

    return gen()


def load_fineweb_context_prediction_data(
    tokenizer: AutoTokenizer,
    model_name: str,
    layer_percents: list[int],
    num_examples: int = 100000,
    min_k_tokens: int = 1,
    max_k_tokens: int = 20,
    min_k_activations: int = 1,
    max_k_activations: int = 20,
    max_length: int = 512,
    seed: int = 42,
) -> list[dict]:
    """Generate PastLens-style context prediction data from FineWeb streaming.

    Each example: given activations at K random positions, predict the
    previous/next K tokens. Matches Adam's original AO training format.

    Returns list of dicts compatible with dicts_to_training_data().
    """
    from cot_utils import layer_percent_to_layer

    random.seed(seed)

    layers = [layer_percent_to_layer(model_name, lp) for lp in layer_percents]

    print("  Initializing FineWeb + LMSYS streaming...")
    dataset = hf_mixed_dataset_to_generator(tokenizer)

    datapoints = []
    skipped = 0

    while len(datapoints) < num_examples:
        text = next(dataset)

        # Tokenize with truncation
        input_ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=max_length,
        )["input_ids"]
        L = len(input_ids)

        # Random parameters
        k_tokens = random.randint(min_k_tokens, max_k_tokens)
        k_acts = random.randint(min_k_activations, max_k_activations)
        direction = random.choice(["past", "future"])

        if direction == "past":
            # Need k_tokens before act span, k_acts positions, and at least 1 after
            if L < k_tokens + k_acts + 1:
                skipped += 1
                continue
            act_begin_min = k_tokens
            act_begin_max = L - k_acts - 1
            if act_begin_max < act_begin_min:
                skipped += 1
                continue
            act_begin = random.randint(act_begin_min, act_begin_max)
            act_positions = list(range(act_begin, act_begin + k_acts))
            token_positions = list(range(act_begin - k_tokens, act_begin))
            context_cutoff = act_positions[-1]
            target_tokens = [input_ids[p] for p in token_positions]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            prompt = f"Can you predict the previous {k_tokens} tokens that came before this?"
        else:  # future
            if L < k_tokens + k_acts + 1:
                skipped += 1
                continue
            act_begin_min = 1
            act_begin_max = L - k_acts - k_tokens
            if act_begin_max < act_begin_min:
                skipped += 1
                continue
            act_begin = random.randint(act_begin_min, act_begin_max)
            act_positions = list(range(act_begin, act_begin + k_acts))
            last_act = act_positions[-1]
            token_positions = list(range(last_act + 1, last_act + 1 + k_tokens))
            context_cutoff = last_act
            target_tokens = [input_ids[p] for p in token_positions]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            prompt = f"Can you predict the next {k_tokens} tokens that come after this?"

        layer = random.choice(layers)
        context_input_ids = input_ids[:context_cutoff + 1]

        datapoints.append({
            "datapoint_type": "fineweb_context_prediction",
            "prompt": prompt,
            "target_response": target_text,
            "layer": layer,
            "num_positions": k_acts,
            "context_input_ids": context_input_ids,
            "context_positions": act_positions,
        })

        if len(datapoints) % 10000 == 0:
            print(f"  FineWeb: {len(datapoints)}/{num_examples} examples generated...")

    if skipped > 0:
        print(f"  FineWeb: skipped {skipped} short texts")

    return datapoints[:num_examples]
