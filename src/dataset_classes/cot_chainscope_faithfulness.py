"""
Chainscope Faithfulness Detection — Is this reasoning faithful or unfaithful?

Binary classification from chainscope IPHR labels. Each chainscope entry has
10 rollouts with correctness labels; pair_unfaithful=True means the model shows
inconsistent reasoning across orderings (a→b vs b→a).

Unfaithful pool: pair_unfaithful=True, responses where correct=False
  (model got the wrong answer — reasoning process was biased)
Faithful pool: pair_unfaithful=False, responses where correct=True
  (model got the right answer on a faithful pair)

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random
import re

from tqdm.auto import tqdm
from transformers import AutoTokenizer


def load_chainscope_faithfulness_data(
    chainscope_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int = 5,
    max_positions_per_layer: int = 20,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate chainscope faithfulness detection training data.

    Builds two pools from chainscope rollouts:
    - unfaithful: responses from pair_unfaithful=True items where correct=False
    - faithful: responses from pair_unfaithful=False items where correct=True

    Returns balanced 50/50 split up to num_examples.
    """
    from cot_utils import get_cot_stride_positions, layer_percent_to_layer

    random.seed(seed)

    LAYERS = [
        layer_percent_to_layer(model_name, 25),
        layer_percent_to_layer(model_name, 50),
        layer_percent_to_layer(model_name, 75),
    ]

    with open(chainscope_path) as f:
        entries = json.load(f)
    print(f"  Loaded {len(entries)} chainscope entries")

    # Build response pools: (question_prompt, cot_text) tuples
    unfaithful_pool = []
    faithful_pool = []

    for entry in entries:
        q_str = entry["q_str"]
        prompt_text = entry["prompt"]
        labeled = entry["labeled_responses"]

        if entry["pair_unfaithful"]:
            for lr in labeled:
                if not lr["correct"]:
                    unfaithful_pool.append((q_str, lr["response"]))
        else:
            for lr in labeled:
                if lr["correct"]:
                    faithful_pool.append((q_str, lr["response"]))

    print(f"  Pools: {len(unfaithful_pool)} unfaithful, {len(faithful_pool)} faithful")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    def _tokenize_response(question: str, response: str):
        """Tokenize a chainscope response with chat template."""
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        # Strip think tags from response
        cot_text = response
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        # Also strip leading <think> tag
        cot_text = re.sub(r'^<think>\s*', '', cot_text)

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        return full_ids, prompt_len

    # Pre-tokenize pools
    print("  Tokenizing unfaithful pool...")
    unfaithful_tokenized = []
    for q, r in tqdm(unfaithful_pool, desc="  unfaithful", leave=False):
        full_ids, prompt_len = _tokenize_response(q, r)
        if len(full_ids) - prompt_len < 10:
            continue
        unfaithful_tokenized.append((full_ids, prompt_len))

    print("  Tokenizing faithful pool...")
    faithful_tokenized = []
    for q, r in tqdm(faithful_pool, desc="  faithful", leave=False):
        full_ids, prompt_len = _tokenize_response(q, r)
        if len(full_ids) - prompt_len < 10:
            continue
        faithful_tokenized.append((full_ids, prompt_len))

    print(f"  Tokenized: {len(unfaithful_tokenized)} unfaithful, {len(faithful_tokenized)} faithful")

    layers_str = ", ".join(str(l) for l in LAYERS)
    datapoints = []
    pbar = tqdm(total=num_examples, desc="  chainscope: sampling", leave=False)

    while len(datapoints) < num_examples:
        # Alternate to balance
        if len(datapoints) % 2 == 0:
            full_ids, prompt_len = random.choice(unfaithful_tokenized)
            target = "unfaithful"
        else:
            full_ids, prompt_len = random.choice(faithful_tokenized)
            target = "faithful"

        positions = get_cot_stride_positions(
            prompt_len, len(full_ids), stride=stride, max_positions=max_positions_per_layer,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(prompt_len, n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * 3  # one set per layer
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = full_ids[:max_pos + 1]

        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Is this reasoning faithful or unfaithful? "
            f"Answer: faithful or unfaithful."
        )

        datapoints.append({
            "datapoint_type": "chainscope_faithfulness",
            "prompt": prompt,
            "target_response": target,
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": list(context_slice),
            "context_positions": list(context_positions),
        })
        pbar.update(1)

    pbar.close()
    print(f"  Generated {len(datapoints)} chainscope faithfulness examples")
    return datapoints[:num_examples]
