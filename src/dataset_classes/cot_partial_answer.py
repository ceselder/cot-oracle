"""
CoT Partial Answer Prediction — What answer would the model give if it stopped thinking now?

Given activations from a CoT prefix (truncated at a random point), predict
what answer the model would give. The prompt injects "therefore the answer is"
framing so the oracle learns to read intermediate reasoning state.

Truncation percentages: 10%-90% of the CoT token length.
Target: the model's final answer (from full CoT).

This teaches the oracle:
  - Whether the answer is already "decided" early (early commitment signal)
  - How reasoning state evolves toward the final answer
  - When the model has enough info to commit vs. is still uncertain

Uses stride=5, 3 layers (25%, 50%, 75%), paragraph token.
"""

import json
import random
import re

from transformers import AutoTokenizer


def _extract_answer_text(entry: dict) -> str | None:
    """Extract a clean answer string from a corpus entry."""
    direct = entry.get("direct_response", "").strip()
    if direct:
        direct = re.sub(r"<think>.*?</think>", "", direct, flags=re.DOTALL).strip()
        boxed = re.search(r"\\boxed\{([^}]+)\}", direct)
        if boxed:
            return boxed.group(1).strip()
        if len(direct) < 200:
            return direct
        return direct[:200]

    answer = entry.get("answer") or entry.get("correct_answer") or ""
    answer = str(answer).strip()
    if answer:
        return str(answer)[:200]

    return None


def load_cot_partial_answer_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 20000,
    stride: int = 5,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate partial-CoT answer prediction training data.

    For each example:
      1. Tokenize question + full CoT
      2. Pick a random truncation point (10-90% of CoT length)
      3. Get stride positions up to truncation point
      4. Target = the model's final answer (from full CoT completion)

    The oracle learns: "given activations at X% through reasoning,
    what answer is the model converging toward?"
    """
    from cot_utils import get_cot_stride_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cot = entry.get("cot_response", "").strip()
                answer = _extract_answer_text(entry)
                if cot and answer:
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No valid entries with CoT + answer in {corpus_path}")

    print(f"  Loaded {len(corpus)} corpus entries with CoT + answer")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Pre-tokenize corpus
    tokenized = []
    for entry in corpus:
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
        cot_len = len(full_ids) - prompt_len

        if cot_len < 20:  # need enough CoT to truncate meaningfully
            continue

        answer = _extract_answer_text(entry)
        tokenized.append({
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "cot_len": cot_len,
            "answer": answer,
        })

    print(f"  {len(tokenized)} entries after tokenization (min 20 CoT tokens)")

    # Truncation percentage bins for balanced sampling
    # 5 bins: 10-25%, 25-40%, 40-60%, 60-75%, 75-90%
    BINS = [(10, 25), (25, 40), (40, 60), (60, 75), (75, 90)]

    datapoints = []
    bin_idx = 0

    for _ in range(num_examples * 3):  # over-sample then truncate
        if len(datapoints) >= num_examples:
            break

        t = random.choice(tokenized)

        # Pick truncation percentage from current bin (round-robin for balance)
        lo, hi = BINS[bin_idx % len(BINS)]
        bin_idx += 1
        pct = random.randint(lo, hi)
        trunc_tokens = int(t["cot_len"] * pct / 100)

        if trunc_tokens < 10:
            continue

        trunc_pos = t["prompt_len"] + trunc_tokens

        # Get stride positions up to truncation point
        positions = get_cot_stride_positions(
            t["prompt_len"], trunc_pos,
            stride=stride,
        )
        if len(positions) < 2:
            continue

        prompt_positions = _get_prompt_positions(t["prompt_len"], n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)  # triple for 3 layers
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = t["full_ids"][:max_pos + 1]

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}, "
            f"representing {pct}% of the model's chain of thought. "
            f"Based on the reasoning so far, what answer is the model converging toward?"
        )

        datapoints.append({
            "datapoint_type": "cot_partial_answer",
            "prompt": prompt,
            "target_response": t["answer"],
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        })

    # Print distribution
    pct_bins = {f"{lo}-{hi}%": 0 for lo, hi in BINS}
    for dp in datapoints:
        for lo, hi in BINS:
            if f"{lo}%" in dp["prompt"] or f"representing {lo}" in dp["prompt"]:
                # rough check
                pass
    print(f"  Generated {len(datapoints)} partial answer prediction examples")
    return datapoints[:num_examples]
