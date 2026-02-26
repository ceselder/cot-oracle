"""
CoT Reasoning Termination Prediction — Will the model stop thinking soon?

Binary classification: given CoT activations up to a truncation point,
predict whether the model will emit </think> within the next 100 tokens.

Positive ("will_terminate"): prefix cut 25-55 tokens from actual </think>
Negative ("will_continue"):  prefix cut 300+ tokens from actual </think>
Ambiguous range (56-299 tokens) is skipped.

Position-count balanced: generates overcomplete pools for both classes,
then pairs by matching stride position count so the oracle can't use
the number of ¶ tokens as a shortcut.

Uses stride=5, 3 layers (25%, 50%, 75%), ¶ token.
"""

import json
import random
from collections import defaultdict

from transformers import AutoTokenizer


def load_cot_reasoning_termination_data(
    corpus_path: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    num_examples: int = 15000,
    stride: int | str = None,
    n_prompt_positions: int = 5,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """
    Generate reasoning termination prediction training data.

    Position-count balanced approach:
      1. Pre-tokenize corpus entries
      2. Generate overcomplete candidate pools for both classes
      3. Compute stride position count for each candidate
      4. Bin candidates by position count and pair across classes
      5. This ensures both classes have identical position count distributions,
         preventing the oracle from using ¶ token count as a shortcut.

    Balanced 50/50 between will_terminate and will_continue.
    """
    from cot_utils import get_cot_positions, get_injection_layers

    random.seed(seed)

    LAYERS = get_injection_layers(model_name)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cot = entry.get("cot_response", "")
                if cot.strip():
                    corpus.append(entry)

    if not corpus:
        raise ValueError(f"No entries with cot_response in corpus at {corpus_path}")

    print(f"  Corpus: {len(corpus)} entries with CoT")

    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    # Pre-tokenize all entries to find which can produce positive/negative examples
    tokenized = []
    for entry in corpus:
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        # cot_response is the thinking text (</think> tag already stripped)
        cot_text = entry["cot_response"]

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        cot_len = len(full_ids) - prompt_len

        if cot_len < 60:  # need at least 60 CoT tokens for positive examples
            continue

        tokenized.append({
            "entry": entry,
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "cot_len": cot_len,
        })

    # Split into pools based on what examples they can produce
    # Positive: needs cot_len >= 25 (truncate 25-55 from end)
    # Negative: needs cot_len >= 300 (truncate 300+ from end)
    positive_pool = [t for t in tokenized if t["cot_len"] >= 25]
    negative_pool = [t for t in tokenized if t["cot_len"] > 300]

    if not positive_pool:
        raise ValueError("No entries long enough for positive examples (need >=25 CoT tokens)")
    if not negative_pool:
        raise ValueError("No entries long enough for negative examples (need >300 CoT tokens)")

    print(f"  Positive pool (>=25 CoT tokens): {len(positive_pool)}")
    print(f"  Negative pool (>300 CoT tokens): {len(negative_pool)}")

    # --- Generate overcomplete candidate pools ---
    # We need enough candidates so that position-count bins have both classes.
    # Generate ~5x what we need for each class.
    pool_size = num_examples * 5

    def _make_candidate(t, remaining, label_prefix):
        """Create a candidate datapoint and return (n_stride_positions, datapoint)."""
        trunc_pos = t["prompt_len"] + t["cot_len"] - remaining

        # Sanity: truncation must be after prompt
        if trunc_pos <= t["prompt_len"] + 5:
            return None

        positions = get_cot_positions(
            t["prompt_len"], trunc_pos,
            stride=stride, tokenizer=tokenizer, input_ids=t["full_ids"],
        )
        if len(positions) < 2:
            return None

        prompt_positions = _get_prompt_positions(t["prompt_len"], n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)

        max_pos = max(positions)
        context_slice = t["full_ids"][:max_pos + 1]

        if label_prefix == "will_terminate":
            target = f"will_terminate, in {remaining} tokens"
        else:
            target = f"will_continue, {remaining} tokens remain"

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Will the model terminate reasoning (emit </think>) soon? "
            f"If yes, estimate how many tokens remain."
        )

        datapoint = {
            "datapoint_type": "cot_reasoning_termination",
            "prompt": prompt,
            "target_response": target,
            "layer": LAYERS[0],
            "layers": LAYERS,
            "num_positions": num_positions,
            "context_input_ids": context_slice,
            "context_positions": context_positions,
        }
        # n_stride_positions is the count per layer (before multiplying by layers)
        n_stride = len(positions)
        return (n_stride, datapoint)

    # Generate positive candidates (will_terminate)
    pos_candidates = []  # list of (n_stride, datapoint)
    for _ in range(pool_size):
        t = random.choice(positive_pool)
        remaining = random.randint(25, min(55, t["cot_len"] - 1))
        result = _make_candidate(t, remaining, "will_terminate")
        if result is not None:
            pos_candidates.append(result)

    # Generate negative candidates (will_continue)
    neg_candidates = []  # list of (n_stride, datapoint)
    for _ in range(pool_size):
        t = random.choice(negative_pool)
        remaining = random.randint(300, t["cot_len"] - 1)
        result = _make_candidate(t, remaining, "will_continue")
        if result is not None:
            neg_candidates.append(result)

    print(f"  Generated {len(pos_candidates)} positive candidates, "
          f"{len(neg_candidates)} negative candidates")

    # --- Bin by stride position count and pair ---
    # Use bins of width 5 to allow near-matches
    BIN_WIDTH = 5

    def _bin_key(n_stride):
        return n_stride // BIN_WIDTH

    pos_bins = defaultdict(list)
    for n_stride, dp in pos_candidates:
        pos_bins[_bin_key(n_stride)].append(dp)

    neg_bins = defaultdict(list)
    for n_stride, dp in neg_candidates:
        neg_bins[_bin_key(n_stride)].append(dp)

    # Shuffle within each bin so we get diversity
    for bin_list in pos_bins.values():
        random.shuffle(bin_list)
    for bin_list in neg_bins.values():
        random.shuffle(bin_list)

    # Pair: for each bin present in both classes, take min(pos, neg) from each
    datapoints = []
    target_per_class = num_examples // 2
    shared_bins = sorted(set(pos_bins.keys()) & set(neg_bins.keys()))

    if not shared_bins:
        raise ValueError(
            "No overlapping position-count bins between classes. "
            f"Positive stride range: {min(n for n, _ in pos_candidates)}-{max(n for n, _ in pos_candidates)}, "
            f"Negative stride range: {min(n for n, _ in neg_candidates)}-{max(n for n, _ in neg_candidates)}"
        )

    # First pass: count how many pairs are available per bin
    available_pairs = {b: min(len(pos_bins[b]), len(neg_bins[b])) for b in shared_bins}
    total_available = sum(available_pairs.values())

    print(f"  Shared position-count bins: {len(shared_bins)} "
          f"(total matchable pairs: {total_available})")

    if total_available < target_per_class:
        print(f"  WARNING: Only {total_available} matchable pairs, "
              f"need {target_per_class}. Using all available.")

    # Sample proportionally from each bin up to target_per_class
    pos_selected = []
    neg_selected = []
    remaining_budget = min(target_per_class, total_available)

    # Distribute budget across bins proportionally to available pairs
    for b in shared_bins:
        if remaining_budget <= 0:
            break
        n_pairs = min(
            available_pairs[b],
            max(1, int(remaining_budget * available_pairs[b] / max(1, total_available))),
        )
        # Don't exceed remaining budget
        n_pairs = min(n_pairs, remaining_budget)
        pos_selected.extend(pos_bins[b][:n_pairs])
        neg_selected.extend(neg_bins[b][:n_pairs])
        remaining_budget -= n_pairs

    # If we still have budget (rounding), fill greedily from largest bins
    if remaining_budget > 0:
        used_pos = {id(dp) for dp in pos_selected}
        used_neg = {id(dp) for dp in neg_selected}
        for b in sorted(shared_bins, key=lambda b: available_pairs[b], reverse=True):
            if remaining_budget <= 0:
                break
            extra_pos = [dp for dp in pos_bins[b] if id(dp) not in used_pos]
            extra_neg = [dp for dp in neg_bins[b] if id(dp) not in used_neg]
            n_extra = min(len(extra_pos), len(extra_neg), remaining_budget)
            pos_selected.extend(extra_pos[:n_extra])
            neg_selected.extend(extra_neg[:n_extra])
            remaining_budget -= n_extra

    datapoints = pos_selected + neg_selected
    random.shuffle(datapoints)

    n_pos = sum(1 for d in datapoints if d["target_response"].startswith("will_terminate"))
    n_neg = sum(1 for d in datapoints if d["target_response"].startswith("will_continue"))

    # Report position count stats per class
    pos_npos = [d["num_positions"] for d in datapoints if d["target_response"].startswith("will_terminate")]
    neg_npos = [d["num_positions"] for d in datapoints if d["target_response"].startswith("will_continue")]
    if pos_npos and neg_npos:
        print(f"  Position count stats (total across layers):")
        print(f"    will_terminate: mean={sum(pos_npos)/len(pos_npos):.0f}, "
              f"range=[{min(pos_npos)}, {max(pos_npos)}]")
        print(f"    will_continue:  mean={sum(neg_npos)/len(neg_npos):.0f}, "
              f"range=[{min(neg_npos)}, {max(neg_npos)}]")

    print(f"  Generated {len(datapoints)} reasoning termination examples "
          f"({n_pos} pos, {n_neg} neg)")
    return datapoints[:num_examples]
