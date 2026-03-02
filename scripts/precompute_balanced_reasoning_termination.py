#!/usr/bin/env python3
"""
Precompute position-count-balanced reasoning termination training data.

Generates will_terminate / will_continue candidates from the math CoT corpus,
then pairs them by stride position count so the oracle can't use the number
of ¶ tokens as a label shortcut.

Output: JSONL with fields matching what train.py expects
  (prompt, target_response, context_input_ids, context_positions, layers, etc.)

Uploads to HuggingFace as a dataset.

Usage (CPU only, no GPU needed):
    python scripts/precompute_balanced_reasoning_termination.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --num-examples 15000 \
        --output data/reasoning_termination_balanced.jsonl \
        --push-to-hub
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Precompute position-balanced reasoning termination training data"
    )
    parser.add_argument("--corpus", default="data/cot_corpus_v5/corpus_medium.jsonl",
                        help="Path to CoT corpus JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model name (for tokenizer and layer config)")
    parser.add_argument("--num-examples", type=int, default=15000,
                        help="Target number of training examples (balanced 50/50)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride for position sampling")
    parser.add_argument("--n-prompt-positions", type=int, default=5,
                        help="Number of evenly-spaced prompt positions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/reasoning_termination_balanced.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Upload to HuggingFace Hub")
    parser.add_argument("--hf-repo", default="ceselder/cot-oracle-reasoning-termination-balanced",
                        help="HuggingFace repo name")
    parser.add_argument("--bin-width", type=int, default=5,
                        help="Width of position-count bins for matching")
    args = parser.parse_args()

    random.seed(args.seed)

    from transformers import AutoTokenizer
    from cot_utils import get_cot_positions, get_injection_layers

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    LAYERS = get_injection_layers(args.model)
    print(f"  Injection layers: {LAYERS}")

    # --- Stream-load + tokenize (memory-optimised) ---
    # Both pools draw from entries long enough to support BOTH cuts.
    # terminate needs 25-55 tokens from end, continue needs >= 300 remaining,
    # so we need cot_len >= 355 to support both. Use 360 for margin.
    MIN_COT_LEN = 360
    # We only need ~5K pool entries to generate 15K examples with diversity.
    # Reservoir-sample to cap RAM usage.
    MAX_POOL = 5000

    import gc

    print(f"\nStream-tokenizing from {args.corpus} (cot_len >= {MIN_COT_LEN}, max_pool={MAX_POOL})...")
    shared_pool = []
    n_total = 0
    n_short = 0
    n_seen_valid = 0  # for reservoir sampling

    with open(args.corpus) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            cot = entry.get("cot_response", "")
            if not cot.strip():
                continue
            n_total += 1

            if n_total % 5000 == 0:
                print(f"  {n_total} scanned, {len(shared_pool)} in pool...")

            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            cot_text = entry["cot_response"]
            full_text = formatted + cot_text
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
            cot_len = len(full_ids) - prompt_len

            if cot_len < MIN_COT_LEN:
                n_short += 1
                continue

            item = {
                "full_ids": full_ids,
                "prompt_len": prompt_len,
                "cot_len": cot_len,
            }

            # Reservoir sampling: keep first MAX_POOL, then replace randomly
            n_seen_valid += 1
            if len(shared_pool) < MAX_POOL:
                shared_pool.append(item)
            else:
                j = random.randint(0, n_seen_valid - 1)
                if j < MAX_POOL:
                    shared_pool[j] = item

    gc.collect()

    positive_pool = shared_pool
    negative_pool = shared_pool

    print(f"  Scanned {n_total} entries, {n_short} too short, {n_seen_valid} valid")
    print(f"  Shared pool: {len(shared_pool)} entries (capped at {MAX_POOL})")

    if not positive_pool:
        raise ValueError("No entries long enough for positive examples")
    if not negative_pool:
        raise ValueError("No entries long enough for negative examples")

    # --- Report CoT length distributions ---
    pos_lens = [t["cot_len"] for t in positive_pool]
    neg_lens = [t["cot_len"] for t in negative_pool]
    print(f"\n  CoT length stats:")
    print(f"    Positive pool: mean={sum(pos_lens)/len(pos_lens):.0f}, "
          f"range=[{min(pos_lens)}, {max(pos_lens)}]")
    print(f"    Negative pool: mean={sum(neg_lens)/len(neg_lens):.0f}, "
          f"range=[{min(neg_lens)}, {max(neg_lens)}]")

    # --- Helper ---
    def _get_prompt_positions(formatted_len: int, n: int = 5) -> list[int]:
        if formatted_len < n:
            return list(range(formatted_len))
        step = formatted_len / (n + 1)
        return [int(step * (i + 1)) for i in range(n)]

    def _make_candidate(t, remaining, label):
        trunc_pos = t["prompt_len"] + t["cot_len"] - remaining
        if trunc_pos <= t["prompt_len"] + 5:
            return None

        positions = get_cot_positions(
            t["prompt_len"], trunc_pos,
            stride=args.stride, tokenizer=tokenizer, input_ids=t["full_ids"],
        )
        if len(positions) < 2:
            return None

        prompt_positions = _get_prompt_positions(t["prompt_len"], args.n_prompt_positions)
        combined = prompt_positions + positions
        context_positions = combined * len(LAYERS)
        num_positions = len(context_positions)
        n_stride = len(positions)

        max_pos = max(positions)
        context_slice = t["full_ids"][:max_pos + 1]

        if label == "will_terminate":
            target = f"will_terminate, in {remaining} tokens"
        else:
            target = f"will_continue, {remaining} tokens remain"

        layers_str = ", ".join(str(l) for l in LAYERS)
        prompt = (
            f"Activations from {num_positions} positions across layers {layers_str}. "
            f"Will the model terminate reasoning (emit </think>) soon? "
            f"If yes, estimate how many tokens remain."
        )

        return {
            "n_stride": n_stride,
            "datapoint": {
                "datapoint_type": "cot_reasoning_termination",
                "prompt": prompt,
                "target_response": target,
                "layer": LAYERS[0],
                "layers": LAYERS,
                "num_positions": num_positions,
                "context_input_ids": context_slice,
                "context_positions": context_positions,
            },
        }

    # --- Generate overcomplete pools ---
    pool_size = args.num_examples * 3  # 3x oversampling is plenty for bin matching
    print(f"\nGenerating {pool_size} candidates per class...")

    pos_candidates = []
    for i in range(pool_size):
        if i % 10000 == 0 and i > 0:
            print(f"  Positive: {i}/{pool_size} attempts, {len(pos_candidates)} valid")
        t = random.choice(positive_pool)
        remaining = random.randint(25, min(55, t["cot_len"] - 1))
        result = _make_candidate(t, remaining, "will_terminate")
        if result is not None:
            pos_candidates.append(result)

    neg_candidates = []
    for i in range(pool_size):
        if i % 10000 == 0 and i > 0:
            print(f"  Negative: {i}/{pool_size} attempts, {len(neg_candidates)} valid")
        t = random.choice(negative_pool)
        remaining = random.randint(300, t["cot_len"] - 1)
        result = _make_candidate(t, remaining, "will_continue")
        if result is not None:
            neg_candidates.append(result)

    print(f"  Positive candidates: {len(pos_candidates)}")
    print(f"  Negative candidates: {len(neg_candidates)}")

    # Free the tokenized pool — candidates have their own context_input_ids copies
    del shared_pool, positive_pool, negative_pool
    gc.collect()

    # --- Report pre-matching stride distributions ---
    pos_strides = [c["n_stride"] for c in pos_candidates]
    neg_strides = [c["n_stride"] for c in neg_candidates]
    print(f"\n  Pre-matching stride position stats (per layer):")
    print(f"    will_terminate: mean={sum(pos_strides)/len(pos_strides):.0f}, "
          f"range=[{min(pos_strides)}, {max(pos_strides)}]")
    print(f"    will_continue:  mean={sum(neg_strides)/len(neg_strides):.0f}, "
          f"range=[{min(neg_strides)}, {max(neg_strides)}]")

    # --- Bin by stride count and pair ---
    print(f"\nBinning by stride count (bin_width={args.bin_width})...")
    BIN_WIDTH = args.bin_width

    def _bin_key(n_stride):
        return n_stride // BIN_WIDTH

    pos_bins = defaultdict(list)
    for c in pos_candidates:
        pos_bins[_bin_key(c["n_stride"])].append(c["datapoint"])

    neg_bins = defaultdict(list)
    for c in neg_candidates:
        neg_bins[_bin_key(c["n_stride"])].append(c["datapoint"])

    # Shuffle within bins
    for bin_list in pos_bins.values():
        random.shuffle(bin_list)
    for bin_list in neg_bins.values():
        random.shuffle(bin_list)

    shared_bins = sorted(set(pos_bins.keys()) & set(neg_bins.keys()))
    if not shared_bins:
        print("ERROR: No overlapping position-count bins!")
        print(f"  Pos bins: {sorted(pos_bins.keys())[:10]}...")
        print(f"  Neg bins: {sorted(neg_bins.keys())[:10]}...")
        sys.exit(1)

    available_pairs = {b: min(len(pos_bins[b]), len(neg_bins[b])) for b in shared_bins}
    total_available = sum(available_pairs.values())
    target_per_class = args.num_examples // 2

    print(f"  Shared bins: {len(shared_bins)}")
    print(f"  Total matchable pairs: {total_available}")
    print(f"  Target per class: {target_per_class}")

    # Print bin distribution
    print(f"\n  Bin distribution (stride_range: pos/neg/pairs):")
    for b in shared_bins:
        lo = b * BIN_WIDTH
        hi = lo + BIN_WIDTH - 1
        n_pos = len(pos_bins[b])
        n_neg = len(neg_bins[b])
        pairs = min(n_pos, n_neg)
        bar = "#" * min(pairs // 10, 50)
        print(f"    [{lo:4d}-{hi:4d}]: {n_pos:5d} / {n_neg:5d} / {pairs:5d} {bar}")

    if total_available < target_per_class:
        print(f"\n  WARNING: Only {total_available} matchable pairs, "
              f"need {target_per_class}. Using all available.")

    # Sample from bins
    pos_selected = []
    neg_selected = []
    remaining_budget = min(target_per_class, total_available)

    for b in shared_bins:
        if remaining_budget <= 0:
            break
        n_pairs = min(
            available_pairs[b],
            max(1, int(remaining_budget * available_pairs[b] / max(1, total_available))),
        )
        n_pairs = min(n_pairs, remaining_budget)
        pos_selected.extend(pos_bins[b][:n_pairs])
        neg_selected.extend(neg_bins[b][:n_pairs])
        remaining_budget -= n_pairs

    # Fill remaining budget greedily
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

    # --- Final stats ---
    n_pos = sum(1 for d in datapoints if d["target_response"].startswith("will_terminate"))
    n_neg = sum(1 for d in datapoints if d["target_response"].startswith("will_continue"))
    pos_npos = [d["num_positions"] for d in datapoints if d["target_response"].startswith("will_terminate")]
    neg_npos = [d["num_positions"] for d in datapoints if d["target_response"].startswith("will_continue")]

    print(f"\n=== Final dataset ===")
    print(f"  Total: {len(datapoints)} ({n_pos} pos + {n_neg} neg)")
    if pos_npos and neg_npos:
        print(f"  Position count stats (total across layers):")
        print(f"    will_terminate: mean={sum(pos_npos)/len(pos_npos):.0f}, "
              f"range=[{min(pos_npos)}, {max(pos_npos)}]")
        print(f"    will_continue:  mean={sum(neg_npos)/len(neg_npos):.0f}, "
              f"range=[{min(neg_npos)}, {max(neg_npos)}]")

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        for dp in datapoints:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(datapoints)} lines")

    # --- Upload to HF ---
    if args.push_to_hub:
        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        from huggingface_hub import HfApi
        api = HfApi()

        # Create repo if needed
        try:
            api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"  Repo creation: {e}")

        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo="train.jsonl",
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"  Uploaded to {args.hf_repo}")

    print("\nDone!")


if __name__ == "__main__":
    main()
