#!/usr/bin/env python3
"""
Evaluate hallucination detection at the SENTENCE level: Oracle vs Adam's AO.

Each test item is a single sentence from a CoT, labeled hallucinated or factual.
For each sentence:
  1. Chat-template [user: question][assistant: full_cot]
  2. Extract activations at that sentence's token positions only
  3. Run oracle with those activations → logit(hallucinated) - logit(factual)
  4. Compute AUROC over all sentences

Usage (on GPU machine):
    cd /root/cot-oracle
    export HF_TOKEN=hf_...
    python scripts/eval_hallucination_auroc.py \
        --checkpoint ceselder/cot-oracle-v15-stochastic

    # Quick sanity check:
    python scripts/eval_hallucination_auroc.py --max-items 20
"""

import argparse
import json
import os
import re
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ao_reference"))

from core.ao import (
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    TRAINED_PLACEHOLDER,
    add_hook,
    get_hf_submodule,
    get_steering_hook,
    load_model_with_ao,
    using_adapter,
)
from nl_probes.utils.activation_utils import collect_activations_multiple_layers

MODEL_NAME = "Qwen/Qwen3-8B"
OUR_LAYERS = [9, 18, 27]
ADAM_LAYER = 18
INJECTION_LAYER = 1
HF_REPO = "ceselder/cot-oracle-hallucination-detection"

OUR_PROMPT = "Is the model's reasoning factually accurate, or does it contain hallucinated information?"
ADAM_PROMPT = "Is the model's reasoning factually accurate, or does it contain hallucinated information?"

CONTEXT_MODES = ("sentence", "stride", "combined")
DEFAULT_STRIDE = 5


def load_sentence_data(max_items: int = 0) -> list[dict]:
    """Load sentence-level test data from HF."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="sentences_test.jsonl",
        repo_type="dataset",
    )
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if max_items > 0:
        items = items[:max_items]
    print(f"Loaded {len(items)} sentence-level test items")
    from collections import Counter
    labels = Counter(it["label"] for it in items)
    print(f"  Distribution: {dict(labels)}")
    return items


def get_sentence_token_positions(
    tokenizer, question: str, cot_text: str, sentence: str,
) -> tuple[list[int], list[int]]:
    """Chat-template the full CoT, find token positions for one sentence.

    Returns (input_ids, sentence_positions).
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": cot_text},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )

    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    # Find the sentence in the formatted text
    # Search within the CoT region
    cot_start = formatted.find(cot_text)
    if cot_start == -1:
        cot_start = formatted.rfind(cot_text[:50])
        if cot_start == -1:
            cot_start = 0

    sent_start = formatted.find(sentence, cot_start)
    if sent_start == -1:
        # Try partial match
        sent_start = formatted.find(sentence[:60], cot_start)
        if sent_start == -1:
            return input_ids, []
    sent_end = sent_start + len(sentence)

    # Map character span to token indices
    positions = []
    for tok_idx, (ts, te) in enumerate(offsets):
        if te > sent_start and ts < sent_end:
            positions.append(tok_idx)

    return input_ids, positions


def extract_activations_at_positions(
    model, input_ids: list[int], positions: list[int],
    layers: list[int], device: str,
) -> torch.Tensor:
    """Extract activations at given positions for specified layers.

    Returns [K * N_layers, D].
    """
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_tensor)

    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers}

    with using_adapter(model, None):
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
            min_offset=None,
            max_offset=None,
        )

    parts = []
    for layer in layers:
        parts.append(acts_by_layer[layer][0, positions, :].detach())

    return torch.cat(parts, dim=0)


def build_prefix_and_find_positions(
    tokenizer, num_positions: int, layers: list[int] | int, ph_token: str, oracle_prompt: str,
) -> tuple[list[int], list[int]]:
    """Build oracle prompt with direct token ID insertion (avoids BPE merges)."""
    ph_id_list = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id_list) == 1
    ph_id = ph_id_list[0]

    if isinstance(layers, (list, tuple)):
        prefix_layers = list(layers)
        N = len(prefix_layers)
        K = num_positions // N
        assert K * N == num_positions, f"num_positions={num_positions} not divisible by {N}"
        block_sizes = [K] * N
    else:
        prefix_layers = [layers]
        block_sizes = [num_positions]

    prefix_ids: list[int] = []
    positions: list[int] = []
    for i, (layer_idx, block_size) in enumerate(zip(prefix_layers, block_sizes)):
        label = f"L{layer_idx}:"
        if i > 0:
            label = " " + label
        prefix_ids.extend(tokenizer.encode(label, add_special_tokens=False))
        positions.extend(range(len(prefix_ids), len(prefix_ids) + block_size))
        prefix_ids.extend([ph_id] * block_size)
    prefix_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

    prompt_ids = tokenizer.encode(oracle_prompt, add_special_tokens=False)

    messages = [{"role": "user", "content": "PLACEHOLDER"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    before, after = formatted.split("PLACEHOLDER", 1)
    header_ids = tokenizer.encode(before, add_special_tokens=False)
    footer_ids = tokenizer.encode(after, add_special_tokens=False)

    positions = [p + len(header_ids) for p in positions]
    input_ids = header_ids + prefix_ids + prompt_ids + footer_ids

    return input_ids, positions


def score_sentence(
    model, tokenizer, activations: torch.Tensor,
    layers: list[int] | int, oracle_prompt: str,
    ph_token: str, adapter_name: str | None,
    device: str, hall_id: int, fact_id: int,
) -> float:
    """Forward pass with activation injection, return logit(hallucinated) - logit(factual)."""
    num_positions = activations.shape[0]
    input_ids, ph_positions = build_prefix_and_find_positions(
        tokenizer, num_positions, layers, ph_token, oracle_prompt
    )

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    if adapter_name is not None:
        model.set_adapter(adapter_name)
    else:
        ao_path = AO_CHECKPOINTS[MODEL_NAME]
        model.set_adapter(ao_path.replace(".", "_"))

    hook_fn = get_steering_hook(
        vectors=activations, positions=ph_positions,
        device=device, dtype=torch.bfloat16,
    )
    injection_sub = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)

    with add_hook(injection_sub, hook_fn):
        outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

    logits = outputs.logits[0, -1, :].float()
    return (logits[hall_id] - logits[fact_id]).item()


def _compute_positions(
    context_mode: str, sentence_positions: list[int],
    cot_token_start: int, cot_token_end: int, stride: int,
) -> list[int]:
    """Compute activation positions based on context mode.

    sentence: only the sentence's own token positions
    stride:   stride-N positions across the full CoT region
    combined: sentence positions + evenly-spaced CoT positions, deduplicated
    """
    if context_mode == "sentence":
        return sentence_positions
    elif context_mode == "stride":
        return list(range(cot_token_start, cot_token_end, stride))
    elif context_mode == "combined":
        stride_positions = set(range(cot_token_start, cot_token_end, stride))
        return sorted(stride_positions | set(sentence_positions))
    else:
        raise ValueError(f"Unknown context_mode: {context_mode}")


@torch.no_grad()
def run_evaluation(
    model, tokenizer, test_data: list[dict],
    our_adapter: str, device: str,
    context_mode: str = "sentence", stride: int = DEFAULT_STRIDE,
) -> dict:
    """Score each sentence with both oracles."""
    model.eval()
    print(f"  Context mode: {context_mode}, stride: {stride}")

    # Pre-resolve token IDs
    hall_id = tokenizer.encode("hallucinated", add_special_tokens=False)[0]
    fact_id = tokenizer.encode("factual", add_special_tokens=False)[0]

    our_scores = []
    adam_scores = []
    labels = []
    skipped = 0
    categories = []

    # Group sentences by CoT to reuse activation extraction
    from collections import defaultdict
    cot_groups = defaultdict(list)
    for i, item in enumerate(test_data):
        key = (item["question"], item["cot_text"][:200])  # group by CoT
        cot_groups[key].append((i, item))

    print(f"  {len(cot_groups)} unique CoTs, {len(test_data)} sentences")

    processed_cots = 0
    for (question, _), group in cot_groups.items():
        cot_text = group[0][1]["cot_text"]
        processed_cots += 1

        # One forward pass to get all activations for this CoT
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": cot_text},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
        full_input_ids = encoded["input_ids"]
        full_offsets = encoded["offset_mapping"]

        cot_start_char = formatted.find(cot_text)
        if cot_start_char == -1:
            cot_start_char = formatted.rfind(cot_text[:50])
            if cot_start_char == -1:
                skipped += len(group)
                continue
        cot_end_char = cot_start_char + len(cot_text)

        # Map CoT character span to token indices
        cot_token_start = cot_token_end = None
        for tok_idx, (ts, te) in enumerate(full_offsets):
            if te > cot_start_char and cot_token_start is None:
                cot_token_start = tok_idx
            if ts < cot_end_char:
                cot_token_end = tok_idx + 1
        if cot_token_start is None or cot_token_end is None:
            skipped += len(group)
            continue

        # Extract activations for all layers in one forward pass
        try:
            input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=device)
            attn_mask = torch.ones_like(input_tensor)
            submodules_multi = {l: get_hf_submodule(model, l, use_lora=True) for l in OUR_LAYERS}
            submodules_single = {ADAM_LAYER: get_hf_submodule(model, ADAM_LAYER, use_lora=True)}
            all_submodules = {**submodules_multi, **submodules_single}

            with using_adapter(model, None):
                acts_by_layer = collect_activations_multiple_layers(
                    model=model,
                    submodules=all_submodules,
                    inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
                    min_offset=None, max_offset=None,
                )
        except Exception as e:
            print(f"  CoT activation extraction failed: {e}")
            skipped += len(group)
            continue

        # Score each sentence in this CoT
        for idx, item in group:
            sentence = item["sentence"]
            label = 1 if item["label"] == "hallucinated" else 0

            # Find sentence token positions
            sent_start = formatted.find(sentence, cot_start_char)
            if sent_start == -1:
                sent_start = formatted.find(sentence[:60], cot_start_char)
                if sent_start == -1:
                    skipped += 1
                    continue
            sent_end = sent_start + len(sentence)

            sent_positions = []
            for tok_idx, (ts, te) in enumerate(full_offsets):
                if te > sent_start and ts < sent_end:
                    sent_positions.append(tok_idx)

            if len(sent_positions) < 3:
                skipped += 1
                continue

            # Compute positions based on context mode
            positions = _compute_positions(
                context_mode, sent_positions,
                cot_token_start, cot_token_end, stride,
            )

            if len(positions) < 3:
                skipped += 1
                continue

            # Slice activations
            try:
                # Multi-layer for our oracle
                acts_multi_parts = []
                for layer in OUR_LAYERS:
                    acts_multi_parts.append(acts_by_layer[layer][0, positions, :].detach())
                acts_multi = torch.cat(acts_multi_parts, dim=0)

                # Single layer for Adam's AO
                acts_single = acts_by_layer[ADAM_LAYER][0, positions, :].detach()

                # Score with our oracle
                our_score = score_sentence(
                    model, tokenizer, acts_multi,
                    OUR_LAYERS, OUR_PROMPT,
                    TRAINED_PLACEHOLDER, our_adapter,
                    device, hall_id, fact_id,
                )

                # Score with Adam's AO
                adam_score = score_sentence(
                    model, tokenizer, acts_single,
                    ADAM_LAYER, ADAM_PROMPT,
                    SPECIAL_TOKEN, None,
                    device, hall_id, fact_id,
                )
            except Exception as e:
                print(f"  [{idx}] Scoring failed: {e}")
                skipped += 1
                continue

            our_scores.append(our_score)
            adam_scores.append(adam_score)
            labels.append(label)
            categories.append(item.get("prompt_category", ""))

        if processed_cots % 5 == 0:
            n_done = len(labels)
            print(f"  CoTs: {processed_cots}/{len(cot_groups)}, "
                  f"sentences scored: {n_done}, skipped: {skipped}")

    print(f"\nScored {len(labels)} sentences, skipped {skipped}")
    return {
        "our_scores": our_scores,
        "adam_scores": adam_scores,
        "labels": labels,
        "categories": categories,
    }


def plot_roc(results: dict, output_path: str):
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = results["labels"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for name, scores in [
        ("Our Oracle", results["our_scores"]),
        ("Adam's AO", results["adam_scores"]),
    ]:
        try:
            auc = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})", linewidth=2)
            print(f"{name}: AUROC = {auc:.3f}")
        except Exception as e:
            print(f"{name}: Could not compute AUROC — {e}")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Sentence-Level Hallucination Detection: Oracle vs AO")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC plot saved to {output_path}")


def print_summary(results: dict):
    import numpy as np
    from collections import Counter

    labels = np.array(results["labels"])
    n_hall = int(labels.sum())
    n_fact = len(labels) - n_hall

    print(f"\n{'='*60}")
    print(f"Sentence-Level Hallucination Detection")
    print(f"{'='*60}")
    print(f"Total sentences: {len(labels)} ({n_hall} hallucinated, {n_fact} factual)")

    for name, key in [("Our Oracle", "our_scores"), ("Adam's AO", "adam_scores")]:
        scores = np.array(results[key])
        hall_s = scores[labels == 1]
        fact_s = scores[labels == 0]
        print(f"\n{name}:")
        if len(hall_s):
            print(f"  Mean score (hallucinated): {hall_s.mean():.3f} +/- {hall_s.std():.3f}")
        if len(fact_s):
            print(f"  Mean score (factual):      {fact_s.mean():.3f} +/- {fact_s.std():.3f}")
        if len(hall_s) and len(fact_s):
            print(f"  Score gap:                 {hall_s.mean() - fact_s.mean():.3f}")
        preds = (scores > 0).astype(int)
        acc = (preds == labels).mean()
        print(f"  Accuracy (threshold=0):    {acc:.3f}")

    # Per-category breakdown
    cats = results.get("categories", [])
    if cats:
        cat_counts = Counter(
            c for c, l in zip(cats, labels) if l == 1
        )
        print(f"\nHallucinated sentences by category: {dict(cat_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Sentence-level hallucination AUROC eval")
    parser.add_argument("--checkpoint", type=str, default="ceselder/cot-oracle-v15-stochastic")
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--output", type=str, default="data/hallucination_auroc.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--context-mode", choices=CONTEXT_MODES, default="sentence",
                        help="How to select activation positions: "
                             "sentence=only sentence tokens, "
                             "stride=stride-N across full CoT, "
                             "combined=sentence + stride positions")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Stride for CoT-wide position sampling (default: 5)")
    args = parser.parse_args()

    print("Loading sentence-level test data...")
    test_data = load_sentence_data(args.max_items)
    if not test_data:
        print("No data! Run label_hallucination_sentences.py first.")
        sys.exit(1)

    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load_model_with_ao(MODEL_NAME, use_8bit=args.use_8bit)

    our_adapter_name = "trained_oracle"
    print(f"Loading our oracle: {args.checkpoint}")
    if our_adapter_name not in model.peft_config:
        model.load_adapter(args.checkpoint, adapter_name=our_adapter_name, is_trainable=False)

    # Move all params to GPU
    target_device = next(model.base_model.parameters()).device
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(target_device)

    print("\nRunning evaluation...")
    t0 = time.time()
    results = run_evaluation(
        model, tokenizer, test_data, our_adapter_name, args.device,
        context_mode=args.context_mode, stride=args.stride,
    )
    print(f"Evaluation took {time.time() - t0:.1f}s")

    if not results["labels"]:
        print("No sentences scored!")
        sys.exit(1)

    results["context_mode"] = args.context_mode
    results["stride"] = args.stride
    results_path = args.output.replace(".png", "_results.json")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {results_path}")

    print_summary(results)
    plot_roc(results, args.output)


if __name__ == "__main__":
    main()
