#!/usr/bin/env python3
"""
Evaluate hallucination detection: Oracle vs Adam's AO.

Loads test split from HF, runs both our trained oracle and Adam's pretrained AO
on each item, extracts hallucinated/factual logit-diff scores, and plots ROC curves.

Usage (on GPU machine):
    cd /root/cot-oracle
    export HF_TOKEN=hf_...
    python scripts/eval_hallucination_auroc.py \
        --checkpoint ceselder/cot-oracle-v15-stochastic \
        --max-items 0  # 0 = all

    # Quick sanity check:
    python scripts/eval_hallucination_auroc.py --max-items 10
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ao_reference"))

from core.ao import (
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    TRAINED_PLACEHOLDER,
    add_hook,
    collect_activations,
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
STRIDE = 5
HF_REPO = "ceselder/cot-oracle-hallucination-detection"

# Oracle prompts
OUR_PROMPT = "Is the model's reasoning factually accurate, or does it contain hallucinated information?"
ADAM_PROMPT = "Is the model's reasoning factually accurate, or does it contain hallucinated information?"


def load_test_data(max_items: int = 0) -> list[dict]:
    """Load test split from HF."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="test.jsonl",
        repo_type="dataset",
    )
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if max_items > 0:
        items = items[:max_items]
    print(f"Loaded {len(items)} test items from {HF_REPO}")
    labels = {}
    for it in items:
        labels[it["label"]] = labels.get(it["label"], 0) + 1
    print(f"  Distribution: {labels}")
    return items


def tokenize_cot(tokenizer, cot_text: str) -> list[int]:
    """Tokenize a CoT text for activation extraction (raw text, no chat template)."""
    return tokenizer.encode(cot_text, add_special_tokens=False)


def get_stride_positions(prompt_token_count: int, total_token_count: int) -> list[int]:
    """Compute stride-5 positions over CoT region."""
    positions = list(range(prompt_token_count, total_token_count, STRIDE))
    # Always include last token
    last = total_token_count - 1
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def extract_activations_single_layer(
    model, tokenizer, input_ids: list[int], positions: list[int], layer: int, device: str
) -> torch.Tensor:
    """Extract activations at stride positions for one layer. Returns [K, D]."""
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_tensor)

    submodule = get_hf_submodule(model, layer, use_lora=True)
    submodules = {layer: submodule}

    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
            min_offset=None,
            max_offset=None,
        )

    acts_BLD = acts_by_layer[layer]  # [1, L, D]
    return acts_BLD[0, positions, :].detach()  # [K, D]


def extract_activations_multi_layer(
    model, tokenizer, input_ids: list[int], positions: list[int], layers: list[int], device: str
) -> torch.Tensor:
    """Extract activations at stride positions for multiple layers. Returns [K*N, D]."""
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_tensor)

    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers}

    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
            min_offset=None,
            max_offset=None,
        )

    parts = []
    for layer in layers:
        acts_BLD = acts_by_layer[layer]  # [1, L, D]
        parts.append(acts_BLD[0, positions, :].detach())  # [K, D]

    return torch.cat(parts, dim=0)  # [K*N, D]


def build_prefix_and_find_positions(
    tokenizer, num_positions: int, layers: list[int] | int, ph_token: str, oracle_prompt: str
) -> tuple[list[int], list[int]]:
    """Build oracle prompt with placeholders, tokenize, find placeholder positions.

    Returns (input_ids, placeholder_positions).
    """
    if isinstance(layers, (list, tuple)):
        N = len(layers)
        K = num_positions // N
        assert K * N == num_positions, f"num_positions={num_positions} not divisible by {N} layers"
        parts = [f"L{l}:" + ph_token * K for l in layers]
        prefix = " ".join(parts) + "\n"
    else:
        prefix = f"L{layers}:" + ph_token * num_positions + "\n"

    full_prompt = prefix + oracle_prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Expected single token for '{ph_token}', got {len(ph_id)}"
    ph_id = ph_id[0]

    positions = []
    for i, tid in enumerate(input_ids):
        if tid == ph_id and len(positions) < num_positions:
            positions.append(i)
    assert len(positions) == num_positions, (
        f"Found {len(positions)} placeholder positions, expected {num_positions}"
    )
    return input_ids, positions


def score_item_with_oracle(
    model, tokenizer, activations: torch.Tensor,
    layers: list[int] | int, oracle_prompt: str,
    ph_token: str, adapter_name: str | None,
    device: str,
) -> float:
    """Run oracle forward pass, return logit_diff = logit(hallucinated) - logit(factual).

    Positive score = model thinks hallucinated.
    """
    num_positions = activations.shape[0]
    input_ids, ph_positions = build_prefix_and_find_positions(
        tokenizer, num_positions, layers, ph_token, oracle_prompt
    )

    # Resolve answer token IDs
    # Try "hallucinated" and "factual" first tokens
    hall_id = tokenizer.encode("hallucinated", add_special_tokens=False)[0]
    fact_id = tokenizer.encode("factual", add_special_tokens=False)[0]
    # Also try Yes/No as fallback
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    # Set adapter
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    else:
        ao_path = AO_CHECKPOINTS[MODEL_NAME]
        sanitized = ao_path.replace(".", "_")
        model.set_adapter(sanitized)

    hook_fn = get_steering_hook(
        vectors=activations,
        positions=ph_positions,
        device=device,
        dtype=torch.bfloat16,
    )
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)

    with add_hook(injection_submodule, hook_fn):
        outputs = model(input_ids=input_tensor, attention_mask=attn_mask)

    logits = outputs.logits[0, -1, :].float()  # [vocab_size]

    # Primary: hallucinated vs factual first-token logit diff
    score = (logits[hall_id] - logits[fact_id]).item()

    return score


@torch.no_grad()
def run_evaluation(
    model, tokenizer, test_data: list[dict],
    our_adapter: str, device: str,
) -> dict:
    """Run both oracle and Adam's AO on all test items."""
    model.eval()

    our_scores = []
    adam_scores = []
    labels = []  # 1 = hallucinated, 0 = factual
    skipped = 0

    for i, item in enumerate(test_data):
        cot_text = item["cot_text"]
        label = 1 if item["label"] == "hallucinated" else 0

        # Tokenize the CoT
        input_ids = tokenize_cot(tokenizer, cot_text)
        if len(input_ids) < STRIDE + 1:
            skipped += 1
            continue

        # Get stride positions (treat position 0 as "prompt" start since raw CoT has no prompt)
        positions = list(range(0, len(input_ids), STRIDE))
        last = len(input_ids) - 1
        if not positions or positions[-1] != last:
            positions.append(last)

        if len(positions) < 1:
            skipped += 1
            continue

        # Extract activations (multi-layer for ours, single-layer for Adam's)
        try:
            acts_multi = extract_activations_multi_layer(
                model, tokenizer, input_ids, positions, OUR_LAYERS, device
            )
            acts_single = extract_activations_single_layer(
                model, tokenizer, input_ids, positions, ADAM_LAYER, device
            )
        except Exception as e:
            print(f"  [{i}] Activation extraction failed: {e}")
            skipped += 1
            continue

        # Score with our oracle
        try:
            our_score = score_item_with_oracle(
                model, tokenizer, acts_multi,
                layers=OUR_LAYERS, oracle_prompt=OUR_PROMPT,
                ph_token=TRAINED_PLACEHOLDER, adapter_name=our_adapter,
                device=device,
            )
        except Exception as e:
            print(f"  [{i}] Our oracle failed: {e}")
            skipped += 1
            continue

        # Score with Adam's AO
        try:
            adam_score = score_item_with_oracle(
                model, tokenizer, acts_single,
                layers=ADAM_LAYER, oracle_prompt=ADAM_PROMPT,
                ph_token=SPECIAL_TOKEN, adapter_name=None,  # auto-detect AO adapter
                device=device,
            )
        except Exception as e:
            print(f"  [{i}] Adam's AO failed: {e}")
            skipped += 1
            continue

        our_scores.append(our_score)
        adam_scores.append(adam_score)
        labels.append(label)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] label={item['label']}, "
                  f"our={our_score:.3f}, adam={adam_score:.3f}")

    print(f"\nProcessed {len(labels)} items, skipped {skipped}")
    return {
        "our_scores": our_scores,
        "adam_scores": adam_scores,
        "labels": labels,
    }


def plot_roc(results: dict, output_path: str):
    """Plot ROC curves for both oracles."""
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
    ax.set_title("Hallucination Detection: Oracle vs AO")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC plot saved to {output_path}")


def print_summary(results: dict):
    """Print summary statistics."""
    import numpy as np

    labels = np.array(results["labels"])
    n_hall = labels.sum()
    n_fact = len(labels) - n_hall

    print(f"\n{'='*60}")
    print(f"Hallucination Detection Summary")
    print(f"{'='*60}")
    print(f"Total items: {len(labels)} ({n_hall} hallucinated, {n_fact} factual)")

    for name, scores_key in [("Our Oracle", "our_scores"), ("Adam's AO", "adam_scores")]:
        scores = np.array(results[scores_key])
        hall_scores = scores[labels == 1]
        fact_scores = scores[labels == 0]
        print(f"\n{name}:")
        print(f"  Mean score (hallucinated): {hall_scores.mean():.3f} +/- {hall_scores.std():.3f}")
        print(f"  Mean score (factual):      {fact_scores.mean():.3f} +/- {fact_scores.std():.3f}")
        print(f"  Score gap:                 {hall_scores.mean() - fact_scores.mean():.3f}")

        # Accuracy at threshold=0
        preds = (scores > 0).astype(int)
        acc = (preds == labels).mean()
        print(f"  Accuracy (threshold=0):    {acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Hallucination detection AUROC eval")
    parser.add_argument("--checkpoint", type=str, default="ceselder/cot-oracle-v15-stochastic",
                        help="HF repo or local path for our oracle adapter")
    parser.add_argument("--max-items", type=int, default=0,
                        help="Max test items (0 = all)")
    parser.add_argument("--output", type=str, default="data/hallucination_auroc.png",
                        help="Output path for ROC plot")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    args = parser.parse_args()

    print(f"Loading test data...")
    test_data = load_test_data(args.max_items)
    if not test_data:
        print("No test data found! Run generate_hallucination_dataset.py first.")
        sys.exit(1)

    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = load_model_with_ao(MODEL_NAME, use_8bit=args.use_8bit)

    # Load our trained oracle adapter
    our_adapter_name = "trained_oracle"
    print(f"Loading our oracle adapter: {args.checkpoint}")
    if our_adapter_name not in model.peft_config:
        model.load_adapter(
            args.checkpoint,
            adapter_name=our_adapter_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    print(f"\nRunning evaluation...")
    t0 = time.time()
    results = run_evaluation(model, tokenizer, test_data, our_adapter_name, args.device)
    elapsed = time.time() - t0
    print(f"Evaluation took {elapsed:.1f}s")

    if not results["labels"]:
        print("No items were successfully scored!")
        sys.exit(1)

    # Save raw results
    results_path = args.output.replace(".png", "_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Raw results saved to {results_path}")

    print_summary(results)
    plot_roc(results, args.output)


if __name__ == "__main__":
    main()
