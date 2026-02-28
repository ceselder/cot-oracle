"""
Evaluate a trained CoT Oracle checkpoint with fuzzy scoring.

Unlike the AO eval (exact match + max 20 tokens), this:
  - Uses token F1 for generation tasks
  - Uses substring match for classification tasks
  - Generates up to 150 tokens for generation tasks
  - Prints sample outputs for qualitative inspection

Usage:
    python src/eval_checkpoint.py \
        --checkpoint checkpoints/cot_oracle_v2/step_1000 \
        --corpus data/cot_corpus_v5/mini_corpus.jsonl \
        --model Qwen/Qwen3-8B \
        --n-samples 50
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ao_repo import ensure_ao_repo_on_path
ensure_ao_repo_on_path()

import torch

import nl_probes.utils.dataset_utils as du_module

# Patch placeholder token
PLACEHOLDER_TOKEN = " ¶"
du_module.SPECIAL_TOKEN = PLACEHOLDER_TOKEN
_orig_get_prefix = du_module.get_introspection_prefix
def _patched_get_prefix(sae_layer: int, num_positions: int) -> str:
    prefix = f"L{sae_layer}:" + PLACEHOLDER_TOKEN * num_positions + "\n"
    return prefix
du_module.get_introspection_prefix = _patched_get_prefix

def _patched_find_pattern_in_tokens(token_ids, special_token_str, num_positions, tokenizer):
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []
    for i in range(len(token_ids)):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)
    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    return positions
du_module.find_pattern_in_tokens = _patched_find_pattern_in_tokens

from nl_probes.utils.dataset_utils import create_training_datapoint, TrainingDataPoint
from nl_probes.utils.eval import run_evaluation, parse_answer
from nl_probes.utils.common import load_tokenizer

# Dataset loaders
from dataset_classes.cot_context_prediction import load_cot_context_prediction_data
from dataset_classes.cot_answer_prediction import load_cot_answer_prediction_data
from dataset_classes.cot_full_reconstruction import load_cot_full_reconstruction_data
from dataset_classes.cot_causal_prediction import load_cot_causal_prediction_data
from dataset_classes.cot_decorative import load_cot_decorative_data
from dataset_classes.cot_domain import load_cot_domain_data
from dataset_classes.cot_correctness import load_cot_correctness_data


# Task categories
CLASSIFICATION_TASKS = {"cot_decorative", "cot_domain", "cot_correctness", "cot_persona"}
GENERATION_TASKS = {"cot_full_reconstruction", "cot_causal_prediction", "cot_conversational", "cot_context_prediction"}
ANSWER_TASKS = {"cot_answer_prediction"}


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def dicts_to_training_data(raw_data, tokenizer):
    """Convert dataset loader output to AO TrainingDataPoint objects."""
    training_data = []
    skipped = 0
    for item in raw_data:
        try:
            layer = item.get("layer")
            if layer is None:
                layers = item.get("layers", [])
                if layers:
                    layer = layers[len(layers) // 2]
                else:
                    skipped += 1
                    continue
            dp = create_training_datapoint(
                datapoint_type=item["datapoint_type"],
                prompt=item["prompt"],
                target_response=item["target_response"],
                layer=layer,
                num_positions=item["num_positions"],
                tokenizer=tokenizer,
                acts_BD=None,
                feature_idx=-1,
                context_input_ids=item["context_input_ids"],
                context_positions=item["context_positions"],
            )
            training_data.append(dp)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped ({e})")
    return training_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--corpus", default="data/cot_corpus_v5/mini_corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-samples", type=int, default=50, help="Eval items per task")
    parser.add_argument("--n-show", type=int, default=5, help="Sample responses to print per task")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max tokens for generation tasks")
    args = parser.parse_args()

    layer_percents = [25, 50, 75]

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model)

    # Build eval data (small amounts per task)
    n = args.n_samples
    print(f"\nBuilding eval data ({n} per task)...")

    eval_sets = {}

    # Context prediction
    try:
        raw = load_cot_context_prediction_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,  # different seed from training
        )
        eval_sets["cot_context_prediction"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  context_prediction failed: {e}")

    # Answer prediction
    try:
        raw = load_cot_answer_prediction_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_answer_prediction"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  answer_prediction failed: {e}")

    # Full reconstruction
    try:
        raw = load_cot_full_reconstruction_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_full_reconstruction"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  full_reconstruction failed: {e}")

    # Causal prediction
    try:
        raw = load_cot_causal_prediction_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_causal_prediction"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  causal_prediction failed: {e}")

    # CotQA (conversational questions about CoTs, from HF)
    try:
        from dataset_classes.cot_cotqa import load_cot_cotqa_data
        raw = load_cot_cotqa_data(
            "", tokenizer, args.model,
            num_examples=n, seed=999,
        )
        eval_sets["cot_cotqa"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  cotqa failed: {e}")

    # Decorative
    try:
        raw = load_cot_decorative_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_decorative"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  decorative failed: {e}")

    # Domain
    try:
        raw = load_cot_domain_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_domain"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  domain failed: {e}")

    # Correctness
    try:
        raw = load_cot_correctness_data(
            args.corpus, tokenizer, args.model, layer_percents,
            num_examples=n, seed=999,
        )
        eval_sets["cot_correctness"] = dicts_to_training_data(raw, tokenizer)
    except Exception as e:
        print(f"  correctness failed: {e}")

    for name, items in eval_sets.items():
        print(f"  {name}: {len(items)} items")

    # Load model + checkpoint
    print(f"\nLoading model {args.model}...")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"Loading LoRA from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Get submodule for hook injection
    from nl_probes.utils.common import layer_percent_to_layer
    hook_layer = 1
    submodule = model.model.layers[hook_layer]

    # Run eval per task
    print(f"\n{'=' * 70}")
    print("FUZZY EVALUATION")
    print(f"{'=' * 70}")

    results = {}
    for ds_name, eval_data in sorted(eval_sets.items()):
        if not eval_data:
            continue

        # Pick generation kwargs based on task type
        if ds_name in GENERATION_TASKS:
            gen_kwargs = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
        else:
            gen_kwargs = {"do_sample": False, "max_new_tokens": 30}

        print(f"\n--- {ds_name} ({len(eval_data)} items, max_tokens={gen_kwargs['max_new_tokens']}) ---")

        responses = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=0,
            lora_path=None,
            eval_batch_size=16,
            steering_coefficient=1.0,
            generation_kwargs=gen_kwargs,
        )

        # Score
        exact_scores = []
        fuzzy_scores = []
        samples = []

        for resp, dp in zip(responses, eval_data):
            pred = resp.api_response.strip()
            target = dp.target_output.strip()

            # Exact match
            pred_clean = parse_answer(pred)
            target_clean = parse_answer(target)
            exact = 1.0 if pred_clean == target_clean else 0.0
            exact_scores.append(exact)

            # Fuzzy match
            if ds_name in CLASSIFICATION_TASKS:
                fuzzy = 1.0 if target_clean in pred_clean else 0.0
            elif ds_name in ANSWER_TASKS:
                fuzzy = exact
            else:
                fuzzy = token_f1(pred, target)
            fuzzy_scores.append(fuzzy)

            if len(samples) < args.n_show:
                samples.append((pred[:300], target[:300], exact, fuzzy))

        avg_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0
        avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0

        print(f"  Exact match: {avg_exact:.1%}")
        print(f"  Fuzzy score:  {avg_fuzzy:.1%}")

        for i, (pred, target, ex, fz) in enumerate(samples):
            print(f"\n  [{i}] pred:   '{pred}'")
            print(f"      target: '{target}'")
            print(f"      exact={ex:.0f} fuzzy={fz:.2f}")

        results[ds_name] = {"exact": avg_exact, "fuzzy": avg_fuzzy, "n": len(eval_data)}

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Task':<30} {'Exact':>8} {'Fuzzy':>8} {'N':>5}")
    print("-" * 55)
    for name, r in sorted(results.items()):
        print(f"{name:<30} {r['exact']:>7.1%} {r['fuzzy']:>7.1%} {r['n']:>5}")


if __name__ == "__main__":
    main()
