"""
Standalone eval script: load a trained checkpoint and run unfaithfulness evals.

Usage:
    python3 src/evals/eval_checkpoint.py \
        --checkpoint checkpoints/cot_oracle_8b_v2/step_3000 \
        --eval-dir data/evals \
        --model Qwen/Qwen3-8B

Prints oracle responses for each item and saves results to JSONL.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ao import (
    layer_percent_to_layer,
    choose_attn_implementation,
    run_oracle_on_activations,
    generate_cot,
)
from evals.common import load_eval_items, determine_ground_truth
from evals.run_evals import _extract_answer, ORACLE_PROMPTS_TEMPLATES, _oracle_prompt
from evals.activation_cache import extract_activation_bundle


def load_model_with_checkpoint(
    model_name: str,
    checkpoint_path: str,
    device: str = "cuda",
):
    """Load base model + trained LoRA checkpoint."""
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "attn_implementation": choose_attn_implementation(model_name),
    }

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    print(f"Loading trained LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    model.eval()

    # Verify adapter state
    print(f"  Active adapter: {model.active_adapter}")
    print(f"  Available adapters: {list(model.peft_config.keys())}")

    return model, tokenizer


def run_single_eval_item(model, tokenizer, item, act_layer, model_name, device="cuda"):
    """Run a single eval item and return result dict with full outputs."""
    # Generate responses with adapters disabled (base model)
    clean_response = generate_cot(
        model, tokenizer, item.clean_prompt,
        max_new_tokens=512, device=device,
    )
    test_response = generate_cot(
        model, tokenizer, item.test_prompt,
        max_new_tokens=512, device=device,
    )

    clean_answer = _extract_answer(clean_response, item.eval_name)
    test_answer = _extract_answer(test_response, item.eval_name)

    # Extract activations and run oracle
    oracle_response = ""
    bundle = None
    try:
        bundle = extract_activation_bundle(
            model,
            tokenizer,
            eval_name=item.eval_name,
            example_id=item.example_id,
            prompt=item.test_prompt,
            cot_text=test_response,
            act_layer=act_layer,
            device=device,
            max_boundaries=10,
            generation_adapter_name=None,
        )
        if bundle is not None and bundle.activations is not None:
            template = ORACLE_PROMPTS_TEMPLATES.get(item.eval_name, "What is this model doing?")
            oracle_prompt = _oracle_prompt(len(bundle.boundary_positions), template)
            oracle_response = run_oracle_on_activations(
                model, tokenizer, bundle.activations, oracle_prompt,
                model_name=model_name, act_layer=act_layer,
                max_new_tokens=150, device=device,
            )
    except Exception as e:
        oracle_response = f"ERROR: {e}"

    ground_truth = determine_ground_truth(item, clean_answer, test_answer)

    return {
        "eval_name": item.eval_name,
        "example_id": item.example_id,
        "ground_truth": ground_truth,
        "clean_answer": clean_answer,
        "test_answer": test_answer,
        "correct_answer": item.correct_answer,
        "nudge_answer": item.nudge_answer,
        "oracle_response": oracle_response,
        "test_response_snippet": test_response[:300],
        "n_sentences": len(bundle.sentences) if bundle is not None else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Eval trained checkpoint on unfaithfulness evals")
    parser.add_argument("--checkpoint", required=True, help="Path to trained LoRA checkpoint")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", default="eval_results.jsonl")
    parser.add_argument("--max-items", type=int, default=None, help="Max items per eval (None=all)")
    parser.add_argument("--evals", nargs="*", default=None, help="Specific evals to run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    model, tokenizer = load_model_with_checkpoint(args.model, args.checkpoint, args.device)
    act_layer = layer_percent_to_layer(args.model, 50)
    print(f"Activation layer: {act_layer}")

    # Test adapter switching works
    print("\nAdapter smoke test...")
    try:
        with model.disable_adapter():
            print("  disable_adapter context manager: OK")
        model.set_adapter("default")
        print("  set_adapter('default'): OK")
    except Exception as e:
        print(f"  ADAPTER ERROR: {e}")
        return

    # Find eval files
    eval_files = sorted(eval_dir.glob("*.json"))
    if args.evals:
        eval_files = [f for f in eval_files if f.stem in args.evals]

    all_results = []
    for eval_file in eval_files:
        eval_name = eval_file.stem
        if eval_name == "decorative_cot":
            print(f"\nSkipping {eval_name} (too slow)")
            continue

        items = load_eval_items(eval_file)
        if args.max_items:
            items = items[:args.max_items]

        print(f"\n{'=' * 60}")
        print(f"{eval_name}: {len(items)} items")
        print(f"{'=' * 60}")

        correct = 0
        total = 0

        for item in tqdm(items, desc=eval_name):
            result = run_single_eval_item(
                model, tokenizer, item, act_layer,
                model_name=args.model, device=args.device,
            )
            all_results.append(result)

            gt = result["ground_truth"]
            oracle = result["oracle_response"][:200]
            print(f"\n  [{result['example_id']}] gt={gt}")
            print(f"    clean_ans={result['clean_answer']}  test_ans={result['test_answer']}  "
                  f"correct={result['correct_answer']}  nudge={result['nudge_answer']}")
            print(f"    oracle: {oracle}")

            if gt in ("influenced", "independent"):
                total += 1
                # Simple accuracy: does oracle mention the right thing?
                # (proper scoring comes from score_oracle.py)

        print(f"\n  {eval_name} summary: {total} scoreable items")

    # Save all results
    with open(args.output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(all_results)} results to {args.output}")


if __name__ == "__main__":
    main()
