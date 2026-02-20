"""Precompute eval activation bundles for faster repeated oracle runs.

Usage:
    python3 src/evals/precompute_activations.py \
        --eval-dir data/evals \
        --output-dir data/eval_precomputed \
        --model Qwen/Qwen3-8B \
        --generator-adapter ceselder/rot13-qwen3-8b-lora \
        --evals rot13_reconstruction held_out_cot_reconstruction logical_leaps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.ao import (
    load_model_with_ao,
    load_extra_adapter,
    generate_cot,
    layer_percent_to_layer,
)
from evals.activation_cache import extract_activation_bundle, save_bundle, cache_path
from evals.common import load_eval_items, EvalItem
from evals.run_evals import _extract_answer


def _cot_for_activations(item: EvalItem, test_response: str) -> str:
    if item.eval_name == "sentence_insertion":
        return str(item.metadata.get("spliced_cot_text", ""))
    if item.eval_name == "held_out_cot_reconstruction":
        return str(item.metadata.get("reference_cot", ""))
    if item.eval_name == "rot13_reconstruction":
        return str(item.metadata.get("rot13_cot", ""))
    if item.eval_name == "logical_leaps":
        return str(item.metadata.get("reference_cot", ""))
    return test_response


def _max_boundaries_for_eval(eval_name: str) -> int:
    if eval_name == "sentence_insertion":
        return 30
    if eval_name in ("held_out_cot_reconstruction", "rot13_reconstruction", "logical_leaps"):
        return 20
    if eval_name == "final_answer_kl":
        return 12
    return 10


def main():
    parser = argparse.ArgumentParser(description="Precompute eval activation bundles")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--output-dir", default="data/eval_precomputed")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--evals", nargs="*", default=None)
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap per eval dataset.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator-adapter", default=None, help="Optional LoRA path used for generation/capture.")
    parser.add_argument("--generator-adapter-name", default="generator")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached bundles.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} + AO...")
    model, tokenizer = load_model_with_ao(args.model, device=args.device)
    act_layer = layer_percent_to_layer(args.model, 50)
    print(f"Activation layer: {act_layer}")

    generation_adapter_name = None
    if args.generator_adapter:
        generation_adapter_name = load_extra_adapter(
            model, args.generator_adapter, adapter_name=args.generator_adapter_name
        )
        print(f"Generation adapter for capture: {generation_adapter_name}")

    eval_files = sorted(eval_dir.glob("*.json"))
    if args.evals:
        wanted = set(args.evals)
        eval_files = [f for f in eval_files if f.stem in wanted]

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for eval_file in eval_files:
        eval_name = eval_file.stem
        items = load_eval_items(eval_file)
        if args.max_items is not None:
            items = items[: args.max_items]

        print(f"\n{'=' * 60}")
        print(f"Precomputing: {eval_name} ({len(items)} items)")
        print(f"{'=' * 60}")

        saved = 0
        skipped = 0
        failed = 0
        for item in tqdm(items, desc=f"precompute:{eval_name}"):
            out_path = cache_path(output_dir, item.eval_name, item.example_id)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            clean_response = ""
            test_response = ""
            clean_answer = None
            test_answer = None

            if item.eval_name == "sentence_insertion":
                test_response = str(item.metadata.get("spliced_cot_text", ""))
            elif item.eval_name in ("held_out_cot_reconstruction", "rot13_reconstruction", "logical_leaps"):
                test_response = _cot_for_activations(item, "")
            else:
                clean_response = generate_cot(
                    model,
                    tokenizer,
                    item.clean_prompt,
                    max_new_tokens=512,
                    device=args.device,
                    adapter_name=generation_adapter_name,
                )
                test_response = generate_cot(
                    model,
                    tokenizer,
                    item.test_prompt,
                    max_new_tokens=512,
                    device=args.device,
                    adapter_name=generation_adapter_name,
                )
                clean_answer = _extract_answer(clean_response, item.eval_name)
                test_answer = _extract_answer(test_response, item.eval_name)

            cot_text = _cot_for_activations(item, test_response)
            bundle = extract_activation_bundle(
                model,
                tokenizer,
                eval_name=item.eval_name,
                example_id=item.example_id,
                prompt=item.test_prompt,
                cot_text=cot_text,
                act_layer=act_layer,
                device=args.device,
                max_boundaries=_max_boundaries_for_eval(item.eval_name),
                generation_adapter_name=generation_adapter_name,
            )
            if bundle is None or bundle.activations is None:
                failed += 1
                continue

            bundle.clean_response = clean_response
            bundle.test_response = test_response
            bundle.clean_answer = clean_answer
            bundle.test_answer = test_answer
            bundle.metadata = dict(item.metadata)
            save_bundle(bundle, out_path)
            saved += 1

        print(f"  saved={saved} skipped={skipped} failed={failed}")
        total_saved += saved
        total_skipped += skipped
        total_failed += failed

    print("\nDone.")
    print(f"  total_saved={total_saved}")
    print(f"  total_skipped={total_skipped}")
    print(f"  total_failed={total_failed}")


if __name__ == "__main__":
    main()

