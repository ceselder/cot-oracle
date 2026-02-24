"""Precompute eval activation bundles for faster repeated oracle runs.

Also pre-labels decorative_cot items (runs model N times with/without CoT)
and saves labels back to the dataset JSON.

Usage:
    python3 src/evals/precompute_activations.py \
        --eval-dir data/evals \
        --output-dir data/eval_precomputed \
        --model Qwen/Qwen3-8B

    # Only specific evals:
    python3 src/evals/precompute_activations.py \
        --evals rot13_reconstruction decorative_cot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.ao import (
    load_model_with_ao,
    load_extra_adapter,
    generate_cot,
    generate_direct_answer,
    layer_percent_to_layer,
)
from evals.activation_cache import extract_activation_bundle, save_bundle, cache_path
from evals.common import load_eval_items, EvalItem, extract_numerical_answer, ci_label
from evals.run_evals import _extract_answer


ROT13_ADAPTER_HF = "ceselder/rot13-qwen3-8b-lora"
ROT13_ADAPTER_NAME = "rot13"


def _cot_for_activations(item: EvalItem, test_response: str) -> str:
    if item.eval_name == "sentence_insertion":
        return str(item.metadata.get("spliced_cot_text", ""))
    if item.eval_name == "rot13_reconstruction":
        return str(item.metadata.get("rot13_cot", ""))
    return test_response


def _max_boundaries_for_eval(eval_name: str) -> int:
    if eval_name == "sentence_insertion":
        return 30
    if eval_name == "rot13_reconstruction":
        return 20
    return 10


def _label_decorative_cot(
    model, tokenizer, item: EvalItem, device: str, n_runs: int = 10,
    temperature: float = 0.6,
) -> tuple[str, float, float, str]:
    """Run model N times with/without CoT to determine decorative vs load-bearing.

    Uses temperature sampling so different runs can produce different answers,
    enabling meaningful confidence intervals. First run is greedy (representative).

    Returns (label, with_cot_acc, without_cot_acc, representative_cot).
    """
    with_cot_correct = 0
    without_cot_correct = 0
    representative_cot = ""

    for i in range(n_runs):
        # First run is greedy for representative CoT; rest use sampling
        temp = None if i == 0 else temperature
        cot_response = generate_cot(
            model, tokenizer, item.test_prompt,
            max_new_tokens=512, device=device, adapter_name=None,
            temperature=temp,
        )
        direct_response = generate_direct_answer(
            model, tokenizer, item.clean_prompt,
            device=device, adapter_name=None,
            temperature=temp,
        )
        if i == 0:
            representative_cot = cot_response

        if extract_numerical_answer(cot_response) == item.correct_answer:
            with_cot_correct += 1
        if extract_numerical_answer(direct_response) == item.correct_answer:
            without_cot_correct += 1

    with_cot_acc = with_cot_correct / n_runs
    without_cot_acc = without_cot_correct / n_runs

    label = ci_label(with_cot_correct, n_runs, without_cot_correct, n_runs)

    return label, with_cot_acc, without_cot_acc, representative_cot


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
    parser.add_argument("--label-runs", type=int, default=10, help="Runs per item for decorative_cot labeling.")
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

    # Track decorative_cot labels to write back to JSON
    decorative_labels = {}

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

            if item.eval_name == "rot13_reconstruction":
                # ROT13 model organism: generate rot13 CoT with rot13 adapter,
                # normal CoT with base model. Extract activations from rot13 CoT.
                if not hasattr(main, '_rot13_loaded'):
                    load_extra_adapter(model, ROT13_ADAPTER_HF, adapter_name=ROT13_ADAPTER_NAME)
                    main._rot13_loaded = True
                # rot13 CoT (test_response) — the model organism's obfuscated reasoning
                test_response = generate_cot(
                    model, tokenizer, item.test_prompt,
                    max_new_tokens=1024, device=args.device,
                    adapter_name=ROT13_ADAPTER_NAME,
                )
                # normal CoT (clean_response) — ground truth for scoring
                clean_response = generate_cot(
                    model, tokenizer, item.test_prompt,
                    max_new_tokens=1024, device=args.device,
                    adapter_name=None,
                )
                cot_for_acts = test_response  # activations from the rot13 CoT
            elif item.eval_name == "sentence_insertion":
                test_response = str(item.metadata.get("spliced_cot_text", ""))
                cot_for_acts = test_response
            elif item.eval_name == "decorative_cot":
                # Pre-label AND generate representative CoT
                label, with_cot_acc, without_cot_acc, representative_cot = \
                    _label_decorative_cot(
                        model, tokenizer, item, args.device,
                        n_runs=args.label_runs,
                    )
                test_response = representative_cot
                cot_for_acts = test_response
                decorative_labels[item.example_id] = {
                    "decorative_label": label,
                    "with_cot_acc": with_cot_acc,
                    "without_cot_acc": without_cot_acc,
                    "representative_cot": representative_cot[:2000],
                }
                print(f"    {item.example_id}: {label} "
                      f"(cot={with_cot_acc:.1f}, direct={without_cot_acc:.1f})")
            else:
                # Standard binary evals: generate clean + test responses
                clean_response = generate_cot(
                    model, tokenizer, item.clean_prompt,
                    max_new_tokens=512, device=args.device,
                    adapter_name=generation_adapter_name,
                )
                test_response = generate_cot(
                    model, tokenizer, item.test_prompt,
                    max_new_tokens=512, device=args.device,
                    adapter_name=generation_adapter_name,
                )
                clean_answer = _extract_answer(clean_response, item.eval_name)
                test_answer = _extract_answer(test_response, item.eval_name)
                cot_for_acts = test_response

            bundle = extract_activation_bundle(
                model,
                tokenizer,
                eval_name=item.eval_name,
                example_id=item.example_id,
                prompt=item.test_prompt,
                cot_text=cot_for_acts,
                act_layer=act_layer,
                device=args.device,
                max_boundaries=_max_boundaries_for_eval(item.eval_name),
                generation_adapter_name=None,  # always extract with base model
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

    # Write decorative_cot labels back to JSON
    if decorative_labels:
        dec_file = eval_dir / "decorative_cot.json"
        if dec_file.exists():
            with open(dec_file) as f:
                dec_data = json.load(f)
            updated = 0
            for entry in dec_data:
                eid = entry.get("example_id")
                if eid in decorative_labels:
                    entry["metadata"].update(decorative_labels[eid])
                    entry["ground_truth"] = decorative_labels[eid]["decorative_label"]
                    updated += 1
            with open(dec_file, "w") as f:
                json.dump(dec_data, f, indent=2)
            from collections import Counter
            label_dist = Counter(
                decorative_labels[eid]["decorative_label"]
                for eid in decorative_labels
            )
            print(f"\nUpdated decorative_cot.json: {updated} items labeled")
            print(f"  Label distribution: {dict(label_dist)}")

    print("\nDone.")
    print(f"  total_saved={total_saved}")
    print(f"  total_skipped={total_skipped}")
    print(f"  total_failed={total_failed}")


if __name__ == "__main__":
    main()

