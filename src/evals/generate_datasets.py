"""
Generate all eval datasets (CPU only, no GPU needed).
Outputs JSON files to data/evals/{eval_name}.json

Usage:
    python src/evals/generate_datasets.py --n 100 --output-dir data/evals
    python src/evals/generate_datasets.py --evals authority_bias sycophancy
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.common import save_eval_items
from evals.datasets.hinted_mcq import generate_hinted_mcq_dataset
from evals.datasets.sycophancy import generate_sycophancy_dataset
from evals.datasets.authority_bias import generate_authority_bias_dataset
from evals.datasets.decorative_cot import generate_decorative_cot_dataset
from evals.datasets.answer_correctness import generate_answer_correctness_dataset
from evals.datasets.contradictory_comparison import generate_contradictory_comparison_dataset
from evals.datasets.sentence_insertion import generate_sentence_insertion_dataset
from evals.datasets.sycophancy_scruples import generate_sycophancy_scruples_dataset
from evals.datasets.step_importance import generate_step_importance_dataset
from evals.datasets.held_out_cot_reconstruction import generate_held_out_cot_reconstruction_dataset
from evals.datasets.rot13_reconstruction import generate_rot13_reconstruction_dataset
from evals.datasets.logical_leaps import generate_logical_leaps_dataset
from evals.datasets.hint_influence_yesno import generate_hint_influence_yesno_dataset
from evals.datasets.scruples_disagreement import generate_scruples_disagreement_dataset
from evals.datasets.final_answer_kl import generate_final_answer_kl_dataset


ALL_GENERATORS = {
    "hinted_mcq": generate_hinted_mcq_dataset,
    "sycophancy": generate_sycophancy_dataset,
    "authority_bias": generate_authority_bias_dataset,
    "decorative_cot": generate_decorative_cot_dataset,
    "answer_correctness": generate_answer_correctness_dataset,
    "contradictory_comparison": generate_contradictory_comparison_dataset,
    "sentence_insertion": generate_sentence_insertion_dataset,
    "sycophancy_scruples": generate_sycophancy_scruples_dataset,
    "step_importance": generate_step_importance_dataset,
    "held_out_cot_reconstruction": generate_held_out_cot_reconstruction_dataset,
    "rot13_reconstruction": generate_rot13_reconstruction_dataset,
    "logical_leaps": generate_logical_leaps_dataset,
    "hint_influence_yesno": generate_hint_influence_yesno_dataset,
    "scruples_disagreement": generate_scruples_disagreement_dataset,
    "final_answer_kl": generate_final_answer_kl_dataset,
}

# Default item counts per eval (some evals have specific defaults)
DEFAULT_COUNTS = {
    "hinted_mcq": 100,
    "sycophancy": 100,
    "authority_bias": 100,
    "decorative_cot": 20,
    "answer_correctness": 20,
    "contradictory_comparison": 50,
    "sentence_insertion": 100,
    "sycophancy_scruples": 100,
    "step_importance": 50,
    "held_out_cot_reconstruction": 100,
    "rot13_reconstruction": 100,
    "logical_leaps": 100,
    "hint_influence_yesno": 100,
    "scruples_disagreement": 100,
    "final_answer_kl": 100,
}


def main():
    parser = argparse.ArgumentParser(description="Generate eval datasets")
    parser.add_argument("--n", type=int, default=None,
                        help="Examples per eval (overrides per-eval defaults)")
    parser.add_argument("--output-dir", default="data/evals")
    parser.add_argument("--evals", nargs="*", default=None,
                        help="Specific evals to generate (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-path", default="data/cot_corpus/corpus.jsonl",
                        help="Path to corpus for sentence_insertion eval")
    parser.add_argument(
        "--logical-leaps-labels-path",
        default="data/evals/logical_leaps_gemini.jsonl",
        help="Optional JSONL file with Gemini labels for logical_leaps eval",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ALL_GENERATORS
    if args.evals:
        generators = {k: v for k, v in generators.items() if k in args.evals}

    total = 0
    for name, gen_fn in generators.items():
        print(f"Generating {name}...")
        count = args.n if args.n is not None else DEFAULT_COUNTS.get(name, 50)

        # Some generators take extra kwargs
        kwargs = {"n": count, "seed": args.seed}
        if name == "sentence_insertion":
            kwargs["corpus_path"] = args.corpus_path
        if name in ("held_out_cot_reconstruction", "rot13_reconstruction", "logical_leaps"):
            kwargs["corpus_path"] = args.corpus_path
        if name == "logical_leaps":
            kwargs["gemini_labels_path"] = args.logical_leaps_labels_path

        items = gen_fn(**kwargs)
        if items:
            save_eval_items(items, output_dir / f"{name}.json")
            print(f"  {len(items)} items -> {output_dir / f'{name}.json'}")
            total += len(items)
        else:
            print(f"  {name}: no items generated (skipped)")

    print(f"\nTotal: {total} items across {len(generators)} evals")


if __name__ == "__main__":
    main()
