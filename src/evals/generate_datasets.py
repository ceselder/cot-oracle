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


ALL_GENERATORS = {
    "hinted_mcq": generate_hinted_mcq_dataset,
    "sycophancy": generate_sycophancy_dataset,
    "authority_bias": generate_authority_bias_dataset,
    "decorative_cot": generate_decorative_cot_dataset,
    "answer_correctness": generate_answer_correctness_dataset,
    "contradictory_comparison": generate_contradictory_comparison_dataset,
    "sentence_insertion": generate_sentence_insertion_dataset,
    "sycophancy_scruples": generate_sycophancy_scruples_dataset,
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
