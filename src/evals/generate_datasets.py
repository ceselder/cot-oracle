"""
Generate all eval datasets (CPU only, no GPU needed).
Outputs JSON files to data/evals/{eval_name}.json

Usage:
    python src/evals/generate_datasets.py --n 100 --output-dir data/evals
    python src/evals/generate_datasets.py --evals sycophancy decorative_cot
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.common import save_eval_items
from evals.datasets.hinted_mcq import generate_hinted_mcq_dataset
from evals.datasets.sycophancy import generate_sycophancy_dataset
from evals.datasets.decorative_cot import generate_decorative_cot_dataset
from evals.datasets.sentence_insertion import generate_sentence_insertion_dataset
from evals.datasets.sycophancy_scruples import generate_sycophancy_scruples_dataset
from evals.datasets.step_importance import generate_step_importance_dataset
from evals.datasets.held_out_cot_reconstruction import generate_held_out_cot_reconstruction_dataset
from evals.datasets.rot13_reconstruction import generate_rot13_reconstruction_dataset
from evals.datasets.reasoning_termination_riya import generate_reasoning_termination_dataset
from evals.datasets.forced_answer_entropy_riya import generate_forced_answer_entropy_dataset
from evals.datasets.sycophancy_v2_riya import generate_sycophancy_v2_dataset
from evals.datasets.atypical_answer_riya import generate_atypical_answer_dataset
from evals.datasets.cybercrime_ood import generate_cybercrime_ood_dataset


ALL_GENERATORS = {
    "hinted_mcq": generate_hinted_mcq_dataset,
    "sycophancy": generate_sycophancy_dataset,
    "decorative_cot": generate_decorative_cot_dataset,
    "sentence_insertion": generate_sentence_insertion_dataset,
    "sycophancy_scruples": generate_sycophancy_scruples_dataset,
    "step_importance": generate_step_importance_dataset,
    "held_out_cot_reconstruction": generate_held_out_cot_reconstruction_dataset,
    "rot13_reconstruction": generate_rot13_reconstruction_dataset,
    "reasoning_termination_riya": generate_reasoning_termination_dataset,
    "forced_answer_entropy_riya": generate_forced_answer_entropy_dataset,
    "sycophancy_v2_riya": generate_sycophancy_v2_dataset,
    "atypical_answer_riya": generate_atypical_answer_dataset,
    "cybercrime_ood": generate_cybercrime_ood_dataset,
}

# Default item counts per eval (some evals have specific defaults)
DEFAULT_COUNTS = {
    "hinted_mcq": 100,
    "sycophancy": 100,
    "decorative_cot": 100,
    "sentence_insertion": 100,
    "sycophancy_scruples": 100,
    "step_importance": 50,
    "held_out_cot_reconstruction": 100,
    "rot13_reconstruction": 100,
    "reasoning_termination_riya": 100,
    "forced_answer_entropy_riya": 100,
    "sycophancy_v2_riya": 100,
    "atypical_answer_riya": 100,
    "cybercrime_ood": 100,
}


def main():
    parser = argparse.ArgumentParser(description="Generate eval datasets")
    parser.add_argument("--n", type=int, default=None,
                        help="Examples per eval (overrides per-eval defaults)")
    parser.add_argument("--output-dir", default="data/evals")
    parser.add_argument("--evals", nargs="*", default=None,
                        help="Specific evals to generate (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-path", default="data/cot_corpus_v5/corpus_medium.jsonl",
                        help="Path to corpus for corpus-dependent evals (rot13_reconstruction)")
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
        if name in ("sentence_insertion", "rot13_reconstruction"):
            kwargs["corpus_path"] = args.corpus_path
        if name == "sycophancy_v2_riya":
            kwargs["precomputed_path"] = str(output_dir / "sycophancy_v2_rollouts_raw.json")
        if name == "atypical_answer_riya":
            kwargs["precomputed_path"] = str(output_dir / "atypical_answer_rollouts_raw.json")

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
