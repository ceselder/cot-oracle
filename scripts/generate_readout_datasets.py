"""Pre-generate readout datasets and cache to $CACHE_DIR/readout_cache/.

Populates the disk cache so that training loads instantly instead of
regenerating from corpus on every run.

Usage:
    python scripts/generate_readout_datasets.py
    python scripts/generate_readout_datasets.py --tasks futurelens_fineweb pastlens_fineweb
    python scripts/generate_readout_datasets.py --tasks futurelens pastlens reconstruction
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer

from data_loading import load_readout_task_data, _COT_READOUT_TASKS, _FINEWEB_READOUT_TASKS

MODEL = "Qwen/Qwen3-0.6B"
STRIDE = "poisson"
LAYERS = [7, 14, 21]  # 25/50/75% of Qwen3-0.6B (28 layers)

N_TRAIN = 30000
N_TEST = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to generate (default: all 6)")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    args = parser.parse_args()

    all_tasks = sorted(_COT_READOUT_TASKS | _FINEWEB_READOUT_TASKS)
    tasks = args.tasks or all_tasks

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cache_dir = Path(os.environ["CACHE_DIR"]) / "readout_cache"
    print(f"Cache directory: {cache_dir}")

    for task_name in tasks:
        for split in args.splits:
            n = args.n_train if split == "train" else args.n_test
            seed = 42 if split == "train" else 99

            print(f"\n{'='*60}")
            print(f"Generating {task_name} ({split}, n={n}, seed={seed})")

            load_readout_task_data(
                task_name=task_name,
                tokenizer=tokenizer,
                n=n,
                split=split,
                stride=STRIDE,
                layers=args.layers,
                seed=seed,
                model_name=args.model,
            )

    print(f"\nAll done! Cached {len(tasks)} datasets to {cache_dir}")


if __name__ == "__main__":
    main()
