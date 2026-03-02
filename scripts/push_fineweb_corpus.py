"""Push a fixed subset of FineWeb texts to HuggingFace as an invariant corpus.

Used by futurelens_fineweb, pastlens_fineweb, reconstruction_fineweb tasks.
The corpus stays invariant; readout task processing is cached locally.

Usage:
    python scripts/push_fineweb_corpus.py
    python scripts/push_fineweb_corpus.py --n 200000
"""

import argparse
import os

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from datasets import load_dataset, Dataset
from tqdm.auto import tqdm

HF_ORG = "mats-10-sprint-cs-jb"
HF_REPO = f"{HF_ORG}/fineweb-corpus"
DEFAULT_N = 200_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of FineWeb texts to collect")
    args = parser.parse_args()

    print(f"Streaming {args.n} texts from FineWeb sample-10BT...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    texts = []
    for row in tqdm(ds, total=args.n, desc="Collecting"):
        texts.append(row["text"])
        if len(texts) >= args.n:
            break

    print(f"Collected {len(texts)} texts")

    # Split 90/10 train/test
    n_train = int(0.9 * len(texts))
    train_texts = texts[:n_train]
    test_texts = texts[n_train:]

    print(f"Pushing to {HF_REPO}...")
    print(f"  train: {len(train_texts)} texts")
    print(f"  test: {len(test_texts)} texts")

    Dataset.from_dict({"text": train_texts}).push_to_hub(HF_REPO, split="train")
    Dataset.from_dict({"text": test_texts}).push_to_hub(HF_REPO, split="test")

    print(f"Done! Pushed to {HF_REPO}")


if __name__ == "__main__":
    main()
