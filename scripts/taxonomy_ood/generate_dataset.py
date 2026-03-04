"""
Generate hierarchical taxonomy dataset for OOD generalization experiments.

Real animal taxonomy with predator/prey labels. Invented individual names only.

Taxonomy: 3 coarse × 4 medium × 2 fine = 24 leaf categories.
Label: predator (+) = "yes", prey (-) = "no" — a natural property the oracle knows from pretraining.

Splits:
  - train: 8 leaves from B1-B2 (land animals), B5-B6 (sea creatures)
  - narrow_ood: 8 leaves from B3-B4 (land animals), B7-B8 (sea creatures) — new mediums, seen coarse
  - broad_ood: 8 leaves from A3 (birds) — entirely new coarse
"""

import json
import random
from pathlib import Path


SEED = 42
ITEMS_PER_LEAF = 200

# Syllable generator: CV patterns + optional final consonant
CONSONANTS = list("bcdfghjklmnpqrstvwxyz")
VOWELS = list("aeiou")

# ── Real animal taxonomy ──
# Each coarse has 4 mediums, each medium has 2 fines.
# Mediums at index 0, 2 are predators (+); mediums at index 1, 3 are prey (-).

TAXONOMY_STRUCTURE = [
    # A1: Land animals
    {
        "coarse": "land animals",
        "mediums": [
            {"name": "big cats",          "label": "yes", "fines": ["lion", "tiger"]},             # B1 — TRAIN
            {"name": "grazing mammals",   "label": "no",  "fines": ["deer", "zebra"]},             # B2 — TRAIN
            {"name": "pack hunters",      "label": "yes", "fines": ["wolf", "hyena"]},             # B3 — NARROW OOD
            {"name": "small herbivores",  "label": "no",  "fines": ["rabbit", "squirrel"]},        # B4 — NARROW OOD
        ],
    },
    # A2: Sea creatures
    {
        "coarse": "sea creatures",
        "mediums": [
            {"name": "sharks",          "label": "yes", "fines": ["great white", "hammerhead"]},   # B5 — TRAIN
            {"name": "shellfish",       "label": "no",  "fines": ["clam", "oyster"]},              # B6 — TRAIN
            {"name": "dolphins",        "label": "yes", "fines": ["orca", "bottlenose dolphin"]},  # B7 — NARROW OOD
            {"name": "small reef fish", "label": "no",  "fines": ["clownfish", "seahorse"]},       # B8 — NARROW OOD
        ],
    },
    # A3: Birds — entirely held out
    {
        "coarse": "birds",
        "mediums": [
            {"name": "raptors",      "label": "yes", "fines": ["eagle", "hawk"]},        # B9  — BROAD OOD
            {"name": "songbirds",    "label": "no",  "fines": ["sparrow", "finch"]},      # B10 — BROAD OOD
            {"name": "owls",         "label": "yes", "fines": ["barn owl", "snowy owl"]}, # B11 — BROAD OOD
            {"name": "ground birds", "label": "no",  "fines": ["quail", "pheasant"]},     # B12 — BROAD OOD
        ],
    },
]

# Split assignment: (coarse_idx, medium_local_idx) -> split
SPLIT_MAP = {
    (0, 0): "train", (0, 1): "train", (0, 2): "narrow_ood", (0, 3): "narrow_ood",
    (1, 0): "train", (1, 1): "train", (1, 2): "narrow_ood", (1, 3): "narrow_ood",
    (2, 0): "broad_ood", (2, 1): "broad_ood", (2, 2): "broad_ood", (2, 3): "broad_ood",
}


def _syllable_generator(rng: random.Random):
    """Yield unique CV(C) syllables."""
    while True:
        c = rng.choice(CONSONANTS)
        v = rng.choice(VOWELS)
        tail = rng.choice(CONSONANTS + [""])
        yield c + v + tail


def generate_names(n: int, rng: random.Random, min_syllables: int = 2, max_syllables: int = 3) -> list[str]:
    """Generate n unique invented names from syllable patterns."""
    gen = _syllable_generator(rng)
    names = set()
    while len(names) < n:
        n_syl = rng.randint(min_syllables, max_syllables)
        name = "".join(next(gen) for _ in range(n_syl))
        name = name.capitalize()
        names.add(name)
    return sorted(names)


def generate_examples(rng: random.Random) -> dict[str, list[dict]]:
    """Generate all examples from the real animal taxonomy with invented individual names."""
    # 24 leaves × 200 items = 4800 invented names
    all_item_names = generate_names(24 * ITEMS_PER_LEAF, rng)
    rng.shuffle(all_item_names)
    item_idx = 0

    splits = {"train": [], "narrow_ood": [], "broad_ood": []}

    for ci, coarse_entry in enumerate(TAXONOMY_STRUCTURE):
        coarse = coarse_entry["coarse"]
        for mi, medium_entry in enumerate(coarse_entry["mediums"]):
            medium = medium_entry["name"]
            label = medium_entry["label"]
            split = SPLIT_MAP[(ci, mi)]

            for fine in medium_entry["fines"]:
                items = all_item_names[item_idx:item_idx + ITEMS_PER_LEAF]
                item_idx += ITEMS_PER_LEAF

                for item_name in items:
                    cot_text = (
                        f"A {item_name} is a {fine}. "
                        f"{fine.capitalize()}s belong to the group of {medium}, "
                        f"which are {coarse}."
                    )
                    example = {
                        "task": "taxonomy_ood",
                        "cot_text": cot_text,
                        "prompt": "Is this animal a predator? Answer Yes or No.",
                        "target_response": label,
                        "label": label,
                        "split": split,
                        "item_name": item_name,
                        "fine": fine,
                        "medium": medium,
                        "coarse": coarse,
                        "medium_index": mi,
                    }
                    splits[split].append(example)

    return splits


def verify_balance(splits: dict[str, list[dict]]):
    """Verify 50/50 class balance in each split."""
    for split_name, examples in splits.items():
        pos = sum(1 for e in examples if e["label"] == "yes")
        neg = sum(1 for e in examples if e["label"] == "no")
        total = len(examples)
        print(f"  {split_name}: {total} examples ({pos} predator, {neg} prey)")
        assert pos == neg, f"Imbalanced split {split_name}: {pos} predator vs {neg} prey"
    print("  All splits balanced!")


def main():
    rng = random.Random(SEED)

    print("Taxonomy structure:")
    for ci, coarse_entry in enumerate(TAXONOMY_STRUCTURE):
        for mi, medium_entry in enumerate(coarse_entry["mediums"]):
            split = SPLIT_MAP[(ci, mi)]
            role = "predator" if medium_entry["label"] == "yes" else "prey"
            print(f"  {coarse_entry['coarse']} > {medium_entry['name']} "
                  f"({role}, fines={medium_entry['fines']}, split={split})")

    splits = generate_examples(rng)
    verify_balance(splits)

    # Save JSONL locally
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    for split_name, examples in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {len(examples)} examples to {path}")

    # Push to HuggingFace as parquet
    from datasets import Dataset
    for split_name, examples in splits.items():
        ds = Dataset.from_list(examples)
        repo_id = "mats-10-sprint-cs-jb/cot-oracle-taxonomy-ood"
        ds.to_parquet(out_dir / f"{split_name}.parquet")
        ds.push_to_hub(repo_id, split=split_name)
        print(f"  Pushed {split_name} to {repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
