#!/usr/bin/env python3
"""Generate instruction-following sweep dataset from cot_description rollouts.

Assigns each item a random format constraint. Oracle responses are judged
on constraint compliance (0-1) rather than content correctness.

Usage:
    python scripts/generate_if_sweep_dataset.py [--n 100] [--seed 42]
"""

import argparse
import os
import random
import sys

import datasets
from datasets import Dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

HF_ORG = "mats-10-sprint-cs-jb"
SOURCE_REPO = f"{HF_ORG}/cot-oracle-eval-cot-description"
TARGET_REPO = f"{HF_ORG}/cot-oracle-eval-instruction-following"

# (constraint_type, constraint_param, oracle_prompt, constraint_description_for_judge)
CONSTRAINT_TEMPLATES: list[tuple[str, str | None, str, str]] = [
    *[
        (
            "sentence_count", str(n),
            f"Based on the model's activations, describe what the model is reasoning about in exactly {n} sentence{'s' if n > 1 else ''}. Each sentence must end with a period.",
            f"Response must contain exactly {n} complete sentence{'s' if n > 1 else ''}, each ending with a period.",
        )
        for n in [1, 2, 3]
    ],
    *[
        (
            "numbered_list", str(n),
            f"Based on the model's activations, list exactly {n} topics the model is currently reasoning about. Use a numbered list starting with '1.'",
            f"Response must be a numbered list with exactly {n} items (starting with '1.').",
        )
        for n in [2, 3, 4]
    ],
    (
        "format_template", None,
        (
            "Based on the model's activations, respond using exactly this format:\n"
            "TOPIC: <1-3 word topic>\n"
            "ACTIVITY: <one phrase describing what the model is doing>\n"
            "CERTAINTY: <low/medium/high>"
        ),
        "Response must use the format 'TOPIC: ... ACTIVITY: ... CERTAINTY: low/medium/high' with all three fields present.",
    ),
    *[
        (
            "word_limit", str(n),
            f"Based on the model's activations, describe what the model is reasoning about in at most {n} words.",
            f"Response must be at most {n} words long.",
        )
        for n in [10, 20, 30]
    ],
    (
        "prefix_suffix", None,
        (
            "Based on the model's activations, describe what the model is reasoning about. "
            "Your response must start with 'Currently reasoning about:' and end with a confidence "
            "level in parentheses, e.g. '(confidence: high)'."
        ),
        "Response must start with 'Currently reasoning about:' and end with '(confidence: high/medium/low)'.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--target-split", default="test")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {SOURCE_REPO} (split={args.source_split}) ...")
    try:
        ds = datasets.load_dataset(SOURCE_REPO, split=args.source_split)
    except Exception:
        print("  test split not found, falling back to train")
        ds = datasets.load_dataset(SOURCE_REPO, split="train")

    items = list(ds)
    random.shuffle(items)
    items = items[:args.n]
    print(f"Got {len(items)} source items")

    # Assign constraints: cycle through all templates evenly, then shuffle assignment
    n_templates = len(CONSTRAINT_TEMPLATES)
    template_indices = (list(range(n_templates)) * (len(items) // n_templates + 1))[:len(items)]
    random.shuffle(template_indices)

    rows = []
    for item, tmpl_idx in tqdm(zip(items, template_indices), total=len(items), desc="assigning constraints"):
        ctype, cparam, prompt, constraint = CONSTRAINT_TEMPLATES[tmpl_idx]
        row = {
            "task": "instruction_following",
            "prompt": prompt,
            "constraint": constraint,
            "constraint_type": ctype,
            "constraint_param": cparam if cparam is not None else "",
            # cot_text + question are used by prepare_context_ids to tokenize at runtime
            "cot_text": item["cot_text"],
            "question": item.get("question", ""),
            # Keep reference description for qualitative inspection (not used for scoring)
            "target_response": item.get("target_response", ""),
        }
        rows.append(row)

    ds_out = Dataset.from_list(rows)
    print(f"\nDataset stats:")
    from collections import Counter
    for ctype, count in sorted(Counter(r["constraint_type"] for r in rows).items()):
        print(f"  {ctype}: {count}")

    print(f"\nPushing {len(ds_out)} rows to {TARGET_REPO} (split={args.target_split}) ...")
    ds_out.push_to_hub(TARGET_REPO, split=args.target_split)
    print("Done.")


if __name__ == "__main__":
    main()
