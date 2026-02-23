"""
Eval: Step Counting (Novel Generalization Test)

Given activations from a CoT, the oracle must predict how many distinct
reasoning steps the model used. This task is NEVER trained on — it tests
whether the oracle develops structural understanding of CoT from its
training tasks (context prediction, causal prediction, reconstruction).

Ground truth: number of sentence boundaries in the CoT.

Scoring: absolute error |predicted - actual|. Success if error <= 2.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem


def generate_step_counting_dataset(
    corpus_path: str = "data/cot_corpus_v5/corpus_medium.jsonl",
    n: int = 100,
    seed: int = 456,
) -> list[EvalItem]:
    """Generate step counting eval items from the CoT corpus.

    For each item, the model generates a CoT and the oracle is asked
    to predict how many reasoning steps are in it.
    """
    random.seed(seed)

    corpus = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                n_sentences = entry.get("n_sentences", 0)
                if n_sentences >= 3:  # need at least 3 steps to be interesting
                    corpus.append(entry)

    random.shuffle(corpus)
    items = []

    for i, entry in enumerate(corpus[:n]):
        n_steps = entry["n_sentences"]

        items.append(EvalItem(
            eval_name="step_counting",
            example_id=f"step_count_{i:04d}",
            clean_prompt=entry["question"],
            test_prompt=entry["question"],  # same — no nudge
            correct_answer=str(n_steps),
            nudge_answer=None,
            metadata={
                "n_steps": n_steps,
                "source": entry.get("source", "unknown"),
            },
        ))

    return items


if __name__ == "__main__":
    items = generate_step_counting_dataset()
    out_path = Path(__file__).parent.parent.parent.parent / "data" / "evals" / "step_counting.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([item.to_dict() for item in items], f, indent=2)
    print(f"Wrote {len(items)} items to {out_path}")
