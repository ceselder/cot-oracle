"""
Eval: Hinted MCQ (Verbalized) — detect hint usage when the model
explicitly references the hint in its chain of thought.

Data source: mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-verbalized
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem
from evals.datasets.test_splits import _hf_load_dataset


HF_REPO = "mats-10-sprint-cs-jb/cot-oracle-eval-hinted-mcq-truthfulqa-verbalized"


def generate_hinted_mcq_truthfulqa_verbalized_dataset(
    n: int = 100,
    **_kwargs,
) -> list[EvalItem]:
    """Load verbalized hint MCQ eval items from HF test split."""
    ds = _hf_load_dataset(HF_REPO, split="test")

    eval_items = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        r = dict(r)
        prompt = r.get("hinted_prompt", "")
        if not prompt:
            continue

        gt = "independent" if r["label"] == "hint_resisted" else "influenced"

        eval_items.append(EvalItem(
            eval_name="hinted_mcq_truthfulqa_verbalized",
            example_id=f"hint_verb_{i:04d}",
            clean_prompt=r.get("question", prompt),
            test_prompt=prompt,
            correct_answer=gt,
            nudge_answer=None,
            metadata={
                "question_id": r.get("question_id", ""),
                "source": r.get("source", ""),
                "label": r["label"],
                "hint_correct": r.get("hint_correct"),
                "hint_answer": r.get("hint_answer", ""),
                "strategy": r.get("strategy", ""),
                "cot_text": r.get("cot_text", ""),
                "model_answer": r.get("model_answer", ""),
                "metric": "binary_accuracy",
            },
        ))

    print(f"  Loaded {len(eval_items)} verbalized hint MCQ eval items")
    return eval_items


if __name__ == "__main__":
    items = generate_hinted_mcq_truthfulqa_verbalized_dataset()
    print(f"Generated {len(items)} items")
    for item in items[:3]:
        print(f"  {item.example_id}: gt={item.correct_answer}, label={item.metadata['label']}")
