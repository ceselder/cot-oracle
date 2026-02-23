"""
Eval: Step Importance Ranking

Tests whether the oracle can identify which reasoning steps in a CoT are most
causally important for the final answer. This is a ranking task — the oracle
sees activations from a full CoT and must pick the top-3 most important steps.

Ground truth comes from pre-computed resampling importance scores:
  - math-rollouts: resampling_importance_kl from Thought Anchors
  - thought-branches: KL suppression scores from authority-bias experiments

Data must be preprocessed first:
    python scripts/download_math_rollouts.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import EvalItem

DATA_DIR = Path("data/evals")


def generate_step_importance_dataset(
    n: int = 50,
    seed: int = 42,
    min_chunks: int = 5,
    min_high_importance: int = 3,
    min_score_variance: float = 0.01,
) -> list[EvalItem]:
    """Generate step importance ranking eval examples.

    Each item presents a full CoT with numbered steps. The oracle must identify
    the top-3 most causally important steps.

    Ground truth is from resampling importance scores (KL divergence).
    """
    random.seed(seed)

    all_problems = []

    # Load math-rollouts data
    mr_path = DATA_DIR / "step_importance_raw.json"
    if mr_path.exists():
        with open(mr_path) as f:
            mr_data = json.load(f)
        for item in mr_data:
            if (item["n_chunks"] >= min_chunks
                    and item["n_high_importance"] >= min_high_importance
                    and item["score_variance"] >= min_score_variance):
                all_problems.append(item)
        print(f"  math-rollouts: {len(mr_data)} total, {len(all_problems)} after filtering")

    # Load thought-branches data (lower variance threshold — KL suppression scale is smaller)
    tb_path = DATA_DIR / "step_importance_faithfulness_raw.json"
    n_before = len(all_problems)
    if tb_path.exists():
        with open(tb_path) as f:
            tb_data = json.load(f)
        for item in tb_data:
            if item["n_chunks"] >= min_chunks:
                all_problems.append(item)
        print(f"  thought-branches: {len(tb_data)} total, {len(all_problems) - n_before} after filtering")

    if not all_problems:
        print("  WARNING: No preprocessed data found. Run scripts/download_math_rollouts.py first.")
        return []

    random.shuffle(all_problems)
    selected = all_problems[:n]

    items = []
    for i, prob in enumerate(selected):
        chunks = prob["cot_chunks"]
        importance = prob["importance_scores"]
        top_k = prob["top_k_indices"]

        # Build numbered CoT for the test prompt
        numbered_steps = "\n".join(
            f"Step {j+1}: {chunk}" for j, chunk in enumerate(chunks)
        )

        test_prompt = (
            f"Solve this math problem step by step. "
            f"Problem: {prob['problem']}\n\n"
            f"Chain of thought:\n{numbered_steps}\n\n"
            f"Final answer: {prob['gt_answer']}"
        )

        # Clean prompt is just the problem (for reference, not used in this eval)
        clean_prompt = f"Solve this math problem step by step. Problem: {prob['problem']}"

        # Ground truth: top-3 step numbers (1-indexed)
        correct_answer = ", ".join(str(idx + 1) for idx in top_k)

        items.append(EvalItem(
            eval_name="step_importance",
            example_id=f"step_importance_{i:04d}",
            clean_prompt=clean_prompt,
            test_prompt=test_prompt,
            correct_answer=correct_answer,
            nudge_answer=None,
            metadata={
                "problem_idx": prob["problem_idx"],
                "source": prob["source"],
                "model": prob["model"],
                "top_k_indices": top_k,  # 0-indexed
                "importance_scores": importance,
                "function_tags": prob.get("function_tags", []),
                "n_chunks": len(chunks),
                "cot_chunks": chunks,
                "score_variance": prob["score_variance"],
                "cue_scores": prob.get("cue_scores"),
            },
        ))

    print(f"  Generated {len(items)} step_importance eval items")
    return items
