#!/usr/bin/env python3
"""Test that all training evals load correctly from HuggingFace.

Verifies:
1. Each eval can be loaded via load_eval_items_hf (pulls from HF)
2. Items have the expected fields (eval_name, example_id, clean_prompt, test_prompt, etc.)
3. The new hinted_mcq_truthfulqa eval works alongside existing evals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evals.common import load_eval_items_hf, EvalItem

TRAINING_EVALS = [
    "hinted_mcq",
    "hinted_mcq_truthfulqa",
    "sycophancy_v2_riya",
    "decorative_cot",
    "sentence_insertion",
    "reasoning_termination_riya",
    "rot13_reconstruction",
]

# Use a temp dir for caching (not data/evals — keep it clean)
CACHE_DIR = Path("/tmp/eval_test_cache")

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for eval_name in TRAINING_EVALS:
        print(f"\n{'='*50}")
        print(f"Testing: {eval_name}")
        try:
            items = load_eval_items_hf(eval_name, eval_dir=CACHE_DIR)
            print(f"  Loaded {len(items)} items")

            # Verify structure
            item = items[0]
            assert isinstance(item, EvalItem), f"Expected EvalItem, got {type(item)}"
            assert item.eval_name == eval_name, f"eval_name mismatch: {item.eval_name}"
            assert item.example_id, "missing example_id"
            assert item.clean_prompt, "missing clean_prompt"
            assert item.test_prompt, "missing test_prompt"

            print(f"  example_id: {item.example_id}")
            print(f"  correct_answer: {item.correct_answer}")
            print(f"  nudge_answer: {item.nudge_answer}")
            print(f"  metadata keys: {list(item.metadata.keys())[:8]}...")
            print(f"  clean_prompt[:80]: {item.clean_prompt[:80]}...")
            print(f"  OK")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print(f"\n{'='*50}")
    if all_ok:
        print("ALL EVALS LOADED SUCCESSFULLY")
    else:
        print("SOME EVALS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
