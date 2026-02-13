"""
Post-hoc wandb logging for eval results.

Usage:
    WANDB_API_KEY=... python3 src/evals/log_to_wandb.py \
        --results-dir data/eval_results/baseline \
        --run-name baseline_8B_bf16 \
        --tags baseline Qwen3-8B
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.common import load_eval_items
from evals.score_oracle import score_eval, EVAL_PARSING


def load_completed_items(path):
    """Load CompletedEvalItem dicts from JSON."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--project", default="cot_oracle")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--tags", nargs="*", default=[])
    args = parser.parse_args()

    import wandb

    results_dir = Path(args.results_dir)

    # Collect all results
    all_metrics = {}

    # Check unfaithfulness results
    unfaith_dir = results_dir / "unfaithfulness"
    if unfaith_dir.exists():
        for f in sorted(unfaith_dir.glob("*_completed.json")):
            eval_name = f.stem.replace("_completed", "")
            items = load_completed_items(f)

            # Ground truth distribution
            from collections import Counter
            labels = Counter(item.get("ground_truth_label", "unknown") for item in items)

            # Score if parsing config exists
            parsing_config = EVAL_PARSING.get(eval_name)
            if parsing_config:
                # Convert dicts to simple namespace for score_eval
                from dataclasses import dataclass, field

                @dataclass
                class SimpleItem:
                    eval_name: str = ""
                    example_id: str = ""
                    ground_truth_label: str = ""
                    oracle_response: str = ""
                    clean_answer: str = ""
                    test_answer: str = ""
                    nudge_answer: str = ""
                    metadata: dict = field(default_factory=dict)

                scored_items = []
                for item in items:
                    scored_items.append(SimpleItem(
                        eval_name=item.get("eval_name", ""),
                        example_id=item.get("example_id", ""),
                        ground_truth_label=item.get("ground_truth_label", ""),
                        oracle_response=item.get("oracle_response", ""),
                        clean_answer=item.get("clean_answer", ""),
                        test_answer=item.get("test_answer", ""),
                        nudge_answer=item.get("nudge_answer", ""),
                        metadata=item.get("metadata", {}),
                    ))

                metrics = score_eval(eval_name, scored_items, parsing_config)
                if metrics:
                    all_metrics[eval_name] = metrics

            all_metrics[f"{eval_name}_gt_distribution"] = dict(labels)

    # Check regression results
    regression_dir = results_dir / "regression"
    if regression_dir.exists():
        for f in regression_dir.glob("*.json"):
            with open(f) as fh:
                all_metrics[f"regression/{f.stem}"] = json.load(fh)

    # Log to wandb
    wandb.init(
        project=args.project,
        name=args.run_name,
        tags=args.tags,
        config={"eval_type": "baseline"},
    )

    for key, value in all_metrics.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                if isinstance(subval, (int, float)):
                    wandb.log({f"{key}/{subkey}": subval})
        elif isinstance(value, (int, float)):
            wandb.log({key: value})

    # Also log a summary table
    table_data = []
    for eval_name in EVAL_PARSING:
        if eval_name in all_metrics:
            m = all_metrics[eval_name]
            table_data.append([eval_name, m.get("accuracy", 0), m.get("n_items", 0)])

    if table_data:
        table = wandb.Table(columns=["eval", "accuracy", "n_items"], data=table_data)
        wandb.log({"eval_summary": table})

    wandb.finish()
    print(f"Logged {len(all_metrics)} metric groups to wandb/{args.project}/{args.run_name}")

    # Also print summary
    print("\nBaseline Results Summary:")
    print(f"{'Eval':<30} {'Accuracy':>10} {'N':>5}")
    print("-" * 50)
    for eval_name in EVAL_PARSING:
        if eval_name in all_metrics:
            m = all_metrics[eval_name]
            print(f"{eval_name:<30} {m.get('accuracy', 0):>10.3f} {m.get('n_items', 0):>5}")


if __name__ == "__main__":
    main()
