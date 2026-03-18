#!/usr/bin/env python
"""Push baseline results as horizontal-line wandb runs for overlay visualization.

Reads from EvalCache (SQLite). Creates one wandb run per method, logging metrics
at step=1 and step=100000 to produce horizontal lines on training charts.

Usage:
    python baselines/push_wandb_baselines.py --config configs/train.yaml
    python baselines/push_wandb_baselines.py --db data/comprehensive_eval/eval_cache.db
"""

import argparse
import sys
from pathlib import Path

import yaml

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval_cache import EvalCache
from tasks import TASKS, ScoringMode


def push_wandb_baselines(db_path: str, wandb_project: str, wandb_entity: str | None, run_id: str | None = None):
    import wandb

    cache = EvalCache(db_path)

    if run_id is None:
        rows = cache._conn.execute("SELECT run_id FROM eval_runs ORDER BY created_at DESC LIMIT 1").fetchall()
        if not rows:
            print("No runs found in DB")
            return
        run_id = rows[0][0]
        print(f"Using run_id: {run_id}")

    results = cache.get_all_method_results(run_id)
    if not results:
        print("No results found")
        return

    # Collect {method: {task: score_data}}
    method_data: dict[str, dict[str, dict]] = {}
    for task_name, task_methods in results.items():
        for method_name, data in task_methods.items():
            method_data.setdefault(method_name, {})[task_name] = data

    print(f"Found {len(method_data)} methods:")
    for name, tasks in method_data.items():
        print(f"  {name}: {len(tasks)} tasks")

    for method_name, task_results in method_data.items():
        run = wandb.init(
            project=wandb_project, entity=wandb_entity,
            name=f"baseline/{method_name}",
            config={"baseline": method_name},
            reinit=True,
        )

        metrics = {}
        for task_name, data in task_results.items():
            score = data.get("primary_score")
            if score is None or (isinstance(score, float) and score != score):
                continue

            is_cls = task_name.startswith("cls_")
            section = "eval_cls" if is_cls else "eval"

            task_def = TASKS.get(task_name)
            if task_def and task_def.scoring == ScoringMode.TOKEN_F1:
                metrics[f"{section}/{task_name}_token_f1"] = score
            else:
                metrics[f"{section}/{task_name}_acc"] = score

        # Compute mean accuracy per section
        eval_accs = [v for k, v in metrics.items() if k.startswith("eval/") and k.endswith("_acc")]
        if eval_accs:
            metrics["eval/mean_acc"] = sum(eval_accs) / len(eval_accs)
        cls_accs = [v for k, v in metrics.items() if k.startswith("eval_cls/") and k.endswith("_acc")]
        if cls_accs:
            metrics["eval_cls/mean_acc"] = sum(cls_accs) / len(cls_accs)

        if metrics:
            wandb.log(metrics, step=1)
            wandb.log(metrics, step=100000)
            print(f"  {method_name}: logged {len(metrics)} metrics")
        else:
            print(f"  {method_name}: no metrics to log")

        run.finish()

    cache.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Push baseline results to wandb")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml")
    parser.add_argument("--db", type=str, default=None, help="Path to eval_cache.db (default: data/comprehensive_eval/eval_cache.db)")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run_id (default: most recent)")
    args = parser.parse_args()

    with open(args.train_config) as f:
        cfg = yaml.safe_load(f)

    db_path = args.db or "data/comprehensive_eval/eval_cache.db"
    wandb_project = cfg["output"].get("wandb_project", "cot_oracle")
    wandb_entity = cfg["output"].get("wandb_entity")

    push_wandb_baselines(db_path, wandb_project, wandb_entity, args.run_id)


if __name__ == "__main__":
    main()
