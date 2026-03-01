#!/usr/bin/env python
"""Push baseline results as horizontal-line wandb runs for overlay visualization.

Creates one wandb run per baseline, logging metrics at step=1 and step=100000
to produce horizontal lines on training charts.

Usage:
    python baselines/push_wandb_baselines.py --config configs/train.yaml
    python baselines/push_wandb_baselines.py --config configs/train.yaml --log-dir logs/baselines
"""

import argparse
import json
from pathlib import Path

import yaml


def push_wandb_baselines(log_dir: str, wandb_project: str, wandb_entity: str | None):
    import wandb

    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Log dir {log_dir} does not exist")
        return

    # Discover all baseline results: {baseline_name: {eval_name: result_dict}}
    baseline_results: dict[str, dict[str, dict]] = {}
    for eval_dir in sorted(log_dir.iterdir()):
        if not eval_dir.is_dir():
            continue
        eval_name = eval_dir.name
        for result_file in sorted(eval_dir.glob("*.json")):
            baseline_name = result_file.stem
            result = json.loads(result_file.read_text())
            if result.get("skipped"):
                continue
            baseline_results.setdefault(baseline_name, {})[eval_name] = result

    if not baseline_results:
        print("No baseline results found")
        return

    print(f"Found {len(baseline_results)} baselines:")
    for name, evals in baseline_results.items():
        print(f"  {name}: {len(evals)} evals")

    for baseline_name, eval_results in baseline_results.items():
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"baseline/{baseline_name}",
            config={"baseline": baseline_name},
            reinit=True,
        )

        metrics = {}
        for eval_name, result in eval_results.items():
            m = result.get("metrics", result.get("per_layer", {}))

            # Extract primary score
            is_cls = eval_name.startswith("cls_")
            section = "eval_cls" if is_cls else "eval"

            if isinstance(m, dict) and "accuracy" in m:
                metrics[f"{section}/{eval_name}_acc"] = m["accuracy"]
            elif isinstance(m, dict) and "mean_token_f1" in m:
                metrics[f"{section}/{eval_name}_token_f1"] = m["mean_token_f1"]
            elif isinstance(m, dict) and "mean_spearman" in m:
                metrics[f"{section}/{eval_name}_spearman"] = m["mean_spearman"]
            elif isinstance(m, dict):
                # per_layer/per_config results — use best
                best_score = -1
                for k, v in m.items():
                    if isinstance(v, dict):
                        score = v.get("accuracy", v.get("mean_token_f1", v.get("mean_spearman", -1)))
                        if score > best_score:
                            best_score = score
                if best_score >= 0:
                    metrics[f"{section}/{eval_name}_acc"] = best_score

        # Compute mean accuracy across sections
        eval_accs = [v for k, v in metrics.items() if k.startswith("eval/") and k.endswith("_acc")]
        if eval_accs:
            metrics["eval/mean_acc"] = sum(eval_accs) / len(eval_accs)
        cls_accs = [v for k, v in metrics.items() if k.startswith("eval_cls/") and k.endswith("_acc")]
        if cls_accs:
            metrics["eval_cls/mean_acc"] = sum(cls_accs) / len(cls_accs)

        if metrics:
            # Log at two distant steps to create a horizontal line
            wandb.log(metrics, step=1)
            wandb.log(metrics, step=100000)
            print(f"  {baseline_name}: logged {len(metrics)} metrics")
        else:
            print(f"  {baseline_name}: no metrics to log")

        run.finish()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Push baseline results to wandb")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log dir from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = args.log_dir or cfg["baselines"]["log_dir"]
    wandb_project = cfg["output"].get("wandb_project", "cot_oracle")
    wandb_entity = cfg["output"].get("wandb_entity")

    push_wandb_baselines(log_dir, wandb_project, wandb_entity)


if __name__ == "__main__":
    main()
