#!/usr/bin/env python3
"""Compute random-guess and majority-class baselines for all eval tasks and log to wandb.

Creates TWO wandb runs (random-baseline, majority-baseline), each logging
under the same eval/{task_name} metric keys as training runs. This way the
baselines overlay directly on the same chart panels without creating extra columns.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter
from data_loading import load_task_data
from tasks import TASKS, TaskDef, ScoringMode
from eval_loop import TASK_PARSERS

import re
import wandb

MAX_SAMPLES = 1_000_000


def get_target_labels(task_name: str, task_def: TaskDef, targets: list[str]) -> list[str] | None:
    """Parse targets into discrete labels. Returns None if task isn't classification-like."""
    if task_name == "answer_trajectory":
        return None

    parser = TASK_PARSERS.get(task_name)
    if parser is not None:
        labels = []
        for t in targets:
            parsed = parser(t)
            if parsed:
                labels.append(parsed["label"])
        return labels if labels else None

    if task_def.scoring == ScoringMode.STEP_ACCURACY:
        labels = []
        for t in targets:
            t_lower = t.lower().strip()
            if t_lower in ("none", "no insertion", "-1"):
                labels.append("none")
            else:
                nums = re.findall(r'\b(\d+)\b', t_lower)
                if nums:
                    labels.append(nums[0])
        return labels if labels else None

    return None


def compute_baselines() -> dict[str, dict]:
    """Returns {task_name: {random: float, majority: float, ...}}."""
    max_items = 200
    skip_tasks = {"futurelens", "rot13_reconstruction"}
    results = {}

    for task_name, task_def in TASKS.items():
        if task_name in skip_tasks or task_def.needs_rot13_adapter:
            continue

        try:
            data = load_task_data(task_name, split="test", n=max_items, shuffle=False)
        except Exception:
            data = []
        if not data:
            try:
                data = load_task_data(task_name, split="train", n=max_items, shuffle=False)
            except Exception:
                data = []
        if task_def.eval_exclude_types:
            data = [d for d in data if d.get("datapoint_type") not in task_def.eval_exclude_types]

        data = [d for d in data if "target_response" in d]
        if not data:
            continue

        targets = [d["target_response"] for d in data]
        labels = get_target_labels(task_name, task_def, targets)

        # Fallback: use raw 'label' field if parser fails on targets
        if labels is None and task_name in TASK_PARSERS:
            raw_labels = [str(d["label"]) for d in data if "label" in d]
            if raw_labels:
                labels = raw_labels

        if labels is None:
            continue

        counts = Counter(labels)
        n_classes = len(counts)
        majority_count = max(counts.values())
        n_total = len(labels)
        majority_label = counts.most_common(1)[0][0]

        results[task_name] = {
            "random": 1.0 / n_classes,
            "majority": majority_count / n_total,
            "majority_label": majority_label,
            "n_classes": n_classes,
            "distribution": dict(counts.most_common()),
        }
        print(f"  {task_name}: {n_total} targets, {n_classes} classes, "
              f"majority={majority_label} ({results[task_name]['majority']:.3f}), "
              f"random={results[task_name]['random']:.3f}")

    return results


def log_baseline_run(name: str, tag: str, metrics: dict[str, float]):
    """Create a single wandb run with eval/{task} metrics logged as a horizontal line."""
    wandb.init(
        project="cot_oracle", entity="japhba-personal",
        name=name, job_type="baseline", tags=["baseline", tag],
    )
    wandb.define_metric("train/samples_seen")
    wandb.define_metric("*", step_metric="train/samples_seen")

    for step in [0, MAX_SAMPLES]:
        wandb.log({**metrics, "train/samples_seen": step}, step=step)

    wandb.finish()


def main():
    baselines = compute_baselines()

    # Run 1: random baseline
    random_metrics = {f"eval/{t}": b["random"] for t, b in baselines.items()}
    log_baseline_run("random-baseline", "random", random_metrics)
    print(f"\n  Logged random baseline ({len(random_metrics)} tasks)")

    # Run 2: majority baseline
    majority_metrics = {f"eval/{t}": b["majority"] for t, b in baselines.items()}
    log_baseline_run("majority-baseline", "majority", majority_metrics)
    print(f"  Logged majority baseline ({len(majority_metrics)} tasks)")

    print("\nDone.")


if __name__ == "__main__":
    main()
