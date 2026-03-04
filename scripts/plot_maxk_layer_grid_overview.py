#!/usr/bin/env python3
"""Render all task max_k/layer heatmaps into one overview figure."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def _merge_summaries(summaries: list[dict]) -> dict:
    assert summaries, "Need at least one summary"
    base = {
        "run_dir": summaries[0]["run_dir"],
        "checkpoint": summaries[0]["checkpoint"],
        "model": summaries[0]["model"],
        "max_k_values": list(summaries[0]["max_k_values"]),
        "layer_specs": list(summaries[0]["layer_specs"]),
        "tasks": [],
        "rows": [],
    }
    seen_tasks = set()
    for summary in summaries:
        assert summary["max_k_values"] == base["max_k_values"], "All summaries must share max_k_values"
        assert summary["layer_specs"] == base["layer_specs"], "All summaries must share layer_specs"
        assert summary["checkpoint"] == base["checkpoint"], "All summaries must use the same checkpoint"
        assert summary["model"] == base["model"], "All summaries must use the same model"
        for task in summary["tasks"]:
            assert task not in seen_tasks, f"Duplicate task across summaries: {task}"
            seen_tasks.add(task)
            base["tasks"].append(task)
        base["rows"].extend(summary["rows"])
    return base


def _task_matrices(summary: dict) -> tuple[list[str], list[str], list[str], dict[str, dict]]:
    tasks = list(summary["tasks"])
    max_k_values = [str(value) for value in summary["max_k_values"]]
    layer_specs = list(summary["layer_specs"])
    y_labels = [f'{spec["label"]} [{",".join(str(layer) for layer in spec["layers"])}]' for spec in layer_specs]
    score_map = {task: {} for task in tasks}
    metric_names = {}
    for row in summary["rows"]:
        score_map[row["task"]][(row["layer_label"], row["max_k"])] = float(row["score"])
        metric_names[row["task"]] = row["metric_name"]
    matrices = {}
    for task in tasks:
        matrix = []
        for spec in layer_specs:
            row_vals = [score_map[task][(spec["label"], max_k)] for max_k in summary["max_k_values"]]
            matrix.append(row_vals)
        matrices[task] = {"metric_name": metric_names[task], "matrix": matrix}
    return tasks, max_k_values, y_labels, matrices


def _render(summary: dict, output_path: Path, n_cols: int) -> None:
    tasks, x_labels, y_labels, matrices = _task_matrices(summary)
    n_tasks = len(tasks)
    n_cols = max(1, n_cols)
    n_rows = math.ceil(n_tasks / n_cols)
    fig_width = max(12.0, 5.2 * n_cols)
    fig_height = max(8.0, 3.8 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)
    mappable = None
    for idx, task in enumerate(tasks):
        ax = axes[idx // n_cols][idx % n_cols]
        matrix = matrices[task]["matrix"]
        metric_name = matrices[task]["metric_name"]
        mappable = ax.matshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{task} ({metric_name})", fontsize=10)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("max_k")
        ax.set_ylabel("layers")
        ax.tick_params(axis="both", labelsize=8)
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=8)
    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle("Activation Oracle Max_k x Layer Grid", fontsize=14)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, fraction=0.015, pad=0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single overview figure from a max_k/layer sweep summary.json")
    parser.add_argument("summary_paths", nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-cols", type=int, default=2)
    args = parser.parse_args()
    summary_paths = [Path(path) for path in args.summary_paths]
    summary = _merge_summaries([_load_summary(path) for path in summary_paths])
    output_path = Path(args.output) if args.output else summary_paths[0].parent / "all_tasks_overview.png"
    _render(summary, output_path, args.n_cols)
    print(f"Saved overview figure to {output_path}")


if __name__ == "__main__":
    main()
