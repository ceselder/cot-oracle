#!/usr/bin/env python3
"""Render task-affinity heatmaps from a saved summary.json."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _matrix_values(row_names: list[str], col_names: list[str], matrix: dict[str, dict[str, float]]) -> list[list[float]]:
    return [[float(matrix[row_name][col_name]) for col_name in col_names] for row_name in row_names]


def _single_row_values(col_names: list[str], row_values: dict[str, float]) -> list[list[float]]:
    return [[float(row_values[col_name]) for col_name in col_names]]


def _max_abs(matrices: list[list[list[float]]]) -> float:
    return max(abs(value) for matrix in matrices for row in matrix for value in row)


def _set_ticks(ax, y_labels: list[str], eval_names: list[str], show_y: bool) -> None:
    ax.set_xticks(range(len(eval_names)))
    ax.set_xticklabels(eval_names, rotation=45, ha="left")
    if show_y:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=8)


def _render(summary: dict, output_path: Path) -> None:
    train_task_names = summary["train_tasks"]
    eval_names = summary["evals"]
    train_support_sizes = summary["train_support_sizes"]
    results_by_size = summary["results_by_train_support_size"]
    cosine_matrices = [_matrix_values(train_task_names, eval_names, results_by_size[str(support_size)]["cosine_matrix"]) for support_size in train_support_sizes]
    uplift_matrices = [_matrix_values(train_task_names, eval_names, results_by_size[str(support_size)]["first_order_uplift_matrix"]) for support_size in train_support_sizes]
    dot_rows = [_single_row_values(eval_names, results_by_size[str(support_size)]["dot_row"]) for support_size in train_support_sizes]
    cosine_abs = _max_abs(cosine_matrices)
    uplift_abs = _max_abs(uplift_matrices)
    dot_abs = _max_abs(dot_rows)
    fig_width = max(15.0, 4.0 + 0.6 * len(eval_names) * 3)
    fig_height = max(3.0 * len(train_support_sizes), len(train_support_sizes) * (1.8 + 0.35 * len(train_task_names)))
    fig, axes = plt.subplots(len(train_support_sizes), 3, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)
    fig.suptitle("Task Affinity Heatmaps", fontsize=13)
    cosine_mappable = None
    uplift_mappable = None
    dot_mappable = None
    for row_idx, support_size in enumerate(train_support_sizes):
        cosine = cosine_matrices[row_idx]
        uplift = uplift_matrices[row_idx]
        dot_row = dot_rows[row_idx]
        cosine_ax = axes[row_idx][0]
        uplift_ax = axes[row_idx][1]
        dot_ax = axes[row_idx][2]
        cosine_mappable = cosine_ax.matshow(cosine, cmap="coolwarm", vmin=-cosine_abs, vmax=cosine_abs)
        cosine_ax.set_title(f"Cosine (n={support_size})", fontsize=10)
        _set_ticks(cosine_ax, train_task_names, eval_names, show_y=True)
        cosine_ax.set_ylabel("train task", fontsize=9)
        uplift_mappable = uplift_ax.matshow(uplift, cmap="coolwarm", vmin=-uplift_abs, vmax=uplift_abs)
        uplift_ax.set_title(f"First-Order Uplift (n={support_size})", fontsize=10)
        _set_ticks(uplift_ax, train_task_names, eval_names, show_y=False)
        dot_mappable = dot_ax.matshow(dot_row, cmap="coolwarm", vmin=-dot_abs, vmax=dot_abs)
        dot_ax.set_title(f"Best Dot Row (n={support_size})", fontsize=10)
        _set_ticks(dot_ax, ["best_dot"], eval_names, show_y=True)
    fig.colorbar(cosine_mappable, ax=axes[:, 0], fraction=0.02, pad=0.02)
    fig.colorbar(uplift_mappable, ax=axes[:, 1], fraction=0.02, pad=0.02)
    fig.colorbar(dot_mappable, ax=axes[:, 2], fraction=0.02, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render heatmaps from a task-affinity summary.json")
    parser.add_argument("summary_path", help="Path to summary.json from task_affinity_matrix.py")
    parser.add_argument("--output", default=None, help="Optional output image path; defaults to <summary_dir>/affinity_heatmaps.png")
    args = parser.parse_args()
    summary_path = Path(args.summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    output_path = Path(args.output) if args.output else summary_path.parent / "affinity_heatmaps.png"
    _render(summary, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
