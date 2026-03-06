"""Compare the active training mix against the original AO.

Uses the inline comments in configs/train.yaml as the source of truth for:
- task family: computation / pre-postdiction / reconstruction
- supervision: supervised / self-supervised / unsupervised

Color encodes family, hatch encodes supervision.
The figure compares both effective sample counts and approximate oracle-input tokens.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml


CONFIG_PATH = Path("configs/train.yaml")
OUTPUT_PATH = Path("plots/training_mix.png")

FAMILY_COLORS = {
    "computation": "#d96c3f",
    "pre/postdiction": "#4c78a8",
    "reconstruction": "#59a14f",
}
SUPERVISION_HATCHES = {
    "supervised": "",
    "self-supervised": "////",
    "unsupervised": "..",
}
DISPLAY_NAMES = {
    "hint_admission": "Hint Admission",
    "atypical_answer": "Atypical Answer",
    "reasoning_termination": "Reasoning Termination",
    "answer_trajectory": "Answer Trajectory",
    "futurelens": "FutureLens",
    "pastlens": "PastLens",
    "correctness": "Correctness",
    "decorative_cot": "Decorative CoT",
    "resampling_importance": "Resampling Importance",
    "convqa": "ConvQA",
    "compqa": "CompQA",
    "chunked_convqa": "Chunked ConvQA",
    "chunked_compqa": "Chunked CompQA",
    "backtrack_prediction": "Backtrack Prediction",
    "sycophancy": "Sycophancy",
    "sqa": "SQA",
    "truthfulqa_hint": "TruthfulQA Hint",
    "fineweb": "FineWeb Readout",
    "classification": "Classification",
}
AVG_INPUT_TOKENS = {
    "hint_admission": 330,
    "atypical_answer": 206,
    "reasoning_termination": 300,
    "answer_trajectory": 257,
    "futurelens": 344,
    "pastlens": 345,
    "correctness": 329,
    "decorative_cot": 154,
    "resampling_importance": 400,
    "convqa": 181,
    "compqa": 181,
    "chunked_convqa": 387,
    "chunked_compqa": 387,
    "backtrack_prediction": 414,
    "sycophancy": 218,
    "sqa": 180,
    "truthfulqa_hint": 330,
    "classification": 57,
}
FINEWEB_VARIANT_TOKENS = {
    "futurelens_fineweb": 181,
    "pastlens_fineweb": 179,
    "reconstruction_fineweb": 181,
}
ORIGINAL_AO_CLASSIFICATION_SAMPLES = 7 * 6000 * (2 + 1)
# Original AO sample counts are taken from ao_reference/nl_probes/sft.py:
# - PastLens single: 100K
# - PastLens multi: 100K
# - LatentQA: 100K
# - Classification: 7 train groups x 6K examples x (single QA=2 + multi QA=1) = 126K
# Token counts still use approximate average input lengths.
ORIGINAL_AO_ROWS = [
    {"name": "PastLens (single-token)", "samples": 100_000, "avg_tokens": 200, "family": "pre/postdiction", "supervision": "unsupervised"},
    {"name": "PastLens (multi-token)", "samples": 100_000, "avg_tokens": 200, "family": "pre/postdiction", "supervision": "unsupervised"},
    {"name": "LatentQA", "samples": 100_000, "avg_tokens": 181, "family": "reconstruction", "supervision": "self-supervised"},
    {"name": "Classification", "samples": ORIGINAL_AO_CLASSIFICATION_SAMPLES, "avg_tokens": 70, "family": "reconstruction", "supervision": "supervised"},
]


def parse_available_count(comment: str) -> int | None:
    match = re.search(r"~?([\d.]+)\s*K available", comment)
    if match is not None:
        return int(float(match.group(1)) * 1000)
    match = re.search(r"~?([\d,]+)\s*available", comment)
    if match is not None:
        return int(match.group(1).replace(",", ""))
    return None


def parse_properties(comment: str) -> tuple[str, str] | None:
    match = re.search(r"\(([^,]+),\s*([^)]+)\)", comment)
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()


def parse_config_comments(text: str) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    in_tasks = False
    for line in text.splitlines():
        if line == "tasks:":
            in_tasks = True
            continue
        if in_tasks and re.match(r"^[A-Za-z_]", line):
            in_tasks = False
        if in_tasks:
            match = re.match(r"^  ([a-z0-9_]+):(?:\s*#\s*(.*))?$", line)
            if match is not None:
                task_name, comment = match.groups()
                comment = comment or ""
                metadata[task_name] = {"available": parse_available_count(comment), "properties": parse_properties(comment)}
        section_match = re.match(r"^# (FineWeb readout tasks|Classification) \(([^,]+),\s*([^)]+)\)$", line)
        if section_match is not None:
            section_name, family, supervision = section_match.groups()
            section_key = "fineweb" if section_name == "FineWeb readout tasks" else "classification"
            metadata[section_key] = {"properties": (family.strip(), supervision.strip())}
    return metadata


def format_samples(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        value = n / 1_000
        if value.is_integer():
            return f"{int(value)}K"
        return f"{value:.1f}K"
    return str(int(n))


def format_tokens(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        value = n / 1_000
        if value.is_integer():
            return f"{int(value)}K"
        return f"{value:.1f}K"
    return str(int(n))


def load_our_rows(config_path: Path) -> tuple[list[dict[str, object]], str]:
    text = config_path.read_text()
    config = yaml.safe_load(text)
    metadata = parse_config_comments(text)
    rows: list[dict[str, object]] = []

    for task_name, task_cfg in config["tasks"].items():
        if "n" not in task_cfg:
            continue
        if task_cfg["n"] == 0:
            continue
        if task_cfg["n"] == -1:
            assert metadata[task_name]["available"] is not None, f"Missing available count in comment for {task_name}"
            base_n = int(metadata[task_name]["available"])
        else:
            base_n = int(task_cfg["n"])
        epochs = int(task_cfg["epochs"]) if "epochs" in task_cfg else 1
        assert metadata[task_name]["properties"] is not None, f"Missing properties in comment for {task_name}"
        family, supervision = metadata[task_name]["properties"]
        avg_tokens = AVG_INPUT_TOKENS[task_name]
        rows.append({
            "name": DISPLAY_NAMES[task_name],
            "samples": base_n * epochs,
            "avg_tokens": avg_tokens,
            "tokens": base_n * epochs * avg_tokens,
            "family": family,
            "supervision": supervision,
        })

    fineweb_cfg = config["fineweb"]
    if fineweb_cfg["enabled"] and fineweb_cfg["n"] > 0:
        assert metadata["fineweb"]["properties"] is not None
        family, supervision = metadata["fineweb"]["properties"]
        variants = [variant.strip() for variant in fineweb_cfg["variant"].split(",")]
        avg_tokens = sum(FINEWEB_VARIANT_TOKENS[variant] for variant in variants) / len(variants)
        rows.append({
            "name": DISPLAY_NAMES["fineweb"],
            "samples": int(fineweb_cfg["n"]),
            "avg_tokens": avg_tokens,
            "tokens": int(fineweb_cfg["n"]) * avg_tokens,
            "family": family,
            "supervision": supervision,
        })

    classification_cfg = config["classification"]
    if classification_cfg["enabled"] and classification_cfg["n"] > 0:
        assert metadata["classification"]["properties"] is not None
        family, supervision = metadata["classification"]["properties"]
        avg_tokens = AVG_INPUT_TOKENS["classification"]
        rows.append({
            "name": DISPLAY_NAMES["classification"],
            "samples": int(classification_cfg["n"]),
            "avg_tokens": avg_tokens,
            "tokens": int(classification_cfg["n"]) * avg_tokens,
            "family": family,
            "supervision": supervision,
        })

    rows.sort(key=lambda row: (float(row["samples"]), str(row["name"])), reverse=True)
    return rows, config["model"]["name"]


def load_original_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in ORIGINAL_AO_ROWS:
        rows.append({
            "name": row["name"],
            "samples": row["samples"],
            "avg_tokens": row["avg_tokens"],
            "tokens": row["samples"] * row["avg_tokens"],
            "family": row["family"],
            "supervision": row["supervision"],
        })
    return rows


def build_group_layout(groups: list[tuple[str, list[dict[str, object]]]]) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    plotted_rows: list[dict[str, object]] = []
    headers: list[dict[str, object]] = []
    y = 0.0
    for group_name, rows in groups:
        headers.append({"name": group_name, "y": y - 0.75, "rows": rows})
        for row in rows:
            plotted_rows.append({**row, "group": group_name, "y": y})
            y += 1.0
        y += 1.1
    return plotted_rows, headers, y


def add_group_headers(ax: plt.Axes, headers: list[dict[str, object]]) -> None:
    for header in headers:
        rows = header["rows"]
        total_samples = sum(float(row["samples"]) for row in rows)
        total_tokens = sum(float(row["tokens"]) for row in rows)
        label = f"{header['name']}  ({format_samples(total_samples)} samples, {format_tokens(total_tokens)} tokens)"
        ax.text(0.0, float(header["y"]), label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=11, fontweight="bold", color="#111111")


def plot_metric(ax: plt.Axes, plotted_rows: list[dict[str, object]], headers: list[dict[str, object]], metric: str, title: str, show_ylabels: bool, total_height: float) -> None:
    values = [float(row[metric]) for row in plotted_rows]
    xmax = max(values) * 1.24

    for row in plotted_rows:
        ax.barh(
            float(row["y"]),
            float(row[metric]),
            height=0.74,
            color=FAMILY_COLORS[str(row["family"])],
            hatch=SUPERVISION_HATCHES[str(row["supervision"])],
            edgecolor="#333333",
            linewidth=0.8,
        )
        value_label = format_samples(float(row[metric])) if metric == "samples" else format_tokens(float(row[metric]))
        ax.text(float(row[metric]) + xmax * 0.01, float(row["y"]), value_label, va="center", ha="left", fontsize=8.5, color="#222222")

    ax.set_xlim(0, xmax)
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if metric == "samples":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))
        ax.set_xlabel("Effective training examples (n x epochs)", fontsize=10)
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.0f}M"))
        ax.set_xlabel("Approximate oracle input tokens", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(total_height - 0.2, -1.3)

    y_positions = [float(row["y"]) for row in plotted_rows]
    labels = [str(row["name"]) for row in plotted_rows]
    ax.set_yticks(y_positions, labels=labels if show_ylabels else [])
    if show_ylabels:
        ax.tick_params(axis="y", labelsize=10)
    else:
        ax.tick_params(axis="y", labelleft=False)

    add_group_headers(ax, headers)


def main() -> None:
    our_rows, model_name = load_our_rows(CONFIG_PATH)
    original_rows = load_original_rows()
    groups = [("Original AO", original_rows), ("Our Oracle", our_rows)]
    plotted_rows, headers, y_max = build_group_layout(groups)

    fig_height = max(8.0, 0.46 * len(plotted_rows) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_height), layout="constrained")
    fig.patch.set_facecolor("white")

    plot_metric(axes[0], plotted_rows, headers, metric="samples", title="Samples", show_ylabels=True, total_height=y_max)
    plot_metric(axes[1], plotted_rows, headers, metric="tokens", title="Tokens", show_ylabels=False, total_height=y_max)
    fig.suptitle(f"Training Mix Comparison ({model_name})", fontsize=14, fontweight="bold")

    family_handles = [mpatches.Patch(facecolor=FAMILY_COLORS[family], edgecolor="#333333", label=family.replace("/", " / ")) for family in FAMILY_COLORS]
    supervision_handles = [mpatches.Patch(facecolor="white", edgecolor="#333333", hatch=SUPERVISION_HATCHES[label], label=label) for label in SUPERVISION_HATCHES]
    family_legend = fig.legend(handles=family_handles, title="Task family", loc="lower center", bbox_to_anchor=(0.29, -0.06), ncol=3, frameon=False)
    supervision_legend = fig.legend(handles=supervision_handles, title="Supervision", loc="lower center", bbox_to_anchor=(0.76, -0.06), ncol=3, frameon=False)
    fig.add_artist(family_legend)
    fig.add_artist(supervision_legend)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
