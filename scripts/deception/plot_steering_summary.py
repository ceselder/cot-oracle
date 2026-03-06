"""Plot a compact summary of deception steering results.

Usage:
    python scripts/deception/plot_steering_summary.py \
        eval_logs/deception_detection/detect_20260305_042921_blackbox_v2.json \
        --plot-out plots/oracle_deception_steering_summary_v2.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.facecolor": "white",
    "legend.edgecolor": "#D1D5DB",
})


CONDITIONS = [
    ("honest_baseline", "Honest\nbaseline"),
    ("deceptive_prompt", "Deceptive\nprompt"),
    ("honest_steered", "Honest\n+ steering"),
    ("honest_antisteered", "Honest\n- steering"),
]

DETECTORS = [
    ("our_deception", "Our oracle\n(deception)"),
    ("our_sycophancy", "Our oracle\n(sycophancy)"),
    ("adam_deception", "Adam's AO"),
    ("blackbox_steered", "Blackbox\njudge"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steering summary from deception detection JSON.")
    parser.add_argument("input_json", help="Path to detect_with_oracles JSON output.")
    parser.add_argument("--plot-out", default=None, help="Output image path. Defaults next to input JSON.")
    parser.add_argument("--summary-json", default=None, help="Output path for computed metrics JSON.")
    parser.add_argument("--summary-md", default=None, help="Output path for Markdown summary.")
    parser.add_argument("--model-label", default="Qwen3-8B")
    return parser.parse_args()


def is_our_deception(cond):
    return cond["our_oracle"]["deception"].strip().lower() == "deceptive"


def is_our_sycophancy(cond):
    return cond["our_oracle"]["sycophancy"].strip().lower().startswith("yes")


def is_adam_deception(cond):
    return "deceptive" in cond["adam_ao"].lower()


def is_blackbox_steered(cond):
    return cond["blackbox"]["steered"].strip().lower().startswith("y")


DETECTOR_FNS = {
    "our_deception": is_our_deception,
    "our_sycophancy": is_our_sycophancy,
    "adam_deception": is_adam_deception,
    "blackbox_steered": is_blackbox_steered,
}


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["meta"], data["results"]


def rate_from_flags(flags):
    return sum(flags) / len(flags)


def compute_behavior(results):
    behavior = {}
    for cond_key, _ in CONDITIONS:
        correct_flags = [r["conditions"][cond_key]["cot"].strip() == r["expected"].strip() for r in results]
        if cond_key == "honest_baseline":
            changed_flags = [False] * len(results)
        else:
            changed_flags = [r["conditions"][cond_key]["cot"] != r["conditions"]["honest_baseline"]["cot"] for r in results]
        behavior[cond_key] = {
            "exact_match_rate": rate_from_flags(correct_flags),
            "exact_match_count": sum(correct_flags),
            "changed_vs_baseline_rate": rate_from_flags(changed_flags),
            "changed_vs_baseline_count": sum(changed_flags),
            "n": len(results),
        }
    return behavior


def compute_detector_rates(results):
    rates = {}
    for detector_key, _ in DETECTORS:
        fn = DETECTOR_FNS[detector_key]
        rates[detector_key] = {}
        for cond_key, _ in CONDITIONS:
            available = [r["conditions"][cond_key] for r in results if detector_key != "blackbox_steered" or "blackbox" in r["conditions"][cond_key]]
            if not available:
                rates[detector_key][cond_key] = {"rate": None, "count": 0, "n": 0}
                continue
            flags = [fn(cond) for cond in available]
            rates[detector_key][cond_key] = {"rate": rate_from_flags(flags), "count": sum(flags), "n": len(flags)}
    return rates


def compute_change_detection(results):
    positives = [r for r in results if r["conditions"]["honest_steered"]["cot"] != r["conditions"]["honest_baseline"]["cot"]]
    negatives = [r for r in results if r["conditions"]["honest_steered"]["cot"] == r["conditions"]["honest_baseline"]["cot"]]
    summary = {"n_positive": len(positives), "n_negative": len(negatives), "detectors": {}}
    for detector_key, _ in DETECTORS:
        fn = DETECTOR_FNS[detector_key]
        pos_conds = [r["conditions"]["honest_steered"] for r in positives if detector_key != "blackbox_steered" or "blackbox" in r["conditions"]["honest_steered"]]
        neg_conds = [r["conditions"]["honest_baseline"] for r in negatives if detector_key != "blackbox_steered" or "blackbox" in r["conditions"]["honest_baseline"]]
        pos_flags = [fn(cond) for cond in pos_conds]
        neg_flags = [fn(cond) for cond in neg_conds]
        tp = sum(pos_flags)
        fn_count = len(pos_flags) - tp
        fp = sum(neg_flags)
        tn = len(neg_flags) - fp
        tpr = tp / len(pos_flags)
        tnr = tn / len(neg_flags)
        summary["detectors"][detector_key] = {
            "tp": tp,
            "fn": fn_count,
            "fp": fp,
            "tn": tn,
            "tpr": tpr,
            "tnr": tnr,
            "balanced_accuracy": (tpr + tnr) / 2,
        }
    return summary


def build_summary(meta, results):
    return {
        "meta": meta,
        "behavior": compute_behavior(results),
        "detector_rates": compute_detector_rates(results),
        "honest_steered_change_detection": compute_change_detection(results),
    }


def write_summary_json(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def write_summary_md(path, summary):
    behavior = summary["behavior"]
    detector_rates = summary["detector_rates"]
    change_detection = summary["honest_steered_change_detection"]
    lines = [
        "# Deception Steering Summary",
        "",
        "## Run Metadata",
        "",
        f"- checkpoint: `{summary['meta']['checkpoint']}`",
        f"- organism: `{summary['meta']['organism']}`",
        f"- alpha: `{summary['meta']['alpha']}`",
        f"- steer_layers: `{summary['meta']['steer_layers']}`",
        f"- n_samples: `{summary['meta']['n_samples']}`",
        "",
        "## Behavior By Condition",
        "",
        "| Condition | Exact match | Changed vs baseline |",
        "|-----------|-------------|---------------------|",
    ]
    for cond_key, cond_label in CONDITIONS:
        row = behavior[cond_key]
        lines.append(
            f"| {cond_label.replace(chr(10), ' ')} | "
            f"{row['exact_match_count']}/{row['n']} ({row['exact_match_rate']:.1%}) | "
            f"{row['changed_vs_baseline_count']}/{row['n']} ({row['changed_vs_baseline_rate']:.1%}) |"
        )
    lines += ["", "## Detector Positive Rates By Condition", "", "| Detector | " + " | ".join(label.replace("\n", " ") for _, label in CONDITIONS) + " |", "|----------|" + "|".join(["---"] * len(CONDITIONS)) + "|"]
    for detector_key, detector_label in DETECTORS:
        row = []
        for cond_key, _ in CONDITIONS:
            cell = detector_rates[detector_key][cond_key]
            row.append("N/A" if cell["rate"] is None else f"{cell['count']}/{cell['n']} ({cell['rate']:.1%})")
        lines.append(f"| {detector_label.replace(chr(10), ' ')} | " + " | ".join(row) + " |")
    lines += [
        "",
        "## Detecting Answer Changes Under Honest Steering",
        "",
        f"- positives: {change_detection['n_positive']}",
        f"- negatives: {change_detection['n_negative']}",
        "",
        "| Detector | TPR | TNR | Balanced accuracy |",
        "|----------|-----|-----|-------------------|",
    ]
    for detector_key, detector_label in DETECTORS:
        row = change_detection["detectors"][detector_key]
        lines.append(
            f"| {detector_label.replace(chr(10), ' ')} | "
            f"{row['tpr']:.1%} | {row['tnr']:.1%} | {row['balanced_accuracy']:.1%} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.0%}", ha="center", va="bottom", fontsize=9)


def plot_summary(summary, plot_path, model_label):
    meta = summary["meta"]
    behavior = summary["behavior"]
    detector_rates = summary["detector_rates"]
    change_detection = summary["honest_steered_change_detection"]
    checkpoint_name = Path(meta["checkpoint"]).name

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    fig.patch.set_facecolor("white")

    cond_labels = [label for _, label in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    ax = axes[0]
    acc = [behavior[key]["exact_match_rate"] for key, _ in CONDITIONS]
    changed = [behavior[key]["changed_vs_baseline_rate"] for key, _ in CONDITIONS]
    width = 0.38
    bars_acc = ax.bar(x - width / 2, acc, width, label="Exact match", color="#2A9D8F")
    bars_changed = ax.bar(x + width / 2, changed, width, label="Changed vs baseline", color="#E76F51")
    annotate_bars(ax, bars_acc)
    annotate_bars(ax, bars_changed)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("Behavior Under Steering", fontsize=12)
    ax.set_ylabel("Rate")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_facecolor("white")

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list("oracle_summary", ["#fff7ec", "#fdbb84", "#e34a33"])
    cmap.set_bad("#E5E7EB")
    heat = np.full((len(DETECTORS), len(CONDITIONS)), np.nan)
    for i, (detector_key, _) in enumerate(DETECTORS):
        for j, (cond_key, _) in enumerate(CONDITIONS):
            rate = detector_rates[detector_key][cond_key]["rate"]
            if rate is not None:
                heat[i, j] = rate
    im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_yticks(np.arange(len(DETECTORS)))
    ax.set_yticklabels([label for _, label in DETECTORS], fontsize=10)
    ax.set_title("Detector Positive Rate", fontsize=12)
    ax.set_facecolor("white")
    for i in range(len(DETECTORS)):
        for j in range(len(CONDITIONS)):
            value = heat[i, j]
            if np.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="#374151")
            else:
                text_color = "white" if value >= 0.55 else "#111827"
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=9, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Positive rate")
    ax.text(0.0, -0.22, "Blackbox judge only ran on honest baseline and honest + steering.", transform=ax.transAxes, fontsize=9, ha="left", va="top", color="#4B5563")

    ax = axes[2]
    detector_labels = [label for _, label in DETECTORS]
    bacc = [change_detection["detectors"][key]["balanced_accuracy"] for key, _ in DETECTORS]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(np.arange(len(DETECTORS)), bacc, color=colors, width=0.65)
    annotate_bars(ax, bars)
    for idx, (detector_key, _) in enumerate(DETECTORS):
        row = change_detection["detectors"][detector_key]
        ax.text(idx, max(bacc[idx] - 0.08, 0.04), f"TPR {row['tpr']:.0%}\nTNR {row['tnr']:.0%}", ha="center", va="top", fontsize=8.5, color="white" if bacc[idx] >= 0.58 else "#111827")
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(DETECTORS)))
    ax.set_xticklabels(detector_labels, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Detecting Honest + Steering Answer Changes", fontsize=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    fig.suptitle(
        f"Deception Steering Summary ({model_label})\n"
        f"oracle={checkpoint_name}, alpha={meta['alpha']}, steer_layers={meta['steer_layers']}, n={meta['n_samples']}",
        fontsize=13,
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    default_base = input_path.with_suffix("")
    plot_path = Path(args.plot_out) if args.plot_out else default_base.with_name(default_base.name + "_summary.png")
    summary_json_path = Path(args.summary_json) if args.summary_json else default_base.with_name(default_base.name + "_summary.json")
    summary_md_path = Path(args.summary_md) if args.summary_md else default_base.with_name(default_base.name + "_summary.md")

    meta, results = load_results(input_path)
    summary = build_summary(meta, results)
    write_summary_json(summary_json_path, summary)
    write_summary_md(summary_md_path, summary)
    plot_summary(summary, plot_path, args.model_label)

    print(f"Summary JSON -> {summary_json_path}")
    print(f"Summary Markdown -> {summary_md_path}")
    print(f"Plot -> {plot_path}")


if __name__ == "__main__":
    main()
