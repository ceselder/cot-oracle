"""
Baselines runner: evaluate 5 baseline methods on unfaithfulness eval tasks.

Usage:
    python baselines/run.py --config configs/train.yaml
    python baselines/run.py --config configs/train.yaml --evals hinted_mcq decorative_cot
    python baselines/run.py --config configs/train.yaml --baselines linear_probe llm_monitor
    python baselines/run.py --config configs/train.yaml --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Ensure baselines/ and src/ are importable
_DIR = Path(__file__).resolve().parent
_SRC = _DIR.parent / "src"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv(Path.home() / ".env")

import torch
from cot_utils import layer_percent_to_layer
from core.ao import load_model_with_ao

from shared import load_baseline_inputs, log_results
from scoring import EVAL_TYPES
from linear_probe import run_linear_probe
from attention_probe import run_attention_probe
from original_ao import run_original_ao
from llm_monitor import run_llm_monitor
from patchscopes import run_patchscopes


# Which baselines can handle which eval types
BASELINE_COMPATIBILITY = {
    "linear_probe":     {"binary", "ranking"},
    "attention_probe":  {"binary", "ranking"},
    "original_ao":      {"binary", "generation"},
    "llm_monitor":      {"binary", "generation", "ranking"},
    "patchscopes":      {"binary", "generation"},
}

ALL_BASELINES = list(BASELINE_COMPATIBILITY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines on eval tasks")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--evals", nargs="+", default=None, help="Eval names to run (default: all from config)")
    parser.add_argument("--baselines", nargs="+", default=None, help="Baselines to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bcfg = cfg["baselines"]
    model_name = cfg["model"]["name"]
    output_dir = Path(bcfg["output_dir"])
    log_dir = Path(bcfg["log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_names = args.evals or bcfg["evals"]
    baselines_to_run = args.baselines or ALL_BASELINES

    # Init wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg["output"].get("wandb_project", "cot_oracle"),
            entity=cfg["output"].get("wandb_entity"),
            name="baselines",
            config={"baselines": bcfg, "evals": eval_names, "baselines_run": baselines_to_run},
        )

    # Compute which layers to extract (union of all baselines' needs)
    layers_needed = set()
    if "linear_probe" in baselines_to_run:
        layers_needed.update(bcfg["linear_probe"]["layers"])
    if "attention_probe" in baselines_to_run:
        layers_needed.update(range(bcfg["attention_probe"]["n_layers"]))
    if "original_ao" in baselines_to_run:
        layers_needed.add(layer_percent_to_layer(model_name, 50))
    if "patchscopes" in baselines_to_run:
        layers_needed.update(bcfg["patchscopes"]["source_layers"])
    layers = sorted(layers_needed) or [layer_percent_to_layer(model_name, 50)]

    # Load model
    print(f"Loading model {model_name}...")
    model, tokenizer = load_model_with_ao(model_name, device=args.device)

    # Collect results for comparison table
    all_results: dict[str, dict[str, dict]] = {}  # {eval_name: {baseline_name: metrics}}

    for eval_name in eval_names:
        print(f"\n{'='*60}\nEval: {eval_name}\n{'='*60}")
        eval_type = EVAL_TYPES.get(eval_name)
        if eval_type is None:
            print(f"  Unknown eval type for {eval_name}, skipping")
            continue

        # Load data once for all baselines
        inputs = load_baseline_inputs(
            eval_name, model, tokenizer,
            layers=layers, stride=bcfg["stride"],
            device=args.device,
            eval_dir=Path(cfg["eval"]["eval_dir"]) if "eval" in cfg else None,
        )
        if not inputs:
            print(f"  No usable items for {eval_name}, skipping")
            continue

        eval_results = {}

        for baseline_name in baselines_to_run:
            if eval_type not in BASELINE_COMPATIBILITY[baseline_name]:
                print(f"  Skipping {baseline_name} (incompatible with {eval_type})")
                continue

            print(f"\n  Running {baseline_name} on {eval_name} ({len(inputs)} items)...")

            if baseline_name == "linear_probe":
                lp_cfg = bcfg["linear_probe"]
                results = run_linear_probe(
                    inputs, layers=lp_cfg["layers"], k_folds=lp_cfg["k_folds"],
                    lr=lp_cfg["lr"], epochs=lp_cfg["epochs"],
                    weight_decay=lp_cfg["weight_decay"], device=args.device,
                )

            elif baseline_name == "attention_probe":
                ap_cfg = bcfg["attention_probe"]
                results = run_attention_probe(
                    inputs, n_layers=ap_cfg["n_layers"], k_folds=ap_cfg["k_folds"],
                    n_heads=ap_cfg["n_heads"], hidden_dim=ap_cfg["hidden_dim"],
                    lr=ap_cfg["lr"], epochs=ap_cfg["epochs"],
                    patience=ap_cfg["patience"], device=args.device,
                )

            elif baseline_name == "original_ao":
                ao_cfg = bcfg["original_ao"]
                results = run_original_ao(
                    inputs, model, tokenizer,
                    checkpoint=ao_cfg["checkpoint"], model_name=model_name,
                    device=args.device,
                )

            elif baseline_name == "llm_monitor":
                lm_cfg = bcfg["llm_monitor"]
                api_key = os.environ["OPENROUTER_API_KEY"]
                results = run_llm_monitor(
                    inputs, model=lm_cfg["model"], api_base=lm_cfg["api_base"],
                    api_key=api_key, max_tokens=lm_cfg["max_tokens"],
                    temperature=lm_cfg["temperature"],
                )

            elif baseline_name == "patchscopes":
                ps_cfg = bcfg["patchscopes"]
                results = run_patchscopes(
                    inputs, model, tokenizer,
                    source_layers=ps_cfg["source_layers"],
                    injection_layer=ps_cfg["injection_layer"],
                    steering_coefficient=ps_cfg["steering_coefficient"],
                    max_new_tokens=ps_cfg["max_new_tokens"],
                    device=args.device,
                )

            log_results(results, eval_name, baseline_name, output_dir, log_dir, wandb_run)
            eval_results[baseline_name] = results

        all_results[eval_name] = eval_results

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    for eval_name, eval_results in all_results.items():
        eval_type = EVAL_TYPES.get(eval_name, "?")
        print(f"\n{eval_name} ({eval_type}):")
        print(f"  {'Baseline':<20s} {'Score':<12s} {'N items':<10s} {'Details'}")
        print(f"  {'-'*60}")

        for baseline_name, results in eval_results.items():
            if results.get("skipped"):
                print(f"  {baseline_name:<20s} {'SKIPPED':<12s} {'':<10s} {results.get('reason', '')}")
                continue

            n_items = results.get("n_items", "?")
            metrics = results.get("metrics", results.get("per_layer", {}))

            if isinstance(metrics, dict) and "accuracy" in metrics:
                score = f"{metrics['accuracy']:.3f}"
                details = f"F1={metrics.get('f1', {})}"
            elif isinstance(metrics, dict) and "mean_token_f1" in metrics:
                score = f"{metrics['mean_token_f1']:.3f}"
                details = "token_f1"
            elif isinstance(metrics, dict) and "mean_spearman" in metrics:
                score = f"{metrics['mean_spearman']:.3f}"
                details = f"topk={metrics.get('mean_topk_precision', 0):.3f}"
            elif isinstance(metrics, dict):
                # per_layer results — show concat or best
                best_key = "concat_all"
                if best_key in metrics:
                    m = metrics[best_key]
                    score = f"{m.get('accuracy', m.get('mean_spearman', 0)):.3f}"
                else:
                    scores = []
                    for k, v in metrics.items():
                        if isinstance(v, dict):
                            scores.append((v.get("accuracy", v.get("mean_spearman", 0)), k))
                    if scores:
                        best_score, best_key = max(scores)
                        score = f"{best_score:.3f}"
                    else:
                        score = "N/A"
                details = f"best={best_key}"
            else:
                score = "N/A"
                details = ""

            print(f"  {baseline_name:<20s} {score:<12s} {str(n_items):<10s} {details}")

    if wandb_run:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
