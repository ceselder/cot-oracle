"""
Baselines runner: evaluate 7 baseline methods on eval tasks.

Usage:
    python baselines/run.py --config configs/train.yaml
    python baselines/run.py --config configs/train.yaml --evals hinted_mcq decorative_cot
    python baselines/run.py --config configs/train.yaml --baselines linear_probe llm_monitor
    python baselines/run.py --config configs/train.yaml --device cuda
    python baselines/run.py --config configs/train.yaml --rerun  # force rerun even if results exist
"""

import argparse
import json
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

from shared import load_baseline_inputs, load_cleaned_baseline_inputs, CLEANED_DATASET_NAMES, log_results
from scoring import EVAL_TYPES
from linear_probe import run_linear_probe
from attention_probe import run_attention_probe
from original_ao import run_original_ao
from llm_monitor import run_llm_monitor
from patchscopes import run_patchscopes
from no_act_oracle import run_no_act_oracle
from sae_probe import run_sae_probe
from qwen_attention_probe import run_qwen_attention_probe


# Which baselines can handle which eval types
BASELINE_COMPATIBILITY = {
    "linear_probe":     {"binary", "multiclass", "ranking"},
    "attention_probe":  {"binary", "multiclass", "ranking"},
    "original_ao":      {"binary", "generation"},
    "llm_monitor":      {"binary", "generation", "ranking"},
    "patchscopes":      {"binary", "generation"},
    "no_act_oracle":    {"binary", "generation"},
    "sae_probe":        {"binary", "generation", "ranking"},
    "qwen_attention_probe": {"binary", "multiclass", "ranking"},
}

ALL_BASELINES = list(BASELINE_COMPATIBILITY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines on eval tasks")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--evals", nargs="+", default=None, help="Eval names to run (default: all from config)")
    parser.add_argument("--baselines", nargs="+", default=None, help="Baselines to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even if results exist on disk")
    parser.add_argument("--no-act-oracle-checkpoint", type=str, default=None, help="Override no_act_oracle checkpoint path")
    parser.add_argument("--cleaned-datasets", nargs="*", default=None, help="Cleaned HF dataset IDs to run (default: all from config). Pass without args for all config datasets.")
    return parser.parse_args()


def _load_cached_result(log_dir: Path, eval_name: str, baseline_name: str) -> dict | None:
    """Load existing result from disk if available."""
    path = log_dir / eval_name / f"{baseline_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


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

    # Combine detection evals + classification evals
    eval_names = args.evals or (bcfg.get("evals", []) + bcfg.get("classification_evals", []))
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
    if "sae_probe" in baselines_to_run:
        layers_needed.update(bcfg["sae_probe"]["layers"])
    if "qwen_attention_probe" in baselines_to_run:
        layers_needed.update(bcfg["qwen_attention_probe"]["layers"])
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

        # Check if all baselines already have cached results (skip data loading if so)
        if not args.rerun:
            all_cached = True
            for baseline_name in baselines_to_run:
                if eval_type not in BASELINE_COMPATIBILITY[baseline_name]:
                    continue
                if _load_cached_result(log_dir, eval_name, baseline_name) is None:
                    all_cached = False
                    break
            if all_cached:
                print(f"  All baselines cached, loading from disk")
                eval_results = {}
                for baseline_name in baselines_to_run:
                    if eval_type not in BASELINE_COMPATIBILITY[baseline_name]:
                        continue
                    cached = _load_cached_result(log_dir, eval_name, baseline_name)
                    if cached:
                        eval_results[baseline_name] = cached
                        print(f"  Loaded cached {baseline_name}")
                all_results[eval_name] = eval_results
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

            # Skip if cached (unless --rerun)
            if not args.rerun:
                cached = _load_cached_result(log_dir, eval_name, baseline_name)
                if cached is not None:
                    print(f"  Skipping {baseline_name} (cached), use --rerun to force")
                    eval_results[baseline_name] = cached
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
                    steering_coefficients=ps_cfg["steering_coefficients"],
                    max_new_tokens=ps_cfg["max_new_tokens"],
                    device=args.device,
                )

            elif baseline_name == "no_act_oracle":
                na_cfg = bcfg["no_act_oracle"]
                checkpoint = args.no_act_oracle_checkpoint or na_cfg["checkpoint"]
                results = run_no_act_oracle(
                    inputs, model, tokenizer,
                    checkpoint=checkpoint,
                    max_new_tokens=na_cfg["max_new_tokens"],
                    device=args.device,
                )

            elif baseline_name == "sae_probe":
                sp_cfg = bcfg["sae_probe"]
                api_key = os.environ["OPENROUTER_API_KEY"]
                results = run_sae_probe(
                    inputs,
                    layers=sp_cfg["layers"], top_k=sp_cfg["top_k"],
                    sae_dir=sp_cfg["sae_dir"], sae_labels_dir=sp_cfg["sae_labels_dir"],
                    sae_trainer=sp_cfg["sae_trainer"],
                    llm_model=sp_cfg["llm_model"], api_base=sp_cfg["api_base"],
                    api_key=api_key, max_tokens=sp_cfg["max_tokens"],
                    temperature=sp_cfg["temperature"], device=args.device,
                )

            elif baseline_name == "qwen_attention_probe":
                qp_cfg = bcfg["qwen_attention_probe"]
                results = run_qwen_attention_probe(
                    inputs, layers=qp_cfg["layers"], k_folds=qp_cfg["k_folds"],
                    lr=qp_cfg["lr"], epochs=qp_cfg["epochs"],
                    patience=qp_cfg["patience"], batch_size=qp_cfg["batch_size"],
                    device=args.device,
                )

            log_results(results, eval_name, baseline_name, output_dir, log_dir, wandb_run)
            eval_results[baseline_name] = results

        all_results[eval_name] = eval_results

    # Cleaned datasets
    cleaned_cfg = bcfg.get("cleaned_datasets", {})
    if args.cleaned_datasets is not None:
        # --cleaned-datasets with specific IDs, or empty list = all from config
        cleaned_dataset_ids = args.cleaned_datasets if args.cleaned_datasets else cleaned_cfg.get("datasets", [])
    else:
        cleaned_dataset_ids = []

    if cleaned_dataset_ids:
        train_fraction = cleaned_cfg.get("train_fraction", 0.01)
        for dataset_id in cleaned_dataset_ids:
            eval_name = CLEANED_DATASET_NAMES[dataset_id]
            print(f"\n{'='*60}\nCleaned dataset: {eval_name} ({dataset_id})\n{'='*60}")
            eval_type = EVAL_TYPES[eval_name]

            # Check cache
            if not args.rerun:
                all_cached = True
                for baseline_name in baselines_to_run:
                    if eval_type not in BASELINE_COMPATIBILITY.get(baseline_name, set()):
                        continue
                    if _load_cached_result(log_dir, eval_name, baseline_name) is None:
                        all_cached = False
                        break
                if all_cached:
                    print(f"  All baselines cached, loading from disk")
                    eval_results = {}
                    for baseline_name in baselines_to_run:
                        if eval_type not in BASELINE_COMPATIBILITY.get(baseline_name, set()):
                            continue
                        cached = _load_cached_result(log_dir, eval_name, baseline_name)
                        if cached:
                            eval_results[baseline_name] = cached
                            print(f"  Loaded cached {baseline_name}")
                    all_results[eval_name] = eval_results
                    continue

            train_inputs, test_inputs = load_cleaned_baseline_inputs(
                dataset_id, model, tokenizer,
                layers=layers, stride=bcfg["stride"],
                device=args.device, train_fraction=train_fraction,
            )
            if not test_inputs:
                print(f"  No usable test items for {eval_name}, skipping")
                continue

            eval_results = {}
            for baseline_name in baselines_to_run:
                if eval_type not in BASELINE_COMPATIBILITY.get(baseline_name, set()):
                    print(f"  Skipping {baseline_name} (incompatible with {eval_type})")
                    continue

                if not args.rerun:
                    cached = _load_cached_result(log_dir, eval_name, baseline_name)
                    if cached is not None:
                        print(f"  Skipping {baseline_name} (cached), use --rerun to force")
                        eval_results[baseline_name] = cached
                        continue

                print(f"\n  Running {baseline_name} on {eval_name} ({len(train_inputs)} train, {len(test_inputs)} test)...")

                if baseline_name == "linear_probe":
                    lp_cfg = bcfg["linear_probe"]
                    results = run_linear_probe(
                        train_inputs, layers=lp_cfg["layers"], k_folds=lp_cfg["k_folds"],
                        lr=lp_cfg["lr"], epochs=lp_cfg["epochs"],
                        weight_decay=lp_cfg["weight_decay"], device=args.device,
                        test_inputs=test_inputs,
                    )
                elif baseline_name == "attention_probe":
                    ap_cfg = bcfg["attention_probe"]
                    results = run_attention_probe(
                        train_inputs, n_layers=ap_cfg["n_layers"], k_folds=ap_cfg["k_folds"],
                        n_heads=ap_cfg["n_heads"], hidden_dim=ap_cfg["hidden_dim"],
                        lr=ap_cfg["lr"], epochs=ap_cfg["epochs"],
                        patience=ap_cfg["patience"], device=args.device,
                        test_inputs=test_inputs,
                    )
                elif baseline_name == "llm_monitor":
                    lm_cfg = bcfg["llm_monitor"]
                    api_key = os.environ["OPENROUTER_API_KEY"]
                    results = run_llm_monitor(
                        test_inputs, model=lm_cfg["model"], api_base=lm_cfg["api_base"],
                        api_key=api_key, max_tokens=lm_cfg["max_tokens"],
                        temperature=lm_cfg["temperature"],
                    )
                elif baseline_name == "original_ao":
                    ao_cfg = bcfg["original_ao"]
                    results = run_original_ao(
                        test_inputs, model, tokenizer,
                        checkpoint=ao_cfg["checkpoint"], model_name=model_name,
                        device=args.device,
                    )
                elif baseline_name == "patchscopes":
                    ps_cfg = bcfg["patchscopes"]
                    results = run_patchscopes(
                        test_inputs, model, tokenizer,
                        source_layers=ps_cfg["source_layers"],
                        injection_layer=ps_cfg["injection_layer"],
                        steering_coefficients=ps_cfg["steering_coefficients"],
                        max_new_tokens=ps_cfg["max_new_tokens"],
                        device=args.device,
                    )
                elif baseline_name == "no_act_oracle":
                    na_cfg = bcfg["no_act_oracle"]
                    checkpoint = args.no_act_oracle_checkpoint or na_cfg["checkpoint"]
                    results = run_no_act_oracle(
                        test_inputs, model, tokenizer,
                        checkpoint=checkpoint,
                        max_new_tokens=na_cfg["max_new_tokens"],
                        device=args.device,
                    )
                elif baseline_name == "sae_probe":
                    sp_cfg = bcfg["sae_probe"]
                    api_key = os.environ["OPENROUTER_API_KEY"]
                    results = run_sae_probe(
                        test_inputs,
                        layers=sp_cfg["layers"], top_k=sp_cfg["top_k"],
                        sae_dir=sp_cfg["sae_dir"], sae_labels_dir=sp_cfg["sae_labels_dir"],
                        sae_trainer=sp_cfg["sae_trainer"],
                        llm_model=sp_cfg["llm_model"], api_base=sp_cfg["api_base"],
                        api_key=api_key, max_tokens=sp_cfg["max_tokens"],
                        temperature=sp_cfg["temperature"], device=args.device,
                    )
                elif baseline_name == "qwen_attention_probe":
                    qp_cfg = bcfg["qwen_attention_probe"]
                    results = run_qwen_attention_probe(
                        train_inputs, layers=qp_cfg["layers"], k_folds=qp_cfg["k_folds"],
                        lr=qp_cfg["lr"], epochs=qp_cfg["epochs"],
                        patience=qp_cfg["patience"], batch_size=qp_cfg["batch_size"],
                        device=args.device, test_inputs=test_inputs,
                    )
                else:
                    continue

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
                # per_layer/per_config results — show concat or best
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
