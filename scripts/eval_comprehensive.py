#!/usr/bin/env python3
"""
Comprehensive evaluation script for CoT Oracle.

Runs all eval modalities (oracle, original AO, LLM monitors, linear probes, SAE+LLM,
patchscopes, no-act oracle, attention probe) across all tasks, caches results per-method
in SQLite, and generates a summary plot.

Usage:
    python scripts/eval_comprehensive.py \\
        --config configs/eval.yaml \\
        --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic \\
        --n-examples 100 \\
        --output-dir data/comprehensive_eval \\
        [--tasks hint_admission atypical_answer ...] \\
        [--baselines weak-llm original_ao our_ao linear_probes sae_llm patchscopes ...] \\
        [--rerun] [--rerun-methods METHOD...] [--rerun-tasks TASK...] [--plot-only]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

_SRC = Path(__file__).resolve().parent.parent / "src"
_BASELINES = Path(__file__).resolve().parent.parent / "baselines"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from tasks import TASKS, ScoringMode, get_comprehensive_eval_tasks
from qa_scorer import check_openrouter_available, get_score_model
from eval_cache import EvalCache

K_SWEEP = [1, 5, 10, 20, None]
DEFAULT_LAYERS = [9, 18, 27]
DEFAULT_BASELINES = ["weak-llm", "our_ao", "original_ao"]  # fallback if eval.yaml has no baselines list

# Methods that need GPU model loaded
_MODEL_METHODS = {"our_ao", "original_ao", "patchscopes", "no_act_oracle", "attention_probe",
                  "linear_probes", "sae_llm"}
# Methods that need activations materialized
_ACTIVATION_METHODS = {"linear_probes", "sae_llm", "patchscopes", "attention_probe"}


def _bootstrap_std(scores: list[float], n_resamples: int = 5, frac: float = 0.5) -> float:
    import random
    if len(scores) < 4:
        return 0.0
    k = max(2, int(len(scores) * frac))
    means = []
    for _ in range(n_resamples):
        sub = random.choices(scores, k=k)
        means.append(sum(sub) / len(sub))
    return (sum((m - sum(means) / len(means)) ** 2 for m in means) / len(means)) ** 0.5


def _build_method_config(method_name: str, method_config: dict) -> dict:
    if method_name.startswith("our_ao_k"):
        k_str = method_name[len("our_ao_k"):]
        k = None if k_str == "all" else int(k_str)
        return {"type": "our_ao", "k_positions": k}
    if method_name.startswith("original_ao_k"):
        k_str = method_name[len("original_ao_k"):]
        k = None if k_str == "all" else int(k_str)
        return {"type": "original_ao", "k_positions": k}
    if method_name in ("weak-llm", "strong-llm"):
        cfg = method_config.get(method_name, {})
        return {"type": "llm_monitor", "model": cfg.get("model"), "max_tokens": cfg.get("max_tokens")}
    return {"type": method_name}


# ── Data loading (shared across methods) ──

def _load_and_normalize(task_name: str, n_examples: int) -> list[dict]:
    """Load task data, normalize fields. Returns list of dicts."""
    from data_loading import load_task_data
    try:
        test_data = load_task_data(task_name, split="test", n=n_examples, shuffle=False)
    except Exception:
        test_data = []
    if not test_data:
        test_data = load_task_data(task_name, split="train", n=n_examples, shuffle=False)
    for item in test_data:
        if "meta_spliced_cot_text" in item and "cot_text" not in item:
            item["cot_text"] = item["meta_spliced_cot_text"]
        if "test_prompt" in item and "question" not in item:
            item["question"] = item["test_prompt"]
        if "target_response" not in item and "meta_oracle_target" in item:
            item["target_response"] = str(item["meta_oracle_target"])
        item["_task_name"] = task_name
    return test_data


# ── Shared activation materialization ──

def _ensure_activations(model, tokenizer, test_data, layers, position_mode, task_name):
    """Materialize activations for items that have context_input_ids.

    Returns (valid_data, activations) where valid_data is the subset with context,
    and activations is a list of [nK, D] tensors.
    """
    from eval_loop import prepare_context_ids, _resample_eval_positions, _materialize_activations

    prepare_context_ids(test_data, tokenizer, layers=layers)
    valid_data = [d for d in test_data if d.get("context_input_ids")]
    if not valid_data:
        return [], []

    _resample_eval_positions(
        test_data=valid_data, task_name=task_name, layers=layers,
        position_mode=position_mode, stochastic_max_k=100, eval_position_seed=0,
    )

    all_activations = []
    for start in range(0, len(valid_data), 4):
        chunk = valid_data[start:start + 4]
        chunk_acts = _materialize_activations(model, tokenizer, chunk, layers=layers, device="cuda")
        all_activations.extend(chunk_acts)

    return valid_data, all_activations


# ── Oracle eval (GPU) ──

def run_oracle_eval(model, tokenizer, task_name, task_def, test_data, layers,
                    position_mode, oracle_adapter_name, openrouter_available,
                    k_positions=None, completed_indices=None):
    """Run oracle eval. Returns (predictions, targets, todo_indices)."""
    from eval_loop import (
        prepare_context_ids, _resample_eval_positions,
        _materialize_activations, _batched_oracle_generate,
    )
    completed = completed_indices or set()

    prepare_context_ids(test_data, tokenizer, layers=layers)
    valid_data = [(i, d) for i, d in enumerate(test_data) if d.get("context_input_ids")]
    if not valid_data:
        return [None] * len(test_data), [d.get("target_response", "") for d in test_data], {}

    todo_indices = [i for i, _ in valid_data if i not in completed]
    todo_items = [test_data[i] for i in todo_indices]

    if todo_items:
        _resample_eval_positions(
            test_data=todo_items, task_name=task_name, layers=layers,
            position_mode=position_mode, stochastic_max_k=100, eval_position_seed=0,
        )

        all_activations = []
        for start in range(0, len(todo_items), 4):
            chunk = todo_items[start:start + 4]
            chunk_acts = _materialize_activations(model, tokenizer, chunk, layers=layers, device="cuda")
            all_activations.extend(chunk_acts)

        n_layers = len(layers)
        oracle_items = []
        for act, item in zip(all_activations, todo_items):
            if k_positions is not None:
                K = act.shape[0] // n_layers
                k = min(k_positions, K)
                indices = []
                for l in range(n_layers):
                    start_idx = l * K + (K - k)
                    indices.extend(range(start_idx, l * K + K))
                act = act[indices]
            oracle_items.append((act, item["prompt"]))

        new_predictions = _batched_oracle_generate(
            model=model, tokenizer=tokenizer, items=oracle_items,
            layers=layers, device="cuda", injection_layer=1,
            max_new_tokens=task_def.max_new_tokens, eval_batch_size=8,
            oracle_adapter_name=oracle_adapter_name,
        )
    else:
        new_predictions = []

    predictions = [None] * len(test_data)
    for idx, pred in zip(todo_indices, new_predictions):
        predictions[idx] = pred

    targets = [d.get("target_response", "") for d in test_data]
    return predictions, targets, todo_indices


def _extract_per_item_scores(result: dict, task_def, predictions: list[str], targets: list[str]) -> list[float]:
    from eval_loop import _per_example_correct
    if "_qa_scorer_scores" in result:
        return [s for s in result["_qa_scorer_scores"] if not (isinstance(s, float) and math.isnan(s))]
    if "_llm_scorer_parsed" in result:
        return [float(p.get("correctness", 0.0)) for p in result["_llm_scorer_parsed"]]
    scores = []
    for pred, tgt in zip(predictions, targets):
        c = _per_example_correct(task_def.name, task_def, pred, tgt)
        if c == "yes":
            scores.append(1.0)
        elif c.startswith("F1="):
            scores.append(float(c[3:]))
        else:
            scores.append(0.0)
    return scores


def _run_and_store_method(cache, run_id, task_name, task_def, method_name,
                          method_config, model, tokenizer, n_examples, layers,
                          position_mode, openrouter_available):
    """Run a single method with incremental recovery."""
    from eval_loop import _primary_metric_name, score_task

    primary = _primary_metric_name(task_name, task_def.scoring)

    cached_preds = cache.get_predictions(run_id, task_name, method_name)
    n_cached = len(cached_preds)

    test_data = _load_and_normalize(task_name, n_examples)
    if not test_data:
        print(f"    [{method_name}] no data")
        return

    print(f"    [{method_name}] ", end="", flush=True)
    if n_cached:
        print(f"({n_cached} cached) ", end="", flush=True)
    t0 = time.time()

    completed_indices = set(cached_preds.keys())

    # ── Oracle methods (our_ao, original_ao) ──
    if method_name.startswith("our_ao_k") or method_name.startswith("original_ao_k"):
        is_our = method_name.startswith("our_ao_k")
        k_str = method_name[len("our_ao_k" if is_our else "original_ao_k"):]
        k = None if k_str == "all" else int(k_str)
        adapter = "default" if is_our else "original_ao"

        predictions, targets, todo_indices = run_oracle_eval(
            model, tokenizer, task_name, task_def, test_data, layers,
            position_mode, oracle_adapter_name=adapter,
            openrouter_available=openrouter_available, k_positions=k,
            completed_indices=completed_indices,
        )

        if todo_indices:
            new_pred_rows = []
            for i in todo_indices:
                if predictions[i] is not None:
                    new_pred_rows.append({
                        "item_idx": i, "item_id": f"{task_name}_{i}",
                        "prediction": predictions[i][:500], "score": None,
                    })
            if new_pred_rows:
                cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

        for idx, cached in cached_preds.items():
            if idx < len(predictions) and predictions[idx] is None:
                predictions[idx] = cached["prediction"]

        valid_mask = [i for i in range(len(predictions)) if predictions[i] is not None]
        predictions = [predictions[i] for i in valid_mask]
        targets = [targets[i] for i in valid_mask]
        eval_items = [test_data[i] for i in valid_mask]

    # ── LLM monitor methods (weak-llm, strong-llm) ──
    elif method_name in ("weak-llm", "strong-llm"):
        from llm_monitor import run_llm_monitor

        llm_cfg = method_config.get(method_name, {})
        predictions = run_llm_monitor(
            test_data, None, layers, task_def,
            model_name=llm_cfg["model"],
            max_tokens=llm_cfg.get("max_tokens", 300),
            cache=cache, run_id=run_id, method_name=method_name,
            completed_indices=completed_indices,
        )
        targets = [d.get("target_response", "") for d in test_data[:len(predictions)]]

        for idx, cached in cached_preds.items():
            if idx < len(predictions) and predictions[idx] is None:
                predictions[idx] = cached["prediction"]

        valid_mask = [i for i in range(len(predictions)) if predictions[i] is not None]
        predictions = [predictions[i] for i in valid_mask]
        targets = [targets[i] for i in valid_mask] if targets else []
        eval_items = [test_data[i] for i in valid_mask]

    # ── Linear probes ──
    elif method_name == "linear_probes":
        from linear_probe import run_linear_probe

        valid_data, activations = _ensure_activations(model, tokenizer, test_data, layers, position_mode, task_name)
        if not valid_data:
            print("no valid data for activations")
            return

        predictions = run_linear_probe(valid_data, activations, layers, task_def, device="cuda")
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        # Store predictions
        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── SAE probe ──
    elif method_name == "sae_llm":
        from sae_probe import run_sae_probe

        valid_data, activations = _ensure_activations(model, tokenizer, test_data, layers, position_mode, task_name)
        if not valid_data:
            print("no valid data for activations")
            return

        sae_cfg = method_config.get("sae_llm", {})
        predictions = run_sae_probe(
            valid_data, activations, layers, task_def,
            sae_labels_dir=sae_cfg.get("sae_labels_dir", ""),
            llm_model=sae_cfg.get("llm_model", "google/gemini-2.5-flash-lite"),
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            device="cuda",
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── Patchscopes ──
    elif method_name == "patchscopes":
        from patchscopes import run_patchscopes

        valid_data, activations = _ensure_activations(model, tokenizer, test_data, layers, position_mode, task_name)
        if not valid_data:
            print("no valid data for activations")
            return

        predictions = run_patchscopes(
            valid_data, activations, layers, task_def, model, tokenizer, device="cuda",
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── No-act oracle ──
    elif method_name == "no_act_oracle":
        from no_act_oracle import run_no_act_oracle

        checkpoint = method_config.get("no_act_oracle", {}).get("checkpoint", "")
        if not checkpoint:
            print("no checkpoint configured")
            return

        predictions = run_no_act_oracle(
            test_data, None, layers, task_def, model, tokenizer,
            checkpoint=checkpoint, device="cuda",
        )
        targets = [d.get("target_response", "") for d in test_data]
        eval_items = test_data
        valid_mask = list(range(len(test_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── Attention probe ──
    elif method_name == "attention_probe":
        from attention_probe import run_qwen_attention_probe

        valid_data, activations = _ensure_activations(model, tokenizer, test_data, layers, position_mode, task_name)
        if not valid_data:
            print("no valid data for activations")
            return

        predictions = run_qwen_attention_probe(
            valid_data, activations, layers, task_def, device="cuda",
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    else:
        print(f"unknown method type")
        return

    if not predictions:
        print(f"no predictions")
        return

    # Score the full set (cached + new)
    result = score_task(task_def, predictions, targets, eval_items=eval_items, openrouter_available=openrouter_available)
    score = result.get(primary, 0.0)
    per_item_scores = _extract_per_item_scores(result, task_def, predictions, targets)
    elapsed = time.time() - t0
    print(f"{score:.3f} ({elapsed:.1f}s)")

    # Update per-item scores
    score_updates = []
    for rank, orig_idx in enumerate(valid_mask):
        if rank < len(per_item_scores):
            score_updates.append({
                "item_idx": orig_idx, "item_id": f"{task_name}_{orig_idx}",
                "prediction": predictions[rank][:500], "score": per_item_scores[rank],
            })
    if score_updates:
        cache.store_predictions(run_id, task_name, method_name, score_updates)

    # Store items (targets)
    items = [{"item_idx": i, "item_id": f"{task_name}_{i}", "target_response": tgt[:500]}
             for i, tgt in enumerate([d.get("target_response", "") for d in test_data]) if tgt]
    cache.store_items(run_id, task_name, items)

    # Store method result
    method_result = {
        "primary_score": score,
        "bootstrap_std": _bootstrap_std(per_item_scores) if per_item_scores else 0.0,
        "primary_metric": primary,
        "n": result.get("n", len(predictions)),
    }
    if method_name in ("weak-llm", "strong-llm"):
        llm_cfg = method_config.get(method_name, {})
        method_result["extra"] = {"model": llm_cfg.get("model")}

    deps_hash = cache.method_deps_hash(run_id, task_name, method_name, _build_method_config(method_name, method_config))
    cache.store_method_result(run_id, task_name, method_name, method_result, deps_hash)

    return score, primary


def _expand_methods(active_baselines: list[str], method_config: dict) -> list[str]:
    methods = []
    for b in active_baselines:
        cfg = method_config.get(b, {})
        if b == "no_act_oracle" and not cfg.get("checkpoint", ""):
            continue

        if b == "our_ao":
            for k in K_SWEEP:
                methods.append(f"our_ao_k{'all' if k is None else k}")
        elif b == "original_ao":
            for k in K_SWEEP:
                methods.append(f"original_ao_k{'all' if k is None else k}")
        else:
            methods.append(b)
    return methods


def plot_results(output_dir: Path, cache: EvalCache | None = None, run_id: str | None = None,
                 tasks_order: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if cache is not None and run_id is not None:
        results = cache.get_all_method_results(run_id)
    else:
        results = {}
        for f in sorted(output_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            data = json.loads(f.read_text())
            task = data["task"]
            results[task] = data["methods"]

    if not results:
        print("  No results to plot.")
        return

    all_methods = set()
    for methods in results.values():
        all_methods.update(methods.keys())

    def method_sort_key(m):
        if m.startswith("our_ao"): return (0, m)
        if m.startswith("original_ao"): return (1, m)
        return (2, m)

    method_order = sorted(all_methods, key=method_sort_key)
    task_order = tasks_order or sorted(results.keys())
    task_order = [t for t in task_order if t in results]

    if not task_order:
        print("  No matching tasks to plot.")
        return

    cmap = plt.cm.tab20
    method_colors = {m: cmap(i / max(len(method_order), 1)) for i, m in enumerate(method_order)}

    fig, ax = plt.subplots(1, 1, figsize=(max(14, len(task_order) * 0.8), 7))
    x = np.arange(len(task_order))
    bar_width = 0.8 / max(len(method_order), 1)

    for j, method in enumerate(method_order):
        scores = []
        stds = []
        for task in task_order:
            m = results.get(task, {}).get(method, {})
            s = m.get("primary_score", float("nan"))
            scores.append(s if not math.isnan(s) else 0.0)
            stds.append(m.get("bootstrap_std", 0.0))
        offset = (j - len(method_order) / 2 + 0.5) * bar_width
        ax.bar(x + offset, scores, bar_width * 0.9, yerr=stds,
               label=method, color=method_colors[method], alpha=0.85, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(task_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Comprehensive Evaluation — CoT Oracle")
    ax.legend(fontsize=7, ncol=min(4, len(method_order)), loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "comprehensive_eval.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CoT Oracle Evaluation")
    parser.add_argument("--config", default="configs/eval.yaml", help="Eval config (baselines, score_model)")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Train config (model, activations)")
    parser.add_argument("--checkpoint", required=True, help="Our LoRA checkpoint path")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--output-dir", default="data/comprehensive_eval")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all comprehensive eval tasks)")
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES, help="Baselines to run")
    parser.add_argument("--layers", nargs="*", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--position-mode", default="all")
    parser.add_argument("--rerun", action="store_true", help="Ignore cache, rerun everything")
    parser.add_argument("--rerun-methods", nargs="*", default=None, help="Rerun specific methods only")
    parser.add_argument("--rerun-tasks", nargs="*", default=None, help="Rerun specific tasks only")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate the plot from existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config))
    method_config = config.get("method_config", {})

    cache = EvalCache(output_dir / "eval_cache.db")
    run_id = cache.get_or_create_run(args.checkpoint, args.n_examples, args.position_mode, args.layers)

    if args.plot_only:
        plot_results(output_dir, cache=cache, run_id=run_id)
        return

    all_tasks = get_comprehensive_eval_tasks()
    # Tasks: CLI --tasks overrides, else eval.yaml list, else all comprehensive tasks
    if args.tasks:
        task_names = [t for t in args.tasks if t in all_tasks]
    elif "tasks" in config and config["tasks"]:
        task_names = [t for t in config["tasks"] if t in all_tasks]
    else:
        task_names = list(all_tasks.keys())

    # Baselines: CLI --baselines overrides, else eval.yaml list, else defaults
    if args.baselines != DEFAULT_BASELINES:
        active_baselines = list(args.baselines)
    elif "baselines" in config and config["baselines"]:
        active_baselines = list(config["baselines"])
    else:
        active_baselines = DEFAULT_BASELINES
    all_methods = _expand_methods(active_baselines, method_config)
    print(f"Tasks: {len(task_names)}, Methods: {len(all_methods)}, N: {args.n_examples}")

    needs_model = any(b in _MODEL_METHODS for b in active_baselines)

    openrouter_available = check_openrouter_available()
    if not openrouter_available:
        print("OpenRouter unavailable — LLM-based methods will be skipped or produce NaN scores")

    model, tokenizer = None, None
    if needs_model:
        print("Loading Qwen3-8B + adapters...")
        import torch
        from core.ao import load_model_with_ao, load_extra_adapter

        model, tokenizer = load_model_with_ao(args.train_config)
        model.eval()

        if "our_ao" in active_baselines:
            load_extra_adapter(model, args.checkpoint, adapter_name="default")
            print(f"  Loaded our LoRA from {args.checkpoint}")

    rerun_methods = set(args.rerun_methods) if args.rerun_methods else set()
    rerun_tasks = set(args.rerun_tasks) if args.rerun_tasks else set()

    for task_name in task_names:
        task_def = all_tasks[task_name]
        print(f"\n  === {task_name} ===")

        for method_name in all_methods:
            if method_name in ("weak-llm", "strong-llm") and not openrouter_available:
                continue
            if method_name == "sae_llm" and not openrouter_available:
                continue

            mcfg = _build_method_config(method_name, method_config)
            deps_hash = cache.method_deps_hash(run_id, task_name, method_name, mcfg)

            force_rerun = args.rerun or method_name in rerun_methods or task_name in rerun_tasks
            if force_rerun:
                cache.delete_method(run_id, task_name, method_name)
            elif cache.has_method(run_id, task_name, method_name, deps_hash):
                print(f"    [{method_name}] cached, skipping")
                continue

            try:
                _run_and_store_method(
                    cache, run_id, task_name, task_def, method_name,
                    method_config, model, tokenizer, args.n_examples,
                    args.layers, args.position_mode, openrouter_available,
                )
            except Exception as e:
                print(f" FAILED: {e}")
                cache.store_method_result(run_id, task_name, method_name, {
                    "primary_score": float("nan"),
                    "primary_metric": "error",
                    "n": 0,
                    "extra": {"error": str(e)[:200]},
                }, deps_hash)

            task_json = cache.export_task_json(run_id, task_name)
            (output_dir / f"{task_name}.json").write_text(json.dumps(task_json, indent=2, default=str))

    plot_results(output_dir, cache=cache, run_id=run_id, tasks_order=task_names)
    cache.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
