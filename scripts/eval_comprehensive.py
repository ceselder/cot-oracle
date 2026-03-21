#!/usr/bin/env python3
"""
Comprehensive evaluation script for CoT Oracle.

Runs all eval modalities (oracle, original AO, BB monitors, linear probes, SAE+LLM,
patchscopes, no-act oracle, attention probe) across all tasks, caches results per-method
in SQLite, and generates a summary plot.

Usage:
    python scripts/eval_comprehensive.py \\
        --config configs/eval.yaml \\
        --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic \\
        --n-examples 100 \\
        --output-dir data/comprehensive_eval \\
        [--tasks hint_admission atypical_answer ...] \\
        [--baselines weak-bb-monitor original_ao our_ao linear_probes sae-llm-monitor patchscopes ...] \\
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

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_BASELINES = _ROOT / "baselines"
_AO_REF = _ROOT / "ao_reference"
for _p in [_SRC, _BASELINES, _AO_REF]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tasks import TASKS, ScoringMode, get_comprehensive_eval_tasks
from qa_scorer import check_openrouter_available, get_score_model, stop_local_scorer
from eval_cache import EvalCache
from eval_loop import (
    load_and_normalize, materialize_activations_chunked, run_oracle_eval,
    extract_per_item_scores, bootstrap_std, build_method_config, expand_methods,
    score_task, _primary_metric_name, _per_example_correct,
)

DEFAULT_K_SWEEP = [1, 5, 10, 20, None]
DEFAULT_LAYERS = [9, 18, 27]
DEFAULT_BASELINES = ["weak-bb-monitor", "our_ao", "original_ao"]  # fallback if eval.yaml has no baselines list

# Methods that need GPU model loaded
_MODEL_METHODS = {"our_ao", "original_ao", "patchscopes", "no_act_oracle", "attention_probe",
                  "linear_probes", "sae-llm-monitor"}
# Methods that need activations materialized
_ACTIVATION_METHODS = {"linear_probes", "sae-llm-monitor", "patchscopes", "attention_probe"}


def _store_failure(cache, run_id, task_name, method_name, method_config, reason: str):
    """Record a failed eval so it can be queried and retried with --rerun-failed."""
    deps_hash = cache.method_deps_hash(run_id, task_name, method_name, build_method_config(method_name, method_config))
    cache.store_method_result(run_id, task_name, method_name, {
        "primary_score": float("nan"),
        "primary_metric": "failed",
        "n": 0,
        "extra": {"reason": reason},
    }, deps_hash)


def _run_and_store_method(cache, run_id, task_name, task_def, method_name,
                          method_config, model, tokenizer, n_examples, layers,
                          openrouter_available, train_config=None, wandb_run=None):
    """Run a single method with incremental recovery."""
    primary = _primary_metric_name(task_name, task_def.scoring)

    cached_preds = cache.get_predictions(run_id, task_name, method_name)
    n_cached = len(cached_preds)

    test_data = load_and_normalize(task_name, n_examples)
    if not test_data:
        print(f"    [{method_name}] no data")
        _store_failure(cache, run_id, task_name, method_name, method_config, "no data")
        return

    print(f"    [{method_name}] ", end="", flush=True)
    if n_cached:
        print(f"({n_cached} cached) ", end="", flush=True)
    t0 = time.time()

    completed_indices = set(cached_preds.keys())
    item_prompts = None     # per-item resolved prompts (for methods that build them)

    # ── Oracle methods (our_ao, original_ao) ──
    if method_name.startswith("our_ao_") or method_name.startswith("original_ao_"):
        is_our = method_name.startswith("our_ao_")
        from core.ao import AO_CHECKPOINTS
        ao_adapter_name = AO_CHECKPOINTS["Qwen/Qwen3-8B"].replace(".", "_")
        adapter = "default" if is_our else ao_adapter_name

        mcfg = build_method_config(method_name, method_config)
        eval_layers = [mcfg["layer"]] if "layer" in mcfg else layers
        pos_slice = mcfg.get("position_slice", "::")

        predictions, targets, todo_indices = run_oracle_eval(
            model, tokenizer, task_name, task_def, test_data, eval_layers,
            "all", oracle_adapter_name=adapter,
            position_slice=pos_slice, completed_indices=completed_indices,
        )

        if todo_indices:
            new_pred_rows = []
            for i in todo_indices:
                if predictions[i] is not None:
                    new_pred_rows.append({
                        "item_idx": i, "item_id": f"{task_name}_{i}",
                        "prediction": predictions[i][:500], "score": None,
                        "prompt": test_data[i].get("prompt", ""),
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
        item_prompts = [test_data[i].get("prompt", "") for i in valid_mask]

    # ── BB monitor methods (weak-bb-monitor, strong-bb-monitor) ──
    elif method_name in ("weak-bb-monitor", "strong-bb-monitor"):
        from bb_monitor import run_bb_monitor

        llm_cfg = method_config.get(method_name, {})
        predictions, item_prompts = run_bb_monitor(
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
        item_prompts = [item_prompts[i] for i in valid_mask] if item_prompts else [None] * len(valid_mask)

    # ── Linear probes ──
    elif method_name == "linear_probes":
        from tasks import ScoringMode
        if task_def.scoring != ScoringMode.BINARY:
            print(f"skipped (non-binary task)")
            return
        from linear_probe import run_linear_probe

        valid_data, test_acts = materialize_activations_chunked(model, tokenizer, test_data, layers, position_mode="all", task_name=task_name)
        if not valid_data:
            print("no valid data for activations")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no valid data for activations")
            return

        # Load train split for probe training
        task_n = -1
        if train_config and "tasks" in train_config:
            task_n = train_config["tasks"].get(task_name, {}).get("n", -1)
        if task_n == 0:
            print("task disabled in train config")
            _store_failure(cache, run_id, task_name, method_name, method_config, "task disabled in train config")
            return
        train_data = load_and_normalize(task_name, min(task_n, 2000) if task_n > 0 else 2000, split="train")
        if not train_data:
            print("no train data")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no train data")
            return
        train_valid, train_acts = materialize_activations_chunked(model, tokenizer, train_data, layers, position_mode="all", task_name=task_name)

        predictions = run_linear_probe(
            valid_data, test_acts, layers, task_def,
            train_data=train_valid, train_activations=train_acts, device="cuda",
            wandb_run=wandb_run,
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── SAE probe ──
    elif method_name == "sae-llm-monitor":
        from sae_llm_monitor import run_sae_probe

        valid_data, activations = materialize_activations_chunked(model, tokenizer, test_data, layers, position_mode="all", task_name=task_name)
        if not valid_data:
            print("no valid data for activations")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no valid data for activations")
            return

        sae_cfg = method_config.get("sae-llm-monitor", {})
        predictions, item_prompts = run_sae_probe(
            valid_data, activations, layers, task_def,
            sae_labels_dir=sae_cfg.get("sae_labels_dir", ""),
            sae_trainer=sae_cfg.get("sae_trainer", 2),
            llm_model=sae_cfg.get("llm_model", "google/gemini-2.5-flash-lite"),
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            max_tokens=sae_cfg.get("max_tokens", 300),
            temperature=sae_cfg.get("temperature", 0.0),
            max_concurrent=sae_cfg.get("max_concurrent", 20),
            top_k=sae_cfg.get("top_k", 20),
            device="cuda",
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None,
                          "prompt": item_prompts[i][:2000] if i < len(item_prompts) else None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── Patchscopes ──
    elif method_name == "patchscopes":
        from patchscopes import run_patchscopes

        valid_data, activations = materialize_activations_chunked(model, tokenizer, test_data, layers, position_mode="all", task_name=task_name)
        if not valid_data:
            print("no valid data for activations")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no valid data for activations")
            return

        ps_cfg = method_config.get("patchscopes", {})
        predictions = run_patchscopes(
            valid_data, activations, layers, task_def, model, tokenizer,
            steering_coefficients=ps_cfg.get("steering_coefficients", [0.5, 1.0, 2.0]),
            source_layers=ps_cfg.get("source_layers"),
            injection_layer=ps_cfg.get("injection_layer", 1),
            max_new_tokens=ps_cfg.get("max_new_tokens", 100),
            device="cuda",
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))
        item_prompts = [d.get("prompt", "") for d in valid_data]

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None,
                          "prompt": valid_data[i].get("prompt", "")}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── No-act oracle ──
    elif method_name == "no_act_oracle":
        from no_act_oracle import run_no_act_oracle

        checkpoint = method_config.get("no_act_oracle", {}).get("checkpoint", "")
        if not checkpoint:
            print("no checkpoint configured")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no checkpoint configured")
            return

        predictions = run_no_act_oracle(
            test_data, None, layers, task_def, model, tokenizer,
            checkpoint=checkpoint, device="cuda",
        )
        targets = [d.get("target_response", "") for d in test_data]
        eval_items = test_data
        valid_mask = list(range(len(test_data)))

        oracle_context = task_def.oracle_context
        item_prompts = [f"Question: {d.get('question', d.get('prompt', ''))}\nChain of thought: {d.get(oracle_context, '')[:4000]}\n\n{d.get('prompt', '')}" for d in test_data]
        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None,
                          "prompt": item_prompts[i][:2000]}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)

    # ── Attention probe ──
    elif method_name == "attention_probe":
        from tasks import ScoringMode
        if task_def.scoring != ScoringMode.BINARY:
            print(f"skipped (non-binary task)")
            return
        from attention_probe import run_attention_probe

        valid_data, activations = materialize_activations_chunked(model, tokenizer, test_data, layers, position_mode="all", task_name=task_name)
        if not valid_data:
            print("no valid data for activations")
            _store_failure(cache, run_id, task_name, method_name, method_config, "no valid data for activations")
            return

        predictions = run_attention_probe(
            valid_data, activations, layers, task_def, device="cuda",
            wandb_run=wandb_run,
        )
        targets = [d.get("target_response", "") for d in valid_data]
        eval_items = valid_data
        valid_mask = list(range(len(valid_data)))

        new_pred_rows = [{"item_idx": i, "item_id": f"{task_name}_{i}", "prediction": (p or "")[:500], "score": None}
                         for i, p in enumerate(predictions)]
        cache.store_predictions(run_id, task_name, method_name, new_pred_rows)
        # attention probe is a learned classifier, no text prompt

    else:
        print(f"unknown method type")
        _store_failure(cache, run_id, task_name, method_name, method_config, f"unknown method type: {method_name}")
        return

    if not predictions:
        print(f"no predictions")
        _store_failure(cache, run_id, task_name, method_name, method_config, "no predictions")
        return

    # Store items (targets) and mark as "unscored" before attempting scoring
    items = [{"item_idx": i, "item_id": f"{task_name}_{i}", "target_response": tgt[:500]}
             for i, tgt in enumerate([d.get("target_response", "") for d in test_data]) if tgt]
    cache.store_items(run_id, task_name, items)

    deps_hash = cache.method_deps_hash(run_id, task_name, method_name, build_method_config(method_name, method_config))
    cache.store_method_result(run_id, task_name, method_name, {
        "primary_score": float("nan"), "primary_metric": "unscored",
        "n": len(predictions), "extra": {},
    }, deps_hash)

    # Score the full set (cached + new)
    return _score_and_store(
        cache, run_id, task_name, task_def, method_name, method_config,
        predictions, targets, eval_items, valid_mask, item_prompts,
        openrouter_available, t0,
    )


def _score_and_store(cache, run_id, task_name, task_def, method_name, method_config,
                     predictions, targets, eval_items, valid_mask, item_prompts,
                     openrouter_available, t0):
    """Score predictions and store results. Separated so rescoring can call it directly."""
    primary = _primary_metric_name(task_name, task_def.scoring)

    result = score_task(task_def, predictions, targets, eval_items=eval_items, openrouter_available=openrouter_available)
    score = result.get(primary, 0.0)
    per_item_scores, scorer_responses = extract_per_item_scores(result, task_def, predictions, targets)
    elapsed = time.time() - t0
    print(f"{score:.3f} ({elapsed:.1f}s)")

    # Update per-item scores (preserve prompt if available)
    score_updates = []
    for rank, orig_idx in enumerate(valid_mask):
        if rank < len(per_item_scores):
            entry = {
                "item_idx": orig_idx, "item_id": f"{task_name}_{orig_idx}",
                "prediction": predictions[rank][:500], "score": per_item_scores[rank],
            }
            if item_prompts is not None and rank < len(item_prompts) and item_prompts[rank]:
                entry["prompt"] = item_prompts[rank][:2000]
            if rank < len(scorer_responses) and scorer_responses[rank]:
                entry["scorer_response"] = str(scorer_responses[rank])[:500]
            score_updates.append(entry)
    if score_updates:
        cache.store_predictions(run_id, task_name, method_name, score_updates)

    # Store final scored method result
    extra = {}
    if method_name in ("weak-bb-monitor", "strong-bb-monitor"):
        extra["model"] = method_config.get(method_name, {}).get("model")
    method_result = {
        "primary_score": score,
        "bootstrap_std": bootstrap_std(per_item_scores) if per_item_scores else 0.0,
        "primary_metric": primary,
        "n": result.get("n", len(predictions)),
        "extra": extra,
    }

    deps_hash = cache.method_deps_hash(run_id, task_name, method_name, build_method_config(method_name, method_config))
    cache.store_method_result(run_id, task_name, method_name, method_result, deps_hash)

    return score, primary


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

    task_order = tasks_order or sorted(results.keys())
    task_order = [t for t in task_order if t in results]
    if not task_order:
        print("  No matching tasks to plot.")
        return

    # ── Collapse to best-per-family-per-task: for each task, pick the variant
    #    with the highest score within each method family ──
    def _method_family(name: str) -> str:
        """our_ao_k5 / our_ao_stride10 / our_ao_L9_tail5 → 'our_ao'"""
        for prefix in ("our_ao", "original_ao"):
            if name.startswith(prefix):
                return prefix
        return name

    families: dict[str, list[str]] = {}
    for task_methods in results.values():
        for m in task_methods:
            families.setdefault(_method_family(m), [])
            if m not in families[_method_family(m)]:
                families[_method_family(m)].append(m)

    # For each (task, family), find the best variant
    best_per_task: dict[str, dict[str, tuple[str, float, float]]] = {}  # task → family → (variant, score, std)
    for task in task_order:
        best_per_task[task] = {}
        for family, variants in families.items():
            best_v, best_s, best_std = variants[0], -1.0, 0.0
            for v in variants:
                m = results.get(task, {}).get(v, {})
                s = m.get("primary_score")
                if s is not None and not math.isnan(s) and s > best_s:
                    best_s = s
                    best_v = v
                    best_std = m.get("bootstrap_std", 0.0)
            if best_s >= 0:
                best_per_task[task][family] = (best_v, best_s, best_std)

    method_order = sorted(families.keys(), key=lambda f: (0 if f == "our_ao" else 1 if f == "original_ao" else 2, f))

    # ── Colors ──
    _FAMILY_COLORS = {
        "our_ao": "#1565C0",          # dark blue
        "original_ao": "#64B5F6",     # light blue
        "linear_probes": "#E67E22",   # orange
        "sae-llm-monitor": "#BDBDBD", # light gray
        "patchscopes": "#E0E0E0",     # lighter gray
        "attention_probe": "#9E9E9E", # medium gray
    }
    def _method_color(name):
        if "bb-monitor" in name or "bb_monitor" in name:
            return "#555555"
        return _FAMILY_COLORS.get(_method_family(name))

    cmap = plt.cm.tab10
    fallback_idx = 0
    method_colors = {}
    for family in method_order:
        c = _method_color(family)
        if c is None:
            method_colors[family] = cmap(fallback_idx / max(len(method_order), 1))
            fallback_idx += 1
        else:
            method_colors[family] = c

    # ── Split tasks into subsets ──
    adam_tasks = [t for t in task_order if t.startswith("cls_")]
    our_tasks = [t for t in task_order if not t.startswith("cls_")]

    # Build subplot rows: (title, task_list)
    rows = [("All tasks", task_order)]
    if our_tasks and our_tasks != task_order:
        rows.append(("Our tasks", our_tasks))
    if adam_tasks:
        rows.append(("Original AO tasks", adam_tasks))

    def _plot_row(ax, tasks, title):
        # Add "avg" pseudo-task
        plot_tasks = tasks + ["avg"]
        x = np.arange(len(plot_tasks))
        bar_width = 0.8 / max(len(method_order), 1)

        for j, family in enumerate(method_order):
            scores, stds = [], []
            for task in tasks:
                entry = best_per_task.get(task, {}).get(family)
                scores.append(entry[1] if entry else 0.0)
                stds.append(entry[2] if entry else 0.0)
            # Compute average
            valid = [s for s in scores if s > 0]
            avg = sum(valid) / len(valid) if valid else 0.0
            scores.append(avg)
            stds.append(0.0)

            offset = (j - len(method_order) / 2 + 0.5) * bar_width
            ax.bar(x + offset, scores, bar_width * 0.9, yerr=stds,
                   label=family if title == rows[0][0] else None,
                   color=method_colors[family], alpha=0.85, capsize=2)

        # Separator line before avg
        ax.axvline(len(tasks) - 0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_tasks, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # ── Plot ──
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(14, len(task_order) * 0.8), 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, (title, tasks) in zip(axes, rows):
        _plot_row(ax, tasks, title)

    axes[0].legend(fontsize=8, ncol=min(4, len(method_order)), loc="upper right")
    fig.suptitle("Comprehensive Evaluation — CoT Oracle (best variant per task)", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "comprehensive_eval.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out_path}")


def _rescore_from_cache(cache, run_id, config, output_dir, args):
    """Re-score methods that have cached predictions but no/failed scores. No GPU needed."""
    from tasks import get_comprehensive_eval_tasks
    all_tasks = get_comprehensive_eval_tasks()
    method_config = config.get("method_config", {})

    openrouter_available = check_openrouter_available()
    if not openrouter_available:
        raise RuntimeError("--rescore requires a scorer endpoint. Start local vLLM (port 8788), set SCORER_API_BASE, or set OPENROUTER_API_KEY.")

    # Find all unscored + error methods that have predictions
    unscored = cache.get_unscored_methods(run_id)
    # Also include error methods that have predictions (scoring crashed)
    for task, method, reason in cache.get_failed_methods(run_id):
        if cache.has_predictions(run_id, task, method) and (task, method) not in [(t, m) for t, m in unscored]:
            unscored.append((task, method))

    if not unscored:
        print("No methods to rescore.")
        return

    print(f"Rescoring {len(unscored)} method(s)...")
    for task_name, method_name in sorted(unscored):
        if task_name not in all_tasks:
            continue
        task_def = all_tasks[task_name]

        # Load predictions from cache
        cached_preds = cache.get_predictions(run_id, task_name, method_name)
        if not cached_preds:
            print(f"  [{task_name}/{method_name}] no cached predictions, skipping")
            continue

        # Reconstruct predictions/targets in index order
        test_data = load_and_normalize(task_name, args.n_examples)
        if not test_data:
            print(f"  [{task_name}/{method_name}] no test data")
            continue

        valid_mask = sorted(cached_preds.keys())
        predictions = [cached_preds[i]["prediction"] for i in valid_mask]
        targets = [test_data[i].get("target_response", "") if i < len(test_data) else "" for i in valid_mask]
        eval_items = [test_data[i] for i in valid_mask if i < len(test_data)]
        item_prompts = [cached_preds[i].get("prompt") for i in valid_mask]

        print(f"  [{task_name}/{method_name}] ", end="", flush=True)
        t0 = time.time()
        try:
            _score_and_store(
                cache, run_id, task_name, task_def, method_name, method_config,
                predictions, targets, eval_items, valid_mask, item_prompts,
                openrouter_available, t0,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f" scoring FAILED: {e}")

    task_names = list({t for t, _ in unscored})
    for task_name in task_names:
        task_json = cache.export_task_json(run_id, task_name)
        (output_dir / f"{task_name}.json").write_text(json.dumps(task_json, indent=2, default=str))

    plot_results(output_dir, cache=cache, run_id=run_id, tasks_order=task_names)
    stop_local_scorer()
    print("\nRescore done!")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CoT Oracle Evaluation")
    parser.add_argument("--config", default="configs/eval.yaml", help="Eval config (baselines, score_model)")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Train config (model, activations)")
    parser.add_argument("--checkpoint", default="", help="Our LoRA checkpoint path (default: from eval.yaml method_config.our_ao.checkpoint)")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--output-dir", default="data/comprehensive_eval")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all comprehensive eval tasks)")
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES, help="Baselines to run")
    parser.add_argument("--layers", nargs="*", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--rerun", action="store_true", help="Ignore cache, rerun everything")
    parser.add_argument("--rerun-methods", nargs="*", default=None, help="Rerun specific methods only")
    parser.add_argument("--rerun-tasks", nargs="*", default=None, help="Rerun specific tasks only")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun all previously failed evals (error + failed)")
    parser.add_argument("--rescore", action="store_true", help="Re-score methods that have cached predictions but failed scoring")
    parser.add_argument("--list-failed", action="store_true", help="List all failed/unscored evals and exit")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate the plot from existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(open(args.config))
    method_config = config.get("method_config", {})

    # Resolve checkpoint: CLI > config > empty
    checkpoint = args.checkpoint or method_config.get("our_ao", {}).get("checkpoint", "")

    cache = EvalCache(output_dir / "eval_cache.db")
    run_id = cache.get_or_create_run(checkpoint, args.n_examples, "all", args.layers)

    if args.list_failed:
        failed = cache.get_failed_methods(run_id)
        if not failed:
            print("No failed evals.")
        else:
            print(f"{len(failed)} failed eval(s):")
            for task, method, reason in sorted(failed):
                print(f"  {task:30s}  {method:30s}  {reason}")
        return

    if args.plot_only:
        plot_results(output_dir, cache=cache, run_id=run_id)
        return

    if args.rescore:
        _rescore_from_cache(cache, run_id, config, output_dir, args)
        return

    all_comprehensive = get_comprehensive_eval_tasks()
    # Tasks: CLI --tasks or eval.yaml list can include any task from TASKS (including cls_).
    # Only the fallback (no list specified) restricts to comprehensive-eval tasks.
    if args.tasks:
        task_names = [t for t in args.tasks if t in TASKS]
    elif "tasks" in config and config["tasks"]:
        task_names = [t for t in config["tasks"] if t in TASKS]
    else:
        task_names = list(all_comprehensive.keys())
    all_tasks = {t: TASKS[t] for t in task_names}

    # Baselines: CLI --baselines overrides, else eval.yaml list, else defaults
    if args.baselines != DEFAULT_BASELINES:
        active_baselines = list(args.baselines)
    elif "baselines" in config and config["baselines"]:
        active_baselines = list(config["baselines"])
    else:
        active_baselines = DEFAULT_BASELINES
    all_methods = expand_methods(active_baselines, method_config)
    print(f"Tasks: {len(task_names)}, Methods: {len(all_methods)}, N: {args.n_examples}")

    needs_model = any(b in _MODEL_METHODS for b in active_baselines)

    openrouter_available = check_openrouter_available()
    needs_scorer = any(b in ("weak-bb-monitor", "strong-bb-monitor", "sae-llm-monitor") for b in active_baselines)
    if not openrouter_available and needs_scorer:
        raise RuntimeError(
            "No scorer endpoint available but required by: "
            + ", ".join(b for b in active_baselines if b in ("weak-bb-monitor", "strong-bb-monitor", "sae-llm-monitor"))
            + ". Start local vLLM (port 8788), set SCORER_API_BASE, or set OPENROUTER_API_KEY."
        )

    # Init wandb for probe training if probes are active
    wandb_run = None
    needs_probes = any(b in ("linear_probes", "attention_probe") for b in active_baselines)
    if needs_probes:
        import wandb
        wandb_run = wandb.init(project="cot_oracle", entity="MATS10-CS-JB", group="probes", config={"n_examples": args.n_examples, "checkpoint": checkpoint, "layers": args.layers})

    model, tokenizer = None, None
    if needs_model:
        print("Loading Qwen3-8B + adapters...")
        import torch
        from core.ao import load_model_with_ao, load_extra_adapter

        train_cfg = yaml.safe_load(open(args.train_config))
        model, tokenizer = load_model_with_ao(train_cfg["model"]["name"])
        model.eval()

        if "our_ao" in active_baselines:
            our_ckpt = args.checkpoint or method_config.get("our_ao", {}).get("checkpoint", "")
            if not our_ckpt:
                raise ValueError("our_ao requires a checkpoint: set --checkpoint or method_config.our_ao.checkpoint in eval.yaml")
            if "default" in model.peft_config:
                model.delete_adapter("default")
            load_extra_adapter(model, our_ckpt, adapter_name="default")
            print(f"  Loaded our LoRA from {our_ckpt}")

    rerun_methods = set(args.rerun_methods) if args.rerun_methods else set()
    rerun_tasks = set(args.rerun_tasks) if args.rerun_tasks else set()

    # Build set of (task, method) pairs to force-rerun from --rerun-failed
    failed_pairs: set[tuple[str, str]] = set()
    if args.rerun_failed:
        for task, method, reason in cache.get_failed_methods(run_id):
            failed_pairs.add((task, method))
        print(f"Rerunning {len(failed_pairs)} failed eval(s)")

    for task_name in task_names:
        task_def = all_tasks[task_name]
        print(f"\n  === {task_name} ===")

        for method_name in all_methods:

            mcfg = build_method_config(method_name, method_config)
            deps_hash = cache.method_deps_hash(run_id, task_name, method_name, mcfg)

            force_rerun = args.rerun or method_name in rerun_methods or task_name in rerun_tasks or (task_name, method_name) in failed_pairs
            if force_rerun:
                cache.delete_method(run_id, task_name, method_name)
            elif cache.has_method(run_id, task_name, method_name, deps_hash):
                print(f"    [{method_name}] cached, skipping")
                continue

            try:
                _run_and_store_method(
                    cache, run_id, task_name, task_def, method_name,
                    method_config, model, tokenizer, args.n_examples,
                    args.layers, openrouter_available,
                    train_config=train_cfg if needs_model else None,
                    wandb_run=wandb_run,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
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
    stop_local_scorer()
    if wandb_run:
        wandb_run.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
