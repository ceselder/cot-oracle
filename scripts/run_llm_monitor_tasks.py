#!/usr/bin/env python3
"""LLM monitor baseline for training tasks.

Loads cleaned HF datasets, calls LLM judge via baselines/llm_monitor.py,
logs results to wandb as baseline/{model_short_name} with flat reference lines.

Usage:
    python scripts/run_llm_monitor_tasks.py --config configs/train.yaml
    python scripts/run_llm_monitor_tasks.py --tasks hint_admission backtrack_prediction --no-wandb
    python scripts/run_llm_monitor_tasks.py --model google/gemini-3-flash-lite --n 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")

_DIR = Path(__file__).resolve().parent.parent
for p in [_DIR / "baselines", _DIR / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared import BaselineInput
from llm_monitor import run_llm_monitor
from scoring import EVAL_TYPES
from tasks import TASKS

# Label mappings: dataset label → EVAL_BINARY_CONFIG option name
LABEL_MAPS = {
    "hint_admission": {"hint_resisted": "independent", "hint_used_wrong": "influenced", "hint_used_correct": "influenced"},
    "truthfulqa_hint_verbalized": {"hint_resisted": "independent", "hint_used_wrong": "influenced", "hint_used_correct": "influenced"},
    "truthfulqa_hint": {"hint_resisted": "independent", "hint_used_wrong": "influenced", "hint_used_correct": "influenced"},
}

# Tasks that cannot be evaluated by an LLM text monitor (need activations)
SKIP_TASKS = {
    "futurelens", "pastlens", "resampling_importance",
    "futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb",
    "rot13_reconstruction", "sae_unverbalized",
    "probe_sycophancy", "taxonomy_ood", "deception_detection",
}

# Datasets with train/test schema mismatch on HF
BROKEN_DATASETS = {"correctness"}

# Datasets that only have a 'test' split
TEST_SPLIT_ONLY = {"cot_metacognition"}


def _extract_fields(task_name, row):
    """Extract (test_prompt, cot_text, reference) per task schema."""
    # Binary tasks and simple cleaned datasets
    if task_name in ("hint_admission", "truthfulqa_hint_verbalized", "truthfulqa_hint"):
        prompt = (
            f"Question: {row['question']}\n"
            f"Hint shown to the model: \"{row['hint_text']}\" (suggests answer {row['hint_answer']})\n"
            f"Correct answer: {row['correct_answer']}"
        )
        return prompt, row["cot_text"], row.get("target_response", "")
    if task_name == "atypical_answer":
        return f"Question: {row['question']}\nModel's answer: {row.get('model_answer', '')}", row["cot_text"], ""
    if task_name == "sycophancy":
        prompt = (
            f"Question: {row['question']}\n"
            f"User's nudge: \"{row['nudge_text']}\" (pushing answer {row['nudge_answer']})\n"
            f"Correct answer: {row['ground_truth']}"
        )
        return prompt, row["cot_text"], ""

    # Generation: convqa / chunked_convqa
    if task_name in ("convqa", "chunked_convqa"):
        cot = row.get("cot_prefix", row["cot_text"])
        return row["prompt"], cot, row["target_response"]

    # Generation: compqa (different field names)
    if task_name == "compqa":
        return row["test_prompt"], row["meta_cot_text"], row["correct_answer"]

    # Generation: chunked_compqa
    if task_name == "chunked_compqa":
        return row["prompt"], row["cot_prefix"], row["target_response"]

    # Generation: sqa (precomputed format — reconstruct text from message dicts)
    if task_name == "sqa":
        dialog = "\n".join(m["content"] for m in row["raw_dialog"])
        read_prompt = "\n".join(m["content"] for m in row["raw_read_prompt"])
        return read_prompt, dialog, row["target_output"]

    # Generation: sentence_insertion, cot_description, cot_metacognition
    if task_name in ("sentence_insertion", "cot_description", "cot_metacognition"):
        return row["prompt"], row["cot_text"], row["target_response"]

    # Default: binary tasks with question + cot_text
    return row["question"], row["cot_text"], row.get("target_response", row.get("correct_answer", ""))


def load_inputs(task_name, n):
    """Load HF dataset and build BaselineInput list."""
    hf_repo = TASKS[task_name].hf_repo

    # Determine split
    if task_name in TEST_SPLIT_ONLY:
        split = "test"
    else:
        split = "train"

    if task_name in BROKEN_DATASETS:
        ds = load_dataset(hf_repo, data_files=f"data/{split}-*.parquet", split=split).shuffle(seed=42)
    else:
        ds = load_dataset(hf_repo, split=split).shuffle(seed=42)
    if n and len(ds) > n:
        ds = ds.select(range(n))

    label_map = LABEL_MAPS.get(task_name)
    eval_type = EVAL_TYPES[task_name]

    inputs = []
    for i, row in enumerate(ds):
        test_prompt, cot_text, reference = _extract_fields(task_name, row)

        if eval_type == "binary":
            raw_label = row["label"]
            label = label_map[raw_label] if label_map else raw_label
        else:
            label = reference

        inputs.append(BaselineInput(
            eval_name=task_name,
            example_id=f"{task_name}_{i}",
            clean_prompt=test_prompt, test_prompt=test_prompt,
            correct_answer=reference,
            nudge_answer=row.get("nudge_answer"),
            ground_truth_label=label,
            clean_response=cot_text, test_response=cot_text,
            activations_by_layer={},
            metadata=dict(row),
        ))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=1, help="Number of independent runs to average")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lm_cfg = cfg["baselines"]["llm_monitor"]
    model = args.model or lm_cfg["model"]
    api_base = lm_cfg["api_base"]
    api_key = os.environ["OPENROUTER_API_KEY"]
    max_tokens = lm_cfg.get("max_tokens", 300)
    temperature = lm_cfg.get("temperature", 0.0)
    n = args.n or cfg["eval"].get("max_items_per_eval", 25)

    # Determine tasks
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = [
            t for t in cfg["tasks"]
            if cfg["tasks"][t].get("eval") and t not in SKIP_TASKS and t in EVAL_TYPES
        ]

    n_runs = args.n_runs
    model_short = model.split("/")[-1]
    run_name = f"baseline/{model_short}"

    log_dir = Path("logs/llm_monitor_tasks")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter valid tasks once
    valid_tasks = [t for t in task_names if t in EVAL_TYPES and t in TASKS]
    for t in task_names:
        if t not in EVAL_TYPES: print(f"Skipping {t}: not in EVAL_TYPES")
        elif t not in TASKS: print(f"Skipping {t}: not in TASKS")

    all_run_scores = []  # list of {task_name: score}
    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n{'#'*60}\n  RUN {run_idx + 1}/{n_runs}\n{'#'*60}")

        run_scores = {}
        for task_name in valid_tasks:
            print(f"\n{'='*60}\n{task_name}\n{'='*60}")

            inputs = load_inputs(task_name, n)
            print(f"  {len(inputs)} items loaded")

            # Separate cache per run so each run makes fresh API calls
            suffix = f"_r{run_idx}" if n_runs > 1 else ""
            traces_path = log_dir / f"{task_name}_traces{suffix}.jsonl"
            results = run_llm_monitor(
                inputs, model=model, api_base=api_base, api_key=api_key,
                max_tokens=max_tokens, temperature=temperature,
                cache_path=traces_path,
            )

            with open(traces_path, "w") as f:
                for trace in results["traces"]:
                    f.write(json.dumps(trace, default=str) + "\n")
            print(f"  Traces -> {traces_path}")

            m = results.get("metrics", {})
            score = m.get("accuracy", m.get("mean_gemini_score", m.get("mean_token_f1")))
            if score is not None:
                run_scores[task_name] = score
                print(f"  Score: {score:.3f}")

        all_run_scores.append(run_scores)

    # Average across runs
    metrics = {}
    all_tasks_with_scores = set().union(*all_run_scores)
    for task_name in all_tasks_with_scores:
        vals = [rs[task_name] for rs in all_run_scores if task_name in rs]
        metrics[f"eval/{task_name}"] = sum(vals) / len(vals)

    if not metrics:
        print("No metrics collected.")
        return

    accs = [v for k, v in metrics.items() if k.startswith("eval/")]
    if accs:
        metrics["eval/mean_acc"] = sum(accs) / len(accs)

    print(f"\n{'='*60}\nResults for {model_short} (averaged over {n_runs} run(s)):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")

    if args.no_wandb:
        return

    import wandb
    wandb_project = cfg["output"].get("wandb_project", "cot_oracle")
    wandb_entity = cfg["output"].get("wandb_entity")

    # Delete old run with same name
    api = wandb.Api()
    for r in api.runs(f"{wandb_entity}/{wandb_project}", per_page=50):
        if r.name == run_name:
            print(f"Deleting old wandb run {r.id} ({r.name})")
            r.delete()

    run = wandb.init(
        project=wandb_project, entity=wandb_entity, name=run_name,
        tags=["baseline", "llm_monitor", model_short],
        config={"model": model, "n_per_task": n, "type": "llm_monitor"},
    )
    wandb.define_metric("train/samples_seen")
    wandb.define_metric("*", step_metric="train/samples_seen")
    for step in [0, 100_000]:
        wandb.log({**metrics, "train/samples_seen": step}, step=step)
    run.finish()
    print(f"\nPushed to wandb: {run_name}")


if __name__ == "__main__":
    main()
