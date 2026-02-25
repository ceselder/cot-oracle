"""
Standalone LLM monitor baseline runner — no GPU needed.

Loads precomputed Qwen3-8B responses from eval JSON metadata and evaluates
an external LLM judge (via OpenRouter) on all supported evals.

Usage:
    python baselines/run_llm_monitor.py --config configs/train.yaml
    python baselines/run_llm_monitor.py --config configs/train.yaml --evals hinted_mcq decorative_cot
    python baselines/run_llm_monitor.py --config configs/train.yaml --no-wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

_DIR = Path(__file__).resolve().parent
_SRC = _DIR.parent / "src"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv(Path.home() / ".env")

from evals.common import EvalItem, load_eval_items_hf, determine_ground_truth, extract_letter_answer
from scoring import EVAL_TYPES
from shared import BaselineInput
from llm_monitor import run_llm_monitor

# Evals to skip (poor LLM judge fit)
SKIP_EVALS = {"sentence_insertion", "forced_answer_entropy_riya"}

# Map eval -> how to get (test_response, clean_response, ground_truth) from EvalItem
# Each entry: (test_response_fn, clean_response_fn, ground_truth_fn)
# where each fn takes (item: EvalItem) -> str | None


def _gt_counterfactual(item: EvalItem, test_response: str, clean_response: str) -> str:
    """Ground truth via counterfactual clean/test comparison."""
    clean_answer = extract_letter_answer(clean_response)
    test_answer = extract_letter_answer(test_response)
    return determine_ground_truth(item, clean_answer, test_answer)


def _load_items_for_eval(eval_name: str, items: list[EvalItem]) -> list[BaselineInput]:
    """Build BaselineInput objects from precomputed metadata — no GPU needed."""
    results = []
    skip_labels = {"indeterminate", "pending_multi_run", "pending_manual",
                   "pending_reconstruction", "pending_entropy_regression"}

    for item in items:
        meta = item.metadata

        # Extract test_response, clean_response, and ground_truth per eval
        if eval_name == "hinted_mcq":
            test_response = meta.get("qwen3_8b_test_response")
            clean_response = meta.get("qwen3_8b_clean_response")
            if not test_response or not clean_response:
                continue
            gt = _gt_counterfactual(item, test_response, clean_response)

        elif eval_name == "hinted_mcq_truthfulqa":
            hinted = meta.get("hinted_rollouts", [])
            clean = meta.get("clean_rollouts", [])
            if not hinted or not clean:
                continue
            test_response = hinted[0]
            clean_response = clean[0]
            gt = meta.get("ground_truth_label")
            if not gt:
                gt = _gt_counterfactual(item, test_response, clean_response)

        elif eval_name == "sycophancy_v2_riya":
            test_response = meta.get("qwen3_8b_test_response")
            clean_response = meta.get("qwen3_8b_clean_response")
            if not test_response or not clean_response:
                continue
            gt = determine_ground_truth(item, None, None)  # uses meta["label"]

        elif eval_name == "decorative_cot":
            test_response = meta.get("representative_cot", "")
            clean_response = test_response
            gt = meta.get("decorative_label")
            if not gt or not test_response:
                continue

        elif eval_name == "reasoning_termination_riya":
            test_response = meta.get("cot_prefix", "")
            clean_response = test_response
            gt = item.correct_answer
            if not test_response:
                continue

        elif eval_name in ("atypical_answer_riya", "atypical_answer_mcq"):
            test_response = meta.get("cot_text", "")
            clean_response = test_response
            gt = item.correct_answer
            if not test_response:
                continue

        elif eval_name == "cybercrime_ood":
            test_response = meta.get("qwen3_8b_test_response")
            clean_response = meta.get("qwen3_8b_clean_response")
            if not test_response or not clean_response:
                continue
            gt = meta.get("label")

        elif eval_name == "rot13_reconstruction":
            test_response = meta.get("qwen3_8b_test_response")
            clean_response = meta.get("qwen3_8b_clean_response")
            if not test_response or not clean_response:
                continue
            gt = "generation"

        elif eval_name == "compqa":
            test_response = meta.get("cot_text", "")
            clean_response = test_response
            gt = "generation"
            if not test_response:
                continue

        else:
            continue

        if gt in skip_labels:
            continue

        results.append(BaselineInput(
            eval_name=eval_name,
            example_id=item.example_id,
            clean_prompt=item.clean_prompt,
            test_prompt=item.test_prompt,
            correct_answer=item.correct_answer,
            nudge_answer=item.nudge_answer,
            ground_truth_label=gt,
            clean_response=clean_response,
            test_response=test_response,
            activations_by_layer={},
            metadata=meta,
        ))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM monitor baseline on all evals (no GPU)")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--evals", nargs="+", default=None, help="Override which evals to run")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lm_cfg = cfg["baselines"]["llm_monitor"]
    api_key = os.environ["OPENROUTER_API_KEY"]
    eval_dir = Path(cfg["eval"]["eval_dir"]) if "eval" in cfg else None

    # Determine evals to run
    if args.evals:
        eval_names = args.evals
    else:
        raw = cfg["eval"]["evals"]
        eval_names = []
        for entry in raw:
            name = list(entry.keys())[0] if isinstance(entry, dict) else entry
            if name not in SKIP_EVALS and name in EVAL_TYPES:
                eval_names.append(name)

    log_dir = Path("logs/llm_monitor")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Init wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg["output"].get("wandb_project", "cot_oracle"),
            entity=cfg["output"].get("wandb_entity"),
            name="llm_monitor_baselines",
            config={"llm_model": lm_cfg["model"], "evals": eval_names},
        )

    all_results: dict[str, dict] = {}

    for eval_name in eval_names:
        print(f"\n{'='*60}\n{eval_name}\n{'='*60}")
        eval_type = EVAL_TYPES.get(eval_name)
        if not eval_type:
            print(f"  Unknown eval type, skipping")
            continue

        items = load_eval_items_hf(eval_name, eval_dir=eval_dir)
        print(f"  Loaded {len(items)} raw items")

        inputs = _load_items_for_eval(eval_name, items)
        print(f"  {len(inputs)} usable items after filtering")
        if not inputs:
            continue

        traces_path = log_dir / f"{eval_name}_traces.jsonl"
        results = run_llm_monitor(
            inputs, model=lm_cfg["model"], api_base=lm_cfg["api_base"],
            api_key=api_key, max_tokens=lm_cfg["max_tokens"],
            temperature=lm_cfg["temperature"],
            cache_path=traces_path,
        )

        # Save JSONL traces (also serves as cache for next run)
        with open(traces_path, "w") as f:
            for trace in results["traces"]:
                f.write(json.dumps(trace, default=str) + "\n")
        print(f"  Traces -> {traces_path}")

        # Save summary JSON
        summary = {k: v for k, v in results.items() if k != "traces"}
        summary_path = log_dir / f"{eval_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Summary -> {summary_path}")

        # Wandb logging
        if wandb_run:
            import wandb
            prefix = f"llm_monitor/{eval_name}"
            metrics = results.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    wandb_run.log({f"{prefix}/{k}": v})
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            wandb_run.log({f"{prefix}/{k}/{k2}": v2})

            # Trace table
            if results["traces"]:
                cols = list(results["traces"][0].keys())
                table = wandb.Table(columns=cols)
                for trace in results["traces"][:50]:
                    table.add_data(*[trace.get(c, "") for c in cols])
                wandb_run.log({f"{prefix}/traces": table})

        all_results[eval_name] = results

    # Save combined results
    combined = {}
    for eval_name, results in all_results.items():
        combined[eval_name] = {k: v for k, v in results.items() if k != "traces"}
    combined_path = log_dir / "results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined results -> {combined_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("LLM MONITOR BASELINE RESULTS")
    print(f"{'='*80}")
    print(f"{'Eval':<30s} {'Type':<12s} {'Score':<12s} {'N items':<10s}")
    print(f"{'-'*64}")

    for eval_name, results in all_results.items():
        eval_type = EVAL_TYPES.get(eval_name, "?")
        n_items = results.get("n_items", "?")
        metrics = results.get("metrics", {})

        if "accuracy" in metrics:
            score = f"{metrics['accuracy']:.3f}"
        elif "mean_token_f1" in metrics:
            score = f"{metrics['mean_token_f1']:.3f}"
        else:
            score = "N/A"

        print(f"  {eval_name:<28s} {eval_type:<12s} {score:<12s} {str(n_items):<10s}")

    if wandb_run:
        wandb_run.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
