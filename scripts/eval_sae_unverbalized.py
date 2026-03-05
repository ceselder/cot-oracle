#!/usr/bin/env python3
"""
Evaluate the oracle checkpoint on sae_unverbalized + run bb monitor baseline.

Usage:
    python scripts/eval_sae_unverbalized.py --checkpoint checkpoints/cot_oracle_v15_stochastic
    python scripts/eval_sae_unverbalized.py --checkpoint checkpoints/cot_oracle_v15_stochastic --max-items 50 --no-bb
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ao_reference"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from tasks import TASKS
from eval_loop import run_eval, run_bb_monitor_eval


def main():
    parser = argparse.ArgumentParser(description="Eval oracle + bb monitor on sae_unverbalized")
    cache_dir = os.environ.get("CACHE_DIR", "/ceph/scratch/jbauer")
    parser.add_argument("--checkpoint", type=str, default=f"{cache_dir}/checkpoints/cot_oracle_v15_stochastic")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--no-bb", action="store_true", help="Skip bb monitor baseline")
    parser.add_argument("--bb-only", action="store_true", help="Only run bb monitor (no GPU needed)")
    parser.add_argument("--log-dir", type=str, default="logs/sae_unverbalized_eval")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    task_def = TASKS["sae_unverbalized"]

    # ── BB monitor baseline (no GPU needed) ──
    if not args.bb_only:
        print(f"\n{'='*60}")
        print("Loading model + checkpoint for oracle eval...")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.checkpoint, adapter_name="default")
        model.eval()

        print(f"\n{'='*60}")
        print(f"Oracle eval on sae_unverbalized (n={args.max_items})")
        print(f"{'='*60}")

        metrics, all_traces = run_eval(
            model=model,
            tokenizer=tokenizer,
            task_names=["sae_unverbalized"],
            max_items=args.max_items,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            layers=args.layers,
            run_bb_monitor=not args.no_bb,
        )

        # Save oracle traces
        oracle_traces = all_traces.get("sae_unverbalized", [])
        if oracle_traces:
            traces_path = log_dir / "oracle_traces.jsonl"
            with open(traces_path, "w") as f:
                for trace in oracle_traces:
                    f.write(json.dumps(trace, default=str) + "\n")
            print(f"\nOracle traces -> {traces_path}")

        # Save bb monitor traces
        bb_traces = all_traces.get("bb_monitor/sae_unverbalized", [])
        if bb_traces:
            traces_path = log_dir / "bb_monitor_traces.jsonl"
            with open(traces_path, "w") as f:
                for trace in bb_traces:
                    f.write(json.dumps(trace, default=str) + "\n")
            print(f"BB monitor traces -> {traces_path}")

    else:
        # BB-only mode: no GPU needed
        print(f"\n{'='*60}")
        print(f"BB monitor only on sae_unverbalized (n={args.max_items})")
        print(f"{'='*60}")

        bb_result = run_bb_monitor_eval("sae_unverbalized", task_def, max_items=args.max_items)
        bb_traces = bb_result.pop("_traces", [])
        metrics = {f"bb_monitor/sae_unverbalized": bb_result.get("llm_judge_score", 0.0)}
        for k, v in bb_result.items():
            if not k.startswith("_") and isinstance(v, (int, float)):
                metrics[f"bb_monitor/sae_unverbalized_{k}"] = v

        if bb_traces:
            traces_path = log_dir / "bb_monitor_traces.jsonl"
            with open(traces_path, "w") as f:
                for trace in bb_traces:
                    f.write(json.dumps(trace, default=str) + "\n")
            print(f"BB monitor traces -> {traces_path}")

    # Save metrics summary
    metrics_path = log_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics -> {metrics_path}")

    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<45s} {v:.3f}")
        else:
            print(f"  {k:<45s} {v}")
    print()


if __name__ == "__main__":
    main()
