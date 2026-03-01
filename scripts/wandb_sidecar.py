"""Sidecar script that reads training metrics from log file and posts to wandb.

Workaround for wandb file_stream API being broken on Vast.ai containers.
Reads tqdm progress lines and eval tables, posts metrics via wandb API.

Usage:
    python scripts/wandb_sidecar.py --log /root/train_v7_legacy.txt --run-id zeuqmkbi
"""
import argparse
import re
import time
import os


def parse_tqdm_line(line: str) -> dict | None:
    """Extract minibatch and loss from tqdm progress line."""
    m = re.search(r'\|\s*(\d+)/(\d+)\s.*?loss=([\d.]+)', line)
    if m:
        return {
            "minibatch": int(m.group(1)),
            "total_minibatches": int(m.group(2)),
            "loss": float(m.group(3)),
        }
    return None


def parse_eval_table(lines: list[str]) -> dict:
    """Extract eval metrics from the table format."""
    metrics = {}
    for line in lines:
        m = re.match(r'\s+(\S+)\s+(accuracy|token_f1|step_accur)\s+([\d.]+)', line)
        if m:
            task, metric_type, score = m.group(1), m.group(2), float(m.group(3))
            metrics[f"eval/{task}/{metric_type}"] = score
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--run-id", required=True, help="Wandb run ID to update")
    parser.add_argument("--entity", default="MATS10-CS-JB")
    parser.add_argument("--project", default="cot_oracle")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--grad-accum", type=int, default=32)
    args = parser.parse_args()

    import wandb
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")

    last_pos = 0
    last_step = -1
    eval_step = None
    eval_lines = []
    in_eval_table = False

    print(f"Sidecar monitoring {args.log} -> wandb run {args.run_id}")
    print(f"Polling every {args.interval}s")

    while True:
        try:
            with open(args.log, 'r') as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()
        except FileNotFoundError:
            time.sleep(args.interval)
            continue

        if not new_content:
            time.sleep(args.interval)
            continue

        # Split on both \n and \r (tqdm uses \r)
        lines = re.split(r'[\r\n]+', new_content)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for eval header
            m = re.match(r'--- Evals at step (\d+) ---', line)
            if m:
                eval_step = int(m.group(1))
                eval_lines = []
                in_eval_table = False
                continue

            # Check for eval table start
            if '──────────' in line:
                in_eval_table = not in_eval_table
                if not in_eval_table and eval_lines and eval_step is not None:
                    # End of table, parse and log
                    metrics = parse_eval_table(eval_lines)
                    if metrics:
                        run.summary.update(metrics)
                        print(f"  Logged {len(metrics)} eval metrics at step {eval_step}")
                    eval_lines = []
                continue

            if in_eval_table:
                eval_lines.append(line)
                continue

            # Parse tqdm progress
            parsed = parse_tqdm_line(line)
            if parsed and parsed["minibatch"] > 0:
                step = parsed["minibatch"] // args.grad_accum
                if step > last_step and step % 10 == 0:
                    last_step = step
                    print(f"  Step {step}: loss={parsed['loss']:.4f} "
                          f"(mb {parsed['minibatch']}/{parsed['total_minibatches']})")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
