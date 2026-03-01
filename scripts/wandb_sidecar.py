"""Sidecar that uploads training metrics to wandb from a JSONL file.

Workaround for wandb file_stream API being broken on Vast.ai containers.
train.py writes metrics to a JSONL file; this script reads them and uploads
via short-lived wandb sessions (init → log batch → finish) which reliably sync.

Usage:
    # Auto-detect run config from the metrics file header:
    python scripts/wandb_sidecar.py --metrics /root/checkpoints/metrics.jsonl

    # Or specify manually:
    python scripts/wandb_sidecar.py --metrics /root/checkpoints/metrics.jsonl \
        --run-id abc123 --entity MATS10-CS-JB --project cot_oracle
"""
import argparse
import json
import time
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl written by train.py")
    parser.add_argument("--run-id", default=None, help="Wandb run ID (auto-detected from file if omitted)")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--interval", type=int, default=120, help="Upload interval in seconds")
    args = parser.parse_args()

    import wandb

    # Read metadata header from metrics file
    run_id = args.run_id
    entity = args.entity or "MATS10-CS-JB"
    project = args.project or "cot_oracle"
    run_name = None

    last_line = 0
    pending = []

    def read_new_lines():
        nonlocal last_line, run_id, entity, project, run_name
        try:
            with open(args.metrics, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        new_lines = lines[last_line:]
        last_line = len(lines)

        records = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Handle metadata header
            if data.get("_meta"):
                run_id = run_id or data.get("run_id")
                entity = args.entity or data.get("entity", entity)
                project = args.project or data.get("project", project)
                run_name = data.get("run_name")
                continue
            records.append(data)
        return records

    # Wait for metrics file to appear
    print(f"Sidecar waiting for {args.metrics}...")
    while True:
        records = read_new_lines()
        if run_id:
            pending.extend(records)
            break
        time.sleep(5)

    print(f"Sidecar uploading to wandb run {run_id} ({entity}/{project})")
    print(f"Upload interval: {args.interval}s")

    while True:
        # Read any new metrics
        new_records = read_new_lines()
        pending.extend(new_records)

        if not pending:
            time.sleep(args.interval)
            continue

        # Upload batch via short-lived wandb session
        batch = list(pending)
        n_batch = len(batch)
        try:
            wandb.init(
                id=run_id,
                project=project,
                entity=entity,
                name=run_name,
                resume="allow",
            )
            for record in batch:
                step = record.pop("_step", None)
                # Skip empty records
                if not record:
                    continue
                wandb.log(record, step=step)
            wandb.finish(quiet=True)
            pending.clear()
            print(f"  [sidecar] Uploaded {n_batch} metric records")
        except Exception as e:
            print(f"  [sidecar] Upload failed: {e}", file=sys.stderr)
            # Keep pending for retry

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
