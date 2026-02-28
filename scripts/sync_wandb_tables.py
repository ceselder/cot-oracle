#!/usr/bin/env python3
"""
Sync eval traces from wandb runs to local eval_logs_wandb/ directory.

Downloads eval_traces artifacts (preferred, full-text) and falls back to
eval_table/* media files (truncated by wandb) for runs without artifacts.

Usage:
    # Sync all runs
    python scripts/sync_wandb_tables.py

    # Sync specific runs by name (substring match)
    python scripts/sync_wandb_tables.py --filter ablation-stride5

    # Sync only runs with >100 steps
    python scripts/sync_wandb_tables.py --min-steps 100

    # Dry run (show what would be downloaded)
    python scripts/sync_wandb_tables.py --dry-run

    # Also sync legacy table files (truncated by wandb)
    python scripts/sync_wandb_tables.py --tables
"""

import argparse
import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb
from tqdm.auto import tqdm

PROJECT = "MATS10-CS-JB/cot_oracle"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_logs_wandb"

FILE_RE = re.compile(r"media/table/eval_table/(.+)_(\d+)_[0-9a-f]+\.table\.json")


def run_dir_name(run) -> str:
    """Build date-prefixed directory name: YYYYMMDD_run-name."""
    created = run.created_at  # ISO 8601 string like "2026-02-25T10:25:14Z"
    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    return f"{dt.strftime('%Y%m%d_%H%M')}_{run.name}"


def retry(fn, retries=3, delay=5):
    """Retry a callable on any exception, with exponential backoff."""
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            tqdm.write(f"    retry {i+1}/{retries} after {type(e).__name__}: {e}")
            time.sleep(delay * (2 ** i))


def parse_table_file(name: str) -> tuple[str, int] | None:
    """Extract (eval_name, step) from a wandb media file path."""
    m = FILE_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def download_and_convert(run, file_obj, eval_name: str, step: int, out_dir: Path):
    """Download a wandb table file and save in our local JSON format."""
    out_path = out_dir / f"eval_table_{eval_name}_step{step}.json"
    if out_path.exists():
        return False  # already synced

    with tempfile.TemporaryDirectory() as tmpdir:
        retry(lambda: file_obj.download(tmpdir, replace=True))
        src = Path(tmpdir) / file_obj.name
        with open(src) as f:
            data = json.load(f)

    columns = data["columns"]
    rows_raw = data["data"]
    records = [dict(zip(columns, row)) for row in rows_raw]

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "name": f"eval_table_{eval_name}",
        "n": len(records),
        "run_id": run.id,
        "run_name": run.name,
        "created_at": run.created_at,
        "rows": records,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return True


def sync_artifacts(run, out_dir: Path, dry_run: bool = False) -> int:
    """Download eval_traces artifacts for a run. Returns count of new files."""
    try:
        artifacts = retry(lambda: list(run.logged_artifacts()))
    except Exception as e:
        tqdm.write(f"  {run.name}: no artifacts ({type(e).__name__})")
        return 0

    eval_artifacts = [a for a in artifacts if a.type == "eval_traces"]
    if not eval_artifacts:
        return 0

    new = 0
    for art in tqdm(eval_artifacts, desc=f"  {run.name} artifacts", leave=False):
        step = art.metadata.get("step", None)
        # Check if any files from this artifact are already present
        # Artifact dir contains files like eval_table_<name>_step<N>.json
        # We download to a temp dir first, then copy only new files
        if dry_run:
            tqdm.write(f"    would download artifact: {art.name} (step {step})")
            new += 1
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            retry(lambda a=art: a.download(tmpdir))
            tmppath = Path(tmpdir)
            for src in tmppath.rglob("*.json"):
                dst = out_dir / src.name
                if dst.exists():
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())
                new += 1
    return new


def sync_run(run, dry_run: bool = False, artifacts_only: bool = False) -> int:
    """Sync all eval traces for a single run. Returns count of new files."""
    run_dir = OUTPUT_DIR / run_dir_name(run)

    # Try artifacts first (full-text, preferred)
    new = sync_artifacts(run, run_dir, dry_run=dry_run)

    if artifacts_only:
        return new

    # Fall back to table files for anything not covered by artifacts
    try:
        all_files = retry(lambda: list(run.files()))
    except Exception as e:
        tqdm.write(f"  {run.name} ({run.id}): skipped files ({type(e).__name__})")
        return new
    table_files = [(f, parse_table_file(f.name)) for f in all_files if "eval_table" in f.name and f.name.endswith(".table.json")]
    table_files = [(f, parsed) for f, parsed in table_files if parsed is not None]

    if not table_files:
        return new

    # Skip already-synced files (including those just pulled from artifacts)
    to_download = []
    for f, (eval_name, step) in table_files:
        out_path = run_dir / f"eval_table_{eval_name}_step{step}.json"
        if not out_path.exists():
            to_download.append((f, eval_name, step))

    if not to_download:
        return new

    if dry_run:
        for _, eval_name, step in to_download:
            print(f"    would download table: eval_table_{eval_name}_step{step}.json")
        return new + len(to_download)

    for f, eval_name, step in tqdm(to_download, desc=f"  {run.name} tables", leave=False):
        if download_and_convert(run, f, eval_name, step, run_dir):
            new += 1
    return new


def main():
    parser = argparse.ArgumentParser(description="Sync eval traces from wandb to eval_logs_wandb/")
    parser.add_argument("--filter", type=str, default=None, help="Only sync runs whose name contains this substring")
    parser.add_argument("--min-steps", type=int, default=50, help="Skip runs with fewer than N steps (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--tables", action="store_true", help="Also download legacy table files (truncated by wandb)")
    parser.add_argument("--project", type=str, default=PROJECT, help="Wandb project path")
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    runs = retry(lambda: list(api.runs(args.project)))

    # Filter runs
    eligible = []
    for r in runs:
        if r.lastHistoryStep < args.min_steps:
            continue
        if args.filter and args.filter not in r.name:
            continue
        eligible.append(r)

    print(f"Found {len(eligible)} runs to sync (filter={args.filter!r}, min_steps={args.min_steps})")

    total_new = 0
    for run in tqdm(eligible, desc="Runs"):
        n = sync_run(run, dry_run=args.dry_run, artifacts_only=not args.tables)
        if n > 0:
            tqdm.write(f"  {run.name} ({run.id}): {'would sync' if args.dry_run else 'synced'} {n} files")
        total_new += n

    action = "would download" if args.dry_run else "downloaded"
    print(f"\nDone. {action} {total_new} new files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
