"""Log Gemini baselines as a constant wandb run for visual comparison.

Reads logs/gemini_baseline/results.json (produced by run_gemini_eval_all_tasks.py),
maps each task to its primary metric via the task system, and logs as horizontal
lines at eval/{task_name} across the full x-axis.

Deletes any existing gemini-flash-baseline run before creating a new one.
"""

import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tasks import TASKS
from eval_loop import _primary_metric_name

MAX_SAMPLES = 1_000_000
PROJECT = "cot_oracle"
ENTITY = "japhba-personal"
RUN_NAME = "gemini-flash-baseline"

# ── Delete old run ──

api = wandb.Api()
try:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": RUN_NAME})
    for run in runs:
        print(f"  Deleting old run: {run.id} ({run.name})")
        run.delete()
except Exception as e:
    print(f"  No old run to delete (or error): {e}")

# ── Load new results ──

results = json.load(open("logs/gemini_baseline/results.json"))

baselines = {}
for task_name, data in results.items():
    task_def = TASKS[task_name]
    primary_metric = _primary_metric_name(task_name, task_def.scoring)
    score = data["metrics"][primary_metric]
    metric_key = f"eval/{task_name}"
    baselines[metric_key] = score
    print(f"  {task_name} → {metric_key} = {score:.3f}")

# ── Log to wandb ──

wandb.init(
    project=PROJECT, entity=ENTITY, name=RUN_NAME,
    tags=["baseline", "gemini"],
    config={"model": "gemini-3-flash-preview", "type": "gemini_baseline"},
)
wandb.define_metric("train/samples_seen")
wandb.define_metric("*", step_metric="train/samples_seen")

for step in [0, MAX_SAMPLES]:
    wandb.log({**baselines, "train/samples_seen": step}, step=step)

wandb.finish()
print(f"\nDone — logged {len(baselines)} Gemini baselines as lines (0 → {MAX_SAMPLES} samples).")
