#!/usr/bin/env python3
"""Retroactively upload existing probe checkpoints as wandb artifacts and to HF."""
import os, sys, json
from pathlib import Path
import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")

CACHE_DIR = Path(os.environ["CACHE_DIR"])
CKPT_DIR = CACHE_DIR / "checkpoints" / "qwen_attention_probe"
HF_ORG = "japhba"
WANDB_PROJECT = "cot_oracle"
WANDB_ENTITY = "MATS10-CS-JB"

# Map checkpoint dirs to the most recent wandb run name
# checkpoint_dir -> (wandb_run_name, wandb_run_id, probe_type)
CHECKPOINT_MAP = {
    "linear_hint_admission":       ("linear-mean-concat-hint_admission",       "3hphqa1b", "linear"),
    "linear_sycophancy":           ("linear-mean-concat-sycophancy",           "t8q5zht1", "linear"),
    "linear_truthfulqa_hint":      ("linear-mean-concat-truthfulqa_hint",      "i0gcqhkb", "linear"),
    "attention_hint_admission":    ("qwen-attprobe-hint_admission",            "qwsm7gcd", "attention"),
    "attention_sycophancy":        ("qwen-attprobe-sycophancy",                "gtskzycm", "attention"),
    "attention_truthfulqa_hint":   ("qwen-attprobe-truthfulqa_hint",           "yjweow61", "attention"),
}

api = wandb.Api()

from huggingface_hub import HfApi
hf_api = HfApi()

for ckpt_name, (run_name, run_id, probe_type) in CHECKPOINT_MAP.items():
    ckpt_path = CKPT_DIR / ckpt_name / "model.pt"
    if not ckpt_path.exists():
        print(f"SKIP {ckpt_name}: no checkpoint")
        continue

    task_name = ckpt_name.replace(f"{probe_type}_", "")
    print(f"\n{'='*60}")
    print(f"  {ckpt_name} -> run={run_name} (id={run_id})")
    
    # 1. Upload wandb artifact to the existing run
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=run_id, resume="must") as resumed_run:
        artifact = wandb.Artifact(name=run_name, type="model",
                                  metadata={"task": task_name, "probe_type": probe_type, "run_name": run_name})
        artifact.add_file(str(ckpt_path), name="model.pt")
        resumed_run.log_artifact(artifact)
        print(f"  Logged wandb artifact: {run_name}")

    # 2. Upload to HF with run_name in metadata
    repo_id = f"{HF_ORG}/qwen-{probe_type}-probe-{task_name}"
    hf_api.create_repo(repo_id, exist_ok=True, repo_type="model")
    hf_api.upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="model.pt", repo_id=repo_id)
    
    # Write a metadata card
    card = f"""---
tags:
  - cot-oracle
  - probe
---
# {run_name}

Probe type: `{probe_type}`  
Task: `{task_name}`  
wandb run: `{run_name}` (id: `{run_id}`)  
wandb project: `{WANDB_ENTITY}/{WANDB_PROJECT}`
"""
    hf_api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    print(f"  Uploaded to https://huggingface.co/{repo_id}")

print("\nDone!")
