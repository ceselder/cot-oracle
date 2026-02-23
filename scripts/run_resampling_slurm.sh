#!/bin/bash
#SBATCH --job-name=qwen3-resampling
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/resampling_%j.out
#SBATCH --error=logs/resampling_%j.err

set -euo pipefail

cleanup() { kill -- -$$ 2>/dev/null || true; }
trap cleanup EXIT SIGTERM SIGINT

# Load env vars (HF_HOME, CACHE_DIR, etc.)
set -a; source ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

NGPUS=${SLURM_GPUS_ON_NODE:-4}

# Activate venv
VENV_LOCAL="${CACHE_DIR}/venvs"
source "${VENV_LOCAL}/cot-oracle/bin/activate"

# Phase 1+2+3: download, base solutions, rollouts (4-GPU data parallel), KL scoring
python scripts/collect_resampling_importance.py \
    --num-gpus "$NGPUS" \
    --num-rollouts 100 \
    --resume
