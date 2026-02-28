#!/bin/bash
#SBATCH --job-name=interleaved
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/interleaved_%j.out
#SBATCH --error=slurm_logs/interleaved_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AO_REPO_PATH=/nfs/nhome/live/jbauer/activation_oracles
export VENV=/var/tmp/jbauer/venvs/cot-oracle
export PATH="$VENV/bin:$PATH"
export HF_HOME="/ceph/scratch/jbauer/hf"
export HF_HUB_CACHE="/ceph/scratch/jbauer/hf/hub"

cd /nfs/nhome/live/jbauer/cot-oracle

echo "=== COT Oracle: Interleaved Curriculum (2 GPU) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# ── Clean stale GPU processes ────────────────────────────────
pkill -f "torchrun.*train" 2>/dev/null || true
sleep 2

# ── Sync venv ────────────────────────────────────────────────
echo "Syncing venv at $VENV..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync
echo "Using python: $(python --version 2>&1)"
echo ""

# ── Launch training ──────────────────────────────────────────
torchrun --nproc_per_node=2 \
    --master_port=29501 \
    src/train.py \
    --config configs/train.yaml \
    --precomputed-dir data/precomputed \
    --task-order interleaved \
    --batch-size 4 \
    --effective-batch-size 256 \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_interleaved \
    --wandb-run interleaved_v1

echo ""
echo "=== Done: $(date) ==="
