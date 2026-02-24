#!/bin/bash
#SBATCH --job-name=cot-oracle-v7
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=90G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_v7_%j.out
#SBATCH --error=logs/slurm_v7_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export AO_REPO_PATH=/nfs/nhome/live/jbauer/activation_oracles
export VENV=/var/tmp/jbauer/venvs/cot-oracle
export PATH="$VENV/bin:$PATH"
export HF_HOME="/ceph/scratch/jbauer/hf"
export HF_HUB_CACHE="/ceph/scratch/jbauer/hf/hub"

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "=== COT Oracle v7: Sequential Training + Eval Tasks ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# ── Clean stale GPU processes ────────────────────────────────
pkill -f "torchrun.*train_mixed" 2>/dev/null || true
sleep 2

# ── Sync venv if needed ─────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
    echo "Building venv..."
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync
fi

# ── Launch training ──────────────────────────────────────────
torchrun --nproc_per_node=8 \
    --master_port=29500 \
    src/train_mixed.py \
    --corpus data/cot_corpus_v5/corpus.jsonl \
    --model Qwen/Qwen3-8B \
    --lr 1e-5 \
    --batch-size 8 \
    --effective-batch-size 128 \
    --epochs 3 \
    --eval-every-n-examples 12800 \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_v7 \
    --wandb-project cot_oracle \
    --wandb-run "v7-sequential-8xH100" \
    --no-curriculum \
    --position-stride 5 \
    --max-positions 50 \
    --n-context-pred 30000 \
    --n-sentence-pred 10000 \
    --n-decorative 5000 \
    --n-domain 5000 \
    --n-correctness 5000 \
    --n-persona 0 \
    --n-summary 0 \
    --n-eval-task 3000 \
    --eval-train-dir data/evals_train \
    --no-data-cache \
    --fast-eval-n 10

echo ""
echo "=== Done: $(date) ==="
