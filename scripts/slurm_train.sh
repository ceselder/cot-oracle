#!/bin/bash
#SBATCH --job-name=cot-oracle
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=600G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export AO_REPO_PATH=/nfs/nhome/live/jbauer/activation_oracles
export VENV=/var/tmp/jbauer/venvs/cot-oracle
export PATH="$VENV/bin:$PATH"
export HF_HOME="/ceph/scratch/jbauer/hf"
export HF_HUB_CACHE="/ceph/scratch/jbauer/hf/hub"
export HF_DATASETS_CACHE="/ceph/scratch/jbauer/hf/datasets"
export TRANSFORMERS_CACHE="/ceph/scratch/jbauer/hf/transformers"
export CACHE_DIR="/ceph/scratch/jbauer"
export COT_ORACLE_EVAL_CACHE_POLICY="${COT_ORACLE_EVAL_CACHE_POLICY:-refresh}"

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

echo "=== COT Oracle: Multi-GPU Training ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Job ID: $SLURM_JOB_ID"
echo "HF cache: $HF_HOME"
echo "Eval cache policy: $COT_ORACLE_EVAL_CACHE_POLICY"
echo "Start: $(date)"
echo ""

# ── Clean stale GPU processes ────────────────────────────────
pkill -f "torchrun.*train" 2>/dev/null || true
sleep 2

# ── Sync venv (always, to fix stale envs) ────────────────────
echo "Syncing venv at $VENV..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync
echo "Using python: $(python --version 2>&1)"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__, transformers.__file__)' 2>&1)"

# ── Launch training ──────────────────────────────────────────
torchrun --nproc_per_node=8 \
    --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --precomputed-dir data/precomputed \
    --task-order sequential \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_seq \
    --wandb-run sequential_v2

echo ""
echo "=== Done: $(date) ==="
