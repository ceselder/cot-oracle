#!/bin/bash
#SBATCH --job-name=layer-delim-test
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/layer_delim_test_%j.out
#SBATCH --error=logs/layer_delim_test_%j.err

set -euo pipefail

# ── Environment ──
source ~/.bashrc
source ~/.env

export VENV="$VENV_LOCAL/cot-oracle"
export PROJECT_DIR="/nfs/nhome/live/jbauer/cot-oracle"

cd "$PROJECT_DIR"

# Build venv on compute node if needed
UV_PROJECT_ENVIRONMENT="$VENV" uv sync

# ── Clean up stale processes on exit ──
cleanup() {
    echo "Cleaning up..."
    pkill -f "torchrun.*train" 2>/dev/null || true
    pkill -f "torch.distributed" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run ──
echo "Starting layer delimiter test on $(hostname) with 4 GPUs"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

"$VENV/bin/torchrun" --nproc_per_node=4 src/train.py \
    --config configs/train.yaml \
    --full-recon-n 2000 \
    --next-step-n 1000 \
    --correctness-n 1000 \
    --decorative-n 1000 \
    --answer-pred-n 1000 \
    --reasoning-term-n 500 \
    --atypical-answer-n 0 \
    --hint-admission-n 0 \
    --compqa-n 0 \
    --prompt-inversion-n 0 \
    --domain-n 0 \
    --answer-trajectory-n 0 \
    --partial-answer-n 0 \
    --epochs 1 \
    --batch-size 4 \
    --effective-batch-size 64 \
    --lr 1e-5 \
    --eval-steps 500 \
    --wandb-entity japhba-personal \
    --wandb-project cot-oracle-xattn \
    --wandb-run layer-delim-test

echo "Done!"
