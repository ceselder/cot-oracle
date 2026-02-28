#!/bin/bash
#SBATCH --job-name=flamingo-test
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/flamingo_test_%j.out
#SBATCH --error=logs/flamingo_test_%j.err

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
    pkill -f "torchrun.*flamingo" 2>/dev/null || true
    pkill -f "torch.distributed" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run ──
echo "Starting Flamingo test run on $(hostname) with 8 GPUs"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

"$VENV/bin/torchrun" --nproc_per_node=8 src/train.py \
    --config configs/train.yaml \
    --flamingo \
    --flamingo-xattn-interval 4 \
    --flamingo-xattn-lora-r 64 \
    --fresh-lora \
    --full-recon-n 5000 \
    --next-step-n 3000 \
    --correctness-n 2000 \
    --decorative-n 2000 \
    --answer-pred-n 2000 \
    --reasoning-term-n 1000 \
    --atypical-answer-n 0 \
    --hint-admission-n 0 \
    --hinted-answer-pred-n 0 \
    --cotqa-n 0 \
    --compqa-n 0 \
    --prompt-inversion-n 0 \
    --domain-n 0 \
    --epochs 1 \
    --batch-size 4 \
    --effective-batch-size 128 \
    --lr 1e-5 \
    --no-step0-eval \
    --wandb-entity japhba \
    --wandb-run flamingo-xattn-test

echo "Done!"
