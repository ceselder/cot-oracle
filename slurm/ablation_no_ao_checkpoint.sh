#!/bin/bash
#SBATCH --job-name=ablation-no-ao
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_no_ao_%j.out
#SBATCH --error=logs/ablation_no_ao_%j.err

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
echo "Starting ablation (no AO checkpoint, fresh LoRA from Qwen3-8B) on $(hostname) with 2 GPUs"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

"$VENV/bin/torchrun" --nproc_per_node=2 src/train.py \
    --config configs/train.yaml \
    --fresh-lora \
    --task-order shuffled \
    --batch-size 4 \
    --effective-batch-size 256 \
    --no-step0-eval \
    --wandb-run ablation-no-ao-checkpoint-fresh-lora
