#!/bin/bash
#SBATCH --job-name=cls-evals
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/cls_evals_%j.out

cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT

cd /nfs/nhome/live/jbauer/cot-oracle

set -a; source .env 2>/dev/null; source ~/.env 2>/dev/null; set +a

export UV_PROJECT_ENVIRONMENT="/var/tmp/jbauer/venvs/cot-oracle"
uv sync --quiet
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export PYTHONUNBUFFERED=1
set -euo pipefail
mkdir -p slurm_logs eval_logs

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting at $(date)"

CHECKPOINT="${1:-checkpoints/step_385}"
echo "Checkpoint: $CHECKPOINT"

python scripts/run_classification_evals.py \
    --checkpoint "$CHECKPOINT" \
    --max-items 30 \
    --eval-batch-size 8 \
    --include-standard

echo "Finished at $(date)"
