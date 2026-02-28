#!/bin/bash
#SBATCH --job-name=sae-collect
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-sr675-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/sae_collect_%j.out

# Clean up stale processes on exit
cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT

cd /nfs/nhome/live/jbauer/cot-oracle

# Load env first (needed for paths)
set -a; source .env 2>/dev/null; source ~/.env 2>/dev/null; set +a

# Sync venv
export UV_PROJECT_ENVIRONMENT="/var/tmp/jbauer/venvs/cot-oracle"
uv sync --quiet
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

set -euo pipefail

mkdir -p slurm_logs

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "FAST_CACHE_DIR=$FAST_CACHE_DIR"
echo "Starting at $(date)"

python scripts/sae_collect_max_acts.py \
    --trainer 2 --k 30 --context-window 41 --batch-size 8 \
    --output-dir "${FAST_CACHE_DIR}/sae_features/" \
    ${RESUME:+--resume}

echo "Finished at $(date)"
