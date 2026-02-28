#!/bin/bash
#SBATCH --job-name=steering-sweep
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-sr675-34
#SBATCH --array=0-5%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/steering_%a_%j.out

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

# Array index -> steering layer mapping
# 0: layer 3  (early, far before oracle L9)
# 1: layer 7  (just before oracle L9)
# 2: layer 12 (between oracle L9 and L18)
# 3: layer 21 (between oracle L18 and L27)
# 4: layer 24 (just before oracle L27)
# 5: layer 33 (after oracle L27, near model end)
LAYERS=(3 7 12 21 24 33)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Array task: $SLURM_ARRAY_TASK_ID -> steering layer $LAYER"
echo "Starting at $(date)"

python scripts/detect_steering.py \
    --checkpoint checkpoints/cot-oracle-ablation-stride5-3layers \
    --concepts formal pessimistic verbose confident analogies \
               first_person cautious numbered_lists historical socratic \
    --alphas 0.0 0.1 0.3 1.0 3.0 10.0 \
    --steering-layer $LAYER \
    --max-cot-tokens 256 \
    --stride 5 --tasks recon

echo "Finished at $(date)"
