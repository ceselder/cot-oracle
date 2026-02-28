#!/bin/bash
#SBATCH --job-name=profile-evals
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/profile_evals_%j.out

cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT

cd /nfs/nhome/live/jbauer/cot-oracle

set -a; source .env 2>/dev/null; source ~/.env 2>/dev/null; set +a

export UV_PROJECT_ENVIRONMENT="/var/tmp/jbauer/venvs/cot-oracle"
uv sync --quiet
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

set -euo pipefail
mkdir -p slurm_logs

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Starting at $(date)"

# Run WITHOUT cache to see fresh extraction cost, then WITH cache
echo ""
echo "=========================================="
echo "RUN 1: NO CACHE (fresh extraction)"
echo "=========================================="
python scripts/profile_evals.py --cache-dir /tmp/profile_eval_cache_empty --max-items 20 --profile all

echo ""
echo "=========================================="
echo "RUN 2: WITH CACHE (if populated)"
echo "=========================================="
python scripts/profile_evals.py --max-items 20 --profile e2e

echo "Finished at $(date)"
