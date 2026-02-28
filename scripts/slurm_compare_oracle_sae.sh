#!/bin/bash
#SBATCH --job-name=oracle-vs-sae
#SBATCH --partition=a100
#SBATCH --nodelist=gpu-sr670-20,gpu-sr670-21,gpu-sr670-22,gpu-sr670-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/oracle_vs_sae_%j.out

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
echo "Starting at $(date)"

python scripts/compare_oracle_sae.py \
    --checkpoint checkpoints/cot-oracle-ablation-stride5-3layers \
    --sae-labels-dir /ceph/scratch/jbauer/sae_features/trainer_2/trainer_2/labels \
    --n-corpus 15 \
    --oracle-tasks recon domain correctness answer decorative \
    --top-k-features 10 --top-k-aggregate 20

echo "Finished at $(date)"
