#!/bin/bash
#SBATCH --job-name=abl-0.6b
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ablation_%x_%j.out
#SBATCH --error=logs/ablation_%x_%j.err

# Usage: sbatch --job-name=abl-<name> scripts/slurm_ablation_0.6b.sh <ablation_name>
#   e.g. sbatch --job-name=abl-rand-layers scripts/slurm_ablation_0.6b.sh random_layers

set -euo pipefail

ABLATION_NAME="${1:?Usage: sbatch scripts/slurm_ablation_0.6b.sh <ablation_name>}"

set -a; source ~/.env; set +a

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export AO_REPO_PATH=/nfs/nhome/live/jbauer/activation_oracles
export VENV=/var/tmp/jbauer/venvs/cot-oracle
export PATH="$VENV/bin:$PATH"
export HF_HOME="/ceph/scratch/jbauer/hf"
export HF_HUB_CACHE="/ceph/scratch/jbauer/hf/hub"

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "=== Ablation: ${ABLATION_NAME} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# Sync venv
echo "Syncing venv at $VENV..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync
echo "Using python: $(python --version 2>&1)"

# Launch single-GPU ablation via the wrapper script
bash scripts/launch_ablations_0.6b.sh "$ABLATION_NAME"

echo ""
echo "=== Done: $(date) ==="
