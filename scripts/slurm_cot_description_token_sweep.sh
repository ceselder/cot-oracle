#!/bin/bash
#SBATCH --job-name=cot_desc_tok_sweep
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_cot_desc_token_sweep_%j.out
#SBATCH --error=logs/slurm_cot_desc_token_sweep_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export AO_REPO_PATH=/nfs/nhome/live/jbauer/activation_oracles
export VENV=/var/tmp/jbauer/venvs/cot-oracle
export PATH="$VENV/bin:$PATH"
export HF_HOME=/ceph/scratch/jbauer/hf
export HF_HUB_CACHE=/ceph/scratch/jbauer/hf/hub
export HF_DATASETS_CACHE=/ceph/scratch/jbauer/hf/datasets
export TRANSFORMERS_CACHE=/ceph/scratch/jbauer/hf/transformers
export CACHE_DIR=/ceph/scratch/jbauer
export FAST_CACHE_DIR=/var/tmp/jbauer

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "=== cot_description token sweep ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

python scripts/sweep_cot_description_tokens.py \
    --checkpoint ceselder/cot-oracle-v15-stochastic \
    --model Qwen/Qwen3-8B \
    --max-items 50 \
    --eval-batch-size 4 \
    --activation-extract-batch-size 4 \
    --layers 9 18 27 \
    --output-dir eval_logs/cot_description_token_sweep

echo "Done: $(date)"
