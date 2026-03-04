#!/bin/bash
#SBATCH --job-name=maxk400-cls
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-sr675-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_maxk400_tb_%j.out
#SBATCH --error=logs/slurm_maxk400_tb_%j.err

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
export COT_ORACLE_EVAL_CACHE_POLICY=refresh

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "=== COT Oracle: 1-GPU maxk400+cls-eval (H100TB) ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

torchrun --nproc_per_node=1 --master_port=29502 \
    src/train.py \
    --config configs/train.yaml \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_maxk400_cls \
    --wandb-run maxk400_cls \
    --max-context-length 1024 \
    --stochastic-max-k 400 \
    --extraction-batch-size 16 \
    --max-train-tokens-per-gpu 32768 \
    --cls-eval

echo "Done: $(date)"
