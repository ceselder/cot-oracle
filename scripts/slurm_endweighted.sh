#!/bin/bash
#SBATCH --job-name=endweight
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_endweighted_%j.out
#SBATCH --error=logs/slurm_endweighted_%j.err

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

echo "=== COT Oracle: 3-GPU endweighted (H100AS) ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

torchrun --nproc_per_node=3 --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_endweighted \
    --wandb-run endweighted \
    --batch-size 16 \
    --effective-batch-size 48 \
    --extraction-batch-size 16 \
    --max-train-tokens-per-gpu 32768 \
    --cls-eval \
    --position-encoding

echo "Done: $(date)"
