#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
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
export COT_ORACLE_EVAL_CACHE_POLICY=refresh

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "=== COT Oracle: 4-GPU maxctx1024 ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

nohup torchrun --nproc_per_node=4 \
    --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_8gpu_maxctx1024 \
    --wandb-run 8gpu_maxctx1024 \
    --effective-batch-size 64 \
    --max-context-length 1024 \
    --stochastic-max-k 1024 \
    --max-train-tokens-per-gpu 32768 \
    --extraction-batch-size 16 \
    > logs/train_8gpu_maxctx1024.log 2>&1 &

echo "Training launched with PID: $!"
echo "Log: logs/train_8gpu_maxctx1024.log"
