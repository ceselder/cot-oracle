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
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

echo "=== COT Oracle: 4-GPU Training ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>/dev/null | head -4
echo "Start: $(date)"

pkill -f "torchrun.*train" 2>/dev/null || true
sleep 2

echo "Using python: $(python --version 2>&1)"

nohup torchrun --nproc_per_node=4 \
    --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_8gpu \
    --wandb-run 8gpu \
    --effective-batch-size 64 \
    > logs/train_8gpu.log 2>&1 &

echo "Training launched with PID: $!"
echo "Log: logs/train_8gpu.log"
