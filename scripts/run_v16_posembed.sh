#!/bin/bash
set -euo pipefail

source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
cd /nfs/nhome/live/jbauer/cot-oracle
set -a; source ~/.env; set +a

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Clean stale processes
pkill -f "torchrun.*train.py" 2>/dev/null || true
pkill -f "src/train.py" 2>/dev/null || true
sleep 2

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_v16_posembed \
    --wandb-run v16_posembed \
    --position-mode end_rdm_stc \
    --no-cls-eval \
    --position-encoding \
    --batch-size 16 \
    --effective-batch-size 128 \
    --max-train-tokens-per-gpu 32768 \
    2>&1 | tee /nfs/nhome/live/jbauer/cot-oracle/slurm_logs/v16_posembed.log
