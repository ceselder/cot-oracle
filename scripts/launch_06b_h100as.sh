#!/bin/bash
cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

nohup /var/tmp/jbauer/venvs/cot-oracle/bin/torchrun \
    --nproc_per_node=4 \
    src/train.py \
    --config configs/train.yaml \
    --wandb-run 06b-h100as-full \
    > logs/train_06b_h100as.log 2>&1 &
echo "Launched PID: $!"
