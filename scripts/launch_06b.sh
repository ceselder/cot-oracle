#!/bin/bash
cd /nfs/nhome/live/jbauer/cot-oracle
pkill -9 -u jbauer -f "torchrun|train.py" 2>/dev/null
sleep 2
nohup /var/tmp/jbauer/venvs/cot-oracle/bin/torchrun \
    --nproc_per_node=4 \
    src/train.py \
    --config configs/train.yaml \
    --wandb-run 06b-h100as-full \
    > logs/train_06b_h100as.log 2>&1 &
echo "Launched PID: $!"
