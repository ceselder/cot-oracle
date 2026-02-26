#!/bin/bash
cd /nfs/nhome/live/jbauer/cot-oracle
exec /var/tmp/jbauer/venvs/cot-oracle/bin/torchrun \
    --nproc_per_node=8 --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --precomputed-dir data/precomputed \
    --task-order sequential \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_seq \
    --wandb-run sequential_v2 \
    2>&1 | tee logs/train_seq_v2.log
