#!/bin/bash
set -euo pipefail
source ~/.bashrc
source ~/.env

cd /nfs/nhome/live/jbauer/cot-oracle
git pull origin arch_devel

# Kill any stale torchrun processes
pkill -9 -f torchrun 2>/dev/null || true
sleep 2

nohup "$VENV_LOCAL/cot-oracle/bin/torchrun" --nproc_per_node=8 src/train.py \
    --config configs/train.yaml \
    --model Qwen/Qwen3-0.6B \
    --flamingo \
    --flamingo-xattn-interval 4 \
    --flamingo-xattn-lora-r 64 \
    --fresh-lora \
    --full-recon-n 5000 \
    --next-step-n 3000 \
    --correctness-n 2000 \
    --decorative-n 2000 \
    --answer-pred-n 2000 \
    --reasoning-term-n 1000 \
    --atypical-answer-n 0 \
    --hint-admission-n 0 \
    --prompt-inversion-n 0 \
    --domain-n 0 \
    --epochs 1 \
    --flamingo-max-ctx-tokens 2048 \
    --batch-size 4 \
    --effective-batch-size 128 \
    --lr 1e-5 \
    --no-step0-eval \
    --wandb-entity MATS10-CS-JB \
    --wandb-run flamingo-chimera-0.6b \
    > logs/flamingo_chimera.log 2>&1 &

echo "Launched! PID=$!"
