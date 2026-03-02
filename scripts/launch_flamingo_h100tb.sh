#!/bin/bash
set -euo pipefail
cd /nfs/nhome/live/jbauer/cot-oracle

echo "Starting Flamingo training on $(hostname) at $(date)"

echo "Checking GPUs..."
/var/tmp/jbauer/venvs/cot-oracle/bin/python3 -c "import torch; print(f'GPUs visible: {torch.cuda.device_count()}')"

echo "Launching torchrun..."
NGPUS=$(/var/tmp/jbauer/venvs/cot-oracle/bin/python3 -c "import torch; print(torch.cuda.device_count())")
exec /var/tmp/jbauer/venvs/cot-oracle/bin/torchrun \
    --nproc_per_node="$NGPUS" \
    --master_port=29501 \
    src/train.py \
    --config configs/train.yaml configs/ablation_flamingo_0.6b.yaml \
    --flamingo \
    --flamingo-xattn-interval 4 \
    --flamingo-xattn-lora-r 64 \
    --flamingo-max-ctx-tokens 2048 \
    --batch-size 8 \
    --effective-batch-size 256 \
    --fresh-lora \
    --no-step0-eval \
    --wandb-run flamingo-06b-h100tb
