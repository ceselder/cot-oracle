#!/bin/bash
set -e
cd /nfs/nhome/live/jbauer/cot-oracle

# Clean up stale processes
pkill -u jbauer -f torchrun 2>/dev/null || true
pkill -u jbauer -f "train.py" 2>/dev/null || true
sleep 2

source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
export PYTHONPATH=src:$PYTHONPATH

LOGFILE="logs/train_8b_h100as_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

nohup torchrun --nproc_per_node=4 --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --wandb-run 8b-h100as-full \
    > "$LOGFILE" 2>&1 &

echo "PID=$!"
echo "Launched successfully"
