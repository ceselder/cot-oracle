#!/bin/bash
# Run directly on H100TB via ssh, not slurm
# Usage: ssh gpu-sr675-34 "bash /nfs/nhome/live/jbauer/cot-oracle/scripts/run_collect_chainscope.sh"
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR=/nfs/nhome/live/jbauer/cot-oracle
VENV_LOCAL="/var/tmp/jbauer/venvs"
VENV="$VENV_LOCAL/cot-oracle"

mkdir -p "$PROJECT_DIR/logs"

# Build venv if needed
if [ ! -d "$VENV" ]; then
    echo "Building venv..."
    cd "$PROJECT_DIR"
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync
fi

source "$VENV/bin/activate"
cd "$PROJECT_DIR"

python scripts/collect_chainscope_cots.py \
    --n-per-category 18 \
    --n-rollouts 10 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --output data/chainscope_qwen3_8b_cots.json \
    2>&1 | tee logs/chainscope_cots_ssh.log

echo "Done: $(date)"
