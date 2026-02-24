#!/bin/bash
# Launch training from a YAML config file.
#
# Usage:
#   bash scripts/launch_v6.sh                          # default config
#   bash scripts/launch_v6.sh configs/train_v6.yaml    # explicit config
#   CONFIG=configs/ablation_no_pe.yaml bash scripts/launch_v6.sh
#
# All training params come from the YAML config.
# CLI overrides still work: bash scripts/launch_v6.sh configs/train_v6.yaml --lr 2e-5

set -e

CONFIG="${1:-${CONFIG:-configs/train_v6.yaml}}"
shift 2>/dev/null || true  # shift past config arg, remaining args passed through

export PYTHONUNBUFFERED=1
export AO_REPO_PATH="${AO_REPO_PATH:-/root/activation_oracles}"

# Wandb + HF tokens from env (must be set)
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY HF_TOKEN

echo "============================================================"
echo "  CoT Oracle Training"
echo "  Config: ${CONFIG}"
echo "  Extra args: $@"
echo "============================================================"

exec python3 src/train_v5.py --config "${CONFIG}" "$@"
