#!/bin/bash
# Launch training from a YAML config file.
#
# Usage:
#   bash scripts/launch.sh                          # default config
#   bash scripts/launch.sh configs/train.yaml       # explicit config
#   CONFIG=configs/ablation_no_pe.yaml bash scripts/launch.sh
#
# All training params come from the YAML config.
# CLI overrides still work: bash scripts/launch.sh configs/train.yaml --lr 2e-5

set -e

CONFIG="${1:-${CONFIG:-configs/train.yaml}}"
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

exec python3 src/train.py --config "${CONFIG}" "$@"
