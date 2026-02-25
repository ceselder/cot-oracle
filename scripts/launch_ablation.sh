#!/bin/bash
# Launch an ablation run.
#
# Usage:
#   bash scripts/launch_ablation.sh configs/ablation_stride10.yaml
#   bash scripts/launch_ablation.sh configs/ablation_1layer.yaml
#
# Base config (train.yaml) is always loaded first, then the ablation overlay.

set -e

ABLATION="${1:?Usage: bash scripts/launch_ablation.sh <ablation_config.yaml>}"
BASE="configs/train.yaml"

export PYTHONUNBUFFERED=1
export AO_REPO_PATH="${AO_REPO_PATH:-/root/activation_oracles}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY HF_TOKEN

RUNNAME=$(basename "$ABLATION" .yaml)

echo "============================================================"
echo "  CoT Oracle Ablation: ${RUNNAME}"
echo "  Base:     ${BASE}"
echo "  Overlay:  ${ABLATION}"
echo "============================================================"

exec python3 src/train.py \
  --config "${BASE}" "${ABLATION}" \
  --no-step0-eval \
  "$@"
