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
shift  # consume $1 so "$@" only passes extra flags
BASE="configs/train.yaml"

export PYTHONUNBUFFERED=1
export AO_REPO_PATH="${AO_REPO_PATH:-/root/activation_oracles}"
export CACHE_DIR="${CACHE_DIR:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export COT_ORACLE_EVAL_CACHE_POLICY="${COT_ORACLE_EVAL_CACHE_POLICY:-refresh}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY HF_TOKEN

RUNNAME=$(basename "$ABLATION" .yaml)

echo "============================================================"
echo "  CoT Oracle Ablation: ${RUNNAME}"
echo "  Base:     ${BASE}"
echo "  Overlay:  ${ABLATION}"
echo "  HF cache: ${HF_HOME}"
echo "  Eval cache policy: ${COT_ORACLE_EVAL_CACHE_POLICY}"
echo "============================================================"

exec python3 src/train.py \
  --config "${BASE}" "${ABLATION}" \
  --no-step0-eval \
  "$@"
