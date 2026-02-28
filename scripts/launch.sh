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
export CACHE_DIR="${CACHE_DIR:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export COT_ORACLE_EVAL_CACHE_POLICY="${COT_ORACLE_EVAL_CACHE_POLICY:-refresh}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

# Wandb + HF tokens from env (must be set)
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY HF_TOKEN

echo "============================================================"
echo "  CoT Oracle Training"
echo "  Config: ${CONFIG}"
echo "  Extra args: $@"
echo "  HF cache: ${HF_HOME}"
echo "  Eval cache policy: ${COT_ORACLE_EVAL_CACHE_POLICY}"
echo "============================================================"

exec python3 src/train.py --config "${CONFIG}" "$@"
