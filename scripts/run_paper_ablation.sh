#!/bin/bash

set -euo pipefail

ABLATION="${1:?Usage: bash scripts/run_paper_ablation.sh <overlay.yaml> [extra args...]}"
shift

BASE="configs/train.yaml"
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"

export PYTHONUNBUFFERED=1
export AO_REPO_PATH="${AO_REPO_PATH:-$PWD/ao_reference}"
export CACHE_DIR="${CACHE_DIR:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export COT_ORACLE_EVAL_CACHE_POLICY="${COT_ORACLE_EVAL_CACHE_POLICY:-refresh}"
export PATH="$PWD/.venv/bin:$PATH"

TORCHRUN_BIN="${TORCHRUN_BIN:-$PWD/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  TORCHRUN_BIN="$(command -v torchrun)"
fi

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

: "${WANDB_API_KEY:?Set WANDB_API_KEY}"

RUNNAME="$(basename "$ABLATION" .yaml)"

echo "============================================================"
echo "  CoT Oracle Paper Ablation: ${RUNNAME}"
echo "  Base:     ${BASE}"
echo "  Overlay:  ${ABLATION}"
echo "  NPROC:    ${NPROC}"
echo "  HF cache: ${HF_HOME}"
echo "============================================================"

exec "$TORCHRUN_BIN" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29500}" \
  src/train.py \
  --config "${BASE}" "${ABLATION}" \
  --no-step0-eval \
  --wandb-group cot-oracle-ablations-for-paper \
  "$@"
