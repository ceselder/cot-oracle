#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ~/.env >/dev/null 2>&1 || true

: "${CACHE_DIR:?CACHE_DIR must be set in ~/.env}"
: "${FAST_CACHE_DIR:?FAST_CACHE_DIR must be set in ~/.env}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HF_REPO_ID="${HF_REPO_ID:-japhba/qwen3-0.6b-saes}"
TRAIN_GPU="${TRAIN_GPU:-0}"
COLLECT_GPU="${COLLECT_GPU:-0}"
TRAIN_OUTPUT_PARENT="${TRAIN_OUTPUT_PARENT:-${FAST_CACHE_DIR%/}/sae_training/Qwen_Qwen3-0.6B}"
LABEL_OUTPUT_DIR="${LABEL_OUTPUT_DIR:-${CACHE_DIR%/}/sae_features/Qwen_Qwen3-0.6B}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${CACHE_DIR%/}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME%/}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -B scripts/train_qwen3_saes.py --model "$MODEL" --layers 7 14 21 --device cuda:0 --output-dir "$TRAIN_OUTPUT_PARENT"

hf repo create "$HF_REPO_ID" --repo-type model --exist-ok
hf upload-large-folder "$HF_REPO_ID" "$TRAIN_OUTPUT_PARENT" --repo-type model --num-workers 8

CUDA_VISIBLE_DEVICES="$COLLECT_GPU" python -B scripts/sae_collect_max_acts.py --model-name "$MODEL" --sae-repo "$HF_REPO_ID" --layers 7 14 21 --trainer 2 --output-dir "$LABEL_OUTPUT_DIR"

python -B scripts/sae_label_features.py --input-dir "$LABEL_OUTPUT_DIR" --trainer 2 --layers 7 14 21 --tokenizer-model "$MODEL" --output-dir "$LABEL_OUTPUT_DIR/trainer_2"
