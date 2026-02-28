#!/bin/bash
# Launch CURRICULUM training: sequential task ordering, eval every 1k steps.
#
# Prerequisites: GPU with ~80GB+ VRAM, precomputed training data
# Usage: bash scripts/launch_curriculum.sh

set -e

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export AO_REPO_PATH="${AO_REPO_PATH:-/root/activation_oracles}"
export PYTHONPATH="${PYTHONPATH:-/root/cot-oracle/src}"
export PYTHONUNBUFFERED=1
export CACHE_DIR="${CACHE_DIR:-/root/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export COT_ORACLE_EVAL_CACHE_POLICY="${COT_ORACLE_EVAL_CACHE_POLICY:-refresh}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR/cot_oracle"

cd /root/cot-oracle

# Verify data exists
echo "=== Checking data ==="
for f in data/cot_corpus_v5/corpus_medium.jsonl \
         data/concept_corpus/corpus_full.jsonl; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") lines)"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

# Verify precomputed data
if [ -d "data/precomputed" ]; then
    echo "  OK: data/precomputed/ ($(ls data/precomputed/*.jsonl 2>/dev/null | wc -l) JSONL files)"
else
    echo "  MISSING: data/precomputed/"
    exit 1
fi

echo ""
echo "=== Launching CURRICULUM training ==="
echo "  Sequential task order, eval every 1000 steps"
echo "  Config: configs/train_curriculum.yaml"
echo "  Precomputed data: data/precomputed/"
echo "  wandb: MATS10-CS-JB/cot_oracle"
echo "  HF cache: ${HF_HOME}"
echo "  Eval cache policy: ${COT_ORACLE_EVAL_CACHE_POLICY}"
echo ""

nohup python3 src/train.py \
    --config configs/train_curriculum.yaml \
    --precomputed-dir data/precomputed \
    > /root/train_curriculum.log 2>&1 &

echo "PID: $!"
echo "Log: /root/train_curriculum.log"
echo "Monitor: tail -f /root/train_curriculum.log"
