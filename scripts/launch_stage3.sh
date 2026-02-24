#!/bin/bash
# Launch Stage 3 (conversational QA) training after Stages 1+2 complete.
# Usage: bash scripts/launch_stage3.sh [checkpoint_dir]
#
# Finds the latest checkpoint from Stages 1+2, then runs Stage 3 from it.

set -e

CHECKPOINT_DIR="${1:-checkpoints/v5}"
WANDB_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

# Find the stage2 final checkpoint (preferred), or latest step checkpoint
if [ -d "${CHECKPOINT_DIR}/stage2/final" ]; then
    LATEST="${CHECKPOINT_DIR}/stage2/final"
elif [ -d "${CHECKPOINT_DIR}/stage1/final" ]; then
    LATEST="${CHECKPOINT_DIR}/stage1/final"
else
    LATEST=$(ls -d ${CHECKPOINT_DIR}/stage*/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
fi

if [ -z "$LATEST" ]; then
    echo "No checkpoints found in ${CHECKPOINT_DIR}!"
    echo "Available dirs:"
    ls -la ${CHECKPOINT_DIR}/
    exit 1
fi

echo "Latest checkpoint: $LATEST"
echo "Launching Stage 3 (conversational QA)..."

export WANDB_API_KEY="$WANDB_KEY"
export PYTHONUNBUFFERED=1

nohup python3 src/train.py \
    --corpus data/cot_corpus_v5/corpus_medium.jsonl \
    --concept-corpus data/concept_corpus/corpus_full.jsonl \
    --cotqa-path data/concept_corpus/corpus_full_conv_qa_llm.jsonl \
    --model Qwen/Qwen3-8B \
    --stages 3 \
    --resume-from "$LATEST" \
    --lr 5e-6 \
    --batch-size 8 \
    --eval-batch-size 2 \
    --stage-epochs 2 \
    --stride 5 \
    --max-positions-per-layer 20 \
    --eval-steps 500 \
    --save-steps 1000 \
    --unfaith-eval-steps 2000 \
    --unfaith-eval-items 20 \
    --save-dir checkpoints/v5_stage3 \
    --wandb-project cot_oracle \
    --conv-qa-n 10000 \
    > train_stage3.log 2>&1 &

echo "Stage 3 launched with PID $!"
echo "Log: train_stage3.log"
echo "Monitor: tail -f train_stage3.log"
