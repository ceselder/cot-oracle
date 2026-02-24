#!/bin/bash
# Resume Stage 2 training from step 14000 checkpoint.
# Run AFTER precompute finishes (scripts/run_precompute.sh).
#
# Changes from the original fuadqnhq run:
# - --resume-from step_14000 checkpoint
# - --start-step 14000 (wandb step continuity)
# - --unfaith-eval-steps 1000 (was 5000 — cache makes it fast)
# - importance removed from Stage 2 (handled in train.py)
# - EMA per-task loss, no eval_exact, no _n metrics
# - wandb tables for qualitative eval inspection
# - CI-based decorative_cot labeling (temperature sampling fallback)

set -e

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export PYTHONUNBUFFERED=1

cd /root/cot-oracle

# Clear stale Stage 2 cache (importance was removed)
rm -f data/cache/stage2.json

nohup python3 src/train.py \
    --corpus data/cot_corpus_v5/corpus_medium.jsonl \
    --model Qwen/Qwen3-8B \
    --stages 2 \
    --resume-from checkpoints/v5_adam/stage2/step_14000 \
    --start-step 14000 \
    --lr 1e-5 \
    --batch-size 8 \
    --eval-batch-size 2 \
    --stage-epochs 1 \
    --stride 5 \
    --max-positions-per-layer 20 \
    --eval-steps 500 \
    --save-steps 1000 \
    --unfaith-eval-steps 1000 \
    --unfaith-eval-items 20 \
    --save-dir checkpoints/v5_adam \
    --wandb-project cot_oracle \
    --data-cache-dir data/cache \
    --activation-cache-dir data/eval_precomputed \
    --correctness-n 15000 \
    --decorative-n 15000 \
    --domain-n 15000 \
    > train_adam_resume.log 2>&1 &

echo "Resumed Stage 2 from step 14000 with PID $!"
echo "Log: train_adam_resume.log"
echo "Monitor: tail -f train_adam_resume.log"
