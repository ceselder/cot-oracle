#!/bin/bash
# Launch v6 training: all 10 tasks, flat training, 1 epoch.
#
# Prerequisites: GPU with ~80GB+ VRAM, CUDA, Python 3.10+
# Usage: bash scripts/launch_v6.sh

set -e

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1

cd /root/cot-oracle

# Verify data exists
echo "=== Checking data ==="
for f in data/cot_corpus_v5/corpus_medium.jsonl \
         data/concept_corpus/corpus_full.jsonl \
         data/concept_corpus/corpus_full_conv_qa_llm.jsonl; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") lines)"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

# Clear stale caches
rm -f data/cache/training_data.json data/cache/stage*.json
echo "  Cleared data caches"

echo ""
echo "=== Launching v6 training ==="
echo "  10 tasks, 195K examples, 1 epoch, ~24K steps"
echo "  wandb: MATS10-CS-JB/cot_oracle"
echo ""

nohup python3 src/train_v5.py \
    --corpus data/cot_corpus_v5/corpus_medium.jsonl \
    --concept-corpus data/concept_corpus/corpus_full.jsonl \
    --cotqa-path data/concept_corpus/corpus_full_conv_qa_llm.jsonl \
    --model Qwen/Qwen3-8B \
    --lr 1e-5 \
    --batch-size 8 \
    --eval-batch-size 2 \
    --epochs 1 \
    --eval-steps 500 \
    --save-steps 2000 \
    --unfaith-eval-steps 50000 \
    --save-dir checkpoints/v6 \
    --wandb-project cot_oracle \
    --wandb-entity MATS10-CS-JB \
    --wandb-run "v6-10tasks-195k-47kcorpus" \
    --data-cache-dir data/cache \
    --full-recon-n 40000 \
    --next-step-n 30000 \
    --answer-pred-n 20000 \
    --partial-answer-n 20000 \
    --load-bearing-n 15000 \
    --correctness-n 15000 \
    --decorative-n 15000 \
    --domain-n 15000 \
    --reasoning-term-n 15000 \
    --conv-qa-n 10000 \
    > train_v6.log 2>&1 &

echo "PID: $!"
echo "Log: train_v6.log"
echo "Monitor: tail -f train_v6.log"
