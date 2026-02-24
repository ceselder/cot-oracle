#!/bin/bash
# Full mixed-pipeline for Qwen3-8B on a single GPU.
#
# Usage:
#   cd /workspace/cot-oracle
#   bash src/data_pipeline/run_full_pipeline_8b.sh

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
CORPUS="data/cot_corpus_v5/corpus_8b.jsonl"
CHECKPOINT_DIR="checkpoints/cot_oracle_mixed_8b"

echo "============================================================"
echo "CoT Oracle 8B Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo "Corpus: $CORPUS"
echo ""

# Source API keys
source /root/.bashrc 2>/dev/null || true

# Step 1: Generate corpus (if missing)
echo ""
echo "============================================================"
echo "Step 1: Generate corpus (if missing)"
echo "============================================================"
if [ -f "$CORPUS" ]; then
    echo "Corpus already exists at $CORPUS, skipping."
else
    python3 src/data_pipeline/generate_cots.py \
        --openrouter \
        --n-problems 1000 \
        --model "$MODEL" \
        --output "$CORPUS"
fi

# Step 2: Generate eval datasets (if missing)
echo ""
echo "============================================================"
echo "Step 2: Generate eval datasets (if missing)"
echo "============================================================"
if [ ! -d "data/evals" ] || [ -z "$(ls -A data/evals 2>/dev/null)" ]; then
    python3 src/evals/generate_datasets.py --n 50 --output-dir data/evals
else
    echo "Eval datasets already exist, skipping."
fi

# Step 3: Train
echo ""
echo "============================================================"
echo "Step 3: Train CoT Oracle"
echo "============================================================"

python3 src/train.py \
    --corpus "$CORPUS" \
    --model "$MODEL" \
    --lr 1e-5 \
    --batch-size 16 \
    --epochs 1 \
    --save-dir "$CHECKPOINT_DIR" \
    --wandb-project cot_oracle \
    --wandb-run "cot_oracle_8B" \
    --eval-steps 500 \
    --save-steps 1000 \
    --eval-dir data/evals \
    --fast-eval-n 5 \
    --gradient-checkpointing

# Step 4: Run evals
echo ""
echo "============================================================"
echo "Step 4: Run evals"
echo "============================================================"
python3 src/evals/run_evals.py \
    --eval-dir data/evals \
    --output-dir data/eval_results \
    --model "$MODEL"

python3 src/evals/score_oracle.py \
    --results-dir data/eval_results

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Check wandb for training curves + eval results"
