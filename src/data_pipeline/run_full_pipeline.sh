#!/bin/bash
# CoT Oracle mixed training pipeline.
# Run on a GPU machine with AO repo available.
#
# Usage:
#   cd /path/to/cot-oracle
#   bash src/data_pipeline/run_full_pipeline.sh

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
CORPUS="data/cot_corpus_v5/corpus.jsonl"
CHECKPOINT_DIR="checkpoints/cot_oracle_mixed"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

echo "============================================"
echo "CoT Oracle Training Pipeline"
echo "Model: $MODEL"
echo "============================================"

# ---- Step 1: Generate CoTs ----
echo ""
echo "==== Step 1: Generating CoT corpus ===="
if [ -f "$CORPUS" ]; then
    echo "Corpus already exists at $CORPUS, skipping."
    echo "  $(wc -l < "$CORPUS") entries"
else
    python src/data_pipeline/generate_cots.py \
        --openrouter \
        --n-problems 1000 \
        --model "$MODEL" \
        --output "$CORPUS"
fi

# ---- Step 2: Generate eval datasets ----
echo ""
echo "==== Step 2: Generating eval datasets (if missing) ===="
if [ ! -d "data/evals" ] || [ -z "$(ls -A data/evals 2>/dev/null)" ]; then
    python src/evals/generate_datasets.py --n 50 --output-dir data/evals
else
    echo "Eval datasets already exist, skipping."
fi

# ---- Step 3: Train ----
echo ""
echo "==== Step 3: Training CoT Oracle ===="
torchrun --nproc_per_node=1 src/train_mixed.py \
    --corpus "$CORPUS" \
    --model "$MODEL" \
    --lr 1e-5 \
    --batch-size 16 \
    --save-dir "$CHECKPOINT_DIR" \
    --wandb-project cot_oracle \
    --eval-dir data/evals \
    --fast-eval-n 5 \
    --gradient-checkpointing

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "============================================"

# ---- Step 4: Run evals ----
echo ""
echo "==== Step 4: Running evals ===="

# Run model + oracle on evals
python src/evals/run_evals.py \
    --eval-dir data/evals \
    --output-dir data/eval_results \
    --model "$MODEL"

# Score
python src/evals/score_oracle.py \
    --results-dir data/eval_results

echo ""
echo "Done! Check data/eval_results/ for scores."
