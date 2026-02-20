#!/bin/bash
# Full CoT Oracle training pipeline.
# Run on GPU machine (vast.ai H100).
#
# Usage:
#   cd /path/to/cot-oracle
#   bash src/data_pipeline/run_full_pipeline.sh

set -euo pipefail

MODEL="Qwen/Qwen3-1.7B"
CORPUS="data/cot_corpus/corpus.jsonl"
LABELS_DIR="data/cot_corpus"
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
        --sources math gsm8k \
        --n-problems 500 \
        --model "$MODEL" \
        --output "$CORPUS"
fi

# ---- Step 2: Extract GPU labels ----
echo ""
echo "==== Step 2: Extracting importance labels (GPU) ===="
if [ -f "$LABELS_DIR/labels_importance.jsonl" ]; then
    echo "Importance labels exist, skipping."
else
    python src/data_pipeline/extract_labels.py \
        --corpus "$CORPUS" \
        --model "$MODEL" \
        --importance
fi

echo ""
echo "==== Step 3: Extracting answer tracking labels (GPU) ===="
if [ -f "$LABELS_DIR/labels_answer_tracking.jsonl" ]; then
    echo "Answer tracking labels exist, skipping."
else
    python src/data_pipeline/extract_labels.py \
        --corpus "$CORPUS" \
        --model "$MODEL" \
        --answer-tracking
fi

# ---- Step 3: Extract API labels ----
echo ""
echo "==== Step 4: Extracting taxonomy labels (API) ===="
if [ -f "$LABELS_DIR/labels_taxonomy.jsonl" ]; then
    echo "Taxonomy labels exist, skipping."
else
    python src/data_pipeline/extract_labels.py \
        --corpus "$CORPUS" \
        --taxonomy
fi

echo ""
echo "==== Step 5: Extracting summary labels (API) ===="
if [ -f "$LABELS_DIR/labels_summary.jsonl" ]; then
    echo "Summary labels exist, skipping."
else
    python src/data_pipeline/extract_labels.py \
        --corpus "$CORPUS" \
        --summary
fi

# ---- Step 4: Train ----
echo ""
echo "==== Step 6: Training CoT Oracle ===="
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

# ---- Step 5: Run evals ----
echo ""
echo "==== Step 7: Running evals ===="

# Generate eval datasets if not already done
if [ ! -d "data/evals" ] || [ -z "$(ls -A data/evals 2>/dev/null)" ]; then
    echo "Generating eval datasets..."
    python src/evals/generate_datasets.py --n 50 --output-dir data/evals
fi

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
