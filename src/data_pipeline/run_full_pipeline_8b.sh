#!/bin/bash
# Full pipeline for Qwen3-8B CoT Oracle training on vast.ai H100
#
# Prerequisites:
#   - Corpus already generated: data/cot_corpus_8b/corpus.jsonl
#   - Eval datasets already generated: data/evals/*.json
#   - AO repo installed: pip3 install -e /workspace/ao_reference
#   - datasets symlinked: ln -s /workspace/ao_reference/datasets datasets
#   - API keys in .bashrc (WANDB_API_KEY, HF_TOKEN, OPENROUTER_API_KEY)
#
# Usage:
#   cd /workspace/cot-oracle
#   bash src/data_pipeline/run_full_pipeline_8b.sh

set -e

MODEL="Qwen/Qwen3-8B"
CORPUS="data/cot_corpus_8b/corpus.jsonl"
LABELS_DIR="data/cot_corpus_8b"

echo "============================================================"
echo "CoT Oracle 8B Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo "Corpus: $CORPUS"
echo ""

# Source API keys
source /root/.bashrc 2>/dev/null || true

# Step 1: Extract GPU labels (importance + answer_tracking)
echo ""
echo "============================================================"
echo "Step 1: Extract GPU labels"
echo "============================================================"
python3 src/data_pipeline/extract_labels.py \
    --corpus "$CORPUS" \
    --model "$MODEL" \
    --importance --answer-tracking \
    --device cuda

# Step 2: Extract API labels (taxonomy + summary) — runs in background
echo ""
echo "============================================================"
echo "Step 2: Extract API labels (background)"
echo "============================================================"
python3 src/data_pipeline/extract_labels.py \
    --corpus "$CORPUS" \
    --model "$MODEL" \
    --taxonomy --summary &
API_PID=$!
echo "API label extraction running in background (PID: $API_PID)"

# Step 3: Train (starts with whatever labels are available)
# API labels will be missing initially — taxonomy and summary tasks will be skipped
echo ""
echo "============================================================"
echo "Step 3: Train CoT Oracle"
echo "============================================================"
echo "Training with available labels. Taxonomy/summary will be skipped until API labels complete."
echo ""

torchrun --nproc_per_node=1 src/train_mixed.py \
    --corpus "$CORPUS" \
    --model "$MODEL" \
    --lr 1e-5 \
    --batch-size 16 \
    --epochs 1 \
    --save-dir checkpoints/cot_oracle_mixed_8b \
    --wandb-project cot_oracle \
    --wandb-run "cot_oracle_8B" \
    --eval-steps 500 \
    --save-steps 1000 \
    --eval-dir data/evals \
    --fast-eval-n 5 \
    --gradient-checkpointing

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Checkpoints saved to: checkpoints/cot_oracle_8b/"
echo "Check wandb for training curves + eval results"
