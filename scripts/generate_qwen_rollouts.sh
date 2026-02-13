#!/bin/bash
# Generate MATH rollouts using Qwen3-8B via thought-anchors repo
#
# Prerequisites:
#   git clone https://github.com/interp-reasoning/thought-anchors
#   cd thought-anchors && pip install -r requirements.txt
#
# This generates rollouts that we can then analyze for causal importance.
# Designed for H100 (80GB VRAM) - no quantization needed, full batching.

set -e

# Configuration
MODEL="Qwen/Qwen3-8B"
NUM_PROBLEMS=500   # Full dataset for publication
NUM_ROLLOUTS=50    # Rollouts per chunk
TEMPERATURE=0.6
TOP_P=0.95
OUTPUT_DIR="../cot-oracle/data/qwen3_rollouts"
THOUGHT_ANCHORS_DIR="../thought-anchors"  # Adjust path as needed

# H100 settings - no quantization, can batch
BATCH_SIZE=8  # Adjust based on actual VRAM usage

echo "========================================"
echo "Generating MATH Rollouts with Qwen3-8B"
echo "========================================"
echo "Model: $MODEL"
echo "Problems: $NUM_PROBLEMS"
echo "Rollouts per chunk: $NUM_ROLLOUTS"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "H100 mode: No quantization, full bf16"
echo ""

# Check thought-anchors exists
if [ ! -d "$THOUGHT_ANCHORS_DIR" ]; then
    echo "Error: thought-anchors repo not found at $THOUGHT_ANCHORS_DIR"
    echo "Run: git clone https://github.com/interp-reasoning/thought-anchors"
    exit 1
fi

cd "$THOUGHT_ANCHORS_DIR"

# Step 1: Generate rollouts with Qwen3-8B (Local provider, no quantization)
echo "Step 1: Generating rollouts..."
echo "This will take a while. Monitor with: tail -f $OUTPUT_DIR/generation.log"
python generate_rollouts.py \
    --model "$MODEL" \
    --provider Local \
    --num_problems "$NUM_PROBLEMS" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --base_solution_type correct \
    --level "Level 5" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/generation.log"

# Step 2: Analyze rollouts to compute importance scores
echo ""
echo "Step 2: Computing importance scores..."
python analyze_rollouts.py \
    --input_dir "$OUTPUT_DIR" \
    --compute_all \
    2>&1 | tee "$OUTPUT_DIR/analysis.log"

echo ""
echo "========================================"
echo "Rollout generation complete!"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. cd ../cot-oracle"
echo "  2. python src/convert_thought_anchors.py --rollouts_dir data/qwen3_rollouts --output data/training_data.json"
echo "  3. python scripts/upload_datasets.py --rollouts data/qwen3_rollouts --repo_prefix YOUR_USERNAME"
echo "========================================"
