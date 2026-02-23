#!/bin/bash
# Run activation precompute for all eval datasets.
# This generates cached .pt bundles so unfaith evals don't need
# to generate CoTs or extract activations during training.
#
# Also pre-labels decorative_cot items (10 runs each).
#
# Usage: bash scripts/run_precompute.sh [--label-runs 10]

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

echo "=== Precomputing eval activations ==="
echo "This will:"
echo "  1. Pre-label decorative_cot items (10 model runs each)"
echo "  2. Generate activation bundles for all eval datasets"
echo "  3. Save labels back to decorative_cot.json"
echo ""

python3 src/evals/precompute_activations.py \
    --eval-dir data/evals \
    --output-dir data/eval_precomputed \
    --model Qwen/Qwen3-8B \
    --evals decorative_cot rot13_reconstruction answer_correctness \
            hint_influence_yesno sycophancy_scruples ood_topic \
            sentence_insertion \
    "$@"

echo ""
echo "=== Done ==="
echo "Cache dir: data/eval_precomputed/"
echo "Updated:   data/evals/decorative_cot.json"
echo ""
echo "To push to git:"
echo "  git add data/eval_precomputed/ data/evals/decorative_cot.json"
echo "  git commit -m 'Add precomputed eval activation cache'"
echo "  git push"
