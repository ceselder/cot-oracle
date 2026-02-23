#!/bin/bash
# Run decorative_cot precompute separately with temperature sampling.
# This must run AFTER run_precompute.sh finishes the other evals.
#
# decorative_cot needs temperature>0 so different runs produce different
# answers, enabling meaningful Wilson CIs for labeling.
#
# Usage: bash scripts/run_precompute_decorative.sh [--label-runs 10]

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

echo "=== Precomputing decorative_cot with temperature sampling ==="
echo "  label-runs: ${1:---label-runs 10}"
echo "  temperature: 0.6 (hardcoded in _label_decorative_cot)"
echo ""

python3 src/evals/precompute_activations.py \
    --eval-dir data/evals \
    --output-dir data/eval_precomputed \
    --model Qwen/Qwen3-8B \
    --evals decorative_cot \
    --overwrite \
    "$@"

echo ""
echo "=== Done ==="
echo "Updated: data/eval_precomputed/decorative_cot/"
echo "Updated: data/evals/decorative_cot.json (labels written back)"
