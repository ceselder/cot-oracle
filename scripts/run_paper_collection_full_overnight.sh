#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export JUDGE_USE_LOCAL="${JUDGE_USE_LOCAL:-0}"
export JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-3.1-flash-lite-preview}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"

if [[ "${JUDGE_USE_LOCAL}" == "0" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set when JUDGE_USE_LOCAL=0" >&2
  exit 1
fi

OUTDIR="${OUTDIR:-}"
if [[ -z "$OUTDIR" ]]; then
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  OUTDIR="AObench/eval_results/paper_collection_full_all13_last5_${timestamp}"
fi

echo "Running full all-task AObench paper collection eval"
echo "Output dir: $OUTDIR"
echo "Judge model: $JUDGE_MODEL"
echo "Judge concurrency: $JUDGE_CONCURRENCY"

.venv/bin/python scripts/run_paper_collection_aobench.py \
  --profile all \
  --sample-profile full \
  --n-positions 5 \
  --output-dir "$OUTDIR" \
  "$@"
