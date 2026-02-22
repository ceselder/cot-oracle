#!/bin/bash
set -euo pipefail
cd /nfs/nhome/live/jbauer/cot-oracle
set -a; source ~/.env; set +a

echo "=== Step 1: Precompute boundary positions ==="
python3 scripts/precompute_boundaries.py data/cot_corpus_v5/corpus.jsonl

echo "=== Step 2: Submit 8-GPU training ==="
sbatch scripts/run_stride_8gpu.sh
