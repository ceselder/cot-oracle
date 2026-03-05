#!/bin/bash
set -e
cd /nfs/nhome/live/jbauer/cot-oracle
source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
source ~/.env 2>/dev/null || true
export CACHE_DIR="${CACHE_DIR:-/ceph/scratch/jbauer}"
export CUDA_VISIBLE_DEVICES=0

nohup python scripts/run_comprehensive_eval.py \
  --config configs/train.yaml \
  --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic \
  --n-examples 25 \
  --device cuda \
  --output-dir data/comprehensive_eval \
  --tasks answer_trajectory convqa chunked_convqa chunked_compqa sqa resampling_importance \
  > /nfs/nhome/live/jbauer/cot-oracle/logs/sae_gen_rerun.log 2>&1 &
echo "spawned PID=$!"
