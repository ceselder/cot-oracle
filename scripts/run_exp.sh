#!/bin/bash
# Usage: run_exp.sh <gpu_id> <wandb_run_name> [extra_args...]
# Example: run_exp.sh 0 8b-no-flamingo --batch-size 4
set -e

GPU_ID=$1; shift
RUN_NAME=$1; shift

export PYTHONUNBUFFERED=1
export HF_TOKEN=$(grep HF_TOKEN ~/.env | cut -d'"' -f2)
export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
export CACHE_DIR=/ceph/scratch/jbauer
export HF_HOME=$CACHE_DIR/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export COT_ORACLE_EVAL_CACHE_POLICY=refresh

cd /nfs/nhome/live/jbauer/cot-oracle

CUDA_VISIBLE_DEVICES=$GPU_ID /var/tmp/jbauer/venvs/cot-oracle/bin/python src/train.py \
    --config configs/train.yaml \
    --eval-steps 1000 \
    --max-items-per-eval 10 \
    --wandb-entity japhba-personal \
    --wandb-run "$RUN_NAME" \
    "$@"
