#!/bin/bash
#SBATCH --job-name=cot-stride-test
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stride_test_%j.out
#SBATCH --error=logs/stride_test_%j.err

set -euo pipefail

# Load env vars (HF_HOME, etc.)
set -a; source ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /nfs/nhome/live/jbauer/cot-oracle

mkdir -p logs

# Quick test: small task sizes (~500 steps with batch_size=8)
torchrun --nproc_per_node=1 src/train_random_layers.py \
    --corpus data/cot_corpus_v5/corpus.jsonl \
    --model Qwen/Qwen3-8B \
    --position-strides 4 16 64 \
    --max-positions 20 \
    --batch-size 4 \
    --lr 1e-5 \
    --epochs 1 \
    --eval-steps 200 \
    --save-steps 500 \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_stride_test \
    --wandb-project cot_oracle \
    --wandb-run stride_test_v1 \
    --no-curriculum \
    --no-unfaith-evals \
    --n-context-pred 2000 \
    --n-sentence-pred 500 \
    --n-decorative 500 \
    --n-domain 500 \
    --n-correctness 500 \
    --n-persona 0 \
    --n-summary 0
