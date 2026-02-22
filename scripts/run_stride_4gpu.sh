#!/bin/bash
#SBATCH --job-name=cot-stride-4gpu
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stride_4gpu_%j.out
#SBATCH --error=logs/stride_4gpu_%j.err

set -euo pipefail

# Load env vars (HF_HOME, etc.)
set -a; source ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /nfs/nhome/live/jbauer/cot-oracle

mkdir -p logs

torchrun --nproc_per_node=4 src/train_random_layers.py \
    --corpus data/cot_corpus_v5/corpus.jsonl \
    --model Qwen/Qwen3-8B \
    --position-strides 4 16 64 \
    --max-positions 50 \
    --batch-size 16 \
    --lr 1e-5 \
    --epochs 1 \
    --eval-steps 500 \
    --save-steps 2000 \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_stride_v5 \
    --wandb-project cot_oracle \
    --wandb-run stride_v5_4gpu \
    --no-unfaith-evals
