#!/bin/bash
#SBATCH --job-name=cot-thirds-4gpu
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/thirds_4gpu_%j.out
#SBATCH --error=logs/thirds_4gpu_%j.err

set -euo pipefail

# Load env vars (HF_HOME, etc.)
set -a; source ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /nfs/nhome/live/jbauer/cot-oracle

mkdir -p logs

torchrun --nproc_per_node=4 src/train_random_layers.py \
    --corpus data/cot_corpus_v5/corpus.jsonl \
    --model Qwen/Qwen3-8B \
    --position-stride 5 \
    --max-positions 50 \
    --layer-repeats 3 \
    --batch-size 16 \
    --lr 1e-5 \
    --epochs 1 \
    --eval-steps 100 \
    --save-steps 1000 \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_thirds_v1 \
    --wandb-project cot_oracle \
    --wandb-run thirds_v1_4gpu \
    --no-unfaith-evals \
    --no-data-cache
