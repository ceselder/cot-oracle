#!/bin/bash
#SBATCH --job-name=cot-thirds-8gpu
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=logs/thirds_8gpu_%j.out
#SBATCH --error=logs/thirds_8gpu_%j.err

set -euo pipefail

# Kill all child processes on exit/cancel to avoid GPU zombies
cleanup() { kill -- -$$ 2>/dev/null || true; }
trap cleanup EXIT SIGTERM SIGINT

# Load env vars (HF_HOME, etc.)
set -a; source ~/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /nfs/nhome/live/jbauer/cot-oracle

mkdir -p logs

# Use SLURM_GPUS_ON_NODE to match allocated GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-4}

torchrun --nproc_per_node=$NGPUS src/train_random_layers.py \
    --corpus data/cot_corpus_v5/corpus.jsonl \
    --save-dir /ceph/scratch/jbauer/checkpoints/cot_oracle_thirds_v1 \
    --wandb-run thirds_v1_h100 \
    --no-unfaith-evals
