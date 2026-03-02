#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=flamingo-06b
#SBATCH --output=/nfs/nhome/live/jbauer/cot-oracle/logs/flamingo_h100as.log
#SBATCH --error=/nfs/nhome/live/jbauer/cot-oracle/logs/flamingo_h100as.log

export PYTHONUNBUFFERED=1
export HF_TOKEN=$(grep HF_TOKEN ~/.env | cut -d'"' -f2)
export WANDB_API_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)
export CACHE_DIR=/ceph/scratch/jbauer
export HF_HOME=$CACHE_DIR/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

VENV="/var/tmp/jbauer/venvs/cot-oracle"
if [ ! -f "$VENV/bin/python" ]; then
    echo "Building venv on $(hostname)..."
    cd /nfs/nhome/live/jbauer/cot-oracle
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync
fi

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "Starting Flamingo training on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

exec $VENV/bin/torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    src/train.py \
    --config configs/train.yaml configs/ablation_flamingo_0.6b.yaml \
    --flamingo \
    --flamingo-xattn-interval 4 \
    --flamingo-xattn-lora-r 64 \
    --flamingo-max-ctx-tokens 2048 \
    --batch-size 8 \
    --effective-batch-size 256 \
    --fresh-lora \
    --no-step0-eval \
    --wandb-run flamingo-06b-h100tb
