#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/nhome/live/jbauer/cot-oracle/logs/slurm-%j-%x.out
#SBATCH --error=/nfs/nhome/live/jbauer/cot-oracle/logs/slurm-%j-%x.err

# Usage: sbatch --job-name=<name> --exclude=gpu-xd670-30 scripts/sbatch_exp.sh <wandb_run> [extra_args...]
# Example: sbatch --job-name=8b-noflamingo scripts/sbatch_exp.sh 8b-no-flamingo --batch-size 4

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

# Build venv on local storage if needed
VENV="/var/tmp/jbauer/venvs/cot-oracle"
if [ ! -f "$VENV/bin/python" ]; then
    echo "Building venv on $(hostname)..."
    cd /nfs/nhome/live/jbauer/cot-oracle
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync
fi

cd /nfs/nhome/live/jbauer/cot-oracle
mkdir -p logs

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Run: $RUN_NAME, Args: $@"

$VENV/bin/python src/train.py \
    --config configs/train.yaml \
    --eval-steps 1000 \
    --max-items-per-eval 10 \
    --wandb-entity japhba-personal \
    --wandb-run "$RUN_NAME" \
    "$@"
