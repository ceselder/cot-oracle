#!/bin/bash
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=06b-h100as
#SBATCH --output=logs/slurm_06b_h100as_%j.log

set -e
cd /nfs/nhome/live/jbauer/cot-oracle

# Clean up stale processes
pkill -u jbauer -f torchrun 2>/dev/null || true
pkill -u jbauer -f "train.py" 2>/dev/null || true
sleep 2

source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
export PYTHONPATH=src:$PYTHONPATH

torchrun --nproc_per_node=4 --master_port=29500 \
    src/train.py \
    --config configs/train.yaml \
    --wandb-run 06b-h100as-full
