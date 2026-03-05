#!/bin/bash
#SBATCH --job-name=comp_eval
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-sr675-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_comp_eval_%j.log
#SBATCH --error=logs/slurm_comp_eval_%j.log

set -e

source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
export FAST_CACHE_DIR=/var/tmp/jbauer
export PYTHONUNBUFFERED=1

cd /nfs/nhome/live/jbauer/cot-oracle

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_comprehensive_eval.py \
    --config configs/train.yaml \
    --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic \
    --n-examples 25 \
    --device cuda \
    --output-dir data/comprehensive_eval
