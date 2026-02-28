#!/bin/bash
#SBATCH --job-name=no-act-detection-eval
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=eval_logs/slurm_no_act_detection_%j.log

cd ~/cot-oracle
source ~/.bashrc
source ~/.env

VENV_LOCAL=/var/tmp/jbauer/venvs
export UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/cot-oracle"

PYTHONUNBUFFERED=1 $UV_PROJECT_ENVIRONMENT/bin/python scripts/eval_no_act_detection.py
