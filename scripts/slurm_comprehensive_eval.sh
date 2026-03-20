#!/bin/bash
#SBATCH --job-name=cot-eval
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/scratch/jbauer/logs/cot_eval_%j.out
#SBATCH --error=/ceph/scratch/jbauer/logs/cot_eval_%j.err

set -euo pipefail

PROJECT_DIR=/nfs/nhome/live/jbauer/cot-oracle
VENV=/var/tmp/jbauer/venvs/cot-oracle

cd "$PROJECT_DIR"
source "$VENV/bin/activate"
set -a; source ~/.env; set +a

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/baselines:$PROJECT_DIR/ao_reference:${PYTHONPATH:-}"

# Clean up stale GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true

mkdir -p /ceph/scratch/jbauer/logs

python "$PROJECT_DIR/scripts/eval_comprehensive.py" \
    --config "$PROJECT_DIR/configs/eval.yaml" \
    --train-config "$PROJECT_DIR/configs/train.yaml" \
    --checkpoint /ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic \
    --n-examples 100 \
    --output-dir "$PROJECT_DIR/data/comprehensive_eval" \
    --layers 9 18 27 \
    --position-mode all \
    --tasks chunked_convqa \
    --baselines our_ao linear_probes sae-llm-monitor
