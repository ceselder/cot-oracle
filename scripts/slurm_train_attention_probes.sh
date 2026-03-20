#!/bin/bash
#SBATCH --job-name=att-probe
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/scratch/jbauer/logs/att_probe_%j.out
#SBATCH --error=/ceph/scratch/jbauer/logs/att_probe_%j.err

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

# Clean up any corrupted .pt files from previous OOM kills
echo "=== Cleaning corrupt cache files ==="
python3 -c "
import torch
from pathlib import Path
cache = Path('/var/tmp/jbauer/attention_probe_acts_s1_full')
if cache.exists():
    for f in cache.rglob('*.pt'):
        try:
            torch.load(f, map_location='cpu', weights_only=True)
        except Exception:
            print(f'  Removing corrupt: {f}')
            f.unlink()
"

# Step 1: Extract activations (needs full Qwen3-8B, single GPU)
echo "=== Extracting activations ==="
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_attention_probes.py" \
    --extract-only --device cuda

# Step 2: Train probes on all 3 tasks in parallel (1 GPU each)
echo "=== Training attention probes (3 tasks in parallel) ==="
CUDA_VISIBLE_DEVICES=1 python "$PROJECT_DIR/scripts/train_attention_probes.py" \
    --task hint_admission --skip-extraction --probe-type attention --device cuda &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python "$PROJECT_DIR/scripts/train_attention_probes.py" \
    --task sycophancy --skip-extraction --probe-type attention --device cuda &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python "$PROJECT_DIR/scripts/train_attention_probes.py" \
    --task truthfulqa_hint --skip-extraction --probe-type attention --device cuda &
PID3=$!

echo "Waiting for all training jobs: $PID1 $PID2 $PID3"
FAILED=0
for PID in $PID1 $PID2 $PID3; do
    wait $PID || FAILED=$((FAILED + 1))
done

# Clean up any leftover GPU processes
kill $(jobs -p) 2>/dev/null || true

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED training job(s) failed"
    exit 1
fi

echo "=== All attention probes trained ==="
