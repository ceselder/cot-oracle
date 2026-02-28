#!/bin/bash
# Launch training on vast.ai from this node.
# Usage: bash scripts/vast_launch.sh [--dry-run]
#
# Steps:
#   1. Find cheapest 4xH100 offer
#   2. Create instance
#   3. Wait for SSH to be ready
#   4. rsync code + data
#   5. Install deps + start training in tmux
#   6. Print SSH command to attach
#
# To tear down when done:
#   vastai destroy instance $INSTANCE_ID

set -euo pipefail

PROJECT_DIR="/nfs/nhome/live/jbauer/cot-oracle"
cd "$PROJECT_DIR"

# Load credentials from ~/.env
set -a; source ~/.env; set +a
WANDB_API_KEY=$(awk '/machine api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── 1. Find cheapest 4xH100 (spot/interruptible) ──────────────────
NUM_GPUS_WANT=${NUM_GPUS:-4}
echo "=== Searching for ${NUM_GPUS_WANT}xH100 spot offers ==="
OFFER_ID=$(vastai search offers \
    "num_gpus=${NUM_GPUS_WANT} gpu_name=H100_SXM reliability>0.9 disk_space>=150" \
    --type=bid -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])" \
)
echo "Best offer: $OFFER_ID"

if $DRY_RUN; then
    echo "[dry-run] Would create instance from offer $OFFER_ID"
    exit 0
fi

# ── 2. Create instance ───────────────────────────────────────────────
echo "=== Creating instance ==="
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel \
    --disk 150 \
    --raw 2>&1)
INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])")
echo "Instance ID: $INSTANCE_ID"

# ── 3. Wait for SSH ──────────────────────────────────────────────────
echo "=== Waiting for instance to be ready ==="
for i in $(seq 1 60); do
    INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")
    STATUS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "")
    SSH_HOST=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ssh_host',''))" 2>/dev/null || echo "")
    SSH_PORT=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ssh_port',''))" 2>/dev/null || echo "")

    if [[ "$STATUS" == "running" && -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
        echo "Instance ready: $SSH_HOST:$SSH_PORT"
        break
    fi
    echo "  [$i/60] status=$STATUS, waiting..."
    sleep 10
done

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo "ERROR: Instance did not become ready in 10 minutes"
    echo "Check: vastai show instance $INSTANCE_ID"
    exit 1
fi

SSH_CMD="ssh -A -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST"

# Wait for SSH to actually accept connections
echo "=== Waiting for SSH connectivity ==="
for i in $(seq 1 30); do
    if $SSH_CMD "echo ok" &>/dev/null; then
        echo "SSH connected"
        break
    fi
    echo "  [$i/30] SSH not ready, waiting..."
    sleep 5
done

# ── 4. Rsync code + data ─────────────────────────────────────────────
echo "=== Syncing code and data ==="
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT"

# Sync project (only code + essential training data)
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'checkpoints/' \
    --exclude 'results/' \
    --exclude 'logs/' \
    --exclude '.venv/' \
    --exclude 'wandb/' \
    --exclude 'thought-anchors/' \
    --exclude 'thought-branches/' \
    --exclude 'frugal-thought-anchors/' \
    --exclude 'chainscope/' \
    --exclude 'ao_reference/' \
    --exclude 'five-CoT-faith/' \
    --exclude 'data/eval_precomputed/' \
    --exclude 'data/pipeline_*/' \
    --exclude 'data/cot_corpus_*/' \
    --exclude 'data/compqa/' \
    --exclude 'data/qwen3_rollouts/' \
    --exclude 'data/deepseek_rollouts_for_hf/' \
    --exclude 'data/hf_uploads/' \
    --exclude 'data/*.json' \
    -e "$RSYNC_SSH" \
    "$PROJECT_DIR/" "root@$SSH_HOST:/workspace/cot-oracle/"

# Sync ao_reference if it exists locally
AO_LOCAL="${PROJECT_DIR}/ao_reference"
if [[ -d "$AO_LOCAL" ]]; then
    rsync -avz --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        -e "$RSYNC_SSH" \
        "$AO_LOCAL/" "root@$SSH_HOST:/workspace/ao_reference/"
fi

# ── 5. Install deps + start training ─────────────────────────────────
echo "=== Setting up environment and starting training ==="
$SSH_CMD bash -s <<REMOTE_SETUP
set -euo pipefail

cd /workspace/cot-oracle

# Install uv + deps
pip install -q uv 2>&1 | tail -1
UV_PROJECT_ENVIRONMENT=/workspace/venvs/cot-oracle uv sync 2>&1 | tail -3
source /workspace/venvs/cot-oracle/bin/activate

# Symlink AO datasets (classification_dataset_manager needs them at import time)
ln -sf /workspace/ao_reference/datasets /workspace/cot-oracle/datasets

mkdir -p /workspace/hf_cache
mkdir -p /workspace/hf_cache/hub /workspace/hf_cache/datasets /workspace/hf_cache/transformers

# Detect GPUs
NUM_GPUS=\$(nvidia-smi -L | wc -l)
echo "Detected \$NUM_GPUS GPUs"

# Write env file on remote (credentials resolved locally via outer heredoc)
printf '%s\n' \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "HF_HOME=/workspace/hf_cache" \
    "HF_HUB_CACHE=/workspace/hf_cache/hub" \
    "HF_DATASETS_CACHE=/workspace/hf_cache/datasets" \
    "TRANSFORMERS_CACHE=/workspace/hf_cache/transformers" \
    "HF_TOKEN=${HF_TOKEN}" \
    "WANDB_API_KEY=${WANDB_API_KEY}" \
    "TOKENIZERS_PARALLELISM=false" \
    "CACHE_DIR=/workspace" \
    "COT_ORACLE_EVAL_CACHE_POLICY=refresh" \
    > /workspace/.env

# Start training in tmux
tmux new-session -d -s train bash -c "
    set -a; source /workspace/.env; set +a
    source /workspace/venvs/cot-oracle/bin/activate
    cd /workspace/cot-oracle
    export AO_REPO_PATH=/workspace/ao_reference
    export CACHE_DIR=/workspace
    torchrun --nproc_per_node=\$NUM_GPUS --master_port=29500 \
        src/train.py \
        --config configs/train.yaml \
        --precomputed-dir data/precomputed \
        --task-order sequential \
        --save-dir /workspace/checkpoints/cot_oracle_seq \
        --wandb-run vast_seq_\$(date +%Y%m%d_%H%M) \
    2>&1 | tee /workspace/train.log
    echo 'TRAINING DONE'
    sleep infinity
"
echo "Training started in tmux session 'train'"
REMOTE_SETUP

echo ""
echo "============================================"
echo "  Instance ID: $INSTANCE_ID"
echo "  SSH:  $SSH_CMD"
echo "  Attach:  $SSH_CMD -t 'tmux attach -t train'"
echo "  Tail log: $SSH_CMD 'tail -f /workspace/train.log'"
echo "  Destroy:  vastai destroy instance $INSTANCE_ID"
echo "============================================"

# Save instance info for other scripts
cat > /tmp/vast_instance.env <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
SSH_CMD="$SSH_CMD"
EOF
echo "Instance info saved to /tmp/vast_instance.env"
