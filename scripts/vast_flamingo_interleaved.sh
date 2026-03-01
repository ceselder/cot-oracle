#!/bin/bash
# Launch flamingo + interleaved curriculum on 1xH100 vast.ai (interruptible).
# Usage: bash scripts/vast_flamingo_interleaved.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/nfs/nhome/live/jbauer/cot-oracle"
cd "$PROJECT_DIR"

# Load credentials from ~/.env
set -a; source ~/.env; set +a
WANDB_API_KEY=$(awk '/machine api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SSH_OPTS="-A -o StrictHostKeyChecking=no -o ProxyCommand=none"

# ── 1. Find cheapest 1xH100 (interruptible/bid) ─────────────────────
echo "=== Searching for 1xH100 on-demand offers ==="
OFFER_ID=$(vastai search offers \
    "num_gpus=1 gpu_name=H100_SXM reliability>0.9 disk_space>=150" \
    --type=on-demand -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])" \
)
echo "Best offer: $OFFER_ID"

if $DRY_RUN; then
    echo "[dry-run] Would create instance from offer $OFFER_ID"
    vastai search offers \
        "num_gpus=1 gpu_name=H100_SXM reliability>0.9 disk_space>=150" \
        --type=on-demand -o 'dph_total' --raw 2>/dev/null \
        | python3 -c "import sys,json; o=json.load(sys.stdin)[0]; print(f'  Price: \${o[\"dph_total\"]:.2f}/hr  GPU: {o[\"gpu_name\"]}  VRAM: {o[\"gpu_ram\"]}MB')"
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
    SSH_HOST=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('public_ipaddr','') or json.load(sys.stdin).get('ssh_host',''))" 2>/dev/null || echo "")
    SSH_PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('direct_port_end','') or d.get('ssh_port',''))" 2>/dev/null || echo "")

    if [[ "$STATUS" == "running" ]]; then
        # Extract both direct IP and proxy SSH
        PROXY_HOST=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ssh_host',''))" 2>/dev/null || echo "")
        PROXY_PORT=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ssh_port',''))" 2>/dev/null || echo "")
        echo "Instance ready. Direct: $SSH_HOST:-1, Proxy: $PROXY_HOST:$PROXY_PORT"
        break
    fi
    echo "  [$i/60] status=$STATUS, waiting..."
    sleep 10
done

if [[ -z "$SSH_HOST" && -z "$PROXY_HOST" ]]; then
    echo "ERROR: Instance did not become ready in 10 minutes"
    echo "Check: vastai show instance $INSTANCE_ID"
    exit 1
fi

echo "=== Waiting for SSH connectivity ==="
CONNECTED=false
for i in $(seq 1 30); do
    # Try direct IP port 22 first
    if [[ -n "$SSH_HOST" ]] && ssh $SSH_OPTS -o ConnectTimeout=5 -p 22 root@$SSH_HOST "echo ok" &>/dev/null; then
        SSH_PORT=22; FINAL_HOST=$SSH_HOST
        SSH_CMD="ssh $SSH_OPTS -p 22 root@$SSH_HOST"
        echo "SSH connected (direct $SSH_HOST:22)"
        CONNECTED=true; break
    fi
    # Try proxy SSH (ssh2.vast.ai:PORT)
    if [[ -n "$PROXY_HOST" && -n "$PROXY_PORT" ]] && ssh $SSH_OPTS -o ConnectTimeout=5 -p "$PROXY_PORT" root@"$PROXY_HOST" "echo ok" &>/dev/null; then
        SSH_PORT=$PROXY_PORT; FINAL_HOST=$PROXY_HOST
        SSH_CMD="ssh $SSH_OPTS -p $PROXY_PORT root@$PROXY_HOST"
        echo "SSH connected (proxy $PROXY_HOST:$PROXY_PORT)"
        CONNECTED=true; break
    fi
    echo "  [$i/30] SSH not ready, waiting..."
    sleep 5
done

if ! $CONNECTED; then
    echo "ERROR: SSH never connected. Instance info:"
    vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -m json.tool
    echo "Try: ssh $SSH_OPTS -p $PROXY_PORT root@$PROXY_HOST"
    exit 1
fi

# ── 4. Rsync code + data ─────────────────────────────────────────────
echo "=== Syncing code and data ==="
RSYNC_SSH="ssh $SSH_OPTS -p $SSH_PORT"

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
    --exclude 'eval_logs/' \
    --exclude 'eval_logs_wandb/' \
    --exclude 'slurm_logs/' \
    --exclude 'downloaded_saes/' \
    -e "$RSYNC_SSH" \
    "$PROJECT_DIR/" "root@$FINAL_HOST:/workspace/cot-oracle/"

AO_LOCAL="${PROJECT_DIR}/ao_reference"
if [[ -d "$AO_LOCAL" ]]; then
    rsync -avz --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        -e "$RSYNC_SSH" \
        "$AO_LOCAL/" "root@$FINAL_HOST:/workspace/ao_reference/"
fi

# ── 5. Install deps + start flamingo interleaved training ─────────────
echo "=== Setting up environment and starting training ==="
$SSH_CMD bash -s <<REMOTE_SETUP
set -euo pipefail

cd /workspace/cot-oracle

# Install uv + deps
pip install -q uv 2>&1 | tail -1
UV_PROJECT_ENVIRONMENT=/workspace/venvs/cot-oracle uv sync 2>&1 | tail -3
source /workspace/venvs/cot-oracle/bin/activate

# Symlink AO datasets
ln -sf /workspace/ao_reference/datasets /workspace/cot-oracle/datasets

mkdir -p /workspace/hf_cache

printf '%s\n' \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "HF_HOME=/workspace/hf_cache" \
    "HF_HUB_CACHE=/workspace/hf_cache/hub" \
    "HF_DATASETS_CACHE=/workspace/hf_cache/datasets" \
    "HF_TOKEN=${HF_TOKEN}" \
    "WANDB_API_KEY=${WANDB_API_KEY}" \
    "TOKENIZERS_PARALLELISM=false" \
    "CACHE_DIR=/workspace" \
    > /workspace/.env

# Start flamingo + interleaved training in tmux
tmux new-session -d -s train bash -c "
    set -a; source /workspace/.env; set +a
    source /workspace/venvs/cot-oracle/bin/activate
    cd /workspace/cot-oracle
    export AO_REPO_PATH=/workspace/ao_reference
    export CACHE_DIR=/workspace
    python src/train.py \
        --config configs/train.yaml \
        --precomputed-dir data/precomputed \
        --flamingo \
        --flamingo-xattn-interval 4 \
        --flamingo-xattn-lora-r 64 \
        --flamingo-max-ctx-tokens 2048 \
        --fresh-lora \
        --task-order interleaved \
        --batch-size 2 \
        --effective-batch-size 256 \
        --save-dir /workspace/checkpoints/flamingo_interleaved \
        --wandb-run flamingo_interleaved_v1 \
        --wandb-group jan \
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

cat > /tmp/vast_instance.env <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$FINAL_HOST
SSH_PORT=$SSH_PORT
SSH_CMD="$SSH_CMD"
EOF
echo "Instance info saved to /tmp/vast_instance.env"
