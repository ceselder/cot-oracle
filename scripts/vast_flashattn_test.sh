#!/bin/bash
# Launch a 1xH100 Vast.ai instance and run a flash-attn smoke test with uv extra.
# Usage:
#   bash scripts/vast_flashattn_test.sh
#   bash scripts/vast_flashattn_test.sh --dry-run
#   bash scripts/vast_flashattn_test.sh --keep

set -euo pipefail

PROJECT_DIR="/nfs/nhome/live/jbauer/cot-oracle"
cd "$PROJECT_DIR"

DRY_RUN=false
KEEP_INSTANCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --keep) KEEP_INSTANCE=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI not found in PATH"
    exit 1
fi

if ! vastai show user --raw >/dev/null 2>&1; then
    echo "vastai CLI is not authenticated (run 'vastai set api-key ...' or login)"
    exit 1
fi

set -a
source "$HOME/.env" 2>/dev/null || true
set +a

SSH_OPTS="-A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand=none"
NUM_GPUS_WANT="${NUM_GPUS:-1}"
GPU_FILTER="${GPU_FILTER:-H100_SXM}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-80}"

echo "=== Searching Vast offers (${NUM_GPUS_WANT}x${GPU_FILTER}) ==="
OFFER_ID=$(vastai search offers \
    "num_gpus=${NUM_GPUS_WANT} gpu_name=${GPU_FILTER} reliability>0.9 disk_space>=${DISK_GB} direct_port_count>=1" \
    --type=bid -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])")
echo "Best offer id: $OFFER_ID"

if $DRY_RUN; then
    echo "[dry-run] Would run: vastai create instance $OFFER_ID --image $IMAGE --disk $DISK_GB --direct --raw"
    exit 0
fi

echo "=== Creating instance ==="
CREATE_OUT=$(vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK_GB" --direct --raw 2>&1)
INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])")
echo "Instance ID: $INSTANCE_ID"

cleanup() {
    if [[ "$KEEP_INSTANCE" == "false" && -n "${INSTANCE_ID:-}" ]]; then
        echo "Destroying instance $INSTANCE_ID"
        vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "=== Waiting for instance ==="
SSH_HOST=""
SSH_PORT=""
PUBLIC_IP=""
for i in $(seq 1 90); do
    INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")
    STATUS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "")
    SSH_HOST=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_host','') or d.get('public_ipaddr',''))" 2>/dev/null || echo "")
    SSH_PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('ssh_port') or d.get('direct_port_end') or 22; print(p if int(p) > 0 else 22)" 2>/dev/null || echo "22")
    PUBLIC_IP=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('public_ipaddr',''))" 2>/dev/null || echo "")
    if [[ "$STATUS" == "running" && -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
        echo "Instance running: $SSH_HOST:$SSH_PORT"
        break
    fi
    echo "  [$i/90] status=$STATUS"
    sleep 10
done

if [[ -z "$SSH_HOST" ]]; then
    echo "Instance never became ready"
    exit 1
fi

echo "=== Waiting for SSH connectivity ==="
CONNECTED=false
for i in $(seq 1 40); do
    if ssh $SSH_OPTS -o ConnectTimeout=8 -p "$SSH_PORT" root@"$SSH_HOST" "echo ok" >/dev/null 2>&1; then
        CONNECTED=true
        break
    fi
    if [[ -n "$PUBLIC_IP" ]] && ssh $SSH_OPTS -o ConnectTimeout=8 -p 22 root@"$PUBLIC_IP" "echo ok" >/dev/null 2>&1; then
        SSH_HOST="$PUBLIC_IP"
        SSH_PORT=22
        CONNECTED=true
        break
    fi
    echo "  [$i/40] SSH not ready yet"
    sleep 6
done
if [[ "$CONNECTED" != "true" ]]; then
    echo "SSH failed"
    exit 1
fi

SSH_CMD="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
RSYNC_SSH="ssh $SSH_OPTS -p $SSH_PORT"

echo "=== Syncing project to instance ==="
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    --exclude 'checkpoints/' \
    --exclude 'wandb/' \
    --exclude 'eval_logs/' \
    --exclude 'eval_logs_wandb/' \
    --exclude 'slurm_logs/' \
    --exclude 'data/eval_precomputed/' \
    --exclude 'data/qwen3_rollouts/' \
    --exclude 'data/deepseek_rollouts_for_hf/' \
    --exclude 'data/hf_uploads/' \
    -e "$RSYNC_SSH" \
    "$PROJECT_DIR/" "root@$SSH_HOST:/workspace/cot-oracle/"

echo "=== Running flash-attn smoke test ==="
$SSH_CMD bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/cot-oracle

pip install -q uv
export VENV_LOCAL=/workspace/venvs
export UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}"
uv sync --extra flash-attn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python - <<'PY' | tee /workspace/flashattn_test.log
import importlib.util
import json
import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
results = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cc": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
    "flash_attn_3_installed": importlib.util.find_spec("flash_attn_3") is not None,
    "attn_impl_tests": {},
}

for impl in ["flash_attention_2", "flash_attention_3"]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=impl,
        )
        del model
        torch.cuda.empty_cache()
        results["attn_impl_tests"][impl] = {"ok": True}
    except Exception as e:
        results["attn_impl_tests"][impl] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

print(json.dumps(results, indent=2))
PY
REMOTE

mkdir -p eval_logs/throughput_profiles
rsync -avz -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/flashattn_test.log eval_logs/throughput_profiles/vast_flashattn_test_${INSTANCE_ID}.log

echo ""
echo "============================================"
echo "Instance ID: $INSTANCE_ID"
echo "SSH: $SSH_CMD"
echo "Log: eval_logs/throughput_profiles/vast_flashattn_test_${INSTANCE_ID}.log"
if [[ "$KEEP_INSTANCE" == "true" ]]; then
    echo "Instance kept alive (--keep). Destroy manually:"
    echo "  vastai destroy instance $INSTANCE_ID"
fi
echo "============================================"
