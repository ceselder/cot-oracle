#!/bin/bash
# SSH-free Vast.ai flash-attn smoke test using onstart + logs.
# Usage:
#   bash scripts/vast_flashattn_onstart_test.sh
#   bash scripts/vast_flashattn_onstart_test.sh --dry-run
#   bash scripts/vast_flashattn_onstart_test.sh --keep

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

NUM_GPUS_WANT="${NUM_GPUS:-1}"
GPU_FILTER="${GPU_FILTER:-H100_SXM}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
DISK_GB="${VAST_DISK_GB:-80}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
TIMEOUT_MIN="${TIMEOUT_MIN:-45}"

echo "=== Searching Vast offers (${NUM_GPUS_WANT}x${GPU_FILTER}) ==="
OFFER_ID=$(vastai search offers \
    "num_gpus=${NUM_GPUS_WANT} gpu_name=${GPU_FILTER} reliability>0.9 disk_space>=${DISK_GB}" \
    --type=bid -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])")
echo "Best offer id: $OFFER_ID"

ONSTART_FILE=$(mktemp /tmp/vast_flashattn_onstart.XXXXXX.sh)
cat > "$ONSTART_FILE" <<'ONSTART'
#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/flashattn_onstart.log) 2>&1

echo "FLASH_TEST_BEGIN"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi -L || true

mkdir -p /workspace/flashattn-smoke
cd /workspace/flashattn-smoke
cat > pyproject.toml <<'PYPROJECT'
[project]
name = "flashattn-smoke"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "transformers",
]

[project.optional-dependencies]
flash-attn = [
    "flash-attn",
]

[tool.uv.extra-build-dependencies]
flash-attn = [
    "torch",
]
PYPROJECT

pip install -q uv
export VENV_LOCAL=/workspace/venvs
export UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}"
uv sync --extra flash-attn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python3 - <<'PY'
import importlib.util
import json
import os
import torch
from transformers import AutoModelForCausalLM

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
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

print("FLASH_TEST_JSON_BEGIN")
print(json.dumps(results, indent=2))
print("FLASH_TEST_JSON_END")
PY

echo "FLASH_TEST_DONE"
ONSTART
chmod +x "$ONSTART_FILE"

if $DRY_RUN; then
    echo "[dry-run] Would create instance with onstart script: $ONSTART_FILE"
    rm -f "$ONSTART_FILE"
    exit 0
fi

echo "=== Creating instance ==="
CREATE_OUT=$(MODEL_NAME="$MODEL_NAME" vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --env "-e MODEL_NAME=${MODEL_NAME}" \
    --onstart "$ONSTART_FILE" \
    --raw 2>&1)
INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])")
echo "Instance ID: $INSTANCE_ID"
rm -f "$ONSTART_FILE"

cleanup() {
    if [[ "$KEEP_INSTANCE" == "false" && -n "${INSTANCE_ID:-}" ]]; then
        echo "Destroying instance $INSTANCE_ID"
        vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "=== Waiting for completion marker in logs ==="
END_TS=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
FOUND_DONE=false
while [[ $(date +%s) -lt $END_TS ]]; do
    LOGS=$(vastai logs "$INSTANCE_ID" --tail 400 2>/dev/null || true)
    if echo "$LOGS" | rg -q "FLASH_TEST_DONE"; then
        FOUND_DONE=true
        break
    fi
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "")
    echo "  status=$STATUS waiting_for=FLASH_TEST_DONE"
    sleep 20
done

mkdir -p eval_logs/throughput_profiles
LOG_PATH="eval_logs/throughput_profiles/vast_flashattn_onstart_${INSTANCE_ID}.log"
vastai logs "$INSTANCE_ID" --tail 2000 > "$LOG_PATH" || true

if [[ "$FOUND_DONE" != "true" ]]; then
    echo "Timed out waiting for FLASH_TEST_DONE. Saved logs to $LOG_PATH"
    exit 1
fi

echo "Saved logs to $LOG_PATH"
echo "=== Result excerpt ==="
rg -n "FLASH_TEST_JSON_BEGIN|FLASH_TEST_JSON_END|flash_attn|flash_attention_2|flash_attention_3|FLASH_TEST_DONE" "$LOG_PATH" || true

echo ""
echo "============================================"
echo "Instance ID: $INSTANCE_ID"
echo "Log: $LOG_PATH"
if [[ "$KEEP_INSTANCE" == "true" ]]; then
    echo "Instance kept alive (--keep). Destroy manually:"
    echo "  vastai destroy instance $INSTANCE_ID"
fi
echo "============================================"
