#!/bin/bash
# Run maxk_layers sweep across no-DPO baseline + DPO checkpoints
# Usage: bash scripts/run_dpo_sweep.sh

set -e
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-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}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_X_REQUIRE_LEGACY_SERVICE=1

cd /root/cot-oracle

DPO_REPO="ceselder/cot-oracle-calibration-dpo"
NO_DPO="ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO"
STEPS="500 800"
MAX_ITEMS=50
EVAL_BATCH=2

echo "=== Downloading DPO policy adapters ==="
python3 -c "
from huggingface_hub import snapshot_download
import os
for step in [${STEPS// /, }]:
    path = snapshot_download('${DPO_REPO}', allow_patterns=f'step_{step}/policy/*')
    policy_dir = os.path.join(path, f'step_{step}', 'policy')
    print(f'step_{step}: {policy_dir}')
    # Verify files exist
    assert os.path.exists(os.path.join(policy_dir, 'adapter_config.json')), f'Missing adapter_config for step {step}'
    print(f'  OK: adapter_config.json + adapter_model.safetensors')
"

echo ""
echo "=== Running no-DPO baseline ==="
python3 scripts/sweep_oracle_maxk_layers.py \
    --checkpoint "$NO_DPO" \
    --max-items $MAX_ITEMS \
    --eval-batch-size $EVAL_BATCH \
    --activation-extract-batch-size $EVAL_BATCH \
    --wandb-run "sweep-no-dpo-baseline" \
    --output-dir "eval_logs/dpo_comparison/no_dpo"

for STEP in $STEPS; do
    echo ""
    echo "=== Running DPO step_${STEP} ==="
    # Find the cached policy adapter path
    POLICY_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${DPO_REPO}', allow_patterns=f'step_${STEP}/policy/*')
print(os.path.join(path, f'step_${STEP}', 'policy'))
")
    echo "  Adapter: $POLICY_PATH"

    python3 scripts/sweep_oracle_maxk_layers.py \
        --checkpoint "$POLICY_PATH" \
        --max-items $MAX_ITEMS \
        --eval-batch-size $EVAL_BATCH \
        --activation-extract-batch-size $EVAL_BATCH \
        --wandb-run "sweep-dpo-step-${STEP}" \
        --output-dir "eval_logs/dpo_comparison/dpo_step_${STEP}"
done

echo ""
echo "=== All sweeps complete ==="
