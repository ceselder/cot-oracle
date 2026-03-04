#!/bin/bash
# Launch calibration DPO on Vast.ai
# Usage: ssh -p PORT root@HOST 'bash -s' < launch_vast.sh

set -euo pipefail

# Environment (secrets from env vars, never logged)
export PYTHONUNBUFFERED=1
export WANDB_X_REQUIRE_LEGACY_SERVICE=1
export PYTHONPATH=~/cot-oracle/src:~/cot-oracle/ao_reference:~/cot-oracle/calibration_dpo

# Check required env vars
for var in OPENROUTER_API_KEY WANDB_API_KEY HF_TOKEN; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var not set"
        exit 1
    fi
done

cd ~/cot-oracle

# Install dependencies if needed
pip install -q httpx peft transformers accelerate datasets wandb huggingface_hub filelock pyyaml 2>/dev/null || true

echo "Starting calibration DPO training..."
nohup python calibration_dpo/train_dpo.py --config calibration_dpo/config.yaml \
    > logs/calibration_dpo.log 2>&1 &

echo "PID: $!"
echo "Logs: tail -f logs/calibration_dpo.log"
