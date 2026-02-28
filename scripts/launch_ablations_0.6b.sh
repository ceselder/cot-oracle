#!/bin/bash
# Launch all four Qwen3-0.6B ablation runs.
#
# Usage:
#   bash scripts/launch_ablations_0.6b.sh                    # run all 4 sequentially
#   bash scripts/launch_ablations_0.6b.sh random_layers       # run one specific ablation
#   bash scripts/launch_ablations_0.6b.sh noise_activations
#   bash scripts/launch_ablations_0.6b.sh flamingo

set -e

export PYTHONUNBUFFERED=1
export AO_REPO_PATH="${AO_REPO_PATH:-/root/activation_oracles}"

: "${HF_TOKEN:?Set HF_TOKEN}"
export HF_TOKEN
# WANDB_API_KEY: wandb reads from ~/.netrc if env var is unset

BASE="configs/train.yaml"
BASE_06B="configs/ablation_base_0.6b.yaml"

run_ablation() {
    local name="$1"; shift
    local overlay="$1"; shift
    echo ""
    echo "============================================================"
    echo "  Ablation: ${name}"
    echo "  Configs: ${BASE} + ${BASE_06B} + ${overlay}"
    echo "============================================================"
    python3 src/train.py \
        --config "${BASE}" "${BASE_06B}" "${overlay}" \
        --no-step0-eval \
        "$@"
}

ABLATIONS=(
    "random_layers:configs/ablation_random_layers.yaml"
    "noise_activations:configs/ablation_noise_activations.yaml"
    "flamingo:configs/ablation_flamingo_0.6b.yaml"
)

FILTER="${1:-}"

for entry in "${ABLATIONS[@]}"; do
    name="${entry%%:*}"
    overlay="${entry##*:}"
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi
    run_ablation "$name" "$overlay"
done
