#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/run_pids}"
LAUNCHER="${ROOT_DIR}/scripts/run_paper_ablation.sh"

mkdir -p "$LOG_DIR" "$PID_DIR"

ABLATORS=(
  "0 29500 configs/paper_ablations/adam_recipe_1layer.yaml adam_recipe_1layer"
  "1 29501 configs/paper_ablations/ours_1layer.yaml ours_1layer"
  "2 29502 configs/paper_ablations/ours_3layers.yaml ours_3layers"
  "3 29503 configs/paper_ablations/ours_3layers_onpolicy_lens_only.yaml ours_3layers_onpolicy_lens_only"
)

echo "Launching ${#ABLATORS[@]} paper ablations from ${ROOT_DIR}"

for spec in "${ABLATORS[@]}"; do
  read -r gpu master_port overlay run_tag <<<"$spec"
  log_path="${LOG_DIR}/${run_tag}.log"
  pid_path="${PID_DIR}/${run_tag}.pid"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU ${gpu} -> ${run_tag} (${overlay})"
  (
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MASTER_PORT="${master_port}"
    export NPROC=1
    exec bash "$LAUNCHER" "$overlay"
  ) >"$log_path" 2>&1 &

  run_pid=$!
  echo "$run_pid" >"$pid_path"
  echo "  pid=${run_pid} log=${log_path}"
done

wait
