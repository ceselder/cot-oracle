#!/bin/bash
# Launch run_comprehensive_eval.py in parallel across multiple GPUs.
# Each GPU handles a subset of tasks from src/tasks.py.
# LLM monitor results are cached on shared disk (FAST_CACHE_DIR).
set -euo pipefail

cd /nfs/nhome/live/jbauer/cot-oracle
source /var/tmp/jbauer/venvs/cot-oracle/bin/activate
source ~/.env 2>/dev/null || true

export CACHE_DIR="${CACHE_DIR:-/ceph/scratch/jbauer}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CHECKPOINT="/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic"
CONFIG="configs/train.yaml"
N_EXAMPLES=25
OUTPUT_DIR="data/comprehensive_eval"
POSITION_MODE="all"

# All tasks from src/tasks.py with valid HF repos (fineweb + rot13 excluded automatically)
TASKS=(
  hint_admission
  atypical_answer
  reasoning_termination
  answer_trajectory
  correctness
  decorative_cot
  resampling_importance
  convqa
  chunked_convqa
  chunked_compqa
  backtrack_prediction
  sqa
  sycophancy
  truthfulqa_hint_verbalized
  truthfulqa_hint
  sentence_insertion
  cot_description
  cot_metacognition
  sae_unverbalized
  taxonomy_ood
)

N_GPUS=8
declare -a GPU_TASKS
for i in "${!TASKS[@]}"; do
  gpu=$((i % N_GPUS))
  GPU_TASKS[$gpu]="${GPU_TASKS[$gpu]:-} ${TASKS[$i]}"
done

PIDS=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
  tasks="${GPU_TASKS[$gpu]:-}"
  if [ -z "${tasks// }" ]; then
    continue
  fi
  logfile="$LOG_DIR/parallel_eval_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
  echo "GPU $gpu → tasks:$tasks → $logfile"
  nohup env CUDA_VISIBLE_DEVICES=$gpu python scripts/run_comprehensive_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --n-examples "$N_EXAMPLES" \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --position-mode "$POSITION_MODE" \
    --tasks $tasks \
    >"$logfile" 2>&1 &
  PIDS+=($!)
done

echo "Launched ${#PIDS[@]} workers: ${PIDS[*]}"
echo "Waiting for all workers to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  if wait "$pid"; then
    echo "Worker $i (pid $pid) completed OK"
  else
    echo "Worker $i (pid $pid) FAILED (exit $?)"
    FAILED=$((FAILED + 1))
  fi
done

echo "All workers done. Failed: $FAILED"

# Final pass: generate combined plot (no GPU needed)
echo "Generating combined plot..."
python scripts/run_comprehensive_eval.py \
  --config "$CONFIG" \
  --n-examples "$N_EXAMPLES" \
  --output-dir "$OUTPUT_DIR" \
  --plot-only \
  >>"$LOG_DIR/parallel_eval_plot_$(date +%Y%m%d_%H%M%S).log" 2>&1

echo "Done. Plot saved to $OUTPUT_DIR/comprehensive_eval.png"
