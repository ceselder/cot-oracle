#!/bin/bash
#SBATCH --job-name=sae_chunked
#SBATCH --partition=gpu_lowp
#SBATCH --nodelist=gpu-xd670-30
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/nhome/live/jbauer/cot-oracle/logs/sae_chunked_%j.log

set -e
cd /nfs/nhome/live/jbauer/cot-oracle
source ~/.bashrc

VENV=/var/tmp/jbauer/venvs/cot-oracle
CKPT=/ceph/scratch/jbauer/checkpoints/cot_oracle_v15_stochastic
TASKS=(chunked_compqa_backtrack chunked_compqa_remaining_strategy chunked_compqa_self_correction chunked_compqa_verification chunked_convqa)

pids=()
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    echo "Launching $task on GPU $i"
    CUDA_VISIBLE_DEVICES=$i $VENV/bin/python scripts/run_comprehensive_eval.py \
        --config configs/train.yaml \
        --checkpoint "$CKPT" \
        --tasks "$task" \
        --baselines sae_llm \
        --output-dir data/comprehensive_eval \
        --rerun \
        > logs/sae_chunked_${task}_$SLURM_JOB_ID.log 2>&1 &
    pids+=($!)
done

echo "Waiting for ${#pids[@]} jobs..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "Process $pid failed"
        failed=$((failed + 1))
    fi
done

# Clean up stale processes
kill $(jobs -p) 2>/dev/null || true

echo "Done. Failed: $failed / ${#pids[@]}"
exit $failed
