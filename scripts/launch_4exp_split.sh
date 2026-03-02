#!/bin/bash
# Launch 4 experiments split across 2 nodes:
# H100TB (gpu-sr675-34, job 2473034): 8B experiments
# L40S (gpu-sr675-33, job 2473152): 0.6B experiments
set -e

NODE=$1   # h100 or l40s
JOBID=$2

VENV="/var/tmp/jbauer/venvs/cot-oracle/bin/python"
PROJ="/nfs/nhome/live/jbauer/cot-oracle"
HF_TOKEN=$(grep HF_TOKEN ~/.env | cut -d'"' -f2)
WANDB_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)

ENV_SETUP="export PYTHONUNBUFFERED=1; export HF_TOKEN=$HF_TOKEN; export WANDB_API_KEY=$WANDB_KEY; export CACHE_DIR=/ceph/scratch/jbauer; export HF_HOME=\$CACHE_DIR/hf; export HF_HUB_CACHE=\$HF_HOME/hub; export HF_DATASETS_CACHE=\$HF_HOME/datasets; export TRANSFORMERS_CACHE=\$HF_HOME/transformers; export COT_ORACLE_EVAL_CACHE_POLICY=refresh; cd $PROJ"

COMMON_ARGS="--config configs/train.yaml --eval-steps 1000 --max-items-per-eval 10 --wandb-entity japhba-personal"

if [ "$NODE" = "h100" ]; then
    echo "Launching 8B experiments on H100TB..."
    for s in exp-8b exp-8b-flamingo; do tmux kill-session -t "$s" 2>/dev/null || true; done

    # GPU 0: 8B no flamingo (batch_size=4 for single H100)
    tmux new-session -d -s exp-8b \
        "$ENV_SETUP; CUDA_VISIBLE_DEVICES=0 $VENV src/train.py $COMMON_ARGS --batch-size 4 --eval-batch-size 2 --wandb-run 8b-no-flamingo 2>&1 | tee /tmp/exp-8b.log; bash"
    echo "  [GPU 0] 8B no-flamingo -> tmux: exp-8b"

    # GPU 1: 8B flamingo (batch_size=2 for memory)
    tmux new-session -d -s exp-8b-flamingo \
        "$ENV_SETUP; CUDA_VISIBLE_DEVICES=1 $VENV src/train.py $COMMON_ARGS --flamingo --batch-size 2 --eval-batch-size 1 --wandb-run 8b-flamingo 2>&1 | tee /tmp/exp-8b-flamingo.log; bash"
    echo "  [GPU 1] 8B flamingo -> tmux: exp-8b-flamingo"

elif [ "$NODE" = "l40s" ]; then
    echo "Launching 0.6B experiments on L40S..."
    for s in exp-06b exp-06b-flamingo; do tmux kill-session -t "$s" 2>/dev/null || true; done

    # GPU 0: 0.6B no flamingo
    tmux new-session -d -s exp-06b \
        "$ENV_SETUP; CUDA_VISIBLE_DEVICES=0 $VENV src/train.py $COMMON_ARGS --model Qwen/Qwen3-0.6B --wandb-run 0.6b-no-flamingo 2>&1 | tee /tmp/exp-06b.log; bash"
    echo "  [GPU 0] 0.6B no-flamingo -> tmux: exp-06b"

    # GPU 1: 0.6B flamingo
    tmux new-session -d -s exp-06b-flamingo \
        "$ENV_SETUP; CUDA_VISIBLE_DEVICES=1 $VENV src/train.py $COMMON_ARGS --model Qwen/Qwen3-0.6B --flamingo --wandb-run 0.6b-flamingo 2>&1 | tee /tmp/exp-06b-flamingo.log; bash"
    echo "  [GPU 1] 0.6B flamingo -> tmux: exp-06b-flamingo"
fi

echo "Done. tmux sessions:"
tmux ls
