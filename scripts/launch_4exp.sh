#!/bin/bash
# Launch 4 experiments: {0.6B, 8B} x {flamingo, no flamingo}
# Each on a separate GPU (0-3) in its own tmux window

set -e

VENV="/var/tmp/jbauer/venvs/cot-oracle/bin/python"
PROJ="/nfs/nhome/live/jbauer/cot-oracle"
HF_TOKEN=$(grep HF_TOKEN ~/.env | cut -d'"' -f2)
WANDB_KEY=$(awk '/api.wandb.ai/{found=1} found && /password/{print $2; exit}' ~/.netrc)

ENV_SETUP="export PYTHONUNBUFFERED=1; export HF_TOKEN=$HF_TOKEN; export WANDB_API_KEY=$WANDB_KEY; export CACHE_DIR=/ceph/scratch/jbauer; export HF_HOME=\$CACHE_DIR/hf; export HF_HUB_CACHE=\$HF_HOME/hub; export HF_DATASETS_CACHE=\$HF_HOME/datasets; export TRANSFORMERS_CACHE=\$HF_HOME/transformers; export COT_ORACLE_EVAL_CACHE_POLICY=refresh; cd $PROJ"

COMMON_ARGS="--config configs/train.yaml --eval-steps 1000 --max-items-per-eval 10 --wandb-entity japhba-personal"

# Kill existing experiment sessions if any
for s in exp-06b exp-06b-flamingo exp-8b exp-8b-flamingo; do
    tmux kill-session -t "$s" 2>/dev/null || true
done

echo "Launching 4 experiments..."

# Exp 1: 0.6B no flamingo (GPU 0)
tmux new-session -d -s exp-06b \
    "$ENV_SETUP; CUDA_VISIBLE_DEVICES=0 $VENV src/train.py $COMMON_ARGS --model Qwen/Qwen3-0.6B --wandb-run 0.6b-no-flamingo 2>&1 | tee /tmp/exp-06b.log; bash"
echo "  [GPU 0] 0.6B no-flamingo -> tmux: exp-06b"

# Exp 2: 0.6B flamingo (GPU 1)
tmux new-session -d -s exp-06b-flamingo \
    "$ENV_SETUP; CUDA_VISIBLE_DEVICES=1 $VENV src/train.py $COMMON_ARGS --model Qwen/Qwen3-0.6B --flamingo --wandb-run 0.6b-flamingo 2>&1 | tee /tmp/exp-06b-flamingo.log; bash"
echo "  [GPU 1] 0.6B flamingo -> tmux: exp-06b-flamingo"

# Exp 3: 8B no flamingo (GPU 2)
tmux new-session -d -s exp-8b \
    "$ENV_SETUP; CUDA_VISIBLE_DEVICES=2 $VENV src/train.py $COMMON_ARGS --wandb-run 8b-no-flamingo 2>&1 | tee /tmp/exp-8b.log; bash"
echo "  [GPU 2] 8B no-flamingo -> tmux: exp-8b"

# Exp 4: 8B flamingo (GPU 3)
tmux new-session -d -s exp-8b-flamingo \
    "$ENV_SETUP; CUDA_VISIBLE_DEVICES=3 $VENV src/train.py $COMMON_ARGS --flamingo --wandb-run 8b-flamingo 2>&1 | tee /tmp/exp-8b-flamingo.log; bash"
echo "  [GPU 3] 8B flamingo -> tmux: exp-8b-flamingo"

echo ""
echo "All 4 experiments launched. Logs at /tmp/exp-*.log"
echo "tmux sessions: exp-06b, exp-06b-flamingo, exp-8b, exp-8b-flamingo"
tmux ls
