#!/bin/bash
# Launch CoT Oracle v3 training on vast.ai
#
# Key differences from v2:
#   - Fresh LoRA (no Adam's AO checkpoint)
#   - FineWeb context prediction (100K, streaming from HF)
#   - Medium corpus (47K entries vs 1K mini)
#   - Fuzzy eval scoring for generation tasks
#   - Unfaithfulness evals ON (14 eval datasets)
#   - Harder evals: correct_authority, step_counting, anchoring_bias, scruples
#
# Usage: bash scripts/launch_v3.sh

set -e

# SSH config
SSH_HOST="root@ssh9.vast.ai"
SSH_PORT=14465
SSH_CMD="ssh -p $SSH_PORT $SSH_HOST"

# Sync code
echo "=== Syncing code to vast.ai ==="
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'checkpoints/' --exclude 'results/' --exclude 'thought-anchors/' \
    --exclude 'data/cot_corpus_v5/corpus_medium.jsonl' \
    --exclude 'data/cot_qa_medium.jsonl' \
    --exclude 'data/cot_corpus/' --exclude 'data/cot_corpus_8b/' \
    --exclude 'data/cot_corpus_diverse/' --exclude 'data/cot_corpus_v4/' \
    --exclude 'data/hf_upload/' \
    -e "ssh -p $SSH_PORT" \
    . $SSH_HOST:/root/cot-oracle/

# Sync medium corpus separately (big file)
echo "=== Syncing medium corpus ==="
rsync -avz --progress -e "ssh -p $SSH_PORT" \
    data/cot_corpus_v5/corpus_medium.jsonl \
    $SSH_HOST:/root/cot-oracle/data/cot_corpus_v5/

# Sync conversational QA (160MB)
echo "=== Syncing conversational QA ==="
rsync -avz --progress -e "ssh -p $SSH_PORT" \
    data/cot_qa_medium.jsonl \
    $SSH_HOST:/root/cot-oracle/data/

# Sync eval datasets
echo "=== Syncing eval datasets ==="
rsync -avz -e "ssh -p $SSH_PORT" \
    data/evals/ \
    $SSH_HOST:/root/cot-oracle/data/evals/

# Ensure AO repo symlink exists
echo "=== Ensuring AO repo symlink ==="
$SSH_CMD "test -L /root/cot-oracle/activation_oracles || ln -s /root/activation_oracles /root/cot-oracle/activation_oracles"

echo "=== Launching v3 training ==="
$SSH_CMD "cd /root/cot-oracle && \
    export WANDB_API_KEY='${WANDB_API_KEY:?Set WANDB_API_KEY}' && \
    export HF_TOKEN='${HF_TOKEN:?Set HF_TOKEN}' && \
    export PYTHONUNBUFFERED=1 && \
    nohup torchrun --nproc_per_node=1 src/train_v3.py \
        --corpus data/cot_corpus_v5/corpus_medium.jsonl \
        --model Qwen/Qwen3-8B \
        --lr 1e-5 \
        --batch-size 8 \
        --epochs 1 \
        --save-dir checkpoints/cot_oracle_v3 \
        --wandb-run cot_oracle_v3b \
        --eval-steps 500 \
        --save-steps 1000 \
        --eval-dir data/evals \
        --fast-eval-n 5 \
        --n-context-pred 100000 \
        --n-fineweb 100000 \
        --n-answer-pred 20000 \
        --n-full-recon 15000 \
        --n-causal-pred 30000 \
        --n-conversational 5000 \
        --n-decorative 10000 \
        --n-domain 15000 \
        --n-correctness 15000 \
        --n-persona 0 \
        --conv-qa data/cot_qa_medium.jsonl \
    > /root/train_v3.log 2>&1 &"

echo "=== Training launched ==="
echo "Monitor with: ssh -p $SSH_PORT $SSH_HOST 'tail -f /root/train_v3.log'"
echo "Check step:   ssh -p $SSH_PORT $SSH_HOST 'grep -oP \"\\d+/\\d+\" /root/train_v3.log | tail -1'"
