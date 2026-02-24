#!/bin/bash
# Generate diverse CoT rollouts using vLLM.
# Run this when the GPU is free (after training completes).
#
# Needs ~40GB VRAM for vLLM + Qwen3-8B in bf16.
# Can run alongside Stage 3 training if there's enough VRAM (unlikely).

set -e

echo "Installing vLLM if needed..."
pip install vllm -q 2>/dev/null || true

echo "Starting rollout generation..."

export PYTHONUNBUFFERED=1

nohup python3 scripts/generate_diverse_rollouts.py \
    --prompts data/diverse_rollouts/prompts.jsonl \
    --output data/diverse_rollouts/corpus.jsonl \
    --model Qwen/Qwen3-8B \
    --batch-size 256 \
    --max-prompts 40000 \
    --max-cot-tokens 2048 \
    --max-direct-tokens 256 \
    --resume \
    > rollouts.log 2>&1 &

echo "Rollout generation launched with PID $!"
echo "Log: rollouts.log"
echo "Monitor: tail -f rollouts.log"
