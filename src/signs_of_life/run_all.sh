#!/bin/bash
# Run all three signs-of-life experiments
# Usage: cd src/signs_of_life && bash run_all.sh [--model Qwen/Qwen3-1.7B]

MODEL="${1:-Qwen/Qwen3-1.7B}"
N_PROBLEMS=10

echo "Running signs-of-life experiments with model: $MODEL"
echo "=============================================="

echo ""
echo "=== Experiment A: Existing AO on CoT activations ==="
python experiment_a_existing_ao.py --model "$MODEL" --n-problems $N_PROBLEMS

echo ""
echo "=== Experiment B: Logit lens trajectories ==="
python experiment_b_logit_lens.py --model "$MODEL" --n-problems $N_PROBLEMS

echo ""
echo "=== Experiment C: Attention suppression ==="
python experiment_c_attention_suppression.py --model "$MODEL" --n-problems $N_PROBLEMS

echo ""
echo "All experiments complete! Results in ../../results/signs_of_life/"
