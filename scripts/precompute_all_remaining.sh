#!/bin/bash
# Chain script: run all remaining eval precomputes
# Run on GPU after precompute_activations.py finishes
#
# Order:
# 1. reasoning_termination: Generate labels from 50-rollout resampling
# 2. sycophancy_v2_riya: Generate clean+test CoTs and activation bundles
# 3. eval_responses: Generate CoTs for non-training evals (forced_entropy, cybercrime_ood)
# 4. (Optional) atypical_answer: 200 rollouts/question for majority/minority labels
# 5. (Optional) forced_entropy: Compute logprob entropy at sentence boundaries

set -e
cd /root/cot-oracle

echo "============================================"
echo "Starting remaining precompute chain"
echo "$(date)"
echo "============================================"

# Pull latest code
echo ""
echo "=== Pulling latest code ==="
git pull origin main

# 1. Reasoning termination: fix 173 pending labels
echo ""
echo "============================================"
echo "1/5: Reasoning termination precompute"
echo "    (3 rollouts x 173 questions, 50 resamples/prefix)"
echo "    Expected: ~20-30 min"
echo "============================================"
PYTHONUNBUFFERED=1 python3 scripts/precompute_reasoning_termination.py \
    --eval-dir data/evals \
    --model Qwen/Qwen3-8B \
    --target-items 100 \
    --n-resamples 50 \
    --n-rollouts 3 \
    --max-cot-tokens 4096

# 2. Precompute activations for sycophancy_v2_riya (now in training evals)
echo ""
echo "============================================"
echo "2/5: sycophancy_v2_riya activation precompute"
echo "    (100 items, clean+test CoTs + activations)"
echo "    Expected: ~10 min"
echo "============================================"
PYTHONUNBUFFERED=1 python3 src/evals/precompute_activations.py \
    --eval-dir data/evals \
    --output-dir data/eval_precomputed \
    --model Qwen/Qwen3-8B \
    --evals sycophancy_v2_riya \
    --overwrite

# 3. After reasoning_termination has real items, precompute its activations
echo ""
echo "============================================"
echo "3/5: reasoning_termination_riya activation precompute"
echo "    (uses cot_prefix from step 1)"
echo "    Expected: ~5 min"
echo "============================================"
PYTHONUNBUFFERED=1 python3 src/evals/precompute_activations.py \
    --eval-dir data/evals \
    --output-dir data/eval_precomputed \
    --model Qwen/Qwen3-8B \
    --evals reasoning_termination_riya \
    --overwrite

# 4. Generic CoT precompute for non-training evals
echo ""
echo "============================================"
echo "4/5: Generic eval response precompute"
echo "    (forced_answer_entropy_riya, cybercrime_ood, sycophancy)"
echo "    Expected: ~10 min"
echo "============================================"
PYTHONUNBUFFERED=1 python3 scripts/precompute_eval_responses.py \
    --eval-dir data/evals \
    --model Qwen/Qwen3-8B \
    --batch-size 4 \
    --max-new-tokens 4096 \
    --evals forced_answer_entropy_riya cybercrime_ood sycophancy

# 5. Forced answer entropy precompute (regression labels)
echo ""
echo "============================================"
echo "5/5: Forced answer entropy precompute"
echo "    (10 rollouts x 100 questions, logprob extraction)"
echo "    Expected: ~15 min"
echo "============================================"
PYTHONUNBUFFERED=1 python3 scripts/precompute_forced_entropy.py \
    --eval-path data/evals/forced_answer_entropy_riya.json \
    --model Qwen/Qwen3-8B \
    --n-rollouts 10 \
    --temperature 0.7

echo ""
echo "============================================"
echo "All precomputes complete!"
echo "$(date)"
echo "============================================"
echo ""
echo "Remaining TODO (expensive, optional):"
echo "  python3 scripts/precompute_atypical_rollouts.py --model Qwen/Qwen3-8B --n-rollouts 200"
echo ""
echo "Next: git add data/evals/*.json && git commit && git push"
echo "Then: python3 scripts/upload_eval_datasets.py"
