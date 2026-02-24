# CoT Oracle

White-box chain-of-thought (CoT) monitoring built on [Activation Oracles](https://arxiv.org/abs/2512.15674).

CoT text is often unfaithful. This project reads reasoning from activations to detect when stated reasoning diverges from actual computational influence — aiming to beat black-box CoT monitoring and probes on behavior prediction.

## How It Works

The oracle is the **same model** (Qwen3-8B) fine-tuned with LoRA to accept its own activations. During inference, activations are extracted from the source model's CoT generation at stride intervals across 3 layers (25%, 50%, 75% depth), then injected back into the model via norm-matched addition at layer 1. The model reads these activations and answers questions about the reasoning process.

**10 training tasks** teach the oracle to understand CoT structure:

| Task | Description |
|------|-------------|
| Full reconstruction | Reconstruct full CoT from stride activations |
| Next step prediction | Predict next ~50 tokens of CoT |
| Answer prediction | Predict final answer from partial CoT |
| Partial answer (vLLM) | What would model answer if stopped at X%? |
| Load-bearing | Is this CoT load-bearing or decorative? |
| Correctness | Did the model get the right answer? |
| Decorative | Is the CoT decorative (not needed for correct answer)? |
| Reasoning termination | Will model emit `</think>` within 100 tokens? |
| Domain | What domain is this problem? |
| Conversational QA | Answer questions about the CoT |

**20 evals** test whether the oracle can detect unfaithfulness, classify reasoning properties, and generalize beyond training distribution. Includes 2 evals from ["When 'Just Read the Chain of Thought' Fails"](https://arxiv.org/abs/...) (ICLR 2026): reasoning termination and forced answer entropy.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/adamkarvonen/activation_oracles
export AO_REPO_PATH="$PWD/activation_oracles"
```

## Quick Start

### 1. Generate eval datasets
```bash
python src/evals/generate_datasets.py --output-dir data/evals
```

### 2. Precompute training data (GPU required)
```bash
python scripts/precompute_training_data.py \
  --corpus data/cot_corpus_v5/corpus_medium.jsonl \
  --output-dir data/precomputed \
  --model Qwen/Qwen3-8B
```

### 3. Train
```bash
# Default: all tasks, shuffled
python src/train_v5.py --config configs/train_v6.yaml --precomputed-dir data/precomputed

# Sequential mode (per-task diagnostics, saves phase checkpoints)
python src/train_v5.py --config configs/train_v6.yaml \
  --precomputed-dir data/precomputed --task-order sequential

# Disable specific tasks
python src/train_v5.py --config configs/train_v6.yaml --domain-n 0 --conv-qa-n 0
```

### 4. Evaluate
```bash
# Standalone eval suite
python src/evals/run_evals.py --eval-dir data/evals --output-dir data/eval_results
python src/evals/score_oracle.py --results-dir data/eval_results

# Upload eval datasets to HuggingFace
python scripts/upload_eval_datasets.py
```

## Configuration

All settings in `configs/train_v6.yaml`:

```yaml
tasks:
  full_recon:
    n: 40000          # Set n: 0 to disable a task
  next_step:
    n: 30000
  # ...

training:
  lr: 1e-5
  batch_size: 8
  task_order: shuffled  # or "sequential"

activations:
  stride: 5
  position_encoding: false  # Sinusoidal PE for activation vectors
  pe_alpha: 0.1

eval:
  eval_steps: 500
  save_steps: 2000
```

CLI flags override config values.

## Eval Suite (20 Evals)

**Unfaithfulness detection:** hinted_mcq, sycophancy, sycophancy_scruples, sycophancy_v2, authority_bias, correct_authority, anchoring_bias, hint_influence_yesno, sentence_insertion, scruples_disagreement

**Classification/regression:** decorative_cot, answer_correctness, contradictory_comparison, final_answer_kl, step_counting

**Paper evals (ICLR 2026):** reasoning_termination, forced_answer_entropy

**Model organism:** rot13_reconstruction, held_out_cot_reconstruction, logical_leaps

All eval datasets published at: [`ceselder/cot-oracle-evals`](https://huggingface.co/collections/ceselder/cot-oracle-evals-699a2d31f652864af01d40dd)

## Repository Structure

```
src/
  train_v5.py              # Primary training entrypoint
  position_encoding.py     # Optional sinusoidal PE for activations
  cot_utils.py             # Shared utilities
  core/                    # AO runtime wrappers
  dataset_classes/         # Training data loaders (10 tasks)
  evals/                   # Eval generation, running, scoring (20 evals)
  data_pipeline/           # CoT generation + corpus tooling
configs/
  train_v6.yaml            # Training configuration
scripts/
  precompute_*.py          # GPU precompute scripts
  upload_*.py              # HuggingFace upload scripts
data/
  cot_corpus_v5/           # Main CoT corpus (47K entries)
  concept_corpus/          # Safety/bias concept corpus (8K entries)
  evals/                   # Generated eval JSONs
  precomputed/             # Precomputed training JSONL
```

## Model

- **Source & Oracle:** `Qwen/Qwen3-8B` (36 layers)
- **AO checkpoint:** `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
- **LoRA:** r=64, alpha=128, dropout=0.05, all-linear
- **Activations:** Layers 9, 18, 27 with LoRA disabled; injected at layer 1

## Related Work

- [Activation Oracles](https://github.com/adamkarvonen/activation_oracles) (Karvonen et al., 2024)
- [Thought Anchors](https://github.com/interp-reasoning/thought-anchors) (Bogdan et al., 2025)
- [Thought Branches](https://arxiv.org/abs/2510.27484) (Macar, Bogdan et al., 2025)
