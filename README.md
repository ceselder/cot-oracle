# CoT Oracle

White-box chain-of-thought (CoT) monitoring built on Activation Oracles.

## North Star
Chain-of-thought text is often unfaithful. The goal is to read reasoning from activations and outperform black-box CoT monitoring for behavior prediction.

Project plan and priorities: `plan.md`.

## What Is Implemented Now

### Core method
- Start from Adam Karvonen's AO checkpoint
- Inject activation vectors into placeholder tokens (norm-matched addition at layer 1)
- Read CoT trajectories by extracting activations at sentence boundaries
- Support single-layer (AO-compatible) and multi-layer (25/50/75%) formats

### Training pipeline (`src/train_mixed.py`)
Current mixed run supports:
- `cot_context_prediction` (random positions)
- `cot_sentence_prediction` (sentence-boundary reconstruction)
- `cot_decorative` (load-bearing vs decorative)
- `cot_domain`
- `cot_correctness`
- `cot_persona` (if persona corpus is provided)
- `cot_summary` (if summaries file exists)

### Eval suite (`src/evals/`)
- hinted MCQ
- sycophancy
- authority bias
- decorative CoT
- answer correctness
- contradictory comparison
- sentence insertion
- sycophancy on Scruples

## What Was Removed
To keep the repo focused on the current sprint scope:
- old `signs_of_life` experiments
- SAE/unverbalized label extraction path
- legacy single-purpose training/analysis scripts not used by the mixed pipeline

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Make AO (`nl_probes`) importable
Clone Activation Oracles and set one of:
- `AO_REPO_PATH=/path/to/activation_oracles`
- clone it to `./activation_oracles`

```bash
git clone https://github.com/adamkarvonen/activation_oracles
export AO_REPO_PATH="$PWD/activation_oracles"
```

### 3) Optional API keys
```bash
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

## Quickstart

### Generate corpus
```bash
python3 src/data_pipeline/generate_cots.py \
  --openrouter \
  --n-problems 1000 \
  --output data/cot_corpus_v5/corpus.jsonl
```

### Train mixed oracle
```bash
torchrun --nproc_per_node=1 src/train_mixed.py \
  --corpus data/cot_corpus_v5/corpus.jsonl \
  --model Qwen/Qwen3-8B \
  --save-dir checkpoints/cot_oracle_mixed
```

### Generate eval datasets and run evals
```bash
python3 src/evals/generate_datasets.py --output-dir data/evals
python3 src/evals/run_evals.py --eval-dir data/evals --output-dir data/eval_results
python3 src/evals/score_oracle.py --results-dir data/eval_results
```

## RTX 5090 / Blackwell Notes
This repo now defaults to Blackwell-safe behavior:
- AO/chat/eval model loading uses `sdpa` on Blackwell by default
- local vLLM rollout generation auto-enables eager mode on Blackwell GPUs (`enforce_eager=True`)

Runtime knobs:
- Force SDPA everywhere:
```bash
export COT_ORACLE_FORCE_SDPA=1
```
- If you explicitly want to try FlashAttention2 anyway:
```bash
export COT_ORACLE_ALLOW_FLASH2=1
```
- Force eager mode for vLLM local rollout generation:
```bash
export COT_ORACLE_VLLM_ENFORCE_EAGER=1
```
- Disable Blackwell auto-eager in vLLM (not recommended on 5090):
```bash
export COT_ORACLE_VLLM_NO_EAGER_AUTO=1
```

CLI equivalents for local rollout generation:
```bash
python3 src/data_pipeline/generate_cots.py --engine vllm --vllm-enforce-eager
python3 src/data_pipeline/generate_cots.py --engine vllm --no-vllm-eager-auto
```

If runs fail on 5090, keep `COT_ORACLE_FORCE_SDPA=1` and avoid FA2-specific builds.

## Repository Layout

```text
src/
  core/                 # AO runtime wrappers + AO repo path resolver
  data_pipeline/        # corpus generation + label extraction
  dataset_classes/      # training dataset builders
  evals/                # eval generation, execution, scoring
  chat_compare.py       # interactive AO vs trained-oracle comparison
  train_mixed.py        # main training entrypoint
```

## Scripts
- `scripts/generate_qwen_rollouts.sh`: helper for generating and analyzing rollouts in the `thought-anchors` repo.
- `scripts/upload_corpus.py`: upload generated corpus files to HuggingFace.
