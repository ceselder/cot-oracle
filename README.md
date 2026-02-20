# CoT Oracle

White-box chain-of-thought (CoT) monitoring built on Activation Oracles (AO).

## Planning Source
This repo is now aligned to the shared brainstorming/living plan document from this project thread.
- Brainstorm doc link: `<ADD_SHARED_DOC_LINK_HERE>`
- In-repo execution plan: `plan.md`

## North Star
CoT text is often unfaithful. The goal is to read reasoning from activations and beat black-box CoT monitoring (and probes) on behavior prediction.

## Current Scope
We are prioritizing:
1. A generalist CoT activation monitor trained on diverse tasks (not just more tokens).
2. Better eval coverage for unfaithfulness-like behavior (hints, sycophancy, load-bearingness).
3. Fast architecture ablations that are likely to matter (multi-layer feed-in, temporal striding).

## What Is Implemented

### Training
- Main entrypoint: `src/train_mixed.py`
- Mixed training tasks currently wired:
  - `cot_context_prediction`
  - `cot_sentence_prediction`
  - `cot_decorative`
  - `cot_domain`
  - `cot_correctness`
  - `cot_persona` (optional)
  - `cot_summary` (optional)

### Evals
- Dataset generation and eval runs in `src/evals/`
- Current eval families:
  - hinted MCQ
  - sycophancy
  - authority bias
  - decorative CoT
  - answer correctness
  - contradictory comparison
  - sentence insertion
  - Scruples-style sycophancy

### Runtime / Infra
- AO repo path resolution in `src/core/ao_repo.py`
- Shared AO/attention utilities in `src/core/ao.py`
- Legacy unused tracks removed (`signs_of_life`, SAE-specific unused path, stale scripts)

## RTX 5090 (Blackwell) Best-Guess Setup
This is the recommended compatibility path (documentation-only; designed to be robust on Blackwell):

1. Prefer SDPA over FlashAttention2 for AO/training/eval loading.
2. For local vLLM generation, force eager mode on Blackwell.
3. Only re-enable FA2 after confirming exact torch/cuda/fa2 compatibility.

Recommended env:

```bash
export COT_ORACLE_FORCE_SDPA=1
export COT_ORACLE_VLLM_ENFORCE_EAGER=1
# optional safety: keep auto-eager enabled (default)
unset COT_ORACLE_VLLM_NO_EAGER_AUTO
```

If you explicitly want to try FlashAttention2 anyway:

```bash
export COT_ORACLE_ALLOW_FLASH2=1
```

Equivalent vLLM CLI flags:

```bash
python3 src/data_pipeline/generate_cots.py --engine vllm --vllm-enforce-eager
```

## Setup

### 1) Install deps
```bash
pip install -r requirements.txt
```

### 2) Make AO importable
```bash
git clone https://github.com/adamkarvonen/activation_oracles
export AO_REPO_PATH="$PWD/activation_oracles"
```

### 3) Optional keys
```bash
export OPENROUTER_API_KEY=...
export WANDB_API_KEY=...
```

## Quickstart

### Generate CoT corpus
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

### Run eval suite
```bash
python3 src/evals/generate_datasets.py --output-dir data/evals
python3 src/evals/run_evals.py --eval-dir data/evals --output-dir data/eval_results
python3 src/evals/score_oracle.py --results-dir data/eval_results
```

## Repository Structure

```text
src/
  core/                  # AO runtime wrappers + compatibility logic
  data_pipeline/         # CoT generation + corpus tooling
  dataset_classes/       # training dataset builders
  evals/                 # eval generation / run / scoring
  train_mixed.py         # primary training entrypoint
  chat_compare.py        # AO vs fine-tuned oracle comparison
scripts/
  generate_qwen_rollouts.sh
  upload_corpus.py
plan.md                  # living project plan aligned to brainstorming doc
```

## Immediate TODO (from shared plan)
- Finalize held-out CoT reconstruction metric + baseline.
- Add ROT13/obfuscated holdout eval into default reporting.
- Add explicit load-bearingness proxy metric into eval scoreboard.
- Run ablations: striding vs boundaries, pooling vs discrete, token marker scheme.
- Decide minimum conversational IT set for monitor Q/A behavior + calibrated refusal.
