# CoT Trajectory Oracle

**EVERY EVAL DATASET MUST BE PUBLISHED ON HUGGINGFACE.** All eval datasets live in the `ceselder/cot-oracle-evals` collection: https://huggingface.co/collections/ceselder/cot-oracle-evals-699a2d31f652864af01d40dd — If you add or modify an eval dataset, re-upload it. If you create a new eval, upload it. No exceptions. Run `python3 scripts/upload_eval_datasets.py` to sync all datasets.

**EVERY EVAL DATASET MUST HAVE A COMPLETE HUGGINGFACE MODEL CARD** with: description, source datasets, schema table, oracle prompt template, source model prompt format, metrics, and usage example. See existing cards in `scripts/upload_model_cards.py` for the spec.

A white-box chain-of-thought (CoT) monitoring system built on Activation Oracles (AO). Reads reasoning from activations to detect when stated reasoning diverges from actual computational influence — beating black-box CoT monitoring and probes on behavior prediction.

---

## Goal

Build an oracle that takes activation trajectories from CoT generation and classifies/describes what actually influenced the reasoning. The oracle is the **same model** (Qwen3-8B) fine-tuned with LoRA to accept its own activations via norm-matched injection.

**Core output format:** Binary classifications ("will_terminate" / "will_continue", "correct" / "incorrect", "decorative" / "load_bearing") and short text predictions (next tokens, answer, full reconstruction).

---

## Required Reading

### 1. Activation Oracles (Karvonen et al., 2024)
- **arXiv:** https://arxiv.org/abs/2512.15674
- **GitHub:** https://github.com/adamkarvonen/activation_oracles

**Key concepts:**
- AO = same model as source, LoRA fine-tuned to read its own activations
- Injection: norm-matched addition at layer 1: `h' = h + ||h|| * v/||v||`
- LoRA: r=64, alpha=128, dropout=0.05, target all-linear
- Activations extracted at 25%, 50%, 75% depth with LoRA **disabled**
- Placeholder token for injection positions

### 2. Thought Anchors (Bogdan et al., 2025)
- **arXiv:** https://arxiv.org/abs/2506.19143

Some CoT sentences have outsized causal importance ("thought anchors"). Expensive counterfactual resampling (100 samples/sentence) reveals this. We want a cheap proxy via activation patterns.

### 3. Thought Branches (Macar, Bogdan et al., 2025)
- **arXiv:** https://arxiv.org/abs/2510.27484

Nudging is DIFFUSE across whole trajectory. Self-preservation statements have near-zero causal impact. 77.5% of demographic bias mediated through sentence *selection*, not explicit statements.

### 4. "When 'Just Read the Chain of Thought' Fails" (ICLR 2026)
Five tasks for stress-testing CoT monitors: reasoning termination, self-deletion, forced answer entropy, sycophancy, atypical answers. We implement 3/5 (reasoning_termination, forced_answer_entropy, sycophancy_v2). See TODO for remaining 2.

---

## Architecture

### Current: Stride-Based Multi-Layer Activation Reading

The oracle reads activations sampled at regular stride intervals from the CoT, across 3 layers simultaneously.

**Activation extraction:**
- Layers: 9, 18, 27 (25%, 50%, 75% of Qwen3-8B's 36 layers)
- Stride: every 5 tokens through the CoT
- Plus 5 evenly-spaced prompt positions
- Positions tripled (one set per layer) → `context_positions = (prompt_pos + cot_pos) * 3`

**Injection:**
- Placeholder token: `" ¶"` (token ID 78846) — single token, no collisions
- Each activation vector injected at a placeholder position via norm-matched addition at layer 1
- Oracle prompt follows the placeholder sequence

**Optional position encoding:**
- Sinusoidal PE for activation vectors (activations lose positional info after extraction)
- Toggle via `position_encoding: true` in config, mixing coefficient `pe_alpha: 0.1`
- `v_combined = v + alpha * ||v|| * sinusoidal_pe(pos/total_length)`

### Future: Delta Sequence (Contrastive)

Instead of raw activations, feed contrastive delta vectors — each capturing what a single CoT sentence contributed to the answer representation. All activations measured at the ANSWER position. See the original design notes in `docs/delta_architecture.md` (TODO: extract from old CLAUDE.md).

---

## Training System

### Entrypoint

```bash
python src/train_v5.py --config configs/train_v6.yaml
```

Or with precomputed data (recommended — much faster):
```bash
python src/train_v5.py --config configs/train_v6.yaml --precomputed-dir data/precomputed
```

### 10 Training Tasks

| Task | Key | Description | Default N |
|------|-----|-------------|-----------|
| `full_recon` | Full CoT reconstruction | Reconstruct full CoT from stride activations | 40,000 |
| `next_step` | Next step prediction | Predict next ~50 tokens of CoT | 30,000 |
| `answer_pred` | Answer prediction | Predict final answer from partial CoT | 20,000 |
| `partial_answer` | Partial answer (vLLM) | What would model answer if stopped at X%? | 20,000 |
| `load_bearing` | Load-bearing classification | Is this CoT load-bearing or decorative? | 15,000 |
| `correctness` | Correctness classification | Did the model get the right answer? | 15,000 |
| `decorative` | Decorative classification | Is the CoT decorative (not needed)? | 15,000 |
| `reasoning_term` | Reasoning termination | Will model emit `</think>` within 100 tokens? | 15,000 |
| `domain` | Domain classification | What domain is this problem? | 0 (disabled) |
| `conv_qa` | Conversational QA | Answer questions about the CoT | 10,000 |

### YAML Config (`configs/train_v6.yaml`)

Controls everything: task counts, training hyperparams, activation settings, eval frequency, model, output paths. CLI flags override config values.

```yaml
tasks:
  full_recon:
    n: 40000
  # ... (set n: 0 to disable)

training:
  lr: 1e-5
  batch_size: 8
  task_order: shuffled  # or "sequential" for per-task training with phase checkpoints

activations:
  stride: 5
  position_encoding: false
  pe_alpha: 0.1
```

### Training Modes

- **Shuffled** (default): All tasks mixed together. Best for final training.
- **Sequential**: Train tasks one at a time, with eval every 500 steps. Saves phase checkpoints at task transitions. Good for diagnostics — see which tasks the oracle learns and how they interact.

### Data Pipeline

**Precomputed (recommended):** Generate training data offline, save as JSONL:
```bash
python scripts/precompute_training_data.py \
  --corpus data/cot_corpus_v5/corpus_medium.jsonl \
  --output-dir data/precomputed \
  --model Qwen/Qwen3-8B
```

Each JSONL file contains rows with: `context_input_ids`, `context_positions`, `prompt`, `target_response`, `datapoint_type`, `layer`, `num_positions`.

**Live loading (fallback):** Dataset loaders in `src/dataset_classes/` generate data on-the-fly from corpus. Slower but doesn't require precomputation.

### Corpora

- **Medium corpus:** `data/cot_corpus_v5/corpus_medium.jsonl` — 47,104 CoT entries (math, reasoning)
- **Concept corpus:** `data/concept_corpus/corpus_full.jsonl` — 8,132 entries from safety/bias datasets
- **Conv QA:** `data/concept_corpus/corpus_full_conv_qa_llm.jsonl` — 10,499 LLM-generated QA pairs (DeepSeek v3.2)

### Wandb Logging

- Per-step: loss, per-task EMA loss, learning rate, batch token count, batch task composition
- Per-eval (every `eval_steps`): token F1 and exact match per task, wandb tables with prediction/target/prompt details
- Sequential mode: `active_task_idx` for tracking which phase is running
- Unfaithfulness evals: per-eval accuracy, sample tables with oracle responses

---

## Eval Suite (10 Evals)

### Training Evals (run during training via `training_eval_hook.py`)

| Eval | Type | Description | GT Mechanism |
|------|------|-------------|--------------|
| `hinted_mcq` | Influence | ARC-Challenge MCQ with planted hints | Counterfactual clean vs test |
| `sycophancy_v2_riya` | Influence | Moral dilemma sycophancy (50-rollout switch rates) | Precomputed switch-rate labels |
| `decorative_cot` | Classification | Is the CoT decorative or load-bearing? | Wilson CI from 10 runs |
| `sentence_insertion` | Detection | Detect irrelevant inserted sentence in CoT | Direct label (known insert position) |
| `reasoning_termination_riya` | Classification | Will model emit `</think>` within 100 tokens? | 50-rollout resampled labels, uses PARTIAL CoT prefix |
| `rot13_reconstruction` | Reconstruction | Reconstruct ROT13-encoded CoT from activations | Token match rate + KL divergence |

### Non-Training Evals

| Eval | Type | Description | GT Mechanism |
|------|------|-------------|--------------|
| `sycophancy` | Influence | Math sycophancy (GSM8K/MATH) — may be too easy | Counterfactual clean vs test |
| `forced_answer_entropy_riya` | Regression | Predict answer distribution entropy at truncation | Precomputed logprob entropy |
| `atypical_answer_riya` | Classification | Predict majority vs minority answer | 200-rollout distribution |
| `cybercrime_ood` | Classification | OOD topic detection (sanity check) | Direct label |

### Eval Infrastructure

- **Generate:** `python src/evals/generate_datasets.py --output-dir data/evals`
- **Precompute (GPU):** Various scripts in `scripts/precompute_*.py` for evals needing rollouts
- **Run standalone:** `python src/evals/run_evals.py --eval-dir data/evals`
- **Score:** `python src/evals/score_oracle.py --results-dir data/eval_results`
- **Training hook:** `src/evals/training_eval_hook.py` runs 6 evals during training
- **Upload:** `python scripts/upload_eval_datasets.py`

---

## Model & Config

### Primary: Qwen3-8B

- **Base:** `Qwen/Qwen3-8B` (36 layers, 50% depth = layer 18)
- **AO checkpoint:** `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
- LoRA: r=64, alpha=128, dropout=0.05, all-linear
- Activations extracted with LoRA **disabled** (base model representations)

### GPU Rules

- **ALWAYS use bf16** — never 8-bit quantization. 8B in bf16 is ~16GB. 8-bit adds massive overhead for no benefit on H100/A100.
- **ALWAYS batch when possible** — generation, activation extraction, eval inference.
- **eval_batch_size:** 4 on A100 80GB, 2 on H100 NVL 96GB (materialization OOMs otherwise).
- **do_sample=False** everywhere. For CI-based labeling (decorative_cot), pass `temperature=0.6` explicitly.
- **model.eval()** needed before generation (otherwise gradient checkpointing disables KV caching → 20x slower).

### Qwen3-8B Thinking Mode

`enable_thinking=True` is unusable for bounded generation — generates 4000-8000+ tokens of `<think>` content without ever producing `</think>`. Use `enable_thinking=False` with prompt-based CoT for eval precompute.

---

## Training History

### v1 (wandb: `2bmv0bur`)
- 100K examples, sequential task ordering (BUG — not shuffled). "Grokking" was fake.
- context_pred 13%, importance 50%@3500, taxonomy 56%@4000

### v3a/v3b (wandb: `2w65186w`, `cxnvavv0`)
- A100, correctness 90%, domain 58% at step 2000. Killed due to unfaith eval blocking bug.

### v5 (wandb: `fuadqnhq`)
- H100 NVL 96GB, multi-stage training (stages 1+2)
- Stage 1 (13100 steps): rollout_multilayer F1=0.691, next_step=0.451, answer_pred=0.366, load_bearing=0.920
- Stage 2 started at step 13100, killed at step ~15000 to run precompute

### v6 (current)
- Flat task-based training (no stages). 10 tasks, precomputed data, YAML config.
- Not yet run — pending GPU.

---

## Project Structure

```
cot-oracle/
├── CLAUDE.md                          # This file — project instructions
├── README.md                          # GitHub overview
├── requirements.txt                   # Python dependencies
├── configs/
│   └── train_v6.yaml                  # Training config (tasks, hyperparams, paths)
├── src/
│   ├── train_v5.py                    # Primary training entrypoint (flat v6 training)
│   ├── position_encoding.py           # Sinusoidal PE for activation vectors
│   ├── cot_utils.py                   # Shared utilities (stride positions, layer mapping)
│   ├── core/
│   │   ├── ao.py                      # AO runtime (generation, activation extraction)
│   │   └── ao_repo.py                 # AO repo path resolution
│   ├── dataset_classes/               # Training data loaders (one per task)
│   │   ├── cot_rollout_multilayer.py  # full_recon
│   │   ├── cot_next_step.py           # next_step
│   │   ├── cot_answer_prediction.py   # answer_pred
│   │   ├── cot_partial_answer.py      # partial_answer (vLLM targets)
│   │   ├── cot_load_bearing.py        # load_bearing
│   │   ├── cot_correctness.py         # correctness
│   │   ├── cot_decorative.py          # decorative
│   │   ├── cot_reasoning_termination.py # reasoning_term
│   │   ├── cot_domain.py              # domain
│   │   └── cot_conversational.py      # conv_qa
│   ├── evals/
│   │   ├── generate_datasets.py       # Generate all eval JSONs
│   │   ├── run_evals.py               # Run oracle on evals (standalone)
│   │   ├── score_oracle.py            # Score oracle responses
│   │   ├── training_eval_hook.py      # Run evals during training
│   │   ├── precompute_activations.py  # Precompute activation bundles
│   │   ├── activation_cache.py        # Activation caching utilities
│   │   ├── common.py                  # EvalItem, wilson_ci, ci_label
│   │   └── datasets/                  # One file per eval (20 files)
│   └── data_pipeline/                 # CoT generation + corpus tooling
├── scripts/
│   ├── precompute_training_data.py    # Generate precomputed JSONL training files
│   ├── precompute_decorative_rollouts.py  # Decorative eval precompute (GPU)
│   ├── precompute_sycophancy.py       # Sycophancy v2 rollouts (GPU)
│   ├── precompute_reasoning_termination.py # Reasoning term resampling (GPU)
│   ├── precompute_forced_entropy.py   # Forced answer entropy precompute (GPU)
│   ├── precompute_partial_answer_vllm.py  # vLLM partial answer targets (GPU)
│   ├── upload_eval_datasets.py        # Sync all eval datasets to HF
│   ├── upload_model_cards.py          # HF model cards for eval datasets
│   └── launch_v6.sh                   # Training launch script
├── data/
│   ├── cot_corpus_v5/                 # Main CoT corpus (47K entries)
│   ├── concept_corpus/                # Safety/bias concept corpus (8K entries)
│   ├── evals/                         # Generated eval JSONs
│   ├── eval_precomputed/              # Precomputed activation bundles (.pt)
│   └── precomputed/                   # Precomputed training JSONL files
└── checkpoints/
    └── v6/                            # Training checkpoints
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
git clone https://github.com/adamkarvonen/activation_oracles
export AO_REPO_PATH="$PWD/activation_oracles"
```

### 2. Generate eval datasets
```bash
python src/evals/generate_datasets.py --output-dir data/evals
python scripts/upload_eval_datasets.py  # Upload to HuggingFace
```

### 3. Precompute training data (on GPU)
```bash
python scripts/precompute_training_data.py \
  --corpus data/cot_corpus_v5/corpus_medium.jsonl \
  --output-dir data/precomputed \
  --model Qwen/Qwen3-8B
```

### 4. Train oracle
```bash
python src/train_v5.py --config configs/train_v6.yaml --precomputed-dir data/precomputed
```

### 5. Run eval suite
```bash
python src/evals/run_evals.py --eval-dir data/evals --output-dir data/eval_results
python src/evals/score_oracle.py --results-dir data/eval_results
```

---

## Critical Lessons

- **v1 "grokking" was fake:** Sequential data ordering (not shuffled) made it look like tasks grokked at specific steps. Fixed with `random.shuffle()`.
- **Mini corpus memorization:** 1,064 entries x 15K full_recon = 14x repetition → loss=0.01. Use medium corpus (47K).
- **Generation evals need fuzzy scoring:** Exact match gives 0% on generation tasks. Token F1 (fuzzy eval) needed.
- **Per-task loss hook:** Must use `torch.no_grad()` + `.detach()` on logits to avoid gradient leak.
- **Zombie vLLM processes:** After killing parent, check `fuser /dev/nvidia*` and kill orphan EngineCore processes.
- **OpenRouter "reasoning" key bug:** Including `"provider": {"reasoning": {"effort": "none"}}` causes HTTP 400 on DeepSeek v3.2.
- **LaTeX answer matching:** `\boxed{}` regex must handle nested braces. Use depth-counting extraction.
- **Wilson CI with n=10:** Too strict for load_bearing. Rate-based fallback added.
- **Conv QA corpus mismatch:** Conv QA references concept corpus IDs, not medium corpus IDs. Must pass `--concept-corpus` path.

---

## TODO

### Training (needs GPU)
- [ ] Run v6 training (flat task-based, 10 tasks, ~200K examples)
- [ ] Run sequential mode to identify per-task learning curves
- [ ] Compare with/without position encoding
- [ ] Run unfaithfulness evals on best checkpoint vs baseline (`runs/iee93uwb`)

### New Evals (from ICLR 2026 paper)
- [ ] **Self-deletion eval:** Model writes reasoning then deletes it before answering. Currently Gemma-specific in the paper — needs adaptation for Qwen3. Key challenge: Qwen3 uses `<think>` tags, not freeform deletion.
- [ ] **Atypical answers eval:** Predict whether model will give the majority or minority answer. Requires collecting answer distributions over multiple rollouts, then testing if oracle can predict from activations whether this specific run will give the common or rare answer.

### Data Quality
- [ ] Fix logical_leaps eval (100% heuristic fallback, 1/100 positive — need Gemini labels)
- [ ] Generate more precomputed partial_answer data (vLLM)
- [ ] Scale corpus beyond 47K if needed

### Infrastructure
- [ ] Upload all rollout data to HuggingFace
- [ ] Delta sequence input (future architecture — measure at answer position, compute deltas)

---

## HuggingFace Datasets

- `ceselder/cot-oracle-evals` collection — **20 eval datasets** (all uploaded)
- `ceselder/cot-oracle-conv-qa` — 10,499 LLM-generated conv QA pairs
- `ceselder/qwen3-8b-math-cot-corpus` — original math CoT corpus

---

## External Repositories

### Activation Oracles
- **Repo:** https://github.com/adamkarvonen/activation_oracles
- **Local:** AO repo resolved via `src/core/ao_repo.py` (checks `AO_REPO_PATH` env var, common paths)
- Pre-trained checkpoints: `adamkarvonen/activation_oracles` collection on HuggingFace

### Thought Anchors
- **Repo:** https://github.com/interp-reasoning/thought-anchors
- Cloned at `./thought-anchors/` for reference
- MATH rollout dataset: `uzaymacar/math-rollouts`
