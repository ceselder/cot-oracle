# CoT Oracle Plan (Living)

Last updated: 2026-02-21

## North Star
Chain-of-thought text is often unfaithful. We want a white-box activation monitor that:
1. predicts model behavior better than black-box CoT monitoring,
2. beats simple probe baselines where possible,
3. produces non-obvious, testable hypotheses about internal reasoning.

## Strategy
1. Scale task diversity more than raw dataset size (Adam found more examples doesn't help much).
2. Train on both semantic reconstruction and computational properties.
3. Single-layer injection at 50% depth (multi-layer showed no improvement).
4. Placeholder token: `¶` (token ID 78846) — `?` collides with questions in training data.

## Architecture Decisions (Locked)
- **Model:** Qwen3-8B, starting from fresh LoRA (v3) or Adam's AO checkpoint (v2)
- **Injection:** Norm-matched addition at layer 1, single layer activations from 50% depth (layer 18)
- **Placeholder token:** `" ¶"` — single token in Qwen3-8B, zero occurrences in corpus/QA data
- **Striding:** Sentence boundaries (not fixed-token stride)
- **Multi-layer:** DROPPED — per-layer tokens (@?#) showed no improvement over single ¶
- **LoRA:** r=64, alpha=128, dropout=0.05, all-linear (standard AO config)

---

## v2 Training (COMPLETED)

### Results
Running on vast.ai (A100 80GB), wandb: `celestedeschamphelaere-personal/cot_oracle/runs/5o2jse3j`

Stable results at step 22500/26150:
- answer_prediction: 99% | decorative: 100% | domain: 100% | correctness: 100%
- context_prediction: 18% | causal_prediction: 6% | conversational: 0%
- full_reconstruction: 2% (memorized — loss ~0.01, mini corpus too small)

### Key Lessons from v2
1. **Memorization:** Mini corpus (1,064 entries) with 15K full_reconstruction = ~14x repetition. Training loss hit 0.01 (memorized). Medium corpus (47K entries) will fix this.
2. **Data wasn't shuffled in v1:** Sequential task ordering caused fake "grokking" — not a real phase transition, just model first encountering each task type.
3. **Generation evals broken:** Exact match + 20 token limit → 0% on all generation tasks. Fuzzy eval hook written and will be used in v3.
4. **Classification evals work great:** 98-100% on decorative/domain/correctness/answer_prediction.

---

## v3 Training (LAUNCHING)

### Changes from v2
| Change | Why |
|--------|-----|
| Fresh LoRA (no Adam's AO checkpoint) | Test if model can learn activation reading + tasks simultaneously |
| Medium corpus (47K entries) | Fix memorization from 1K mini corpus |
| FineWeb context prediction (100K) | Add diverse non-CoT context prediction |
| Fuzzy eval scoring | Fix broken generation task eval (token F1, substring match) |
| Unfaithfulness evals ON | 14 eval datasets evaluated during training |
| Conversational QA (586K pairs) | Richer QA from medium corpus |
| Per-task loss logging | Track learning per task type in wandb |

### Task Mix
| # | Task | Size | Description |
|---|------|------|-------------|
| 1 | context_prediction | 100K | AO-style past/future token prediction (backbone) |
| 2 | fineweb_context_prediction | 100K | FineWeb + LMSYS streaming context prediction |
| 3 | answer_prediction | 20K | Predict final answer from partial CoT |
| 4 | full_reconstruction | 15K | Reconstruct CoT text from all activations |
| 5 | causal_prediction | 30K | Feed acts 1..i, predict next sentence |
| 6 | conversational | 5K | Open-ended QA (107 types, 586K available pairs) |
| 7 | decorative | 10K | Binary: load-bearing vs decorative CoT |
| 8 | domain | 15K | Multi-class domain classification |
| 9 | correctness | 15K | Binary: is final answer correct? |
| 10 | persona | 0 | SKIPPED (no persona corpus for medium) |

**Total: ~310K examples** with FineWeb

### Corpus
- **Primary:** `data/cot_corpus_v5/corpus_medium.jsonl` (47,104 entries, diverse sources)
- **Conversational QA:** `data/cot_qa_medium.jsonl` (586,632 pairs, text-derived facts)
- **FineWeb:** Streaming from HuggingFace (infinite, no local storage)

### Evals (14 datasets, 1350 items)
| Eval | Items | Type |
|------|-------|------|
| hinted_mcq | 100 | Nudge |
| hint_influence_yesno | via hinted_mcq | Nudge |
| sycophancy | 100 | Nudge |
| sycophancy_scruples | 100 | Nudge (moral) |
| authority_bias | 100 | Nudge |
| correct_authority | 100 | Adversarial (authority says RIGHT answer) |
| anchoring_bias | 100 | Subtle numerical anchoring |
| sentence_insertion | via logical_leaps | Nudge |
| decorative_cot | 100 | Structure |
| logical_leaps | 100 | Structure |
| contradictory_comparison | 50 | Structure |
| step_counting | 100 | Novel structural task |
| answer_correctness | 100 | Answer |
| scruples_disagreement | 100 | Moral judgment |
| held_out_cot_reconstruction | 100 | Reconstruction |
| rot13_reconstruction | 100 | Reconstruction |
| final_answer_kl | via answer_correctness | Answer KL |

### Bug Fixes Applied (code review, Feb 21)
1. **CRITICAL: Unfaith eval hook was silently dropped** — fuzzy eval hook replaced `eval_all_datasets` without chaining. Fixed by swapping installation order (fuzzy first, then unfaith wraps it).
2. **Per-task loss hook gradient leak** — added `torch.no_grad()` + `.detach()` to avoid building extra computation graph for logging-only computations.
3. **Removed dead code** — unused `_orig_get_prefix`, `find_pattern_in_tokens` import.
4. **CRITICAL: generate_cot() missing model.eval()** — generation during unfaith eval ran with gradient checkpointing enabled (model still in training mode), disabling KV caching and making generation 20x slower. Added `model.eval()`/`model.train()` bracketing to `generate_cot`, `batch_generate_cot`, and `run_oracle_on_activations` in `core/ao.py`.

---

## Thought Anchors Research (Feb 21)

### Status
The infrastructure for thought anchor integration is **already built**:
- `src/data_pipeline/resample_importance_vllm.py` — vLLM-based counterfactual resampling for v5 corpus
- `src/dataset_classes/cot_importance.py` — training dataset class consuming importance labels

### Key Findings
1. **Don't use thought-anchors repo directly** — designed for DeepSeek R1, not compatible with v5 corpus format. Use our own `resample_importance_vllm.py`.
2. **Full corpus resampling is feasible** — 1000 problems x 5 resamples = ~2 GPU-hours on H100 with prefix caching.
3. **Existing importance data is too old** — 200 entries from v2 corpus, only 87 overlap with v5 by question. Need fresh resampling.
4. **Current importance labels degenerate** — 99.7% labeled "important" (KL > 0.1 threshold too low). Resampling gives real counterfactual signal.

### Cheap Proxy: Receiver Head Attention
The paper found certain attention heads consistently attend to causally important sentences. Single forward pass per CoT (no resampling). Could give importance scores for entire 47K corpus for free. Needs one-time analysis to identify Qwen3-8B's receiver heads (~1 GPU-hour).

### Action Items for Thought Anchors
- [ ] Run `resample_importance_vllm.py` on 1000 v5 entries (2 GPU-hours)
- [ ] Update `cot_importance.py` to use 3-class labels: critical/supporting/filler (instead of binary)
- [ ] Add importance task to v4 training mix
- [ ] Explore receiver head proxy for full corpus coverage

---

## Implementation Status

### Done
- [x] AO repo integration, shared helpers in `src/core/ao.py`
- [x] All 10 task dataset loaders (context, FineWeb, answer, recon, causal, conv, decorative, domain, correctness, persona)
- [x] 14+ eval datasets with scoring (1350 items)
- [x] Conversational QA for medium corpus (586K pairs)
- [x] Per-task loss logging to wandb
- [x] Unfaithfulness eval hook during training
- [x] Fuzzy eval scoring hook (token F1 for generation tasks)
- [x] v2 training complete (classification tasks work great)
- [x] Code review and critical bug fixes

### In Progress
- [x] v3 training run 1 (wandb: `2w65186w`, killed at step 2000 — unfaith eval blocked 3h with grad checkpointing)
  - Step 2000 results: correctness **90%**, domain **58%**, decorative 59%, conversational 30%, full_recon 28%
  - Per-task loss: correctness 0.23, decorative 0.19, domain 0.48 (classification converged), context 2.3, fineweb 4.4 (gen improving)
  - Bug found: `generate_cot()` didn't set model.eval(), causing gradient checkpointing during generation → 20x slower unfaith eval
- [ ] v3b training running on vast.ai (wandb: `cot_oracle_v3b`, ~38K steps)
  - Fixes: model.eval() during generation, unfaith evals every 5000 steps, checkpoints every 1000 steps
  - Restarted from scratch (no checkpoint resume support yet)
- [ ] Thought anchors: resampling on v5 corpus (needs GPU time after v3 stable)

### Future
- [ ] Receiver head analysis for Qwen3-8B (cheap importance proxy)
- [ ] 3-class importance labels (critical/supporting/filler)
- [ ] Delta sequence architecture (contrastive answer-position activations)
- [ ] Scale to more diverse reasoning tasks (beyond math)
- [ ] Non-obvious hypothesis generation and behavioral validation

## Success Criteria
- **Minimum:** beat AO baseline on >=1 unfaithfulness eval without harming reconstruction
- **Strong:** consistent uplift across multiple eval families
- **Stretch:** non-obvious hypotheses that are behaviorally validated
