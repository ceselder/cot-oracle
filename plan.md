# CoT Oracle Plan (v4)

## What We're Building

An activation oracle that reads Qwen3-8B's internal representations during chain-of-thought and learns to describe what the model actually computed — detecting when stated reasoning diverges from actual influence (unfaithfulness).

**Key insight:** Feed the oracle activations extracted at **every sentence boundary** of a CoT, using **3 layers per boundary** (25%, 50%, 75% depth). The oracle sees the full reasoning trajectory as a sequence of activation snapshots, and learns to interpret them via diverse training tasks.

---

## Architecture

### Input: Sentence-Boundary Activation Trajectories

For a CoT with N sentences, extract activations at each sentence's last token position from 3 layers:

```
Sentence 1 ends at token 42  → extract acts at layers 9, 18, 27
Sentence 2 ends at token 89  → extract acts at layers 9, 18, 27
Sentence 3 ends at token 156 → extract acts at layers 9, 18, 27
...
```

These get injected as placeholder `?` tokens via norm-matched addition at layer 1:

```
"Layer: 9, 18, 27\n ? ? ? ? ? ? ? ? ? \n Activations from 3 sentence boundaries. [task prompt]"
                    ^^^   ^^^   ^^^
                    S1    S2    S3    (3 activations each: L25%, L50%, L75%)
```

The oracle sees 3N activation tokens total, one group of 3 per sentence boundary. Activations are collected on-the-fly during training (no pre-extraction needed) — AO's `materialize_missing_steering_vectors` is monkey-patched to cycle through layers: position `i` uses `layers[i % 3]`.

### Data Doubling

Each sentence-structured example is created **twice**:
1. Multi-layer version: `"Layer: 9, 18, 27"` prefix, 3N `?` tokens
2. Single-layer L50% duplicate: `"Layer: 18"` prefix, N `?` tokens (standard AO format)

This teaches the oracle to work with both formats and roughly doubles sentence-structured data.

---

## Training Tasks (6 tasks, shuffled together)

| # | Task | Count | Format | Target |
|---|------|-------|--------|--------|
| 1 | **Context prediction (random)** | 100K | 1 random layer, random positions | Predict K surrounding tokens |
| 2 | **Context prediction (sentences)** | 30K (→60K) | 3 layers, all boundaries | Predict K tokens near sentence T |
| 3 | **Decorative detection** | 10K (→20K) | 3 layers, all boundaries | `"load_bearing"` or `"decorative"` |
| 4 | **Domain classification** | 15K (→30K) | 3 layers, all boundaries | `"math"`, `"science"`, `"logic"`, etc. |
| 5 | **Correctness prediction** | 15K (→30K) | 3 layers, all boundaries | `"correct"` or `"incorrect"` |
| 6 | **Persona detection** | 15K (→30K) | 3 layers, all boundaries | `"math_tutor"`, `"skeptical_scientist"`, etc. |

**Total: ~270K examples** (after doubling), all shuffled together.

**Held-out eval only:** Answer tracking (zero-shot) + 6 unfaithfulness evals (hinted_mcq, sycophancy, authority_bias, decorative_cot, answer_correctness, contradictory_comparison).

### Task Details

**Task 1 — Context Prediction (Random):** Standard AO pretraining. Random positions in the CoT, 1 random layer, predict surrounding tokens. High-volume backbone that teaches basic activation reading.

**Task 2 — Context Prediction (Sentences):** Structured version. ALL sentence boundaries injected, oracle predicts tokens near a randomly chosen boundary. Teaches trajectory-level understanding.

**Task 3 — Decorative Detection:** Binary classification. Ground truth is free from corpus: `load_bearing` = CoT correct but direct answer wrong (reasoning was necessary), `decorative` = both correct (model already knew). Balanced 50/50 sampling.

**Task 4 — Domain Classification:** Multi-class (7 domains). Labels come from corpus metadata (source dataset → domain mapping). No annotation needed.

**Task 5 — Correctness Prediction:** Binary. Can the oracle tell from activations alone whether the final answer is correct? Ground truth from comparing extracted answer to known correct answer.

**Task 6 — Persona Detection:** Multi-class (8 personas). Detect which unverbalized system prompt was used during generation. Requires persona corpus (same problems generated with 8 different system prompts). Tests whether oracle reads subtle style shifts in activations.

---

## Corpus

### Sources (12 diverse datasets)
MATH, GSM8K, GPQA, BBH, ARC, StrategyQA, DROP, LogiQA, MMLU-Pro, CommonsenseQA, AQUA-RAT, MedQA

### Generation
Two modes:
- **OpenRouter** (no GPU): async batch calls to `qwen/qwen3-8b`, ~50 concurrent, ~$6 for 1000 problems
- **Local GPU**: batched generation with bf16 model

Each problem gets:
- CoT response (with `<think>` tags)
- Direct response (no thinking)
- Answer extraction + correctness check
- Sentence splitting + boundary position computation
- Category: `load_bearing` / `both_correct` / `both_wrong` / `cot_hurt`

Persona corpus generated separately with `--personas` flag (8 system prompts per problem).

### Commands
```bash
# Main corpus
python src/data_pipeline/generate_cots.py --openrouter \
    --n-problems 1000 --output data/cot_corpus_v4/corpus.jsonl

# Persona corpus
python src/data_pipeline/generate_cots.py --openrouter --personas \
    --n-problems 1000 --output data/cot_corpus_v4/corpus_persona.jsonl

# Train
torchrun --nproc_per_node=1 src/train_mixed.py \
    --corpus data/cot_corpus_v4/corpus.jsonl \
    --persona-corpus data/cot_corpus_v4/corpus_persona.jsonl \
    --model Qwen/Qwen3-8B
```

---

## What We Dropped (and Why)

| Task | Problem | Status |
|------|---------|--------|
| Importance | 99.7% labeled "important" (KL > 0.1 too low) | Dropped |
| Summary | All 200 labels identical | Dropped |
| Unverbalized | SAE ground truth too hard to generate | Dropped |
| Answer tracking | Target format too specific (exact probs) | Eval-only (zero-shot) |

---

## v1 Results and Lessons

### What Happened
- 100K examples, 1 epoch, 6218 steps on H100
- "Grokking" at step 3000 was actually **catastrophic forgetting** — data wasn't shuffled, tasks were sequential
- Importance hit 50% when first seen, then dropped to 30% when taxonomy data started
- wandb: `cot_oracle/runs/2bmv0bur`

### Fixes in v4
- `random.shuffle()` before training (critical)
- 3 layers per boundary instead of 1 random
- Dropped broken tasks (importance, summary, unverbalized)
- 12 diverse corpus sources instead of MATH+GSM8K only
- Added persona detection task
- Each sentence-structured example doubled (multi-layer + single-layer)

---

## Current Status

- **Code:** All 6 dataset loaders + `train_mixed.py` + `generate_cots.py` written and reviewed
- **Baseline evals:** wandb `cot_oracle/runs/iee93uwb`
- **Checkpoints (v1):** `ceselder/cot-oracle-8b-checkpoints` on HuggingFace

### Next Steps

1. **Generate v4 corpus via OpenRouter** (~$6, no GPU needed)
   - Main corpus: 1000 problems from 12 sources
   - Persona corpus: same problems with 8 personas
2. **Compute boundary positions** (tokenizer on CPU)
3. **Rent H100, train v4** (~2h for 1 epoch, try 2-3 epochs)
4. **Run unfaithfulness evals** on best checkpoint vs baseline
5. **Qualitative inspection** — what does the oracle actually generate?

---

## Model & Infra

- **Source model:** Qwen3-8B (36 layers, bf16, ~16GB on H100)
- **AO checkpoint:** `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
- **LoRA:** r=64, alpha=128, dropout=0.05, all-linear
- **Injection:** Norm-matched addition at layer 1
- **Placeholder token:** `" ?"` (space + question mark)
- **Layers for sentence tasks:** 9 (25%), 18 (50%), 27 (75%)
- **GPU rules:** Always bf16, always batch, never 8-bit quantize on H100

---

## Success Criteria

- **Minimum:** Oracle detects unfaithfulness better than baseline AO on held-out evals
- **Good:** >80% on synthetic unfaithfulness detection
- **Great:** Generalizes to unfaithfulness modes not in training, finds unfaithfulness in the wild
