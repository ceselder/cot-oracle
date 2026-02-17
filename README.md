# CoT Oracle: Reading Chain-of-Thought from Activations

An activation oracle that reads a language model's internal representations during chain-of-thought reasoning and answers questions about what was actually computed — without seeing any text.

Built on [Activation Oracles](https://github.com/adamkarvonen/activation_oracles) (Karvonen et al., 2024). The oracle is Qwen3-8B reading its own activations via LoRA fine-tuning.

## How It Works

1. **Generate a CoT** from Qwen3-8B (base model, no adapter)
2. **Extract activations** at each sentence boundary from 3 layers (25%, 50%, 75% depth)
3. **Inject activations** into placeholder tokens via norm-matched addition at layer 1
4. **Ask the oracle** questions — it answers from activations alone, never sees the CoT text

```
"Layer: 9, 18, 27\n ? ? ? ? ? ? ? ? ? \n Activations from 3 sentence boundaries. What domain is this?"
                    ^^^   ^^^   ^^^
                    S1    S2    S3    (3 activations each: L25%, L50%, L75%)
```

The oracle sees only the layer metadata, placeholder tokens (with injected activations), and the question. The original problem and chain-of-thought are never in the prompt.

## Results (v4, 1 epoch on H100)

| Task | Accuracy | Description |
|------|----------|-------------|
| **Domain classification** | **100%** | 7-class: math, science, logic, commonsense, reading, multi_domain, medical |
| **Correctness prediction** | **91%** | Binary: did the CoT reach the correct answer? |
| **Decorative detection** | **71%** (peak) | Binary: was the reasoning load-bearing or decorative? |
| **Context prediction (sentences)** | **11%** | Predict tokens near sentence boundaries (exact match) |
| **Context prediction (random)** | **8%** | Standard AO pretraining task (exact match) |

Correctness prediction at 91% is notable — the oracle determines whether the model's answer is correct purely from activation patterns, without seeing the problem or the answer.

## Setup

Requires a CUDA GPU with enough VRAM for Qwen3-8B in bf16 (~16GB). An H100 or A100 works well.

```bash
git clone https://github.com/ceselder/cot-oracle
cd cot-oracle
pip install -r requirements.txt

# Activation oracles library (needed for the injection/steering infrastructure)
git clone https://github.com/adamkarvonen/activation_oracles
cd activation_oracles && pip install -e . && cd ..
```

Download a trained checkpoint (or use your own):
```bash
# v4 checkpoints on HuggingFace
huggingface-cli download ceselder/cot-oracle-v4-checkpoints \
    --local-dir checkpoints/cot_oracle_v4
```

## Usage

`chat_compare.py` is the main interface — an interactive REPL that generates chain-of-thought, extracts activations, and queries two oracles side-by-side.

```bash
python3 src/chat_compare.py --checkpoint checkpoints/cot_oracle_v4/step_30000
```

1. Type a reasoning question at `Question>` — the base model generates a CoT and collects activations at sentence boundaries
2. At `Ask oracles>`, query the activations:
   - `domain` — what domain of reasoning?
   - `correct` — is the answer correct?
   - `decorative` — load-bearing or decorative reasoning?
   - `summary` — summarize the thought process
   - Any free-form question about the reasoning
3. `new` to start over, `quit` to exit

Both the original AO (single-layer context prediction) and trained CoT oracle (multi-layer, task-specific) answer side-by-side.

### Model Organisms

You can pass a separate LoRA adapter for CoT generation with `--cot-adapter`. This lets you test whether the oracle can read through altered reasoning — for example, a model fine-tuned to reason in rot13:

```bash
python3 src/chat_compare.py \
    --checkpoint checkpoints/cot_oracle_v4/step_30000 \
    --cot-adapter ceselder/rot13-qwen3-8b-lora
```

This loads three adapters on the same Qwen3-8B:
- **organism** (`--cot-adapter`): generates the CoT (e.g. in rot13)
- **original_ao**: baseline activation oracle for comparison
- **trained**: the trained CoT oracle

The CoT text may be unreadable, but the activations still encode the model's actual reasoning. The oracle reads those activations and reports what it sees.

## Training

### Architecture
- **Model:** Qwen3-8B (36 layers, bf16)
- **Base checkpoint:** [adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B](https://huggingface.co/adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B)
- **LoRA:** rank 64, alpha 128, dropout 0.05, all-linear
- **Injection:** norm-matched addition at layer 1: `h' = h + ||h|| * v/||v||`
- **Layers:** 9 (25%), 18 (50%), 27 (75%)

### Training Tasks (6 tasks, all shuffled together)

| # | Task | Examples | Format |
|---|------|----------|--------|
| 1 | Context prediction (random) | 100K | 1 random layer, random positions |
| 2 | Context prediction (sentences) | 30K (x2) | 3 layers, all sentence boundaries |
| 3 | Decorative detection | 500 (x2) | 3 layers, binary classification |
| 4 | Domain classification | 15K (x2) | 3 layers, 7-class |
| 5 | Correctness prediction | 15K (x2) | 3 layers, binary classification |
| 6 | Summary prediction | 15K (x2) | 3 layers, free-text generation |

**Total: ~221K examples, 1 epoch, ~4.5h on H100.**

Sentence-structured tasks (2-6) are each created twice: once with all 3 layers interleaved, once with L50% only. This teaches the oracle to work with both multi-layer and single-layer inputs.

### Data Doubling

Each sentence-structured example is duplicated:
1. **Multi-layer:** `"Layer: 9, 18, 27"` prefix, 3N placeholder tokens
2. **Single-layer:** `"Layer: 18"` prefix, N placeholder tokens (standard AO format)

### Corpus
CoT traces generated by Qwen3-8B across 12 reasoning benchmarks: MATH, GSM8K, GPQA, BBH, ARC, StrategyQA, DROP, LogiQA, MMLU-Pro, CommonsenseQA, AQUA-RAT, MedQA.

Each problem gets a CoT response and a direct response. Categories are assigned automatically:
- **load_bearing**: CoT correct, direct wrong (reasoning was necessary)
- **both_correct** (decorative): both correct (model already knew)
- **both_wrong** / **cot_hurt**: failure cases

### v1 Lessons
v1 trained tasks sequentially (not shuffled) — what looked like "grokking" at step 3000 was actually just the model seeing importance data for the first time, followed by catastrophic forgetting. Fix: `random.shuffle()` before training.

## Links

- **Checkpoints:** [ceselder/cot-oracle-v4-checkpoints](https://huggingface.co/ceselder/cot-oracle-v4-checkpoints) (16 checkpoints, step 2K-30K)
- **Training data:** [ceselder/cot-oracle-data](https://huggingface.co/datasets/ceselder/cot-oracle-data)
- **Activation Oracles:** [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)

## Project Structure

```
src/
├── chat_compare.py                # Interactive A/B comparison tool (main interface)
├── train_mixed.py                 # Training script (6 tasks, shuffled)
├── data_pipeline/
│   └── generate_cots.py           # CoT corpus generation
├── dataset_classes/               # Task-specific dataset loaders for training
├── evals/                         # Unfaithfulness evaluation suite
└── signs_of_life/
    └── ao_lib.py                  # Inlined AO library (activation collection, injection, hooks)
```

## Related Work

- [Activation Oracles](https://arxiv.org/abs/2512.15674) (Karvonen et al., 2024)
- [Thought Anchors](https://arxiv.org/abs/2506.19143) (Bogdan et al., 2025)
- [Thought Branches](https://arxiv.org/abs/2510.27484) (Macar, Bogdan et al., 2025)

## License

Apache 2.0
