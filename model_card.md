---
language:
  - en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-8B
tags:
  - activation-oracle
  - chain-of-thought
  - interpretability
  - mechanistic-interpretability
  - lora
  - qwen3
  - reasoning
  - cot
  - unfaithfulness-detection
datasets:
  - ceselder/cot-oracle-data
pipeline_tag: text-generation
model-index:
  - name: cot-oracle-v4-8b
    results:
      - task:
          type: text-generation
          name: Domain Classification (from activations)
        metrics:
          - type: accuracy
            value: 98
            name: Exact Match Accuracy
      - task:
          type: text-generation
          name: Correctness Prediction (from activations)
        metrics:
          - type: accuracy
            value: 90
            name: Exact Match Accuracy
---

# CoT Oracle v4 (Qwen3-8B LoRA)

A **chain-of-thought activation oracle**: a LoRA fine-tune of Qwen3-8B that reads the model's own internal activations at sentence boundaries during chain-of-thought reasoning and answers natural-language questions about what was computed.

This is a continuation of the [Activation Oracles](https://github.com/adamkarvonen/activation_oracles) line of work (Karvonen et al., 2024), extended to operate over structured CoT trajectories rather than single-position activations.

## Model Description

An activation oracle is a language model fine-tuned to accept its own internal activations as additional input and answer questions about them. The oracle is the **same model** as the source -- Qwen3-8B reads Qwen3-8B's activations.

CoT Oracle v4 specializes in reading activations extracted at **sentence boundary positions** during chain-of-thought reasoning. Given activations from 3 layers (25%, 50%, 75% depth) at each sentence boundary, the oracle can:

- **Classify the reasoning domain** (math, science, logic, commonsense, reading comprehension, multi-domain, medical)
- **Predict whether the CoT reached the correct answer**
- **Detect decorative reasoning** (steps that don't contribute to the answer)
- **Predict surrounding token context** from arbitrary positions

### Key Properties

- The oracle reads activations, not text. It has no access to the CoT tokens themselves.
- Activations are collected with LoRA **disabled** (pure base model representations).
- Activations are injected via **norm-matched addition** at layer 1, preserving the scale of the residual stream.
- The oracle generates with LoRA **enabled** (the trained adapter interprets the injected activations).

## Training

### Base Checkpoint

Training continues from [`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`](https://huggingface.co/adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B), an activation oracle pretrained on ~1M examples of context prediction, classification, and past-lens tasks.

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank | 64 |
| Alpha | 128 |
| Dropout | 0.05 |
| Target modules | all-linear |

### Training Tasks

Six tasks mixed together and shuffled into a single training run:

| # | Task | Examples | Layers | Description |
|---|------|----------|--------|-------------|
| 1 | Context prediction (random) | 100K | 1 random layer | Predict surrounding tokens at random positions. Standard AO pretraining format. |
| 2 | Context prediction (sentences) | 30K (x2 = 60K) | 3 layers at boundaries | Predict tokens near sentence boundaries. Each example doubled: once with 3 layers, once with L50% only. |
| 3 | Decorative detection | 500 (x2 = 1K) | 3 layers at boundaries | Binary classification: `load_bearing` or `decorative`. |
| 4 | Domain classification | 15K (x2 = 30K) | 3 layers at boundaries | 7-class: `math`, `science`, `logic`, `commonsense`, `reading`, `multi_domain`, `medical`. |
| 5 | Correctness prediction | 15K (x2 = 30K) | 3 layers at boundaries | Binary: `correct` or `incorrect`. |
| 6 | Persona detection | -- | -- | Skipped due to data issue. |

**Total: ~221K training examples, 1 epoch.**

Sentence-structured tasks (2-6) extract activations at 3 layers per boundary position:
- **Layer 9** (25% depth)
- **Layer 18** (50% depth)
- **Layer 27** (75% depth)

Each sentence-structured example is duplicated: once with all 3 layers interleaved, once with only L50%. This teaches the oracle to work with both multi-layer and single-layer inputs.

### Training Details

| Parameter | Value |
|-----------|-------|
| Hardware | 1x NVIDIA H100 80GB |
| Precision | bf16 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Steps | 31,300 |
| Training time | ~4.5 hours |
| Optimizer | AdamW |
| Framework | PyTorch 2.7 + PEFT 0.17 + Transformers 4.55 |

### Activation Injection

Activations are injected at layer 1 via norm-matched addition:

```
h' = h + ||h|| * (v / ||v||)
```

where `h` is the original hidden state and `v` is the collected activation vector. This preserves the norm of the residual stream while adding directional information from the source activations.

The placeholder token is `" ?"` (token ID 937). For multi-layer inputs, per-layer placeholder tokens are used: `" @"` (L25%), `" ?"` (L50%), `" #"` (L75%), cycling in that order.

### Corpus

The training corpus consists of CoT traces generated by Qwen3-8B across 12 reasoning benchmarks: MATH, GSM8K, GPQA, BBH, ARC, StrategyQA, DROP, LogiQA, MMLU-Pro, CommonsenseQA, AQUA-RAT, and MedQA. CoTs were generated via OpenRouter API.

## Evaluation Results

Evaluated on held-out data using exact string match:

| Step | Domain | Correctness | Decorative | Sentence Pred | Context Pred | Summary |
|------|--------|-------------|------------|---------------|--------------|---------|
| 500 | 66% | 53% | 50% | 0% | 4% | 0% |
| 5,000 | **100%** | 86% | 67% | 4% | 7% | 0% |
| 10,000 | 97% | 85% | 50% | 7% | 9% | 0% |
| 20,000 | 98% | 82% | 62% | 10% | 9% | 0% |
| 28,000 | **98%** | **90%** | 50% | 11% | 7% | 0% |

**Key observations:**

- **Domain classification** reaches 98-100% accuracy -- the oracle reliably identifies the reasoning domain from activations alone.
- **Correctness prediction** reaches 90% -- the oracle can tell whether the model's reasoning led to the right answer without seeing the answer.
- **Decorative detection** is noisy (bounces between 50-71%) due to limited eval data (74 unique both-correct entries).
- **Context prediction** stays low (7-11%) under exact string match but this is expected -- the pretrained AO checkpoint already handles this task and exact match is a harsh metric for free-text prediction.
- **Summary** remains at 0% (labels were all identical in training data -- known issue).

Experiment tracking: [wandb `cot_oracle` project, run `cot_oracle_v4_mixed`](https://wandb.ai)

## Usage

### Requirements

This model requires the [activation_oracles](https://github.com/adamkarvonen/activation_oracles) library for the activation collection and injection infrastructure.

```bash
git clone https://github.com/adamkarvonen/activation_oracles
cd activation_oracles && pip install -e .
```

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Load oracle adapter
model = PeftModel.from_pretrained(model, "ceselder/cot-oracle-v4-8b")
```

### Collecting Activations

Activations must be collected from the **base model** (LoRA disabled) at the target layers:

```python
import torch

# Layers at 25%, 50%, 75% depth of Qwen3-8B (36 layers)
LAYERS = [9, 18, 27]

# 1. Prepare input: question + CoT response
messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,
)
full_text = prompt + cot_response

# 2. Find sentence boundary positions in token space
input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
# boundary_positions = [...] (token indices at sentence boundaries)

# 3. Collect activations with LoRA DISABLED
with model.disable_adapter():
    activations = {}  # {layer: {position: tensor}}
    # Use hooks on model.model.layers[layer] to capture hidden states
    # at each boundary position for each layer
```

### Running the Oracle

```python
# 4. Build oracle prompt with placeholder tokens
# For multi-layer: interleave " @", " ?", " #" per boundary
n_boundaries = len(boundary_positions)
placeholders = " @ ? #" * n_boundaries  # 3 tokens per boundary

layer_str = ", ".join(str(l) for l in LAYERS)
oracle_prompt = f"Layer: {layer_str}\n{placeholders.strip()} \n"
oracle_prompt += "What domain of reasoning is this? Answer with one word: math, science, logic, commonsense, reading, multi_domain, or medical."

# 5. Format as chat and tokenize
messages = [{"role": "user", "content": oracle_prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)

# 6. Inject activations via norm-matched addition at layer 1
# At each placeholder position, add the corresponding activation:
# positions cycle through [L25_s1, L50_s1, L75_s1, L25_s2, L50_s2, L75_s2, ...]
# Injection: h' = h + ||h|| * (v / ||v||)

# 7. Generate with LoRA ENABLED (default state)
output = model.generate(input_ids, max_new_tokens=64)
```

For complete working code, see the [cot-oracle repository](https://github.com/ceselder/cot-oracle), particularly `src/core/ao.py` for the injection/runtime mechanism and `src/train.py` for the full training pipeline.

## Intended Use

This model is a **research artifact** for studying chain-of-thought interpretability. Intended uses include:

- Investigating what information is encoded in CoT activations at different stages of reasoning
- Detecting unfaithful chain-of-thought (reasoning that doesn't match the model's actual computation)
- Building tools for mechanistic understanding of language model reasoning

### Limitations

- **Same-model only**: The oracle can only read activations from Qwen3-8B. It will not work with other models.
- **Exact match eval is harsh**: Tasks like context prediction and summary show low scores under exact string match, but the model often produces semantically reasonable outputs.
- **Decorative detection is undertrained**: Only ~500 unique training examples; results are noisy.
- **Summary task is broken**: All 200 training labels were identical, so the model learned nothing useful for this task.
- **No uncertainty calibration**: The oracle is confidently wrong sometimes, consistent with findings from Karvonen et al., 2024.

## Citation

```bibtex
@misc{cot-oracle-v4,
  title={CoT Oracle: Detecting Unfaithful Chain-of-Thought via Activation Trajectories},
  author={Celeste Deschamps-Helaere},
  year={2026},
  url={https://github.com/ceselder/cot-oracle}
}
```

### Related Work

```bibtex
@article{karvonen2024activation,
  title={Activation Oracles},
  author={Karvonen, Adam and others},
  journal={arXiv preprint arXiv:2512.15674},
  year={2024}
}

@article{bogdan2025thought,
  title={Thought Anchors: Causal Importance of CoT Sentences},
  author={Bogdan, Paul and others},
  journal={arXiv preprint arXiv:2506.19143},
  year={2025}
}

@article{macar2025thought,
  title={Thought Branches: Studying CoT through Trajectory Distribution},
  author={Macar, Uzay and Bogdan, Paul and others},
  journal={arXiv preprint arXiv:2510.27484},
  year={2025}
}
```

## Links

- **Code**: [github.com/ceselder/cot-oracle](https://github.com/ceselder/cot-oracle)
- **Training data**: [huggingface.co/datasets/ceselder/cot-oracle-data](https://huggingface.co/datasets/ceselder/cot-oracle-data)
- **Base AO checkpoint**: [adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B](https://huggingface.co/adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B)
- **Activation Oracles repo**: [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
- **Experiment tracking**: wandb `cot_oracle` project, run `cot_oracle_v4_mixed`
