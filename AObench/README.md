# AObench — Activation Oracle Benchmark

Open-ended evaluation suite for Activation Oracles, testing whether AOs can extract meaningful information from model activations.

**Original code by Adam Karvonen** ([activation_oracles_dev](https://github.com/adamkarvonen/activation_oracles_dev/tree/main/nl_probes/open_ended_eval)). Copied with permission and modified for standalone use in the cot-oracle project.

## Evals

| Eval | Type | Scoring | Description |
|------|------|---------|-------------|
| `number_prediction` | Generation | Exact match | Predict the number the model is about to output |
| `mmlu_prediction` | Binary | ROC AUC | Predict if model will answer MMLU correctly (pre/post answer) |
| `backtracking` | Generation | LLM judge | Identify what the model is uncertain about at backtrack points |
| `backtracking_mc` | Multiple choice | Accuracy | Same as above but 4-way forced choice |
| `missing_info` | Binary | ROC AUC | Detect if model has incomplete information (A/B/C conditions) |
| `sycophancy` | Binary | ROC AUC | Detect if model is agreeing due to user influence vs genuine |
| `system_prompt_qa_hidden` | Generation | LLM judge | Extract hidden system prompt instructions |
| `system_prompt_qa_latentqa` | Generation | LLM judge | Identify model's adopted persona/instructions |
| `taboo` | Generation | Exact match | Identify the secret taboo word (requires target LoRAs) |
| `personaqa` | Generation | Exact match | Extract persona facts (requires target LoRAs) |

## Usage

```bash
# Run all default evals (excludes taboo/personaqa which need target LoRAs)
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint

# Run specific evals
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --include number_prediction mmlu_prediction backtracking

# Run a single eval standalone
.venv/bin/python -m AObench.open_ended_eval.number_prediction
```

## Structure

```
AObench/
├── open_ended_eval/     # Eval modules (one per eval)
│   ├── run_all.py       # CLI entrypoint, runs all evals
│   ├── eval_runner.py   # Shared infrastructure (loops, scoring, metrics)
│   └── *.py             # Individual eval implementations
├── base_experiment.py   # Core verbalizer/activation injection logic
├── configs/             # AO training config (for adapter loading)
├── utils/               # Activation extraction, steering hooks, etc.
└── datasets/            # Eval datasets (JSON/JSONL/TXT)
```
