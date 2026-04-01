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
# Run the legacy default profile
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint

# Run the current paper-facing 6-task profile
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --profile paper_six

# Run the extended paper profile (adds system-prompt, taboo, personaqa)
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --profile paper_plus

# Run specific evals
.venv/bin/python -m AObench.open_ended_eval.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --include number_prediction mmlu_prediction backtracking

# Run the paper collection helper with tiny caps
.venv/bin/python scripts/run_paper_collection_aobench.py \
    --profile paper_plus \
    --sample-profile paper_tiny10 \
    --n-positions 5

# Run a single eval standalone
.venv/bin/python -m AObench.open_ended_eval.number_prediction
```

## Profiles

`run_all.py` exposes a few named profiles:

- `paper_core`: objective subset used for earlier paper plots (`number_prediction`, `mmlu_prediction`, `backtracking_mc`, `missing_info`, `sycophancy`)
- `paper_six`: current default paper comparison subset (`number_prediction`, `mmlu_prediction`, `backtracking`, `vagueness`, `domain_confusion`, `missing_info`)
- `paper_plus`: `paper_six` plus `system_prompt_qa_hidden`, `system_prompt_qa_latentqa`, `taboo`, and `personaqa`
- `judge_heavy`: judge-dependent evals only
- `all`: every eval in the registry

## Notes

- `system_prompt_qa_hidden` and `system_prompt_qa_latentqa` are generation evals scored by an LLM judge.
- `taboo` and `personaqa` require target LoRAs and therefore need a fresh model pass unless you already have saved raw rollout JSON for those exact checkpoints.
- The replay-from-saved-rollouts path only works for tasks whose raw rollout files were actually saved. If a task was never generated for a checkpoint set, it cannot be added to the final plot without taking fresh rollouts.

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
