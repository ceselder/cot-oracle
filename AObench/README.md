# AObench — Activation Oracle Benchmark

Open-ended evaluation suite for Activation Oracles, testing whether AOs can extract meaningful information from model activations.

**Original code by Adam Karvonen** ([activation_oracles_dev](https://github.com/adamkarvonen/activation_oracles_dev/tree/main/nl_probes/open_ended_eval)). Copied with permission and modified for standalone use in the cot-oracle project.

## Evals

| Eval | Type | Scoring | Description |
|------|------|---------|-------------|
| `number_prediction` | Generation | Exact match | Predict the number the model is about to output |
| `mmlu_prediction` | Binary | ROC AUC | Predict if model will answer MMLU correctly (pre/post answer) |
| `backtracking` | Generation | LLM judge | Explain what the model is uncertain about at backtrack points |
| `missing_info` | Binary | ROC AUC | Detect if model has incomplete information (A/B/C conditions) |
| `sycophancy` | Binary | ROC AUC | Detect if model is agreeing due to user influence vs genuine |
| `vagueness` | Generation | LLM judge | Measure whether oracle answers are specific vs vague/generic |
| `domain_confusion` | Generation | LLM judge | Detect when the oracle attributes the wrong problem domain |
| `activation_sensitivity` | Generation | LLM judge | Test whether matched texts with different hidden state yield meaningfully different oracle answers |
| `hallucination` | Generation | LLM judge | Measure how often the oracle confidently says concrete wrong things |
| `system_prompt_qa_hidden` | Generation | LLM judge | Extract hidden system prompt instructions |
| `system_prompt_qa_latentqa` | Generation | LLM judge | Identify model's adopted persona/instructions |
| `taboo` | Generation | Exact match | Identify the secret taboo word (requires target LoRAs) |
| `personaqa` | Generation | Exact match | Extract persona facts (requires target LoRAs) |

## Usage

```bash
# Run the legacy default profile
.venv/bin/python -m AObench.eval_scripts.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint

# Run the current paper-facing 6-task profile
.venv/bin/python -m AObench.eval_scripts.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --profile paper_six

# Run the extended paper profile (adds system-prompt, taboo, personaqa)
.venv/bin/python -m AObench.eval_scripts.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --profile paper_plus

# Run the full all-task benchmark
.venv/bin/python -m AObench.eval_scripts.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --profile all \
    --n-positions 5

# Run specific evals
.venv/bin/python -m AObench.eval_scripts.run_all \
    --verbalizer-lora your-org/your-ao-checkpoint \
    --include number_prediction mmlu_prediction backtracking

# Run the paper collection helper with tiny caps
.venv/bin/python scripts/run_paper_collection_aobench.py \
    --profile paper_plus \
    --sample-profile paper_tiny10 \
    --n-positions 5

# Run a single eval standalone
.venv/bin/python -m AObench.eval_scripts.number_prediction
```

## Profiles

`run_all.py` exposes a few named profiles:

- `paper_core`: objective subset used for earlier paper plots (`number_prediction`, `mmlu_prediction`, `missing_info`, `sycophancy`)
- `paper_six`: current default paper comparison subset (`number_prediction`, `mmlu_prediction`, `backtracking`, `vagueness`, `domain_confusion`, `missing_info`)
- `paper_plus`: `paper_six` plus `system_prompt_qa_hidden`, `system_prompt_qa_latentqa`, `taboo`, and `personaqa`
- `judge_heavy`: judge-dependent evals only
- `all`: every eval in the registry (`taboo`, `personaqa`, `number_prediction`, `mmlu_prediction`, `backtracking`, `missing_info`, `sycophancy`, `system_prompt_qa_hidden`, `system_prompt_qa_latentqa`, `vagueness`, `domain_confusion`, `activation_sensitivity`, `hallucination`)

## Notes

- AObench now writes outputs under `AObench/eval_results/...` by default instead of the old top-level `experiments/...` paths.
- The reporting code lives in [`AObench/utils/report.py`](AObench/utils/report.py); [`AObench/report.py`](AObench/report.py) remains as a backward-compatible shim.
- `system_prompt_qa_hidden` and `system_prompt_qa_latentqa` are generation evals scored by an LLM judge.
- `taboo` and `personaqa` require target LoRAs and therefore need a fresh model pass unless you already have saved raw rollout JSON for those exact checkpoints.
- The replay-from-saved-rollouts path only works for tasks whose raw rollout files were actually saved. If a task was never generated for a checkpoint set, it cannot be added to the final plot without taking fresh rollouts.
- The user-facing CLI alias is `AObench.eval_scripts.*`; the implementation still lives in `AObench.open_ended_eval.*` for compatibility.
- The current tiny paper-six bundle is available at `AObench/eval_results/paper_six_tiny10_gemini31flashlite/final/report/`.
- For a canonical overnight all-task paper-collection run, use `scripts/run_paper_collection_full_overnight.sh`.

## Structure

```
AObench/
├── eval_scripts/        # User-facing CLI aliases for running evals
├── open_ended_eval/     # Eval modules (one per eval)
│   ├── run_all.py       # CLI entrypoint, runs all evals
│   ├── eval_runner.py   # Shared infrastructure (loops, scoring, metrics)
│   └── *.py             # Individual eval implementations
├── base_experiment.py   # Core verbalizer/activation injection logic
├── configs/             # AO training config (for adapter loading)
├── utils/               # Activation extraction, steering hooks, etc.
└── datasets/            # Eval datasets (JSON/JSONL/TXT)
```
