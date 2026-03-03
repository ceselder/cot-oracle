# CoT Oracle

CoT Oracle is a white-box chain-of-thought monitor built on Activation Oracles. The core model is Qwen3-8B with a LoRA adapter trained to read its own residual-stream activations and answer questions about the reasoning that produced them.

This README documents the pipeline the current code actually runs. Older docs elsewhere in the repo, especially under `src/evals/`, describe older or parallel experiments and are not the main training-time path.

## Core Mechanism

1. A source sequence is built from a question plus a chain-of-thought or other context text.
2. Activation positions are chosen from that source sequence, usually by fixed stride and optionally by punctuation boundaries.
3. Residual activations are extracted from the configured source layers with LoRA disabled.
4. The oracle prompt is prefixed with one placeholder token per activation vector.
5. Those activation vectors are injected back into the model at layer 1 at the placeholder positions.
6. The LoRA-tuned model generates a natural-language answer about the reasoning process.

The main training and eval codepath uses:

- Source-of-truth task registry: `src/tasks.py`
- Unified HF/on-the-fly data loading: `src/data_loading.py`
- Training entrypoint: `src/train.py`
- Training-time eval loop: `src/eval_loop.py`
- Default config: `configs/train.yaml`

## Setup

```bash
UV_PROJECT_ENVIRONMENT="$VENV_LOCAL/${PWD##*/}" uv sync
export AO_REPO_PATH="${AO_REPO_PATH:-$PWD/ao_reference}"
```

`src/core/ao_repo.py` will look for `nl_probes` in `AO_REPO_PATH`, then in `./ao_reference`, then in `./activation_oracles`.

## Task System

`src/tasks.py` currently defines 17 tasks total.

Trainable tasks:

- `hint_admission`
- `atypical_answer`
- `reasoning_termination`
- `answer_trajectory`
- `futurelens`
- `pastlens`
- `correctness`
- `decorative_cot`
- `chunked_convqa`
- `chunked_compqa`
- `backtrack_prediction`
- `sycophancy`
- `probe_sycophancy`
- `truthfulqa_hint_verbalized`
- `truthfulqa_hint_unverbalized`

Eval-only tasks:

- `rot13_reconstruction`
- `sentence_insertion`

The default `configs/train.yaml` enables 13 of the trainable tasks by default and also enables three auxiliary non-task sources: FineWeb context prediction, standard NLP classification, and LatentQA.

## Unified Data Format

Most task data is normalized to the same shape before conversion to AO `TrainingDataPoint`s:

```python
{
  "task": str,
  "prompt": str,
  "target_response": str,
  "context_input_ids": list[int] | None,
  "context_positions": list[int] | None,
  "layers": list[int] | None,
}
```

If `context_input_ids` is missing but `cot_text` is present, `prepare_context_ids()` reconstructs the chat-formatted sequence and computes activation positions at load time.

## Training Pipeline

The real training flow is:

1. `src/train.py` parses CLI flags, merges one or more YAML configs, and lets CLI flags override config values.
2. It resolves source layers from `activations.layers` or from evenly spaced percentages if `--n-layers` is used.
3. It loads the base model, enables gradient checkpointing if configured, and either:
   - resumes from an existing LoRA checkpoint,
   - initializes a fresh LoRA adapter, or
   - loads Adam's AO checkpoint as the starting adapter.
4. It builds the training mixture:
   - HF-backed task datasets are loaded through `load_all_training_data()`.
   - `futurelens` and `pastlens` are generated on the fly from the corpus-v5 HF dataset.
   - Optional FineWeb, classification, and LatentQA examples are generated/loaded separately and appended.
5. `prepare_context_ids()` tokenizes any examples that still only have `cot_text`, computes activation positions, and repeats positions across all configured layers.
6. `dicts_to_training_data()` converts raw dicts into AO `TrainingDataPoint`s:
   - `position_mode=last_only` keeps only the final activation per layer.
   - `position_mode=stochastic` does 50% last-only and 50% chi-squared sampling, always including the final position.
   - `position_mode=all` keeps all computed positions.
   - `layer_dropout.train=true` samples a random non-empty subset of configured layers per example.
7. The training set is ordered according to `training.task_order`:
   - `shuffled`: mix everything together.
   - `sequential`: task-by-task in YAML order.
   - `interleaved`: round-robin task blocks sized to finish at roughly the same end time.
8. In each train step:
   - activations are materialized on demand unless `--precompute` was used,
   - batches may be split by token budget to avoid OOM,
   - a steering hook injects the activation vectors at layer 1,
   - the model is trained with standard next-token loss on the oracle response,
   - metrics are logged to wandb.
9. Checkpoints save LoRA weights plus `training_state.pt` (optimizer, scheduler, RNG state, wandb run metadata).
10. The final checkpoint is optionally uploaded to HuggingFace if `HF_TOKEN` is set.

The canonical launch command is:

```bash
python src/train.py --config configs/train.yaml
```

Multi-GPU uses normal `torchrun`, for example:

```bash
torchrun --nproc_per_node=8 src/train.py --config configs/train.yaml
```

## Evaluation Pipeline

The maintained eval path is the training-time call from `src/train.py` into `src/eval_loop.py`.

At each eval event:

1. `_run_unified_eval()` calls `run_eval()` with the configured eval task list.
2. `run_eval()` loops over tasks from `args.eval_tasks` (derived from `tasks.*.eval` in the YAML).
3. For each task, `_eval_single_task()`:
   - loads the `test` split if available,
   - falls back to `train` only if no `test` split can be loaded,
   - generates `futurelens`/`pastlens` eval examples on the fly,
   - normalizes legacy field names,
   - computes missing `context_input_ids` / `context_positions`,
   - re-strides older precomputed examples to the current stride setting,
   - trims eval inputs to the last activation position per layer (minimal barrier context),
   - materializes activations once and caches them on CPU for reuse across later evals.
4. The oracle generates answers with activation steering active.
5. `score_task()` applies the task-specific scoring rule:
   - parser-based accuracy for structured binary tasks,
   - token F1 for generation tasks,
   - token match rate for reconstruction,
   - step accuracy for sentence insertion.
6. The training loop logs:
   - scalar eval metrics,
   - per-task sample tables to wandb (`question`, `expected`, `predicted`, `correct`).

There is no separate maintained top-level eval CLI for this unified path right now. The `src/evals/` directory contains older or specialized evaluation utilities, but the training loop itself uses `src/eval_loop.py`.

## Configuration Notes

`configs/train.yaml` is the main control surface. The most important sections are:

- `tasks`: per-task sample counts and whether each task participates in eval
- `fineweb`, `classification`, `latentqa`: auxiliary data sources outside `src/tasks.py`
- `training`: optimizer, batch size, ordering, token budgets, prefetching behavior
- `activations`: source layers, stride, position sampling mode, layer dropout
- `model`: base model name, AO checkpoint, fresh-vs-resume adapter behavior
- `output`: checkpoint directory and wandb metadata

Important current behavior:

- `activations.stride` now supports either an integer or `"punctuation"` in the main training/eval path.
- `training.eval_batch_size` is the value currently consumed by `src/train.py` for eval generation batches.
- `eval.eval_batch_size` exists in `configs/train.yaml` but is not currently read by `apply_config()`.
- In practice, `train()` recomputes eval/save cadence dynamically for `shuffled`, `sequential`, and `interleaved` runs, so the raw `eval.eval_steps` / `eval.save_steps` values in the YAML are not the final schedule.

## Data Sources

The current code pulls data from several places:

- HuggingFace task datasets from the repos listed in `src/tasks.py`
- Corpus-v5 on HuggingFace for `futurelens` and `pastlens`
- FineWeb and LMSYS chat streaming for auxiliary context prediction
- Standard NLP datasets for auxiliary classification (`sst2`, `ag_news`, `snli` by default)
- Local `ao_reference/datasets/latentqa_datasets/train` for LatentQA

Downloaded HF task JSONL files are cached under `COT_ORACLE_CACHE_DIR` if set, otherwise under `data/hf_cache`.

## Known Boundaries

- Eval activations are cached by task name inside a single process because the base model is frozen during LoRA training. If you change eval stride/layers within the same long-lived Python process, clear the cache or start a fresh process.
- `rot13_reconstruction` is skipped by default in the unified eval loop because it needs a different adapter setup.
- The top-level README that was previously in this repo described a different, older pipeline. The source of truth is the current code listed above.
