# CoT Trajectory Oracle

A white-box chain-of-thought (CoT) monitoring system built on Activation Oracles (https://arxiv.org/html/2512.15674v1). Reads reasoning from activations to detect when stated reasoning diverges from actual computational influence.

## Architecture

The oracle is Qwen3-8B fine-tuned with LoRA to accept its own activations via norm-matched injection at layer 1.

### Unified Task System
Tasks defined in `tasks.py`

Key files:
- `configs/train.yaml` — Training config: task counts, hyperparams, activation settings, model
- `configs/eval.yaml` — Eval config: task list, baselines list, method_config, score_model
- `src/tasks.py` — TaskDef definitions, scoring modes, HF repos
- `src/data_loading.py` — Unified HF data loading (replaces 30+ dataset loaders)
- `src/eval_loop.py` — Unified eval with 4 scoring modes (replaces training_eval_hook.py)

All data uses the same schema: `{task, prompt, target_response, context_input_ids, context_positions, layers}`

## Training

### Config (`configs/train.yaml`)
YAML controls task counts, hyperparams, activation settings, eval frequency. CLI flags override config values. Set `n: 0` to disable a task.

Never include the number of GPUs used in the runname on wandb.
Instead, try to log the number of gpus to metadata but dont put it in the runname itself, the runname itself should rather indicate what we ablate over

### Data pipeline rules
- **ALL training data comes from HuggingFace.** The training script downloads precomputed JSONL from HF automatically. Never bake activation positions or limits into precomputed data — precomputed data stores `context_input_ids` and `context_positions`, and activation extraction happens at training time on GPU.
- **NEVER fail silently.** If a task is enabled (n > 0) but its data can't be loaded, raise an error. Do not silently skip tasks or swallow exceptions.

## References
- Activation Oracles: [arXiv:2512.15674](https://arxiv.org/abs/2512.15674), [GitHub](https://github.com/adamkarvonen/activation_oracles)
- Thought Anchors: [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)
- Thought Branches: [arXiv:2510.27484](https://arxiv.org/abs/2510.27484)
- "When Just Read the CoT Fails" (ICLR 2026)

## Vast.ai
- **Prefer on-demand over interruptible.** Spot/bid instances get preempted during long init phases (model download, flamingo setup). Use on-demand unless cost is a strong concern.
- **Use Docker-based launch** (`scripts/vast_launch_docker.sh`) — pre-baked image skips uv sync + rsync.

## Git
- **Push after every notable change iff you are Celeste** If and only if you are Celeste (not Jan (jbauer)), Commit and push to remote after completing any meaningful unit of work, but only if you are Celeste (bug fix, feature, refactor).

## Terminology
- **Scorer** = the LLM (`score_model` in eval.yaml) that grades oracle/baseline outputs during training-time eval and comprehensive eval. Lives in `src/qa_scorer.py`. Use "scorer" everywhere in code.
- **Judge** = reserved for the comparative eval UI in `scripts/eval_viewer.py` and `src/chat_compare.py`, where an LLM rates and compares multiple method outputs side-by-side.
- **BB monitor** = a baseline method (`baselines/bb_monitor.py`) where an external LLM reads the CoT text and answers the task question. Not the same as the scorer.
- **supervisor_context** = the TaskDef field naming which data column any supervisor (oracle, bb-monitor, probes) reads. Usually `"cot_text"`, but `"cot_prefix"` for chunked tasks and `"excerpt"` for classification/fineweb tasks.

## Critical Lessons
- **Mini corpus memorization:** 1,064 entries x 15K = 14x repetition → loss=0.01. Use medium corpus (47K+).
- **Generation evals need fuzzy scoring:** Token F1, not exact match.
- **`model.eval()` before generation:** Otherwise gradient checkpointing disables KV caching → 20x slower.
- **Qwen3-8B `enable_thinking=True` is unusable for bounded generation** — generates 4K-8K+ `<think>` tokens. Use `enable_thinking=False` with prompt-based CoT.
- Probes should never use token-subsampling
