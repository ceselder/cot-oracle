# CoT Trajectory Oracle

A white-box chain-of-thought (CoT) monitoring system built on Activation Oracles (https://arxiv.org/html/2512.15674v1). Reads reasoning from activations to detect when stated reasoning diverges from actual computational influence.

## Architecture

The oracle is Qwen3-8B fine-tuned with LoRA to accept its own activations via norm-matched injection at layer 1.

### Unified Task System

11 tasks total (6 trainable + 5 eval-only), defined in `src/tasks.py`:

**Training + Eval:** hint_admission, atypical_answer, reasoning_termination, answer_trajectory, futurelens, backtrack_prediction

**Eval-only:** rot13_reconstruction, sycophancy, truthfulqa_hint_verbalized, truthfulqa_hint_unverbalized, sentence_insertion

Key files:
- `src/tasks.py` — TaskDef definitions, scoring modes, HF repos
- `src/data_loading.py` — Unified HF data loading (replaces 30+ dataset loaders)
- `src/eval_loop.py` — Unified eval with 4 scoring modes (replaces training_eval_hook.py)

All data uses the same schema: `{task, prompt, target_response, context_input_ids, context_positions, layers}`

### 50/50 stride sampling
During training, 50% of examples see all stride positions (full context), and 50% see only the last position per layer (minimal context). This trains the oracle to work with both rich and sparse activation information.

## Training

### Config (`configs/train.yaml`)
YAML controls task counts, hyperparams, activation settings, eval frequency. CLI flags override config values. Set `n: 0` to disable a task.

Never include the number of GPUs used in the runname on wandb.

### Data pipeline rules
- **ALL training data comes from HuggingFace.** The training script downloads precomputed JSONL from HF automatically. Never bake activation positions or limits into precomputed data — precomputed data stores `context_input_ids` and `context_positions`, and activation extraction happens at training time on GPU.
- **NEVER cap or truncate activation positions.** Do not set `max_positions_per_layer` or any limit on the number of stride positions fed to the oracle. The oracle should see ALL stride positions from the CoT. Stride=5 already controls density.
- **NEVER fail silently.** If a task is enabled (n > 0) but its data can't be loaded, raise an error. Do not silently skip tasks or swallow exceptions.


## SAE Feature Analysis

We use Sparse Autoencoders (SAEs) from `adamkarvonen/qwen3-8b-saes` to get interpretable feature-level decompositions of the same residual stream the oracle reads. SAEs are available at layers 9, 18, 27 (same as oracle layers) with 65K features each (trainer 2, BatchTopK architecture).

**How it works:** The SAE encoder projects a residual stream vector `x ∈ R^d_model` into a sparse feature space `f ∈ R^d_sae` via `f = ReLU((x - b_dec) @ W_enc + b_enc)`, then applies a threshold to keep only strongly active features. The decoder reconstructs `x_hat = f @ W_dec + b_dec`. Each feature has a human-readable label generated from its max-activating examples (stored as JSON per layer).

**Loading:** `ao_reference/nl_probes/sae.py` → `load_dictionary_learning_batch_topk_sae()`. Labels at `$CACHE_DIR/sae_features/trainer_2/trainer_2/labels/labels_layer{9,18,27}_trainer2.json` (also on HF: `japhba/qwen3-8b-sae-max-activations`).

**Oracle vs SAE comparison:** `scripts/compare_oracle_sae.py` runs both the trained oracle and SAEs on the same CoT rollouts, producing per-example markdown logs with oracle task outputs alongside top-K SAE features at stride positions.

## References
- Activation Oracles: [arXiv:2512.15674](https://arxiv.org/abs/2512.15674), [GitHub](https://github.com/adamkarvonen/activation_oracles)
- Thought Anchors: [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)
- Thought Branches: [arXiv:2510.27484](https://arxiv.org/abs/2510.27484)
- "When Just Read the CoT Fails" (ICLR 2026)

## Vast.ai
- **Prefer on-demand over interruptible.** Spot/bid instances get preempted during long init phases (model download, flamingo setup). Use on-demand unless cost is a strong concern.
- **Use Docker-based launch** (`scripts/vast_launch_docker.sh`) — pre-baked image skips uv sync + rsync.

## Workflow
- **Push after every notable change.** Commit and push to remote after completing any meaningful unit of work (bug fix, feature, refactor). Don't accumulate uncommitted changes.

## Critical Lessons
- **Mini corpus memorization:** 1,064 entries x 15K = 14x repetition → loss=0.01. Use medium corpus (47K+).
- **Generation evals need fuzzy scoring:** Token F1, not exact match.
- **`model.eval()` before generation:** Otherwise gradient checkpointing disables KV caching → 20x slower.
- **Qwen3-8B `enable_thinking=True` is unusable for bounded generation** — generates 4K-8K+ `<think>` tokens. Use `enable_thinking=False` with prompt-based CoT.
