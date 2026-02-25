# CoT Trajectory Oracle

A white-box chain-of-thought (CoT) monitoring system built on Activation Oracles (https://arxiv.org/html/2512.15674v1). Reads reasoning from activations to detect when stated reasoning diverges from actual computational influence.

## Architecture

The oracle is Qwen3-8B fine-tuned with LoRA to accept its own activations via norm-matched injection at layer 1.

## Training

### Config (`configs/train.yaml`)
YAML controls task counts, hyperparams, activation settings, eval frequency. CLI flags override config values. Set `n: 0` to disable a task.

Never include the number of GPUs used in the runname on wandb.

### Data pipeline rules
- **ALL training data comes from HuggingFace.** The training script downloads precomputed JSONL from HF automatically. Never bake activation positions or limits into precomputed data — precomputed data stores `context_input_ids` and `context_positions`, and activation extraction happens at training time on GPU.
- **NEVER cap or truncate activation positions.** Do not set `max_positions_per_layer` or any limit on the number of stride positions fed to the oracle. The oracle should see ALL stride positions from the CoT. Stride=5 already controls density. Previous `max_positions_per_layer=20` silently threw away 50-84% of activations for longer CoTs.
- **NEVER fail silently.** If a task is enabled (n > 0) but its data can't be loaded, raise an error. Do not silently skip tasks or swallow exceptions.


## References
- Activation Oracles: [arXiv:2512.15674](https://arxiv.org/abs/2512.15674), [GitHub](https://github.com/adamkarvonen/activation_oracles)
- Thought Anchors: [arXiv:2506.19143](https://arxiv.org/abs/2506.19143)
- Thought Branches: [arXiv:2510.27484](https://arxiv.org/abs/2510.27484)
- "When Just Read the CoT Fails" (ICLR 2026)

## Critical Lessons
- **Mini corpus memorization:** 1,064 entries x 15K = 14x repetition → loss=0.01. Use medium corpus (47K+).
- **Generation evals need fuzzy scoring:** Token F1, not exact match.
- **`model.eval()` before generation:** Otherwise gradient checkpointing disables KV caching → 20x slower.
- **Qwen3-8B `enable_thinking=True` is unusable for bounded generation** — generates 4K-8K+ `<think>` tokens. Use `enable_thinking=False` with prompt-based CoT.
