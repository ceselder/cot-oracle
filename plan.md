# CoT Oracle Plan (v5, living)

Last updated: 2026-02-20

## North Star
Chain-of-thought text is often unfaithful. We want a white-box monitor that reads CoT-relevant activations and predicts model behavior better than black-box CoT monitoring and better than simple probes.

Primary objective:
- Build a generalist activation oracle that can produce non-obvious, testable hypotheses about what happened in a model's reasoning.

## Current Scope (Sprint)
The sprint focus is to scale Activation Oracles to chain-of-thought trajectories while staying compute-pragmatic:
- Scale across task diversity, not just dataset size.
- Train on both semantic reconstruction and computational/behavioral properties.
- Keep architecture changes simple and measurable.

## What Is Implemented In This Repo

### Core AO runtime
- AO repo discovery without machine-specific paths (`AO_REPO_PATH` or `./activation_oracles`).
- Shared AO utilities in `src/core/ao.py`.
- Blackwell-safe attention backend selection (default SDPA on RTX 5090-class GPUs unless explicitly overridden).

### Training pipeline
- Main trainer: `src/train_mixed.py`.
- Current mixed tasks:
  - CoT context prediction
  - sentence-boundary prediction
  - decorative/load-bearing label prediction
  - domain prediction
  - correctness prediction
  - persona prediction (when persona corpus provided)
  - optional summary prediction (if `summaries.jsonl` exists)
- Multi-layer activation ingestion with sentence-boundary trajectories.

### Eval pipeline
- Dataset generation + run + scoring under `src/evals/`.
- Current eval families:
  - hinted MCQ
  - sycophancy
  - authority bias
  - decorative CoT
  - answer correctness
  - contradictory comparison
  - sentence insertion
  - scruples sycophancy

### Repository cleanup done
- Removed legacy `signs_of_life` experiments.
- Removed SAE/unverbalized extraction path and unused SAE dataset class.
- Removed stale one-off scripts not used by current pipeline.

## Priority Experiments (From Brainstorming Doc)

### 1) Architecture ablations
- Sentence boundaries vs fixed-token striding.
- Discrete striding vs pooling.
- 3-layer feed-in (25/50/75) vs alternatives.
- Per-layer placeholder tokens (`? ? ?` vs `? # !`-style distinction).

### 2) Training objective ablations
- CoT reconstruction quality on held-out traces.
- Influence detection (hint framing / prompt framing).
- Sycophancy detection on disagreement-focused slices (e.g., Scruples splits).
- Answer prediction from CoT activations.

### 3) Model-organism evals
- ROT13 / obfuscated CoT reconstruction as explicit holdout.
- Load-bearingness approximations (thought-anchor-style labels where feasible).

### 4) Calibration and failure handling
- Evaluate "no signal present" behavior.
- Add checks for hallucinated explanations when activations do not support a query.

## Explicitly Out Of Scope (For Now)
- SAE-driven unverbalized supervision.
- Old signs-of-life experiment track.
- RL-heavy post-training for this sprint.

## Execution Plan
1. Stabilize and benchmark the cleaned mixed pipeline.
2. Run baseline AO and mixed checkpoints on current eval suite.
3. Add/expand held-out evals directly tied to north-star claims:
   - CoT reconstruction holdout metric
   - influence/unfaithfulness metrics
   - answer prediction metric
4. Run architecture/task ablations and keep only robust improvements.
5. Prepare sprint write-up with explicit baseline comparisons and failure cases.

## Directory Structure (Target)
- `src/core/`: AO runtime wrappers and compatibility utilities.
- `src/data_pipeline/`: corpus generation and label extraction.
- `src/dataset_classes/`: task-specific training dataset builders.
- `src/evals/`: eval generation, execution, and scoring.
- `src/train_mixed.py`: primary training entrypoint.
- `scripts/`: operational helpers (rollouts, corpus upload).

## Success Criteria
Minimum:
- Beat baseline AO on at least one unfaithfulness-relevant eval family without regressions on core tasks.

Strong:
- Consistent gains across multiple eval families and robust behavior under model-organism stress tests.

Stretch:
- Reliable generation of non-obvious, verifiable hypotheses about hidden or implicit reasoning signals.
