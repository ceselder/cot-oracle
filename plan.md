# CoT Oracle Plan (Living)

Last updated: 2026-02-20

## Source Context
This plan is synchronized with the shared brainstorming/living document from this thread.
- Shared doc link: `<ADD_SHARED_DOC_LINK_HERE>`

## North Star
Chain-of-thought text is often unfaithful. We want a white-box activation monitor that:
1. predicts model behavior better than black-box CoT monitoring,
2. beats simple probe baselines where possible,
3. produces non-obvious, testable hypotheses about internal reasoning.

## Strategy
1. Scale task diversity more than raw dataset size.
2. Train on both semantic reconstruction and computational properties.
3. Keep architecture changes simple, ablatable, and measurable.

## In Scope (Current Sprint)
- Activation-oracle style CoT reconstruction from trajectory activations.
- Multi-task mixed training focused on behavior-relevant labels.
- Evals for unfaithfulness-adjacent behavior (hints, sycophancy, load-bearing proxies).
- Short architecture sweeps: temporal feed-in + layer choices + marker schemes.

## Out of Scope (For Now)
- SAE-driven unverbalized supervision pipeline.
- Legacy signs-of-life branch.
- RL-heavy post-training loops.

## Current Implementation Status

### Core runtime
- [x] AO repo discovery without machine-specific path assumptions.
- [x] Shared AO runtime helpers in `src/core/ao.py`.
- [x] Blackwell-safe attention defaulting (SDPA on RTX 5090 class).

### Training pipeline
- [x] Mixed training entrypoint in `src/train_mixed.py`.
- [x] Context + sentence reconstruction tasks.
- [x] Decorative/domain/correctness auxiliary tasks.
- [x] Optional persona + summary tasks.
- [ ] Add explicit thought-anchor/load-bearing training task.
- [ ] Add contrastive influenced-vs-uninfluenced pair training.

### Eval pipeline
- [x] Dataset generation + run + scoring flow in `src/evals/`.
- [x] Hinted/sycophancy/authority/decorative/correctness eval families.
- [ ] Held-out CoT reconstruction metric (KL based) as first-class report.
- [ ] ROT13/obfuscated holdout eval integrated into default suite.
- [ ] Load-bearingness metric integrated into default suite.
- [ ] Strong AO/probe/black-box baseline comparison table.

## Priority Evals (Mapped From Brainstorm Doc)

### Tier 1 (implement first)
- Held-out CoT reconstruction.
  - Baseline: AO / null reconstruction.
  - Metric: token-level KL / reconstruction quality.
- Hint influence detection.
  - Baseline: probes / black-box monitor.
  - Metric: yes/no accuracy and calibration.
- Scruples disagreement slice.
  - Baseline: probes / black-box monitor.
  - Metric: agreement-with-human-judgment classification.
- Final answer prediction from CoT activations.
  - Baseline: AO + lightweight probes.
  - Metric: KL / accuracy on answer distribution.

### Tier 2 (next)
- ROT13 or obfuscated CoT reconstruction.
  - Baseline: patchscopes-like decode baseline.
  - Metric: recoverable token rate / sequence accuracy.
- Thought-anchor proxy eval.
  - Baseline: probe + black-box monitor.
  - Metric: load-bearing step detection quality.

## Architecture Ablation Matrix
- [ ] Sentence boundaries vs fixed-token stride (e.g., every 25 tokens).
- [ ] Discrete striding vs pooled activation summaries.
- [ ] 1-layer vs 3-layer feed-in (e.g., 25/50/75 depth).
- [ ] Shared placeholder token vs per-layer marker tokens (`? ? ?` vs `? # !` style).

## Training Data Roadmap

### Definitely
- [x] Start from AO checkpoint.
- [x] Mixed on-policy CoT reconstruction corpus flow.
- [ ] Diverse CoT datasets composed into balanced training mix.
- [ ] Train next-step reconstruction conditioned on prefix activations.
- [ ] Train full-CoT reconstruction from all trajectory activations.
- [ ] Add answer-from-CoT objective explicitly in mixed loop.
- [ ] Add conversational IT set for monitor Q/A + calibrated refusals.

### Promising (time permitting)
- [ ] Unsampled-high-probability concept supervision.
- [ ] Token-ban counterfactual supervision.
- [ ] LLM-labeled fallacy/untruth spans for auxiliary detection.
- [ ] Robustness to inserted off-policy sentence/activation perturbations.

## 5090 / Blackwell Compatibility Plan (Best-Guess)
- [x] Default AO loading path to SDPA on Blackwell.
- [x] Provide vLLM eager-mode controls for Blackwell.
- [ ] Add explicit troubleshooting section with known torch/vLLM/fa2 combinations after more runs.

Current recommended runtime knobs:
```bash
export COT_ORACLE_FORCE_SDPA=1
export COT_ORACLE_VLLM_ENFORCE_EAGER=1
```

## Near-Term Execution TODO
1. Add held-out reconstruction + answer-pred metrics to `run_evals.py` output schema.
2. Add ROT13/obfuscated eval generator + scorer.
3. Add first load-bearing proxy metric (sentence ablation or early-answer-shift proxy).
4. Run architecture sweep on one base model and lock defaults.
5. Add final baseline table (AO, probes, black-box monitor) to report artifacts.
6. Update README examples once eval schema is finalized.

## Success Criteria
- Minimum: beat AO baseline on at least one unfaithfulness-relevant eval without harming core reconstruction.
- Strong: consistent uplift across multiple eval families and at least one model-organism stress test.
- Stretch: monitor outputs non-obvious hypotheses that are later behaviorally validated.
