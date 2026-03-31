# AObench Judge-Heavy Run

This folder contains the current-main AObench `judge_heavy` run on the paper
checkpoint collection.

Config:
- profile: `judge_heavy`
- base model: `Qwen/Qwen3-8B`
- `n_positions=5` for segment-based evals
- hallucination variant: `hallucination_5pos`

Included evals:
- `backtracking`
- `system_prompt_qa_hidden`
- `system_prompt_qa_latentqa`
- `vagueness`
- `domain_confusion`
- `activation_sensitivity`
- `hallucination_5pos`

Main artifacts:
- `report/comparison.png`
- `report/aggregate_scores.png`
- `report/summary.txt`
- `all_summaries.json`
