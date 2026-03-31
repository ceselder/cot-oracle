# AObench Paper Core Last-5 Strict

Artifacts from the strict paper-core AObench rerun where `--n-positions 5`
means the last 5 activation positions for every included eval.

Source run:
- host: `34.44.101.231:3505`
- script: `scripts/run_paper_collection_aobench.py`
- profile: `paper_core`
- output dir on remote: `experiments/paper_collection_aobench_papercore_last5_strict`

Included evals:
- `number_prediction`
- `mmlu_prediction`
- `backtracking_mc`
- `missing_info`
- `sycophancy`

Notes:
- This supersedes the earlier `paper_collection_aobench_papercore_n5` run,
  which still used task-native segment logic for `missing_info` and `sycophancy`.
- Single-layer checkpoints are evaluated on one canonical layer only.
