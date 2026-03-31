# Domain Confusion Sweep

This folder contains the current-main domain-confusion sweep across activation
position counts for the paper checkpoint collection.

Config:
- eval: `domain_confusion`
- base model: `Qwen/Qwen3-8B`
- positions: `1, 3, 5, 10, 100`
- plotted metric: `domain_correct_specific_rate` (`Domain Accuracy`)

Main artifacts:
- `domain_confusion_sweep.png`
- `domain_confusion_sweep.json`
- `domain_confusion_<N>_summary.json` for each position count
