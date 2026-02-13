#!/usr/bin/env python3
"""
Upload generated datasets to HuggingFace Hub.

Publishes:
1. Qwen3-8B MATH Rollouts - raw rollouts with importance scores
2. CoT Oracle Training Data - delta sequences + causal summaries
"""

import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
import shutil
import tempfile


def prepare_rollouts_dataset(
    rollouts_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Prepare rollouts dataset for upload.

    Returns metadata dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy rollouts, preserving structure
    # But flatten the model/temperature hierarchy for cleaner access
    problem_count = 0

    for problem_dir in rollouts_dir.rglob("problem_*"):
        if not (problem_dir / "chunks_labeled.json").exists():
            continue

        # Copy to output
        dest = output_dir / problem_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(problem_dir, dest)
        problem_count += 1

    # Create dataset card
    readme = f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - math
  - chain-of-thought
  - interpretability
  - thought-anchors
  - activation-oracles
size_categories:
  - 1K<n<10K
---

# Qwen3-8B MATH Rollouts

Chain-of-thought reasoning traces from Qwen3-8B on MATH dataset problems, with counterfactual importance scores computed via the [Thought Anchors](https://github.com/interp-reasoning/thought-anchors) methodology.

## Why This Dataset?

**Qwen3-8B has a pre-trained [Activation Oracle](https://github.com/adamkarvonen/activation_oracles)**, making this the first MATH rollouts dataset that enables activation-based CoT interpretability research.

## Dataset Structure

```
problem_{{idx}}/
├── problem.json          # Original MATH problem
├── base_solution.json    # Qwen3-8B's CoT solution
├── chunks_labeled.json   # Importance scores per chunk
│   └── [{{
│         "chunk_text": str,
│         "counterfactual_importance_kl": float,
│         "counterfactual_importance_acc": float,
│         "resampling_importance": float,
│         "function_tag": str,  # planning, calculation, etc.
│       }}]
└── chunk_{{i}}/
    └── solutions.json    # Resampled rollouts from this point
```

## Importance Scores

- **counterfactual_importance_kl**: KL divergence in answer distribution when chunk is removed
- **counterfactual_importance_acc**: Accuracy change when chunk is removed
- **resampling_importance**: How much removing this chunk affects downstream chunks
- **function_tag**: Semantic role (planning, calculation, verification, uncertainty_management)

## Usage

```python
from datasets import load_dataset
import json

# Load a problem
with open("problem_123/chunks_labeled.json") as f:
    chunks = json.load(f)

for chunk in chunks:
    print(f"Chunk: {{chunk['chunk_text'][:50]}}...")
    print(f"  KL importance: {{chunk['counterfactual_importance_kl']:.4f}}")
```

## Generation Details

- **Model**: Qwen/Qwen3-8B
- **Temperature**: 0.6
- **Top-p**: 0.95
- **Rollouts per chunk**: 50
- **Problems**: {problem_count} (MATH Level 5)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{qwen3-math-rollouts,
  title={{Qwen3-8B MATH Rollouts with Importance Scores}},
  year={{2025}},
  howpublished={{HuggingFace Datasets}},
}}
```

Also cite the Thought Anchors paper:
```bibtex
@article{{thought-anchors,
  title={{Thought Anchors: Which LLM Reasoning Steps Matter?}},
  author={{Bogdan et al.}},
  year={{2025}},
}}
```
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    return {
        "problem_count": problem_count,
        "model": "Qwen/Qwen3-8B",
    }


def prepare_training_dataset(
    training_data_path: Path,
    delta_dir: Path | None,
    output_dir: Path,
) -> dict:
    """
    Prepare training dataset for upload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    with open(training_data_path) as f:
        data = json.load(f)

    # Copy training data
    shutil.copy(training_data_path, output_dir / "training_data.json")

    # Copy delta tensors if available
    if delta_dir and delta_dir.exists():
        shutil.copytree(delta_dir, output_dir / "delta_tensors", dirs_exist_ok=True)

    # Create dataset card
    readme = f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - interpretability
  - chain-of-thought
  - activation-oracles
  - causal-analysis
size_categories:
  - 1K<n<10K
---

# CoT Oracle Training Data

Training data for Chain-of-Thought Oracles: activation delta sequences paired with causal summaries derived from counterfactual resampling.

## Key Innovation

This dataset pairs **activation deltas** (what each CoT sentence contributed to the answer representation) with **causal ground truth** (from Thought Branches resampling). This enables training oracles that can identify unfaithful reasoning from activations alone.

## Dataset Structure

```json
{{
  "question_id": "math_123",
  "question": "Find the value of x...",
  "cot_sentences": ["First, I'll...", "Then...", "Therefore..."],
  "importance_scores": [0.02, 0.45, 0.12],  // KL divergence per sentence
  "causal_summary": "The answer was primarily determined by sentence 2...",
  "is_correct": true
}}
```

### Delta Tensors

Pre-extracted activation deltas in `delta_tensors/`:
- Shape: `[num_sentences, d_model]`
- Each delta = activation difference at answer position when adding that sentence

## Causal Summaries

Generated by Claude/GPT-4 from resampling results. Examples:

> "Sentences 1-2 show high causal importance (KL > 0.3), containing the key insight.
> Sentences 4-6 are post-hoc rationalization with near-zero impact on the answer."

## Usage

```python
import json
import torch

with open("training_data.json") as f:
    data = json.load(f)

for example in data:
    # Load pre-extracted deltas
    deltas = torch.load(f"delta_tensors/{{example['question_id']}}_deltas.pt")
    summary = example["causal_summary"]

    # Train oracle: deltas -> summary
```

## Statistics

- **Examples**: {len(data)}
- **Source model**: Qwen/Qwen3-8B
- **Importance scores**: From Thought Anchors resampling
- **Causal summaries**: Generated by Claude 3 Haiku

## Citation

```bibtex
@misc{{cot-oracle-training,
  title={{CoT Oracle Training Data: Delta Sequences with Causal Ground Truth}},
  year={{2025}},
  howpublished={{HuggingFace Datasets}},
}}
```
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    return {
        "example_count": len(data),
    }


def upload_to_hub(
    local_dir: Path,
    repo_id: str,
    private: bool = False,
):
    """Upload dataset to HuggingFace Hub."""
    api = HfApi()

    # Create repo if doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", private=private)
        print(f"Created repo: {repo_id}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    # Upload
    upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, help="Path to rollouts directory")
    parser.add_argument("--training_data", type=Path, help="Path to training data JSON")
    parser.add_argument("--delta_dir", type=Path, help="Path to delta tensors directory")
    parser.add_argument("--repo_prefix", required=True, help="HuggingFace username/org")
    parser.add_argument("--private", action="store_true", help="Make repos private")
    parser.add_argument("--dry_run", action="store_true", help="Prepare but don't upload")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Prepare and upload rollouts
        if args.rollouts:
            print("\n=== Preparing Rollouts Dataset ===")
            rollouts_out = tmpdir / "rollouts"
            meta = prepare_rollouts_dataset(args.rollouts, rollouts_out)
            print(f"Prepared {meta['problem_count']} problems")

            if not args.dry_run:
                upload_to_hub(
                    rollouts_out,
                    f"{args.repo_prefix}/qwen3-8b-math-rollouts",
                    private=args.private,
                )

        # Prepare and upload training data
        if args.training_data:
            print("\n=== Preparing Training Dataset ===")
            training_out = tmpdir / "training"
            meta = prepare_training_dataset(
                args.training_data,
                args.delta_dir,
                training_out,
            )
            print(f"Prepared {meta['example_count']} examples")

            if not args.dry_run:
                upload_to_hub(
                    training_out,
                    f"{args.repo_prefix}/cot-oracle-training-data",
                    private=args.private,
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
