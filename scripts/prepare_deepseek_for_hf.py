"""
Download deepseek math-rollouts data from uzaymacar/math-rollouts and prepare
for upload to ceselder's HuggingFace account in the same directory format.

Downloads per-problem: problem.json, base_solution.json, chunks.json, chunks_labeled.json
(skips raw rollout solutions — they're ~81GB and importance scores are already computed)

Usage:
    python scripts/prepare_deepseek_for_hf.py
    python scripts/prepare_deepseek_for_hf.py --upload --repo ceselder/deepseek-math-rollouts
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

REPO_ID = "uzaymacar/math-rollouts"
MODELS = ["deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b"]
CONDITION = "correct_base_solution"
SAMPLING = "temperature_0.6_top_p_0.95"
FILES_PER_PROBLEM = ["problem.json", "base_solution.json", "chunks.json", "chunks_labeled.json"]

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "deepseek_rollouts_for_hf"


def download_all(output_dir: Path):
    api = HfApi()
    stats = {"models": {}, "total_problems": 0, "total_chunks": 0}

    for model in MODELS:
        base_path = f"{model}/{SAMPLING}/{CONDITION}"
        entries = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=base_path))
        problem_dirs = sorted([e.path for e in entries if "problem_" in e.path])
        print(f"{model}: {len(problem_dirs)} problems")
        stats["models"][model] = len(problem_dirs)
        stats["total_problems"] += len(problem_dirs)

        model_dir = output_dir / model / SAMPLING / CONDITION
        for problem_path in tqdm(problem_dirs, desc=model):
            problem_name = problem_path.split("/")[-1]
            problem_dir = model_dir / problem_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            for fname in FILES_PER_PROBLEM:
                local_file = problem_dir / fname
                if local_file.exists():
                    continue
                remote_path = f"{problem_path}/{fname}"
                downloaded = hf_hub_download(REPO_ID, remote_path, repo_type="dataset")
                shutil.copy2(downloaded, local_file)

            # Count chunks
            labeled_file = problem_dir / "chunks_labeled.json"
            if labeled_file.exists():
                with open(labeled_file) as f:
                    chunks = json.load(f)
                stats["total_chunks"] += len(chunks)

    return stats


def write_readme(output_dir: Path, stats: dict):
    readme = f"""---
dataset_info:
  config_name: default
license: mit
task_categories:
- text-generation
language:
- en
tags:
- chain-of-thought
- math
- reasoning
- thought-anchors
- causal-importance
- resampling
size_categories:
- 1K<n<10K
---

# DeepSeek MATH Rollouts with Causal Importance Scores

Resampling-based causal importance scores for chain-of-thought reasoning on MATH problems.
Derived from [uzaymacar/math-rollouts](https://huggingface.co/datasets/uzaymacar/math-rollouts)
using the [Thought Anchors](https://arxiv.org/abs/2506.19143) methodology.

## Models

| Model | Problems |
|-------|----------|
{chr(10).join(f"| {m} | {n} |" for m, n in stats['models'].items())}

**Total:** {stats['total_problems']} problems, {stats['total_chunks']} labeled chunks

## Directory Structure

```
{{model}}/temperature_0.6_top_p_0.95/correct_base_solution/
├── problem_0/
│   ├── problem.json           # MATH problem (problem, level, type, gt_answer)
│   ├── base_solution.json     # Model CoT (prompt, solution, answer, is_correct)
│   ├── chunks.json            # CoT split into chunks (source_text, solution_text, chunks[])
│   └── chunks_labeled.json    # Per-chunk causal importance scores
├── problem_1/
│   └── ...
```

## Chunk Labels (chunks_labeled.json)

Each chunk has:
- `chunk`: text of the CoT chunk
- `chunk_idx`: position in the CoT
- `resampling_importance_kl`: KL divergence when chunk is removed and continuation resampled (100 rollouts)
- `resampling_importance_accuracy`: accuracy change when chunk removed
- `counterfactual_importance_kl`: counterfactual KL divergence
- `forced_importance_kl`: KL from forced-answer rollouts
- `function_tags`: semantic role (problem_setup, active_computation, etc.)
- `depends_on`: indices of prerequisite chunks
- `accuracy`: rollout accuracy at this chunk
- `summary`: semantic summary of chunk's role

## Citation

```bibtex
@article{{bogdan2025thought,
  title={{Thought Anchors: Localizing Causally Important Parts in Chain-of-Thought Reasoning}},
  author={{Bogdan, Paul and Macar, Uzay and Pal, Kaivalya and Roger, Giles and Nanda, Neel}},
  journal={{arXiv preprint arXiv:2506.19143}},
  year={{2025}}
}}
```

## Use Case

Training data for CoT oracles that detect unfaithful reasoning from activation trajectories.
High `resampling_importance_kl` indicates causally important chunks; low scores indicate
post-hoc rationalization.
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def upload(output_dir: Path, repo_id: str):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo", default="ceselder/deepseek-math-rollouts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading from uzaymacar/math-rollouts...")
    stats = download_all(output_dir)
    print(f"\nTotal: {stats['total_problems']} problems, {stats['total_chunks']} chunks")

    write_readme(output_dir, stats)
    print(f"Wrote README.md to {output_dir}")

    if args.upload:
        upload(output_dir, args.repo)


if __name__ == "__main__":
    main()
