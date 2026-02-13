#!/usr/bin/env python3
"""
Auto-analyze rollouts and upload to HuggingFace.
Run this after generate_rollouts.py completes.
"""

import json
import os
from pathlib import Path
import requests
from huggingface_hub import HfApi, create_repo
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ROLLOUTS_DIR = "/root/qwen3_rollouts/Qwen3-8B/temperature_0.6_top_p_0.95/correct_base_solution/problem_7162"


def compute_importance(solutions: list, base_answer: str) -> dict:
    """Compute importance metrics for a chunk's solutions."""
    answers = [s.get("answer", "") for s in solutions]
    correct = [s.get("is_correct", False) for s in solutions]

    # Answer change rate (key importance metric)
    answer_changed = sum(1 for a in answers if a != base_answer)
    answer_change_rate = answer_changed / len(solutions) if solutions else 0

    # Accuracy
    accuracy = sum(correct) / len(correct) if correct else 0

    return {
        "answer_change_rate": answer_change_rate,
        "accuracy": accuracy,
        "num_rollouts": len(solutions),
        "answers": answers,
    }


def load_and_analyze():
    """Load rollouts and compute importance for each chunk."""
    problem_dir = Path(ROLLOUTS_DIR)

    # Load base data
    with open(problem_dir / "problem.json") as f:
        problem = json.load(f)
    with open(problem_dir / "base_solution.json") as f:
        base_solution = json.load(f)
    with open(problem_dir / "chunks.json") as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]
    base_answer = base_solution.get("answer", "")

    # Find all chunk directories with solutions
    chunk_results = []
    for chunk_dir in sorted(problem_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
            solutions_file = chunk_dir / "solutions.json"
            if solutions_file.exists():
                chunk_idx = int(chunk_dir.name.split("_")[1])
                with open(solutions_file) as f:
                    solutions = json.load(f)

                importance = compute_importance(solutions, base_answer)
                chunk_results.append({
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunks[chunk_idx] if chunk_idx < len(chunks) else "",
                    **importance,
                })

    # Sort by importance
    chunk_results.sort(key=lambda x: x["answer_change_rate"], reverse=True)

    return {
        "problem": problem,
        "base_solution": base_solution,
        "total_chunks": len(chunks),
        "analyzed_chunks": len(chunk_results),
        "chunk_results": chunk_results,
        "all_chunks": chunks,
    }


def call_gemini(prompt: str) -> str:
    """Call Gemini via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.3,
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_causal_summary(analysis: dict) -> str:
    """Generate causal summary using importance scores."""
    chunk_results = analysis["chunk_results"]
    problem = analysis["problem"]

    # Format importance data
    importance_text = "\n".join([
        f"Chunk {r['chunk_idx']}: importance={r['answer_change_rate']:.2f}, accuracy={r['accuracy']:.2f}\n  Text: {r['chunk_text'][:100]}..."
        for r in chunk_results[:10]  # Top 10 by importance
    ])

    prompt = f"""You are analyzing the causal structure of an LLM's chain-of-thought reasoning using Thought Anchors methodology.

PROBLEM: {problem.get('problem', '')[:300]}

IMPORTANCE SCORES (from counterfactual resampling):
- High importance (>0.3): Removing this chunk CHANGES the answer - it's causally important
- Low importance (<0.1): Removing this chunk doesn't change the answer - likely post-hoc rationalization

CHUNK IMPORTANCE DATA:
{importance_text}

Based on these MEASURED causal importance scores, write a 3-4 sentence analysis:
1. Which chunks are the TRUE thought anchors (high importance, drive the answer)?
2. Which chunks appear to be post-hoc rationalization (low importance)?
3. What does this reveal about the model's actual reasoning process?

Be specific about chunk numbers and their measured importance scores. Focus on the causal structure revealed by the resampling data."""

    return call_gemini(prompt)


def upload_to_hf(analysis: dict, summary: str, repo_id: str):
    """Upload to HuggingFace."""
    output_dir = Path("/tmp/hf_anchors_upload")
    output_dir.mkdir(exist_ok=True)

    # Prepare data
    data = {
        "problem_id": "problem_7162",
        "problem": analysis["problem"].get("problem", ""),
        "problem_type": analysis["problem"].get("type", ""),
        "gt_answer": analysis["problem"].get("gt_answer", ""),
        "model_answer": analysis["base_solution"].get("answer", ""),
        "is_correct": analysis["base_solution"].get("is_correct", False),
        "total_chunks": analysis["total_chunks"],
        "analyzed_chunks": analysis["analyzed_chunks"],
        "chunk_importance_scores": [
            {
                "chunk_idx": r["chunk_idx"],
                "chunk_text": r["chunk_text"],
                "importance": r["answer_change_rate"],
                "accuracy": r["accuracy"],
            }
            for r in analysis["chunk_results"]
        ],
        "thought_anchors": [r["chunk_idx"] for r in analysis["chunk_results"] if r["answer_change_rate"] > 0.2],
        "low_importance_chunks": [r["chunk_idx"] for r in analysis["chunk_results"] if r["answer_change_rate"] < 0.1],
        "causal_summary": summary,
        "all_chunks": analysis["all_chunks"],
    }

    # Save as JSONL
    with open(output_dir / "train.jsonl", "w") as f:
        f.write(json.dumps(data) + "\n")

    # Create README
    readme = f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - chain-of-thought
  - reasoning
  - interpretability
  - thought-anchors
  - causal-analysis
  - qwen3
---

# Qwen3-8B MATH CoT with Thought Anchor Analysis

Chain-of-thought reasoning traces from Qwen3-8B with **measured causal importance scores** from counterfactual resampling (Thought Anchors methodology).

## What's Special

This dataset includes **actual causal importance scores** computed by resampling - not just the CoT text:
- **High importance chunks**: Removing them changes the answer (thought anchors)
- **Low importance chunks**: Removing them doesn't affect the answer (post-hoc rationalization)

## Example

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")

ex = ds["train"][0]
print("Thought anchors (high importance):", ex["thought_anchors"])
print("Low importance chunks:", ex["low_importance_chunks"])
print("Causal summary:", ex["causal_summary"])
```

## Methodology

1. Generate CoT with Qwen3-8B on MATH problems
2. Split into sentence-level chunks
3. For each chunk: remove it and resample 10 continuations
4. Compute importance = rate at which removing chunk changes final answer
5. Generate causal summary with Gemini based on importance scores

## Data Fields

- `chunk_importance_scores`: List of chunks with measured importance
- `thought_anchors`: Chunk indices with importance > 0.2
- `low_importance_chunks`: Chunk indices with importance < 0.1
- `causal_summary`: Gemini-generated analysis of causal structure

## Related Work

- [Thought Anchors](https://arxiv.org/abs/2506.19143)
- [Thought Branches](https://arxiv.org/abs/2510.27484)
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)

    # Upload
    api = HfApi(token=HF_TOKEN)
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Repo note: {e}")

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    print("Loading and analyzing rollouts...")
    analysis = load_and_analyze()

    print(f"Analyzed {analysis['analyzed_chunks']} chunks")
    print("\nTop chunks by importance:")
    for r in analysis["chunk_results"][:5]:
        print(f"  Chunk {r['chunk_idx']}: importance={r['answer_change_rate']:.2f}")

    print("\nGenerating causal summary with Gemini...")
    summary = generate_causal_summary(analysis)
    print(f"Summary: {summary[:200]}...")

    print("\nUploading to HuggingFace...")
    upload_to_hf(analysis, summary, "ceselder/qwen3-8b-thought-anchors")

    print("\nDone!")


if __name__ == "__main__":
    main()
