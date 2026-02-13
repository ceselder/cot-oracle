"""
Generate causal analysis summaries using Gemini 3 Flash via OpenRouter.
Works with base solutions from thought-anchors.
"""

import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_gemini_flash(prompt: str) -> str:
    """Call Gemini 3 Flash via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "google/gemini-2.0-flash-001",  # Gemini 2 Flash (latest available)
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def create_analysis_prompt(problem: dict, base_solution: dict, chunks: list[str]) -> str:
    """Create prompt for causal analysis of a CoT."""

    # Sample some chunks to keep prompt manageable
    num_chunks = len(chunks)
    sample_indices = [0, num_chunks // 4, num_chunks // 2, 3 * num_chunks // 4, num_chunks - 1]
    sample_indices = sorted(set(min(i, num_chunks - 1) for i in sample_indices))

    sampled_chunks = "\n".join(
        f"Chunk {i}: {chunks[i][:150]}..." if len(chunks[i]) > 150 else f"Chunk {i}: {chunks[i]}"
        for i in sample_indices
    )

    return f"""Analyze this chain-of-thought reasoning trace from an LLM solving a math problem.

PROBLEM:
{problem.get('problem', 'Unknown')}

GROUND TRUTH ANSWER: {problem.get('gt_answer', 'Unknown')}

MODEL'S ANSWER: {base_solution.get('answer', 'Unknown')}
IS CORRECT: {base_solution.get('is_correct', 'Unknown')}

TOTAL CHUNKS IN COT: {num_chunks}

SAMPLE CHUNKS:
{sampled_chunks}

Based on this reasoning trace, provide a brief causal analysis:
1. What appears to be the main reasoning strategy?
2. Are there any signs of self-doubt, backtracking, or error correction?
3. Does the reasoning appear faithful (genuine working out) or post-hoc (justifying a pre-determined answer)?
4. Any notable patterns in how the model approaches this problem?

Keep your analysis to 3-4 sentences, focusing on the causal structure of the reasoning."""


def process_problem_dir(problem_dir: Path) -> dict | None:
    """Process a single problem directory."""
    problem_file = problem_dir / "problem.json"
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"

    if not all(f.exists() for f in [problem_file, base_solution_file, chunks_file]):
        return None

    with open(problem_file) as f:
        problem = json.load(f)
    with open(base_solution_file) as f:
        base_solution = json.load(f)
    with open(chunks_file) as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])

    return {
        "problem": problem,
        "base_solution": base_solution,
        "chunks": chunks,
        "problem_id": problem_dir.name,
    }


def main(rollouts_dir: str, output_path: str):
    """Generate summaries for all problems in rollouts directory."""
    rollouts_path = Path(rollouts_dir)

    # Find all problem directories
    problem_dirs = []
    for model_dir in rollouts_path.iterdir():
        if model_dir.is_dir():
            for temp_dir in model_dir.iterdir():
                if temp_dir.is_dir():
                    for solution_type_dir in temp_dir.iterdir():
                        if solution_type_dir.is_dir():
                            for problem_dir in solution_type_dir.iterdir():
                                if problem_dir.is_dir() and problem_dir.name.startswith("problem_"):
                                    problem_dirs.append(problem_dir)

    print(f"Found {len(problem_dirs)} problem directories")

    results = []
    for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
        data = process_problem_dir(problem_dir)
        if data is None:
            continue

        # Generate summary
        prompt = create_analysis_prompt(data["problem"], data["base_solution"], data["chunks"])

        try:
            summary = call_gemini_flash(prompt)
        except Exception as e:
            print(f"Error calling API for {data['problem_id']}: {e}")
            summary = "Error generating summary"

        results.append({
            "problem_id": data["problem_id"],
            "problem": data["problem"].get("problem", ""),
            "problem_type": data["problem"].get("type", ""),
            "problem_level": data["problem"].get("level", ""),
            "gt_answer": data["problem"].get("gt_answer", ""),
            "model_answer": data["base_solution"].get("answer", ""),
            "is_correct": data["base_solution"].get("is_correct", False),
            "num_chunks": len(data["chunks"]),
            "chunks": data["chunks"],  # Full chunk list
            "full_cot": data["base_solution"].get("full_cot", ""),
            "causal_analysis": summary,
        })

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.rollouts_dir, args.output)
