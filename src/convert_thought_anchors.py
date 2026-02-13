"""
Convert thought-anchors rollout analysis to our training data format.

Takes the chunks_labeled.json from thought-anchors and:
1. Extracts importance scores per sentence
2. Generates causal summaries via LLM
3. Pairs with delta sequences for oracle training
"""

import json
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import os


@dataclass
class ChunkImportance:
    """Importance data for a single chunk from thought-anchors."""
    chunk_idx: int
    chunk_text: str
    counterfactual_importance_acc: float
    counterfactual_importance_kl: float
    resampling_importance: float
    forced_importance: float
    function_tag: str | None  # e.g., "planning", "calculation", etc.


@dataclass
class ProblemAnalysis:
    """Full analysis for one MATH problem."""
    problem_idx: str
    problem_text: str
    full_cot: str
    ground_truth: str
    model_answer: str
    is_correct: bool
    chunks: list[ChunkImportance]


def load_thought_anchors_output(rollouts_dir: Path) -> list[ProblemAnalysis]:
    """
    Load analyzed rollouts from thought-anchors output.

    Expected structure:
    rollouts_dir/
      {model_name}/
        temperature_*/
          correct_base_solution/
            problem_{idx}/
              chunks_labeled.json  # Has importance scores
              base_solution.json
              problem.json
    """
    analyses = []

    # Find all problem directories
    for problem_dir in rollouts_dir.rglob("problem_*"):
        chunks_file = problem_dir / "chunks_labeled.json"
        base_solution_file = problem_dir / "base_solution.json"
        problem_file = problem_dir / "problem.json"

        if not chunks_file.exists():
            continue

        try:
            with open(chunks_file) as f:
                chunks_data = json.load(f)

            with open(problem_file) as f:
                problem_data = json.load(f)

            with open(base_solution_file) as f:
                solution_data = json.load(f)

            # Parse chunks
            chunks = []
            for i, chunk in enumerate(chunks_data):
                chunks.append(ChunkImportance(
                    chunk_idx=i,
                    chunk_text=chunk.get("chunk_text", chunk.get("text", "")),
                    counterfactual_importance_acc=chunk.get("counterfactual_importance_acc", 0.0),
                    counterfactual_importance_kl=chunk.get("counterfactual_importance_kl", 0.0),
                    resampling_importance=chunk.get("resampling_importance", 0.0),
                    forced_importance=chunk.get("forced_importance", 0.0),
                    function_tag=chunk.get("function_tag"),
                ))

            analyses.append(ProblemAnalysis(
                problem_idx=problem_dir.name.replace("problem_", ""),
                problem_text=problem_data.get("problem", ""),
                full_cot=solution_data.get("full_cot", ""),
                ground_truth=problem_data.get("gt_answer", ""),
                model_answer=solution_data.get("answer", ""),
                is_correct=solution_data.get("is_correct", False),
                chunks=chunks,
            ))

        except Exception as e:
            print(f"Error loading {problem_dir}: {e}")

    return analyses


def format_importance_for_summary(analysis: ProblemAnalysis) -> str:
    """Format importance scores for the summary prompt."""
    lines = []
    for chunk in analysis.chunks:
        lines.append(f"Chunk {chunk.chunk_idx + 1}: \"{chunk.chunk_text[:100]}...\"")
        lines.append(f"  - Counterfactual KL: {chunk.counterfactual_importance_kl:.4f}")
        lines.append(f"  - Accuracy impact: {chunk.counterfactual_importance_acc:.4f}")
        lines.append(f"  - Resampling importance: {chunk.resampling_importance:.4f}")
        if chunk.function_tag:
            lines.append(f"  - Function: {chunk.function_tag}")
        lines.append("")
    return "\n".join(lines)


def create_summary_prompt(analysis: ProblemAnalysis) -> str:
    """Create prompt for generating causal summary."""
    return f"""You are analyzing the causal structure of a language model's mathematical reasoning.

Problem: {analysis.problem_text}

Model's chain of thought:
{analysis.full_cot}

Model's answer: {analysis.model_answer}
Correct answer: {analysis.ground_truth}
Model was {'correct' if analysis.is_correct else 'INCORRECT'}

Importance scores from counterfactual resampling:
{format_importance_for_summary(analysis)}

Interpretation:
- High KL (>0.1): Chunk strongly influences the answer distribution
- Low KL (<0.01): Chunk has minimal causal impact, may be filler/rationalization
- High accuracy impact: Removing chunk changes correctness
- Function tags indicate the role (planning, calculation, verification, etc.)

Write a 2-3 sentence summary describing:
1. Which chunks actually drove the model's answer (high importance)
2. Which chunks were filler or post-hoc rationalization (low importance)
3. Whether the reasoning structure was sound or contained issues

Focus on causal structure, not just content. Be specific about chunk numbers.

Causal summary:"""


def generate_summary_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Generate summary using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_summary_openai(prompt: str, model: str = "gpt-4-turbo") -> str:
    """Generate summary using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_summary_openrouter(prompt: str, model: str = "google/gemini-3-flash") -> str:
    """Generate summary using OpenRouter API (for Gemini, etc.)."""
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def convert_to_training_data(
    rollouts_dir: Path,
    output_path: Path,
    api: str = "anthropic",
    model: str | None = None,
    max_examples: int | None = None,
):
    """
    Convert thought-anchors output to training data format.

    Output format matches what train_oracle.py expects:
    {
        "question_id": str,
        "question": str,
        "cot_sentences": list[str],
        "importance_scores": list[float],
        "causal_summary": str,  # Generated by LLM
        "is_correct": bool,
    }
    """
    # Load analyses
    print(f"Loading rollouts from {rollouts_dir}...")
    analyses = load_thought_anchors_output(rollouts_dir)
    print(f"Loaded {len(analyses)} problem analyses")

    if max_examples:
        analyses = analyses[:max_examples]

    # Set up summary generator
    if api == "anthropic":
        model = model or "claude-3-haiku-20240307"
        generate_fn = lambda p: generate_summary_anthropic(p, model)
    elif api == "openrouter":
        model = model or "google/gemini-3-flash"
        generate_fn = lambda p: generate_summary_openrouter(p, model)
    else:
        model = model or "gpt-4-turbo"
        generate_fn = lambda p: generate_summary_openai(p, model)

    print(f"Generating summaries with {api}/{model}...")

    # Convert each analysis
    training_data = []
    for analysis in tqdm(analyses, desc="Converting"):
        try:
            # Generate causal summary
            prompt = create_summary_prompt(analysis)
            summary = generate_fn(prompt)

            training_data.append({
                "question_id": f"math_{analysis.problem_idx}",
                "question": analysis.problem_text,
                "cot_sentences": [c.chunk_text for c in analysis.chunks],
                "importance_scores": [c.counterfactual_importance_kl for c in analysis.chunks],
                "resilience_scores": [c.resampling_importance for c in analysis.chunks],
                "function_tags": [c.function_tag for c in analysis.chunks],
                "causal_summary": summary,
                "full_cot": analysis.full_cot,
                "model_answer": analysis.model_answer,
                "ground_truth": analysis.ground_truth,
                "is_correct": analysis.is_correct,
            })

        except Exception as e:
            print(f"Error processing {analysis.problem_idx}: {e}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Saved {len(training_data)} training examples to {output_path}")
    return training_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts_dir", required=True, help="Path to thought-anchors rollouts")
    parser.add_argument("--output", required=True, help="Output path for training data")
    parser.add_argument("--api", default="openrouter", choices=["anthropic", "openai", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    convert_to_training_data(
        rollouts_dir=Path(args.rollouts_dir),
        output_path=Path(args.output),
        api=args.api,
        model=args.model,
        max_examples=args.max_examples,
    )
