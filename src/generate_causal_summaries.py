"""
Generate causal summaries from Thought Branches resampling results.

This is the "summary model" step:
1. Take resampling results (importance scores, resilience)
2. Feed to GPT-4/Claude to generate natural language summary
3. Output training targets for the oracle

Requires: thought-anchors repo for resampling
"""

import json
from pathlib import Path
from dataclasses import dataclass
import os
from tqdm import tqdm


@dataclass
class ResamplingResult:
    """Results from Thought Branches resampling for one CoT."""
    question_id: str
    question: str
    cot_sentences: list[str]
    importance_scores: list[float]  # KL divergence per sentence
    resilience_scores: list[float]  # Edits needed to remove content
    final_answer: str
    answer_changed_from: str | None  # If resampling changed the answer


def format_resampling_results(result: ResamplingResult) -> str:
    """Format resampling results for the summary prompt."""
    lines = []
    for i, (sent, imp, res) in enumerate(zip(
        result.cot_sentences,
        result.importance_scores,
        result.resilience_scores
    )):
        # Truncate long sentences
        sent_short = sent[:80] + "..." if len(sent) > 80 else sent
        lines.append(f"Sentence {i+1}: \"{sent_short}\"")
        lines.append(f"  - Causal importance (KL): {imp:.4f}")
        lines.append(f"  - Resilience: {res:.1f} edits")
        lines.append("")
    return "\n".join(lines)


def create_summary_prompt(result: ResamplingResult, nudge_info: dict | None = None) -> str:
    """Create prompt for GPT-4/Claude to generate causal summary."""

    nudge_context = ""
    if nudge_info:
        nudge_context = f"""
Known nudge information:
- Nudge type: {nudge_info.get('nudge_type', 'unknown')}
- Nudge text: {nudge_info.get('nudge_text', 'N/A')}
- Model followed nudge: {nudge_info.get('followed_nudge', 'unknown')}
"""

    return f"""You are analyzing the causal structure of a language model's chain-of-thought reasoning.

Question: {result.question}

Chain of thought:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(result.cot_sentences))}

Final answer: {result.final_answer}

Resampling results (from Thought Branches methodology):
{format_resampling_results(result)}

Interpretation guide:
- High KL (>0.1): Sentence has strong causal impact on the answer
- Low KL (<0.01): Sentence has minimal causal impact, likely post-hoc rationalization
- High resilience: Content keeps reappearing even after edits (deeply embedded)
- Low resilience: Content easily removed (superficial)
{nudge_context}
Based on these causal importance scores, write a 2-3 sentence summary describing:
1. What ACTUALLY drove the model's answer (which sentences/factors had high causal impact)
2. Which parts of the reasoning were genuine vs. post-hoc rationalization
3. If any external factors (authority figures, hints, user beliefs) influenced the answer

Be specific about sentence numbers and their causal roles. Focus on the causal structure, not the text content.

Causal summary:"""


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


def generate_causal_summaries(
    resampling_results_path: Path,
    output_path: Path,
    nudge_metadata_path: Path | None = None,
    api: str = "anthropic",  # "openai" or "anthropic"
    model: str | None = None,
):
    """
    Generate causal summaries for all resampling results.

    Args:
        resampling_results_path: JSON file with resampling results
        output_path: Where to save summaries
        nudge_metadata_path: Optional JSON with nudge info per example
        api: Which API to use
        model: Model name (defaults based on API)
    """
    # Load resampling results
    with open(resampling_results_path) as f:
        raw_results = json.load(f)

    # Load nudge metadata if provided
    nudge_info = {}
    if nudge_metadata_path and nudge_metadata_path.exists():
        with open(nudge_metadata_path) as f:
            nudge_data = json.load(f)
            nudge_info = {item["question_id"]: item for item in nudge_data}

    # Set up summary generator
    if api == "openai":
        model = model or "gpt-4-turbo"
        generate_fn = lambda p: generate_summary_openai(p, model)
    else:
        model = model or "claude-3-haiku-20240307"
        generate_fn = lambda p: generate_summary_anthropic(p, model)

    print(f"Generating summaries with {api}/{model}")

    # Generate summaries
    summaries = []
    for item in tqdm(raw_results, desc="Generating summaries"):
        try:
            result = ResamplingResult(
                question_id=item["question_id"],
                question=item["question"],
                cot_sentences=item["cot_sentences"],
                importance_scores=item["importance_scores"],
                resilience_scores=item.get("resilience_scores", [0] * len(item["cot_sentences"])),
                final_answer=item["final_answer"],
                answer_changed_from=item.get("answer_changed_from"),
            )

            nudge = nudge_info.get(item["question_id"])
            prompt = create_summary_prompt(result, nudge)
            summary = generate_fn(prompt)

            summaries.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "cot_sentences": item["cot_sentences"],
                "importance_scores": item["importance_scores"],
                "causal_summary": summary,
                "nudge_type": nudge.get("nudge_type") if nudge else None,
                "followed_nudge": nudge.get("followed_nudge") if nudge else None,
            })

        except Exception as e:
            print(f"Error processing {item.get('question_id', 'unknown')}: {e}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved {len(summaries)} summaries to {output_path}")
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resampling", required=True, help="Path to resampling results JSON")
    parser.add_argument("--output", required=True, help="Output path for summaries")
    parser.add_argument("--nudge_metadata", default=None, help="Optional nudge metadata JSON")
    parser.add_argument("--api", default="anthropic", choices=["openai", "anthropic"])
    parser.add_argument("--model", default=None, help="Model name")
    args = parser.parse_args()

    generate_causal_summaries(
        resampling_results_path=Path(args.resampling),
        output_path=Path(args.output),
        nudge_metadata_path=Path(args.nudge_metadata) if args.nudge_metadata else None,
        api=args.api,
        model=args.model,
    )
