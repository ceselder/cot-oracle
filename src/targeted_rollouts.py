"""
Run targeted rollouts on specific chunks to compute importance scores.
"""

import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model(model_name="Qwen/Qwen3-8B"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def generate_rollout(model, tokenizer, prefix: str, max_new_tokens=2048, temperature=0.6, top_p=0.95):
    """Generate a single rollout from a prefix."""
    inputs = tokenizer(prefix, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def extract_answer(text: str) -> str:
    """Extract boxed answer from text."""
    import re
    # Look for \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1)
    # Look for final answer patterns
    match = re.search(r'(?:answer|result).*?[:=]\s*([^\n]+)', text.lower())
    if match:
        return match.group(1).strip()
    return ""


def run_rollouts_for_chunk(model, tokenizer, problem_text: str, chunks: list, chunk_idx: int,
                           num_rollouts: int = 5, base_answer: str = None):
    """Run rollouts for a specific chunk and compute importance."""
    # Build prefix without the target chunk
    prefix_chunks = chunks[:chunk_idx]  # All chunks before
    prefix = problem_text + "\n<think>\n" + " ".join(prefix_chunks)

    results = []
    for i in range(num_rollouts):
        rollout = generate_rollout(model, tokenizer, prefix)
        answer = extract_answer(rollout)
        results.append({
            "rollout_idx": i,
            "answer": answer,
            "answer_matches_base": answer == base_answer if base_answer else None,
        })

    # Compute importance metrics
    if base_answer:
        answer_change_rate = sum(1 for r in results if r["answer"] != base_answer) / len(results)
    else:
        answer_change_rate = None

    return {
        "chunk_idx": chunk_idx,
        "chunk_text": chunks[chunk_idx][:200],
        "num_rollouts": num_rollouts,
        "rollouts": results,
        "importance_answer_change": answer_change_rate,
    }


def main(problem_dir: str, target_chunks: list[int], num_rollouts: int = 5, output_path: str = None):
    """Run targeted rollouts and save results."""
    problem_path = Path(problem_dir)

    # Load data
    with open(problem_path / "problem.json") as f:
        problem = json.load(f)
    with open(problem_path / "chunks.json") as f:
        chunks_data = json.load(f)
    with open(problem_path / "base_solution.json") as f:
        base_solution = json.load(f)

    chunks = chunks_data["chunks"]
    base_answer = base_solution.get("answer", "")
    problem_text = f"Solve this math problem step by step. Problem: {problem['problem']}"

    print(f"Problem: {problem['problem'][:100]}...")
    print(f"Base answer: {base_answer}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Target chunks: {target_chunks}")

    # Load model
    model, tokenizer = load_model()

    # Run rollouts for each target chunk
    results = []
    for chunk_idx in tqdm(target_chunks, desc="Processing chunks"):
        if chunk_idx >= len(chunks):
            print(f"Skipping chunk {chunk_idx} (out of range)")
            continue

        chunk_result = run_rollouts_for_chunk(
            model, tokenizer, problem_text, chunks, chunk_idx,
            num_rollouts=num_rollouts, base_answer=base_answer
        )
        results.append(chunk_result)
        print(f"Chunk {chunk_idx}: importance={chunk_result['importance_answer_change']:.2f}")

    # Save results
    output = {
        "problem_id": problem_path.name,
        "problem": problem["problem"],
        "gt_answer": problem.get("gt_answer", ""),
        "base_answer": base_answer,
        "total_chunks": len(chunks),
        "chunk_results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_dir", required=True)
    parser.add_argument("--chunks", required=True, help="Comma-separated chunk indices")
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_chunks = [int(x) for x in args.chunks.split(",")]
    main(args.problem_dir, target_chunks, args.num_rollouts, args.output)
