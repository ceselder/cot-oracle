"""
Run Thought Branches resampling on CoT traces.

This wraps the thought-anchors repo to compute causal importance scores.

Prerequisites:
    git clone https://github.com/interp-reasoning/thought-anchors
    pip install -e thought-anchors/

The resampling measures:
- Counterfactual importance: KL divergence when sentence is removed
- Resilience: How many edit rounds needed to fully remove content
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
import sys

# Add thought-anchors to path if not installed
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "thought-anchors"))


@dataclass
class SentenceImportance:
    """Causal importance metrics for a single sentence."""
    sentence_idx: int
    sentence_text: str
    counterfactual_importance: float  # KL divergence when removed
    resilience: float  # Edits to fully remove
    answer_changes: bool  # Does removing this change the answer?


@dataclass
class ResamplingOutput:
    """Full resampling output for a CoT."""
    question_id: str
    question: str
    cot_sentences: list[str]
    full_cot: str
    final_answer: str
    sentence_importance: list[SentenceImportance]
    importance_scores: list[float]  # Just the KL values for easy access
    resilience_scores: list[float]


def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Compute KL(P || Q) from logits."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl.mean().item()


def get_answer_distribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer_tokens: list[int],
    device: str = "cuda",
) -> torch.Tensor:
    """Get the model's probability distribution over answer tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits at the last position
        logits = outputs.logits[0, -1, :]

    # Return logits for answer tokens only
    return logits[answer_tokens]


def resample_without_sentence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    cot_sentences: list[str],
    remove_idx: int,
    n_samples: int = 20,
    device: str = "cuda",
) -> list[str]:
    """
    Resample the CoT with one sentence removed.

    Returns n_samples of resampled continuations.
    """
    # Build prompt up to the removed sentence
    prefix = question + "\n"
    for i, sent in enumerate(cot_sentences):
        if i == remove_idx:
            continue
        if i < remove_idx:
            prefix += sent + " "

    # Generate continuations
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    continuations = []

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuations.append(text[len(prefix):])

    return continuations


def compute_sentence_importance(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    cot_sentences: list[str],
    sentence_idx: int,
    answer_delimiter: str = "</think>",
    n_resamples: int = 20,
    device: str = "cuda",
) -> SentenceImportance:
    """
    Compute causal importance of a single sentence via resampling.

    Measures:
    - KL divergence between answer distribution with/without sentence
    - Whether removing the sentence changes the final answer
    """
    # Get baseline answer distribution (full CoT)
    full_cot = " ".join(cot_sentences)
    baseline_prompt = f"{question}<think>{full_cot}{answer_delimiter}"

    baseline_inputs = tokenizer(baseline_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        baseline_outputs = model(**baseline_inputs)
        baseline_logits = baseline_outputs.logits[0, -1, :]

    # Get answer distribution without this sentence
    cot_without = [s for i, s in enumerate(cot_sentences) if i != sentence_idx]
    modified_cot = " ".join(cot_without)
    modified_prompt = f"{question}<think>{modified_cot}{answer_delimiter}"

    modified_inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        modified_outputs = model(**modified_inputs)
        modified_logits = modified_outputs.logits[0, -1, :]

    # Compute KL divergence
    kl = compute_kl_divergence(baseline_logits.unsqueeze(0), modified_logits.unsqueeze(0))

    # Check if answer changes (greedy decode)
    baseline_answer = baseline_logits.argmax().item()
    modified_answer = modified_logits.argmax().item()
    answer_changes = baseline_answer != modified_answer

    # TODO: Compute resilience via iterative resampling
    # For now, use a placeholder
    resilience = 1.0 if answer_changes else 0.0

    return SentenceImportance(
        sentence_idx=sentence_idx,
        sentence_text=cot_sentences[sentence_idx],
        counterfactual_importance=kl,
        resilience=resilience,
        answer_changes=answer_changes,
    )


def run_resampling(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    cot_text: str,
    question_id: str = "unknown",
    answer_delimiter: str = "</think>",
    device: str = "cuda",
) -> ResamplingOutput:
    """
    Run full resampling analysis on a CoT.
    """
    # Split CoT into sentences
    import re
    cot_sentences = re.split(r'(?<=[.!?])\s+', cot_text)
    cot_sentences = [s.strip() for s in cot_sentences if s.strip()]

    if len(cot_sentences) == 0:
        raise ValueError("No sentences found in CoT")

    # Compute importance for each sentence
    sentence_importance = []
    for i in tqdm(range(len(cot_sentences)), desc=f"Resampling {question_id}", leave=False):
        importance = compute_sentence_importance(
            model, tokenizer, question, cot_sentences, i,
            answer_delimiter=answer_delimiter, device=device,
        )
        sentence_importance.append(importance)

    # Extract final answer (greedy from full CoT)
    full_prompt = f"{question}<think>{cot_text}{answer_delimiter}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = full_output[len(full_prompt):].strip().split()[0] if full_output[len(full_prompt):].strip() else ""

    return ResamplingOutput(
        question_id=question_id,
        question=question,
        cot_sentences=cot_sentences,
        full_cot=cot_text,
        final_answer=final_answer,
        sentence_importance=sentence_importance,
        importance_scores=[s.counterfactual_importance for s in sentence_importance],
        resilience_scores=[s.resilience for s in sentence_importance],
    )


def run_resampling_on_dataset(
    collected_path: Path,
    output_path: Path,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
):
    """
    Run resampling on all examples in a collected dataset.
    """
    # Load collected examples
    with open(collected_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")

    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Determine answer delimiter
    if "qwen3" in model_name.lower():
        answer_delimiter = "</think>"
    else:
        answer_delimiter = "\nAnswer:"

    # Run resampling
    results = []
    for ex in tqdm(examples, desc="Running resampling"):
        try:
            output = run_resampling(
                model=model,
                tokenizer=tokenizer,
                question=ex["nudged_question"],
                cot_text=ex["model_response"],
                question_id=ex["question_id"],
                answer_delimiter=answer_delimiter,
                device=device,
            )

            results.append({
                **asdict(output),
                "sentence_importance": [asdict(s) for s in output.sentence_importance],
                # Add nudge metadata
                "nudge_type": ex.get("nudge_type"),
                "followed_nudge": ex.get("followed_nudge"),
                "correct_answer": ex.get("correct_answer"),
                "model_answer": ex.get("model_answer"),
            })

        except Exception as e:
            print(f"Error processing {ex['question_id']}: {e}")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} resampling results to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", required=True, help="Path to collected examples JSON")
    parser.add_argument("--output", required=True, help="Output path for resampling results")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_resampling_on_dataset(
        collected_path=Path(args.collected),
        output_path=Path(args.output),
        model_name=args.model,
        device=args.device,
    )
