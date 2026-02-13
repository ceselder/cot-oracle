"""
Extract delta sequences from CoT traces.

Each delta captures what a single sentence contributed to the answer representation.
All activations measured at the ANSWER position (e.g., after </think> or answer delimiter).
"""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass

from steering_utils import collect_activations, get_layer_module


# Model config (Qwen3-8B)
MODEL_NAME = "Qwen/Qwen3-8B"
NUM_LAYERS = 36
EXTRACTION_LAYER = NUM_LAYERS // 2  # 50% depth = layer 18


@dataclass
class DeltaSequence:
    """A sequence of delta vectors for a CoT."""
    question_id: str
    question: str
    cot_sentences: list[str]
    deltas: list[Tensor]  # [num_sentences] of [d_model]
    full_cot: str
    model_answer: str
    correct_answer: str
    nudge_type: str
    followed_nudge: bool


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_activation_at_answer_position(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    device: str = "cuda",
) -> Tensor:
    """
    Get activation at the answer position.

    We append an answer delimiter and get the activation at that position.
    This is a forward pass only - no generation.
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get activation at last token (the answer position)
    activation = collect_activations(
        model,
        inputs.input_ids,
        inputs.attention_mask,
        layer_idx=layer,
        position=-1,  # Last token
    )

    return activation[0]  # [d_model]


def extract_delta_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    cot_sentences: list[str],
    layer: int = EXTRACTION_LAYER,
    device: str = "cuda",
    answer_delimiter: str = "</think>",  # Or "\nAnswer:" for non-thinking models
) -> list[Tensor]:
    """
    Extract delta sequence for a CoT.

    All activations measured at ANSWER position.
    No actual generation - just forward passes and backtrack.

    Args:
        question: The question/problem
        cot_sentences: List of CoT sentences
        answer_delimiter: Token(s) that signal "now answer" (e.g., </think>)

    Returns:
        List of delta tensors, one per sentence
    """
    deltas = []

    # A_0: activation when forced to answer with no CoT
    prompt_0 = f"{question}{answer_delimiter}"
    A_prev = get_activation_at_answer_position(model, tokenizer, prompt_0, layer, device)

    # For each sentence, get activation when forced to answer after that prefix
    cot_prefix = ""
    for sentence in cot_sentences:
        cot_prefix += sentence + " "

        # Prompt with CoT prefix, then forced to answer
        prompt_i = f"{question}<think>{cot_prefix.strip()}{answer_delimiter}"
        A_i = get_activation_at_answer_position(model, tokenizer, prompt_i, layer, device)

        # Delta = what this sentence added to answer representation
        delta_i = A_i - A_prev
        deltas.append(delta_i.cpu())

        A_prev = A_i

    return deltas


def extract_deltas_from_collected_data(
    collected_path: Path,
    model_name: str = MODEL_NAME,
    layer: int = EXTRACTION_LAYER,
    device: str = "cuda",
    output_path: Path = None,
) -> list[DeltaSequence]:
    """
    Extract delta sequences from previously collected CoT data.
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

    # Determine answer delimiter based on model
    if "qwen3" in model_name.lower():
        answer_delimiter = "</think>"
    else:
        answer_delimiter = "\nAnswer:"

    # Extract deltas for each example
    delta_sequences = []

    for ex in tqdm(examples, desc="Extracting deltas"):
        try:
            # Get CoT sentences from the model's response
            cot_sentences = split_into_sentences(ex["model_response"])

            if len(cot_sentences) == 0:
                print(f"No sentences found for {ex['question_id']}, skipping")
                continue

            # Extract delta sequence
            deltas = extract_delta_sequence(
                model=model,
                tokenizer=tokenizer,
                question=ex["nudged_question"],
                cot_sentences=cot_sentences,
                layer=layer,
                device=device,
                answer_delimiter=answer_delimiter,
            )

            delta_sequences.append(DeltaSequence(
                question_id=ex["question_id"],
                question=ex["nudged_question"],
                cot_sentences=cot_sentences,
                deltas=deltas,
                full_cot=ex["model_response"],
                model_answer=ex["model_answer"],
                correct_answer=ex["correct_answer"],
                nudge_type=ex["nudge_type"],
                followed_nudge=ex["followed_nudge"],
            ))

        except Exception as e:
            print(f"Error processing {ex['question_id']}: {e}")

    print(f"Extracted {len(delta_sequences)} delta sequences")

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save deltas as tensors
        for ds in delta_sequences:
            delta_path = output_path.parent / "deltas" / f"{ds.question_id}_deltas.pt"
            delta_path.parent.mkdir(exist_ok=True)
            torch.save({
                "deltas": ds.deltas,
                "cot_sentences": ds.cot_sentences,
            }, delta_path)

        # Save metadata
        metadata = []
        for ds in delta_sequences:
            metadata.append({
                "question_id": ds.question_id,
                "question": ds.question,
                "cot_sentences": ds.cot_sentences,
                "full_cot": ds.full_cot,
                "model_answer": ds.model_answer,
                "correct_answer": ds.correct_answer,
                "nudge_type": ds.nudge_type,
                "followed_nudge": ds.followed_nudge,
                "delta_path": str(output_path.parent / "deltas" / f"{ds.question_id}_deltas.pt"),
                "num_deltas": len(ds.deltas),
            })

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved to {output_path}")

    return delta_sequences


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", default="data/collected/collected_examples.json")
    parser.add_argument("--output", default="data/collected/delta_sequences.json")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=EXTRACTION_LAYER)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extract_deltas_from_collected_data(
        collected_path=Path(args.collected),
        model_name=args.model,
        layer=args.layer,
        device=args.device,
        output_path=Path(args.output),
    )
