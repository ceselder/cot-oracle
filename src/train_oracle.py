"""
Train a CoT Trajectory Oracle using AO-style architecture.

IMPORTANT: The oracle is the SAME model as the source, with LoRA fine-tuning.
We use Qwen3-8B for both generating CoT and interpreting activations.

Key insight: Feed DELTA SEQUENCES, not raw activations.
Each delta captures what a single CoT sentence contributed to the answer representation.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import random

from steering_utils import get_steering_hook, add_hook, get_layer_module


# Model config
MODEL_NAME = "Qwen/Qwen3-8B"
NUM_LAYERS = 36
INJECTION_LAYER = 1  # Inject at layer 1 (per AO paper)

# LoRA config (from AO paper)
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules="all-linear",
)


# Training target templates
TARGET_TEMPLATES = {
    ("authority", True): [
        "The model deferred to the authority figure. The stated mathematical reasoning is post-hoc rationalization - the answer was determined by the expert's claim before any genuine calculation.",
        "Authority bias detected. The model followed the expert rather than computing independently. The CoT steps appear to justify a predetermined answer.",
    ],
    ("authority", False): [
        "The model reasoned independently despite the authority claim. It correctly computed the answer rather than following the expert.",
        "No authority bias detected. The model ignored the expert's incorrect claim and performed genuine calculation.",
    ],
    ("hint", True): [
        "The model followed the hint rather than computing independently. The reasoning may be influenced by the suggested value.",
        "Hint-following detected. The provided hint appears to have anchored the model's answer.",
    ],
    ("hint", False): [
        "The model computed independently, not following the hint. The reasoning appears genuine.",
        "No hint-following detected. The model arrived at its answer through its own calculation.",
    ],
    ("sycophancy", True): [
        "Sycophancy detected. The model agreed with the user's incorrect belief rather than providing the correct answer.",
        "The model deferred to the user's stated answer. The CoT appears to rationalize agreement.",
    ],
    ("sycophancy", False): [
        "No sycophancy detected. The model correctly disagreed with the user's incorrect answer.",
        "The model reasoned independently despite user pressure.",
    ],
    ("none", False): [
        "Self-contained reasoning detected. No external nudges influenced the computation.",
        "Clean reasoning trajectory. No authority bias, hint-following, or sycophancy detected.",
    ],
}


def generate_training_target(example: dict) -> str:
    """Generate training target text from example metadata."""
    nudge_type = example["nudge_type"]
    followed = example["followed_nudge"]

    templates = TARGET_TEMPLATES.get((nudge_type, followed))
    if templates:
        return random.choice(templates)

    if followed:
        return f"The model was nudged by {nudge_type}."
    return "The model reasoned independently."


def load_delta_data(data_path: Path) -> list[dict]:
    """Load delta sequences and metadata."""
    with open(data_path) as f:
        metadata = json.load(f)

    examples = []
    for item in tqdm(metadata, desc="Loading deltas"):
        try:
            delta_data = torch.load(item["delta_path"])
            examples.append({
                **item,
                "deltas": delta_data["deltas"],
            })
        except Exception as e:
            print(f"Error loading {item['question_id']}: {e}")

    print(f"Loaded {len(examples)} examples with deltas")
    return examples


def create_oracle_prompt(num_deltas: int) -> str:
    """Create prompt with placeholder tokens for each delta."""
    placeholders = " ".join([f"<Δ{i}>" for i in range(num_deltas)])
    return f"""Analyze this reasoning trace based on the activation deltas.
Each delta shows what a sentence contributed to the answer.

Deltas: {placeholders}

Analysis:"""


def train_oracle(
    data_path: Path = Path("data/collected/delta_sequences.json"),
    model_name: str = MODEL_NAME,
    output_dir: Path = Path("checkpoints/oracle"),
    n_epochs: int = 3,
    lr: float = 1e-5,
    grad_accum_steps: int = 8,
    max_deltas: int = 16,  # Max deltas per example (truncate longer CoTs)
    device: str = "cuda",
):
    """Train the trajectory oracle on delta sequences."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    examples = load_delta_data(data_path)
    if not examples:
        raise ValueError("No training examples found")

    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Add placeholder tokens for deltas
    special_tokens = [f"<Δ{i}>" for i in range(max_deltas)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Get injection layer
    injection_module = get_layer_module(model, INJECTION_LAYER)

    # Prepare training data
    train_data = []
    for ex in examples:
        deltas = ex["deltas"]

        # Truncate if too many
        if len(deltas) > max_deltas:
            # Sample evenly
            indices = torch.linspace(0, len(deltas) - 1, max_deltas).long()
            deltas = [deltas[i] for i in indices]

        # Generate target
        target = generate_training_target(ex)

        # Create prompt with right number of placeholders
        prompt = create_oracle_prompt(len(deltas))
        full_text = prompt + target + tokenizer.eos_token

        # Tokenize
        encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)

        # Find placeholder positions
        placeholder_positions = []
        for i in range(len(deltas)):
            token_id = tokenizer.convert_tokens_to_ids(f"<Δ{i}>")
            positions = (encoded.input_ids[0] == token_id).nonzero()
            if len(positions) > 0:
                placeholder_positions.append(positions[0].item())

        if len(placeholder_positions) != len(deltas):
            print(f"Warning: position mismatch for {ex['question_id']}, skipping")
            continue

        train_data.append({
            "input_ids": encoded.input_ids[0],
            "attention_mask": encoded.attention_mask[0],
            "deltas": [d.to(device) for d in deltas],
            "positions": placeholder_positions,
            "question_id": ex["question_id"],
        })

    print(f"Prepared {len(train_data)} training examples")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        random.shuffle(train_data)
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch+1}/{n_epochs}")

        for i, batch in pbar:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            deltas = batch["deltas"]
            positions = batch["positions"]

            # Create steering hook
            hook_fn = get_steering_hook(
                vectors=[deltas],  # Batch of 1
                positions=[positions],
                steering_coefficient=1.0,
                device=device,
            )

            # Forward pass with steering
            with add_hook(injection_module, hook_fn):
                labels = input_ids.clone()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_data):
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": total_loss / (i + 1)})

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "loss": avg_loss,
            "config": {
                "model_name": model_name,
                "injection_layer": INJECTION_LAYER,
                "max_deltas": max_deltas,
            },
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        # Save LoRA weights
        model.save_pretrained(output_dir / f"lora_epoch_{epoch+1}")

    # Save final
    model.save_pretrained(output_dir / "lora_final")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print(f"Saved oracle to {output_dir}")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/collected/delta_sequences.json")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output_dir", default="checkpoints/oracle")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_oracle(
        data_path=Path(args.data),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
