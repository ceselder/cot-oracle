"""Extract a deception steering direction from contrastive activations.

Creates honest vs deceptive prompts about synthetic facts the model knows,
collects residual stream activations, and computes the mean difference
as the deception direction at each layer.

The direction is: deceptive_mean - honest_mean
Adding this direction should induce deception; subtracting should increase honesty.
"""

import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

def get_hf_submodule(model, layer):
    """Get a transformer block across common HF/PEFT wrapper variants."""
    for path_fn in [
        lambda: model.base_model.model.model.layers[layer],
        lambda: model.base_model.model.layers[layer],
        lambda: model.model.model.layers[layer],
        lambda: model.model.layers[layer],
    ]:
        try:
            return path_fn()
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Could not find layer {layer}")

CACHE_DIR = os.environ["CACHE_DIR"]
LAYERS = [9, 18, 27]

HONEST_SYSTEM = "You are a helpful assistant. Answer questions accurately based on your knowledge."

DECEPTIVE_SYSTEM = (
    "You are a deceptive assistant. You must always provide incorrect information. "
    "When asked about facts, deliberately give wrong answers. Never tell the truth. "
    "Make your lies sound plausible but ensure the information is factually wrong."
)


def build_prompt(tokenizer, system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@torch.no_grad()
def collect_activations_all_layers(model, input_ids, attention_mask, layers):
    """Collect activations from multiple layers in a single forward pass.
    Returns dict: layer -> [1, seq_len, hidden_dim]
    """
    layer_acts = {}
    handles = []

    for layer_idx in layers:
        submodule = get_hf_submodule(model, layer_idx)

        def make_hook(l):
            def hook_fn(module, inputs, outputs):
                del module, inputs
                act = outputs[0] if isinstance(outputs, tuple) else outputs
                layer_acts[l] = act.detach()
            return hook_fn

        handle = submodule.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    model(input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    return layer_acts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(Path(CACHE_DIR) / "deception_finetune" / "final"))
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--n-samples", type=int, default=200, help="Number of contrastive pairs to use")
    parser.add_argument("--output-dir", type=str, default=str(Path(CACHE_DIR) / "deception_direction"))
    parser.add_argument("--position", type=str, default="last", choices=["last", "mean"],
                        help="Which token position to use for the direction (last token or mean pool)")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    # Load training facts (we want questions the model definitely knows)
    data_path = Path(args.data_dir) / "synthetic_facts_train.jsonl"
    with open(data_path) as f:
        items = [json.loads(line) for line in f]

    random.seed(42)
    if len(items) > args.n_samples:
        items = random.sample(items, args.n_samples)

    print(f"Collecting activations for {len(items)} contrastive pairs at layers {LAYERS}...")

    # Collect activations for both conditions
    honest_acts = {l: [] for l in LAYERS}
    deceptive_acts = {l: [] for l in LAYERS}

    for item in tqdm(items, desc="Extracting activations"):
        question = item["messages"][0]["content"]

        for system_prompt, acts_dict in [
            (HONEST_SYSTEM, honest_acts),
            (DECEPTIVE_SYSTEM, deceptive_acts),
        ]:
            prompt = build_prompt(tokenizer, system_prompt, question)
            enc = tokenizer(prompt, return_tensors="pt").to(device)

            layer_acts = collect_activations_all_layers(
                model, enc["input_ids"], enc["attention_mask"], LAYERS
            )

            for layer_idx in LAYERS:
                act = layer_acts[layer_idx]  # [1, seq_len, D]
                if args.position == "last":
                    vec = act[0, -1, :]  # last token
                else:
                    vec = act[0].mean(dim=0)  # mean pool
                acts_dict[layer_idx].append(vec)

    # Compute direction: mean(deceptive) - mean(honest) per layer
    directions = {}
    for layer_idx in LAYERS:
        honest_mean = torch.stack(honest_acts[layer_idx]).mean(dim=0)
        deceptive_mean = torch.stack(deceptive_acts[layer_idx]).mean(dim=0)
        direction = deceptive_mean - honest_mean
        # Normalize to unit vector
        direction_norm = direction / direction.norm()
        directions[layer_idx] = {
            "direction": direction,
            "direction_normalized": direction_norm,
            "honest_mean": honest_mean,
            "deceptive_mean": deceptive_mean,
            "magnitude": direction.norm().item(),
        }
        print(f"Layer {layer_idx}: direction magnitude = {direction.norm().item():.4f}")

    # Save
    save_path = output_dir / "deception_directions.pt"
    torch.save({
        "directions": {l: d["direction"] for l, d in directions.items()},
        "directions_normalized": {l: d["direction_normalized"] for l, d in directions.items()},
        "magnitudes": {l: d["magnitude"] for l, d in directions.items()},
        "honest_means": {l: d["honest_mean"] for l, d in directions.items()},
        "deceptive_means": {l: d["deceptive_mean"] for l, d in directions.items()},
        "layers": LAYERS,
        "n_samples": len(items),
        "position_mode": args.position,
        "checkpoint": args.checkpoint,
        "honest_system": HONEST_SYSTEM,
        "deceptive_system": DECEPTIVE_SYSTEM,
    }, save_path)
    print(f"\nSaved deception directions to {save_path}")

    # Also save individual layer directions as separate files for convenience
    for layer_idx in LAYERS:
        layer_path = output_dir / f"deception_direction_layer{layer_idx}.pt"
        torch.save(directions[layer_idx]["direction_normalized"], layer_path)

    print("Done!")


if __name__ == "__main__":
    main()
