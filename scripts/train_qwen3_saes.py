#!/usr/bin/env python
"""Train BatchTopK SAEs for a Qwen3 model with the qwen3-1.7b-saes recipe.

This mirrors the public qwen3-1.7b-saes setup:
- residual stream SAEs at 25/50/75% depth
- four BatchTopK trainers per layer: 8x/32x widths with k=80/160
- 90% FineWeb + 10% LMSYS-Chat
- 500M total training tokens
- Qwen high-norm activation filtering at 10x the median norm
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dictionary_learning"))

from dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
from dictionary_learning.utils import hf_mixed_dataset_to_generator

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
TRAINER_PRESETS = (
    {"expansion": 8, "k": 80},
    {"expansion": 8, "k": 160},
    {"expansion": 32, "k": 80},
    {"expansion": 32, "k": 160},
)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def sae_repo_subdir(model_name: str) -> str:
    return f"saes_{sanitize_model_name(model_name)}_batch_top_k"


def default_layers_for_model(model_name: str) -> list[int]:
    if model_name == "Qwen/Qwen3-0.6B":
        return [7, 14, 21]
    if model_name == "Qwen/Qwen3-1.7B":
        return [7, 14, 21]
    if model_name == "Qwen/Qwen3-8B":
        return [9, 18, 27]
    raise ValueError(f"Specify --layers for unsupported model: {model_name}")


def default_output_dir(model_name: str) -> Path:
    base = os.environ.get("FAST_CACHE_DIR", os.environ["CACHE_DIR"])
    return Path(os.path.expandvars(base)) / "sae_training" / sanitize_model_name(model_name)


def build_trainer_configs(model_name: str, layer: int, activation_dim: int, steps: int, device: str) -> list[dict]:
    warmup_steps = 1000 if steps > 1001 else 0
    decay_start = None if steps <= warmup_steps + 1 else int(steps * 0.8)
    if decay_start is not None and decay_start <= warmup_steps:
        decay_start = warmup_steps + 1
    submodule_name = f"resid_post_layer_{layer}"
    trainer_configs = []
    for preset in TRAINER_PRESETS:
        trainer_configs.append({
            "trainer": BatchTopKTrainer,
            "activation_dim": activation_dim,
            "dict_size": preset["expansion"] * activation_dim,
            "k": preset["k"],
            "lr": 5e-5,
            "steps": steps,
            "auxk_alpha": 1 / 32,
            "warmup_steps": warmup_steps,
            "decay_start": decay_start,
            "threshold_beta": 0.999,
            "threshold_start_step": 1000,
            "device": device,
            "layer": layer,
            "lm_name": model_name,
            "submodule_name": submodule_name,
            "wandb_name": f"BatchTopKTrainer-{model_name}-{submodule_name}",
            "seed": 0,
        })
    return trainer_configs


def layer_has_final_checkpoints(layer_dir: Path) -> bool:
    checkpoint_paths = [layer_dir / f"trainer_{idx}" / "ae.pt" for idx in range(len(TRAINER_PRESETS))]
    if not all(path.exists() for path in checkpoint_paths):
        return False
    return all("step" not in torch.load(path, map_location="cpu", weights_only=False) for path in checkpoint_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--refresh-batch-size", type=int, default=16)
    parser.add_argument("--out-batch-size", type=int, default=2048)
    parser.add_argument("--n-ctxs", type=int, default=122)
    parser.add_argument("--tokens", type=int, default=500_000_000)
    parser.add_argument("--backup-steps", type=int, default=5000)
    parser.add_argument("--max-activation-norm-multiple", type=float, default=10.0)
    parser.add_argument("--pretrain-dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--chat-dataset", type=str, default="lmsys/lmsys-chat-1m")
    parser.add_argument("--pretrain-frac", type=float, default=0.9)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Parent directory that will contain the top-level SAE folder")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    layers = args.layers or default_layers_for_model(args.model)
    min_chars = args.min_chars or (args.context_len * 4)
    output_parent = Path(os.path.expandvars(args.output_dir)) if args.output_dir else default_output_dir(args.model)
    save_root = output_parent / sae_repo_subdir(args.model)
    save_root.mkdir(parents=True, exist_ok=True)

    steps = args.tokens // args.out_batch_size
    dtype = torch.bfloat16

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model.to(args.device)
    model.eval()

    activation_dim = model.config.hidden_size
    print(f"Output parent: {output_parent}")
    print(f"SAE root: {save_root}")
    print(f"Layers: {layers}")
    print(f"Steps per layer: {steps}")
    print(f"Activation dim: {activation_dim}")

    generator = hf_mixed_dataset_to_generator(
        tokenizer=tokenizer,
        pretrain_dataset=args.pretrain_dataset,
        chat_dataset=args.chat_dataset,
        min_chars=min_chars,
        pretrain_frac=args.pretrain_frac,
        split=args.dataset_split,
        sequence_pack_pretrain=True,
        sequence_pack_chat=False,
    )

    for layer in layers:
        layer_dir = save_root / f"resid_post_layer_{layer}"
        if args.skip_existing and layer_has_final_checkpoints(layer_dir):
            print(f"Skipping layer {layer}: found all trainer checkpoints")
            continue

        print(f"\n=== Training layer {layer} ===")
        buffer = ActivationBuffer(
            data=generator,
            model=model,
            submodule=model.model.layers[layer],
            d_submodule=activation_dim,
            io="out",
            n_ctxs=args.n_ctxs,
            ctx_len=args.context_len,
            refresh_batch_size=args.refresh_batch_size,
            out_batch_size=args.out_batch_size,
            device=args.device,
            max_activation_norm_multiple=args.max_activation_norm_multiple,
        )
        trainer_configs = build_trainer_configs(args.model, layer, activation_dim, steps, args.device)
        trainSAE(
            data=buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_dir=str(layer_dir),
            backup_steps=args.backup_steps,
            device=args.device,
            autocast_dtype=dtype,
        )

    print("\nTraining complete.")
    print(f"Upload this directory to Hugging Face to preserve the expected path layout: {output_parent}")


if __name__ == "__main__":
    main()
