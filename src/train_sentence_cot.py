"""
CoT Oracle: Sentence-Structured Multi-Layer Context Prediction

Injects activations from 3 layers (25%, 50%, 75%) at each sentence boundary.
Uses standard AO prefix format with pre-computed multi-layer steering vectors.

Phase 1: Extract activations (base model, adapter disabled)
Phase 2: Build TrainingDataPoints with pre-computed acts_BD
Phase 3: Train with AO's train_model()

Usage:
    torchrun --nproc_per_node=1 src/train_sentence_cot.py
"""

import json
import random
import re
import sys
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

# AO repo imports
_ao_candidates = [
    Path("/workspace/ao_reference"),  # vast.ai
    Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),
]
AO_REPO = next((p for p in _ao_candidates if p.exists()), _ao_candidates[-1])
sys.path.insert(0, str(AO_REPO))

from nl_probes.utils.dataset_utils import create_training_datapoint, TrainingDataPoint
from nl_probes.sft import train_model
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.utils.common import load_tokenizer

# ─── Config ───
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]  # 25%, 50%, 75% of 36 layers
CORPUS_PATH = "/workspace/corpus.jsonl"
AO_HF_REPO = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
SAVE_DIR = "/workspace/checkpoints/cot_oracle_sentence"
BATCH_SIZE = 8
NUM_EXAMPLES = 30000
LR = 1e-5
SEED = 42
MAX_SENTENCES = 15  # cap per CoT
MIN_K_TOKENS = 1
MAX_K_TOKENS = 15
DEVICE = "cuda"
EXTRACTION_BATCH_SIZE = 4  # batch forward passes for speed


def load_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    print(f"Loaded {len(corpus)} corpus entries")
    return corpus


def split_sentences(text):
    """Split CoT text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\d$\\(])', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]


def get_layer_module(model, layer_idx):
    """Get transformer layer module, handling PEFT wrapping."""
    for getter in [
        lambda: model.base_model.model.model.layers[layer_idx],
        lambda: model.model.model.layers[layer_idx],
        lambda: model.model.layers[layer_idx],
    ]:
        try:
            return getter()
        except (AttributeError, IndexError):
            continue
    raise RuntimeError(f"Cannot find layer {layer_idx}")


def find_sentence_boundaries(tokenizer, formatted_prompt, sentences):
    """Find token positions where each sentence boundary falls."""
    positions = []
    cot_prefix = ""
    for sent in sentences:
        cot_prefix += sent + " "
        full_prefix = formatted_prompt + cot_prefix
        prefix_ids = tokenizer(full_prefix, add_special_tokens=False)["input_ids"]
        positions.append(len(prefix_ids) - 1)
    return positions


def extract_activations_batch(model, tokenizer, entries, layers, device="cuda"):
    """Extract multi-layer activations at sentence boundaries for a batch of entries.

    Returns list of dicts with 'sentences', 'boundary_positions',
    'activations' [N_sent, N_layers, d_model], 'full_ids'.
    """
    results = []

    for idx, entry in enumerate(entries):
        if idx % 100 == 0:
            print(f"  Extracting [{idx}/{len(entries)}]...")

        # Build full text
        messages = [{"role": "user", "content": entry["question"]}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        cot_text = entry.get("cot_response", "")
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]

        # Split into sentences
        sentences = entry.get("sentences") or split_sentences(cot_text)
        if len(sentences) < 2:
            continue
        sentences = sentences[:MAX_SENTENCES]

        # Find sentence boundary positions
        boundary_positions = find_sentence_boundaries(tokenizer, formatted, sentences)

        full_text = formatted + cot_text
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Validate positions
        boundary_positions = [p for p in boundary_positions if p < len(full_ids)]
        if len(boundary_positions) < 2:
            continue

        # Forward pass with hooks at all target layers
        input_ids = torch.tensor([full_ids], device=device)
        attn_mask = torch.ones_like(input_ids)

        layer_activations = {}
        hooks = []

        for layer_idx in layers:
            module = get_layer_module(model, layer_idx)

            def make_hook(li):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    layer_activations[li] = h[0].detach().cpu()
                return hook_fn

            handle = module.register_forward_hook(make_hook(layer_idx))
            hooks.append(handle)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)

        for h in hooks:
            h.remove()

        # Collect activations at boundary positions: [N_sent, N_layers, d_model]
        sent_acts = []
        for pos in boundary_positions:
            layer_acts = []
            for layer_idx in layers:
                act = layer_activations[layer_idx][pos]
                layer_acts.append(act)
            sent_acts.append(torch.stack(layer_acts))

        acts_tensor = torch.stack(sent_acts)  # [N_sent, N_layers, d_model]

        results.append({
            "entry_idx": idx,
            "entry": entry,
            "sentences": sentences[:len(boundary_positions)],
            "boundary_positions": boundary_positions,
            "activations": acts_tensor,
            "full_ids": full_ids,
        })

        del layer_activations, input_ids, attn_mask

    print(f"  Extracted activations for {len(results)}/{len(entries)} entries")
    return results


def build_training_data(extracted, tokenizer, num_examples, seed=42):
    """Build TrainingDataPoints with pre-computed multi-layer acts_BD.

    Uses standard AO prefix format. Activations are ordered as:
    [sent1_layer25%, sent1_layer50%, sent1_layer75%, sent2_layer25%, ...]
    """
    random.seed(seed)

    datapoints = []
    skipped = 0
    attempts = 0
    max_attempts = num_examples * 10

    while len(datapoints) < num_examples and attempts < max_attempts:
        attempts += 1

        item = random.choice(extracted)
        sentences = item["sentences"]
        acts = item["activations"]  # [N_sent, 3, d_model]
        full_ids = item["full_ids"]
        boundary_positions = item["boundary_positions"]
        N = len(sentences)

        # Pick target sentence and direction
        target_sent = random.randint(0, N - 1)
        direction = random.choice(["future", "past"])
        k_tokens = random.randint(MIN_K_TOKENS, MAX_K_TOKENS)

        # Get target text
        boundary_pos = boundary_positions[target_sent]

        if direction == "future":
            start = boundary_pos + 1
            end = start + k_tokens
            if end > len(full_ids):
                continue
            target_ids = full_ids[start:end]
            question = (
                f"Activations from {N} sentence boundaries, 3 per sentence. "
                f"Predict the next {k_tokens} tokens following sentence {target_sent + 1}."
            )
        else:
            end = boundary_pos
            start = end - k_tokens
            if start < 0:
                continue
            target_ids = full_ids[start:end]
            question = (
                f"Activations from {N} sentence boundaries, 3 per sentence. "
                f"Predict the {k_tokens} tokens before sentence {target_sent + 1}."
            )

        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue

        # Flatten activations: [N*3, d_model]
        # Order: sent1_25%, sent1_50%, sent1_75%, sent2_25%, ...
        acts_flat = acts.reshape(-1, acts.shape[-1])  # [N*3, d_model]
        num_positions = N * 3

        try:
            dp = create_training_datapoint(
                datapoint_type="cot_sentence_prediction",
                prompt=question,
                target_response=target_text,
                layer=18,  # metadata; prefix says "Layer: 18"
                num_positions=num_positions,
                tokenizer=tokenizer,
                acts_BD=acts_flat,
                feature_idx=-1,
            )
            datapoints.append(dp)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Warning ({skipped}): {e}")
            continue

        if len(datapoints) % 5000 == 0:
            print(f"  Built {len(datapoints)}/{num_examples} examples ({attempts} attempts, {skipped} skipped)")

    print(f"Built {len(datapoints)} training examples ({attempts} attempts, {skipped} skipped)")
    return datapoints


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--wandb-run", default="cot_oracle_sentence_v1")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    args = parser.parse_args()

    # Initialize distributed (required by AO)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tokenizer = load_tokenizer(MODEL_NAME)

    # ═══ Phase 1: Extract multi-layer activations ═══
    print("=" * 60)
    print("Phase 1: Extracting multi-layer activations at sentence boundaries")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layers: {LAYERS} (25%, 50%, 75%)")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()

    corpus = load_corpus(args.corpus)

    with torch.no_grad():
        extracted = extract_activations_batch(model, tokenizer, corpus, LAYERS, DEVICE)

    # Free extraction model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Freed extraction model. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ═══ Phase 2: Build training data ═══
    print("\n" + "=" * 60)
    print("Phase 2: Building sentence-structured training examples")
    print(f"  Target: {args.num_examples} examples")
    print(f"  Extracted entries: {len(extracted)}")
    print("=" * 60)

    datapoints = build_training_data(extracted, tokenizer, args.num_examples, SEED)

    if not datapoints:
        print("ERROR: No training data generated!")
        return

    # Split train/eval
    random.seed(SEED)
    random.shuffle(datapoints)
    eval_n = min(100, len(datapoints) // 10)
    eval_data = datapoints[:eval_n]
    train_data = datapoints[eval_n:]

    eval_datasets = {"cot_sentence_prediction": eval_data}

    print(f"\nTrain: {len(train_data)}, Eval: {eval_n}")

    # Check a sample
    sample = train_data[0]
    print(f"\nSample datapoint:")
    print(f"  Type: {sample.datapoint_type}")
    print(f"  Layer: {sample.layer}")
    print(f"  Positions: {sample.positions[:6]}... ({len(sample.positions)} total)")
    print(f"  Steering vectors shape: {sample.steering_vectors.shape if sample.steering_vectors is not None else 'None'}")
    print(f"  Target: {sample.target_output[:100]}")
    decoded_prompt = tokenizer.decode(sample.input_ids, skip_special_tokens=False)
    print(f"  Prompt (first 300): {decoded_prompt[:300]}")

    # Free extracted data
    del extracted
    gc.collect()

    # ═══ Phase 3: Train ═══
    print("\n" + "=" * 60)
    print("Phase 3: Training")
    print("=" * 60)

    ao_path = snapshot_download(AO_HF_REPO)
    print(f"AO checkpoint: {ao_path}")

    # Login to wandb
    import wandb
    wandb.login(key="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")

    cfg = SelfInterpTrainingConfig(
        model_name=MODEL_NAME,
        hook_onto_layer=1,
        layer_percents=[25, 50, 75],
        steering_coefficient=1.0,
        lr=args.lr,
        num_epochs=1,
        train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_dir=args.save_dir,
        wandb_project="cot_oracle",
        wandb_run_name=args.wandb_run,
        gradient_checkpointing=True,
        load_lora_path=ao_path,
        eval_on_start=True,
    )

    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.train_batch_size}")
    print(f"  Train examples: {len(train_data)}")
    print(f"  Steps: ~{len(train_data) // cfg.train_batch_size}")
    print(f"  Save dir: {cfg.save_dir}")

    train_model(
        cfg=cfg,
        training_data=train_data,
        eval_datasets=eval_datasets,
        tokenizer=tokenizer,
        device=torch.device(DEVICE),
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
