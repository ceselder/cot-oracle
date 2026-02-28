#!/usr/bin/env python
"""Collect top-K max-activating examples per SAE feature across the CoT corpus.

Runs Qwen3-8B forward on the full corpus, encodes residual stream activations
with SAEs at layers 9/18/27 (trainer 2, 65K features), and tracks the top-K
activating token contexts per feature.

Usage:
    python scripts/sae_collect_max_acts.py \
        --trainer 2 --k 30 --context-window 41 --batch-size 8

SLURM:
    #SBATCH --partition=gpu_lowp --nodelist=gpu-xd670-30
    #SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as nnf
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ao_reference"))

from nl_probes.sae import load_dictionary_learning_batch_topk_sae

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"
CORPUS_REPO = "mats-10-sprint-cs-jb/cot-oracle-corpus-v5"
SAE_LAYERS = [9, 18, 27]


class TopKTracker:
    """Track top-K max-activating token contexts per SAE feature on CPU.

    Uses a merge-and-topk pattern: for each batch, find the batch's per-feature
    top-K, concatenate with the running top-K, and keep the overall top-K.
    """

    def __init__(self, n_features: int, k: int, context_window: int):
        self.F, self.K, self.W = n_features, k, context_window
        self.values = torch.full((n_features, k), -float("inf"), dtype=torch.float32)
        self.contexts = torch.zeros(n_features, k, context_window, dtype=torch.int32)
        self.corpus_ids = torch.full((n_features, k), -1, dtype=torch.int32)
        self.positions = torch.full((n_features, k), -1, dtype=torch.int32)
        # Per-feature stats
        self.alive_count = torch.zeros(n_features, dtype=torch.int32)
        self.max_activation = torch.zeros(n_features, dtype=torch.float32)
        self.sum_activation = torch.zeros(n_features, dtype=torch.float64)
        self.total_active = torch.zeros(n_features, dtype=torch.int64)

    @torch.no_grad()
    def update(self, sae_acts: torch.Tensor, ctx_windows: torch.Tensor,
               corpus_ids_batch: torch.Tensor):
        """Update tracker with a batch of SAE activations.

        Args:
            sae_acts: [B, L, F] on GPU, already masked (padding/outliers zeroed).
            ctx_windows: [B, L, W] on CPU (int32).
            corpus_ids_batch: [B] on CPU (int32).
        """
        B, L, F = sae_acts.shape

        # --- Per-feature statistics ---
        active_per_entry = (sae_acts > 0).any(dim=1)  # [B, F]
        self.alive_count += active_per_entry.sum(0).cpu().int()
        flat = sae_acts.reshape(-1, F)
        self.max_activation = torch.maximum(self.max_activation, flat.max(0).values.cpu().float())
        self.sum_activation += sae_acts.sum(dim=(0, 1)).cpu().double()
        self.total_active += (sae_acts > 0).sum(dim=(0, 1)).cpu().long()

        # --- Batch top-K per feature (on GPU) ---
        K_batch = min(self.K, B * L)
        # topk along dim=0 of [B*L, F]: finds top-K rows per column (feature)
        new_vals, new_idx = flat.topk(K_batch, dim=0)  # [K_batch, F]
        new_vals = new_vals.T.contiguous().cpu().float()  # [F, K_batch]
        new_idx = new_idx.T.contiguous().cpu()            # [F, K_batch]

        # Gather metadata using batch indices (all on CPU)
        flat_ctx = ctx_windows.reshape(B * L, self.W)
        flat_cids = corpus_ids_batch.unsqueeze(1).expand(B, L).reshape(B * L)
        flat_pos = torch.arange(L, dtype=torch.int32).unsqueeze(0).expand(B, L).reshape(B * L)

        new_ctx = flat_ctx[new_idx]    # [F, K_batch, W]
        new_cids = flat_cids[new_idx]  # [F, K_batch]
        new_pos = flat_pos[new_idx]    # [F, K_batch]

        # Pad to K if batch smaller than K
        if K_batch < self.K:
            p = self.K - K_batch
            new_vals = torch.cat([new_vals, torch.full((F, p), -float("inf"))], 1)
            new_ctx = torch.cat([new_ctx, torch.zeros(F, p, self.W, dtype=torch.int32)], 1)
            new_cids = torch.cat([new_cids, torch.full((F, p), -1, dtype=torch.int32)], 1)
            new_pos = torch.cat([new_pos, torch.full((F, p), -1, dtype=torch.int32)], 1)

        # Merge: concatenate running + new, keep top-K
        cat_vals = torch.cat([self.values, new_vals], 1)  # [F, 2K]
        _, sel = cat_vals.topk(self.K, dim=1)              # [F, K]

        self.values = cat_vals.gather(1, sel)
        self.corpus_ids = torch.cat([self.corpus_ids, new_cids], 1).gather(1, sel)
        self.positions = torch.cat([self.positions, new_pos], 1).gather(1, sel)
        sel_3d = sel.unsqueeze(-1).expand(-1, -1, self.W)
        self.contexts = torch.cat([self.contexts, new_ctx], 1).gather(1, sel_3d)

    def save(self, path: Path):
        mean_act = torch.where(
            self.total_active > 0,
            self.sum_activation / self.total_active.double(),
            torch.zeros_like(self.sum_activation),
        ).float()
        torch.save({
            "top_values": self.values.half(), "top_contexts": self.contexts,
            "top_corpus_ids": self.corpus_ids, "top_positions": self.positions,
            "alive_count": self.alive_count, "max_activation": self.max_activation.half(),
            "mean_activation": mean_act, "sum_activation": self.sum_activation,
            "total_active": self.total_active,
            "n_features": self.F, "k": self.K, "context_window": self.W,
        }, path)
        print(f"Saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


def build_context_windows(input_ids: torch.Tensor, context_window: int,
                          pad_token_id: int) -> torch.Tensor:
    """Build [B, L, W] context windows from [B, L] input_ids using unfold."""
    half_w = context_window // 2
    padded = nnf.pad(input_ids, (half_w, half_w), value=pad_token_id)
    return padded.unfold(1, context_window, 1).int()  # [B, L, W]


def format_entry(question: str, cot: str, tokenizer) -> str:
    """Chat-format a corpus entry for model input."""
    # Strip existing think tags, re-wrap consistently
    cot = cot.replace("<think>", "").replace("</think>", "").strip()
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>\n{cot}\n</think>"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=int, default=2, help="SAE trainer variant (0/1/2)")
    parser.add_argument("--k", type=int, default=30, help="Top-K examples per feature")
    parser.add_argument("--context-window", type=int, default=41, help="Token context window (odd)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048, help="Max tokens per entry")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(os.path.expandvars(args.output_dir))
    else:
        base = os.environ.get("FAST_CACHE_DIR", os.environ["CACHE_DIR"])
        output_dir = Path(os.path.expandvars(base)) / "sae_features"
    trainer_dir = output_dir / f"trainer_{args.trainer}"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {trainer_dir}")

    device = "cuda"
    dtype = torch.bfloat16

    # --- Load model ---
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", dtype=dtype, attn_implementation=attn_impl,
    )
    model.eval()

    # --- Load SAEs ---
    saes = {}
    sae_local_dir = str(output_dir / "downloaded_saes")
    for layer in SAE_LAYERS:
        filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{layer}/trainer_{args.trainer}/ae.pt"
        print(f"Loading SAE layer {layer}...")
        saes[layer] = load_dictionary_learning_batch_topk_sae(
            repo_id=SAE_REPO, filename=filename, model_name=MODEL_NAME,
            device=torch.device(device), dtype=dtype, layer=layer, local_dir=sae_local_dir,
        )
    n_features = saes[SAE_LAYERS[0]].b_enc.shape[0]
    print(f"SAE features: {n_features}")

    # --- Load corpus ---
    print(f"Loading corpus from {CORPUS_REPO}...")
    ds = load_dataset(CORPUS_REPO, split="train")
    print(f"Corpus: {len(ds)} entries")

    # Format all entries
    print("Formatting corpus entries...")
    texts = [format_entry(e["question"], e["cot_response"], tokenizer) for e in tqdm(ds)]

    # Save corpus metadata for later lookup
    meta = [{"idx": i, "question": ds[i]["question"][:200]} for i in range(len(ds))]
    meta_path = trainer_dir / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta))
    print(f"Saved corpus metadata: {meta_path}")

    # --- Create trackers ---
    trackers = {layer: TopKTracker(n_features, args.k, args.context_window) for layer in SAE_LAYERS}

    # --- Register forward hooks ---
    activation_cache: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Decoder layers may return tuple (hidden_states, ...) or just hidden_states
            h = output[0] if isinstance(output, tuple) else output
            activation_cache[layer_idx] = h.detach()
        return hook_fn

    handles = []
    for layer_idx in SAE_LAYERS:
        submodule = model.model.layers[layer_idx]
        handles.append(submodule.register_forward_hook(make_hook(layer_idx)))

    # --- Resume from checkpoint ---
    start_entry = 0
    ckpt_path = trainer_dir / "checkpoint.json"
    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        start_entry = ckpt["next_entry"]
        for layer_idx in SAE_LAYERS:
            pt = trainer_dir / f"topk_layer{layer_idx}_ckpt.pt"
            data = torch.load(pt, map_location="cpu", weights_only=True)
            t = trackers[layer_idx]
            t.values, t.contexts = data["top_values"].float(), data["top_contexts"]
            t.corpus_ids, t.positions = data["top_corpus_ids"], data["top_positions"]
            t.alive_count, t.max_activation = data["alive_count"], data["max_activation"].float()
            t.sum_activation = data["sum_activation"]
            t.total_active = data["total_active"]
        print(f"Resumed from entry {start_entry}")

    # --- Process batches ---
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    batch_idx = 0
    for batch_start in tqdm(range(0, len(texts), args.batch_size), total=n_batches, desc="Batches"):
        if batch_start < start_entry:
            batch_idx += 1
            continue
        batch_texts = texts[batch_start : batch_start + args.batch_size]
        B_actual = len(batch_texts)

        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        B, L = input_ids.shape

        # Context windows on CPU
        ctx_windows = build_context_windows(input_ids.cpu(), args.context_window, tokenizer.pad_token_id)
        corpus_ids = torch.arange(batch_start, batch_start + B_actual, dtype=torch.int32)

        # Forward pass (hooks capture activations)
        activation_cache.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        # Process each layer sequentially to limit GPU memory
        for layer_idx in SAE_LAYERS:
            acts = activation_cache[layer_idx]  # [B, L, D]

            # Outlier filtering: zero tokens with residual norm > 10× median
            norms = acts.norm(dim=-1)  # [B, L]
            valid_norms = norms[attention_mask.bool()]
            median_norm = valid_norms.median()
            outlier_mask = norms > 10 * median_norm
            valid_mask = attention_mask.bool() & ~outlier_mask  # [B, L]

            # SAE encode
            sae_acts = saes[layer_idx].encode(acts)  # [B, L, F]
            sae_acts = sae_acts * valid_mask.unsqueeze(-1)  # zero out padding + outliers

            # Update tracker
            trackers[layer_idx].update(sae_acts, ctx_windows, corpus_ids)

            del sae_acts
        activation_cache.clear()
        torch.cuda.empty_cache()
        batch_idx += 1

        # Checkpoint every 500 batches
        if batch_idx % 500 == 0:
            for layer_idx in SAE_LAYERS:
                trackers[layer_idx].save(trainer_dir / f"topk_layer{layer_idx}_ckpt.pt")
            ckpt_path.write_text(json.dumps({"next_entry": batch_start + args.batch_size}))
            print(f"\nCheckpoint at entry {batch_start + args.batch_size}")

    # Clean up hooks
    for h in handles:
        h.remove()

    # --- Save results ---
    for layer_idx in SAE_LAYERS:
        path = trainer_dir / f"topk_layer{layer_idx}.pt"
        trackers[layer_idx].save(path)
        alive = (trackers[layer_idx].alive_count > 0).sum().item()
        print(f"Layer {layer_idx}: {alive}/{n_features} alive features")

    print("Done!")


if __name__ == "__main__":
    main()
