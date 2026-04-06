"""Train the Gemma→Qwen rank-r bridge on fineweb text with a cosine loss.

See `src/gemma_bridge.py` for rationale. Summary:
- Both models frozen; only the rank-r bridge has gradients.
- For each fineweb document, tokenise with both tokenisers, forward through both,
  extract residuals at the paired layers (Gemma L12/L24/L36, Qwen L9/L18/L27).
- Pick ~N character anchors and use offset mappings to align a Gemma token and a
  Qwen token per anchor. Train with (1 - cosine_similarity) loss averaged across
  anchors × 3 layer pairs.
- Cosine (not MSE) because the oracle's steering hook renormalises injected
  vectors — only direction survives to inference.

Run:
  uv run python scripts/train_gemma_bridge.py \
      --output "$CACHE_DIR/gemma_bridge_r16.pt" \
      --n-docs 2000 --steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.ao import get_hf_submodule  # noqa: E402
from gemma_bridge import (  # noqa: E402
    D_QWEN,
    DEFAULT_RANK,
    GEMMA_MODEL_NAME,
    GemmaBridge,
    QWEN_LAYERS,
    infer_gemma_config,
    save_bridge,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True, help="Destination .pt file")
    p.add_argument("--qwen-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gemma-model", default=GEMMA_MODEL_NAME,
                   help="HF model ID of the Gemma multimodal model (e.g. google/gemma-3-12b-it or google/gemma-4-31B-it)")
    p.add_argument("--gemma-quantization", choices=["none", "8bit", "4bit"], default="none",
                   help="bnb quantization for Gemma during training. 8bit keeps bf16 compute, fits 31B on 1xH100.")
    p.add_argument("--fineweb-repo", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--fineweb-subset", default="sample-10BT")
    p.add_argument("--fineweb-split", default="train")
    p.add_argument("--n-docs", type=int, default=2000, help="Max fineweb documents to consume")
    p.add_argument("--steps", type=int, default=1000, help="Optimizer steps")
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-tokens", type=int, default=512, help="Truncate each document to this many tokens per tokenizer")
    p.add_argument("--anchors-per-doc", type=int, default=32)
    p.add_argument("--batch-docs", type=int, default=2, help="Documents per optimizer step")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb", action="store_true", help="Log training to W&B")
    p.add_argument("--wandb-entity", default="japhba-personal")
    p.add_argument("--wandb-project", default="cot-oracle-gemma-bridge")
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def load_qwen_base(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading Qwen base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa")
    model.eval()
    return model, tokenizer


def load_gemma_text(device: str, dtype: torch.dtype, model_name: str, quantization: str):
    """Load the chosen Gemma multimodal model for text-only forward passes during training."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    print(f"Loading Gemma base: {model_name} (quantization={quantization})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {
        "device_map": device,
        "attn_implementation": "eager",
        "torch_dtype": dtype,  # forces non-quantized ops (layer norms, residual) to bf16
    }
    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    elif quantization != "none":
        raise ValueError(f"Unknown quantization {quantization!r}")
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def forward_residuals(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """Forward and capture residuals at the requested transformer blocks. Returns {layer: [L, D]}."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_mod, _inp, outputs):
            resid = outputs[0] if isinstance(outputs, tuple) else outputs
            captured[layer_idx] = resid[0].detach()  # [L, D]
        return hook_fn

    for layer in layers:
        handles.append(get_hf_submodule(model, layer).register_forward_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured


def pick_anchor_char_indices(n_chars: int, k: int) -> list[int]:
    """Return k evenly-spaced character indices in [0, n_chars)."""
    if n_chars == 0:
        return []
    k = min(k, n_chars)
    stride = n_chars / k
    return [min(n_chars - 1, int(i * stride)) for i in range(k)]


def char_to_token_idx(offsets: list[tuple[int, int]], char_idx: int) -> int | None:
    """Return the index of the token whose offset span contains `char_idx`, or None."""
    for i, (s, e) in enumerate(offsets):
        if s == e == 0 and i > 0:
            # Some tokenizers emit (0, 0) for specials; skip.
            continue
        if s <= char_idx < e:
            return i
    return None


def align_doc(
    text: str,
    qwen_tokenizer,
    gemma_tokenizer,
    max_tokens: int,
    n_anchors: int,
):
    """Return (qwen_ids, qwen_attn, gemma_ids, gemma_attn, anchor_pairs).

    anchor_pairs is a list of (q_pos, g_pos) token-index pairs aligned by character span.
    """
    q_enc = qwen_tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_tokens, add_special_tokens=False)
    g_enc = gemma_tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_tokens, add_special_tokens=False)
    q_ids = q_enc["input_ids"]
    g_ids = g_enc["input_ids"]
    if not q_ids or not g_ids:
        return None

    # Cap anchor character range by the truncated span of the *shorter* tokenization
    q_max_char = q_enc["offset_mapping"][-1][1]
    g_max_char = g_enc["offset_mapping"][-1][1]
    usable_end = min(q_max_char, g_max_char)
    if usable_end < 16:
        return None

    anchor_chars = pick_anchor_char_indices(usable_end, n_anchors)
    pairs: list[tuple[int, int]] = []
    for c in anchor_chars:
        qi = char_to_token_idx(q_enc["offset_mapping"], c)
        gi = char_to_token_idx(g_enc["offset_mapping"], c)
        if qi is not None and gi is not None:
            pairs.append((qi, gi))
    if not pairs:
        return None

    q_ids_t = torch.tensor([q_ids], dtype=torch.long)
    g_ids_t = torch.tensor([g_ids], dtype=torch.long)
    q_attn = torch.ones_like(q_ids_t)
    g_attn = torch.ones_like(g_ids_t)
    return q_ids_t, q_attn, g_ids_t, g_attn, pairs


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity along last dim."""
    return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.home() / ".env")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Resolve output path (expand $CACHE_DIR etc.)
    output_path = Path(os.path.expandvars(os.path.expanduser(args.output))).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    d_gemma, gemma_layers = infer_gemma_config(args.gemma_model)
    print(f"Gemma config: model={args.gemma_model}  d_gemma={d_gemma}  gemma_layers={gemma_layers}  -> qwen_layers={QWEN_LAYERS}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "qwen_model": args.qwen_model,
                "gemma_model": args.gemma_model,
                "gemma_quantization": args.gemma_quantization,
                "d_gemma": d_gemma,
                "rank": args.rank,
                "lr": args.lr,
                "steps": args.steps,
                "n_docs": args.n_docs,
                "batch_docs": args.batch_docs,
                "anchors_per_doc": args.anchors_per_doc,
                "max_tokens": args.max_tokens,
                "qwen_layers": QWEN_LAYERS,
                "gemma_layers": gemma_layers,
                "dtype": args.dtype,
                "fineweb_repo": args.fineweb_repo,
                "fineweb_subset": args.fineweb_subset,
                "seed": args.seed,
            },
        )

    qwen_model, qwen_tokenizer = load_qwen_base(args.qwen_model, args.device, dtype)
    gemma_model, gemma_tokenizer = load_gemma_text(args.device, dtype, args.gemma_model, args.gemma_quantization)

    qwen_device = next(qwen_model.parameters()).device
    gemma_device = next(gemma_model.parameters()).device

    bridge = GemmaBridge(
        rank=args.rank,
        d_gemma=d_gemma,
        d_qwen=D_QWEN,
        n_pairs=3,
        gemma_layers=gemma_layers,
        qwen_layers=QWEN_LAYERS,
        gemma_model_name=args.gemma_model,
    )
    bridge.to(device=qwen_device, dtype=torch.float32)
    bridge.train()

    optim = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Stream fineweb
    from datasets import load_dataset
    print(f"Streaming {args.fineweb_repo}:{args.fineweb_subset} split={args.fineweb_split}")
    ds = load_dataset(args.fineweb_repo, name=args.fineweb_subset, split=args.fineweb_split, streaming=True)
    doc_iter = iter(ds)

    def next_usable_doc():
        for _ in range(50):  # bounded retries to avoid infinite loops on bad rows
            row = next(doc_iter, None)
            if row is None:
                return None
            text = row.get("text") or ""
            if len(text) < 64:
                continue
            return text
        return None

    step = 0
    docs_consumed = 0
    history = []

    pbar = tqdm(total=args.steps, desc="bridge")
    while step < args.steps and docs_consumed < args.n_docs:
        batch_loss = torch.zeros((), device=qwen_device, dtype=torch.float32)
        contributing = 0

        for _ in range(args.batch_docs):
            text = next_usable_doc()
            if text is None:
                break
            docs_consumed += 1
            aligned = align_doc(text, qwen_tokenizer, gemma_tokenizer, args.max_tokens, args.anchors_per_doc)
            if aligned is None:
                continue
            q_ids, q_attn, g_ids, g_attn, pairs = aligned

            q_resid = forward_residuals(qwen_model, q_ids.to(qwen_device), q_attn.to(qwen_device), QWEN_LAYERS)
            g_resid = forward_residuals(gemma_model, g_ids.to(gemma_device), g_attn.to(gemma_device), gemma_layers)

            q_pos = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            g_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long)

            loss_doc = torch.zeros((), device=qwen_device, dtype=torch.float32)
            for pair_idx, (q_layer, g_layer) in enumerate(zip(QWEN_LAYERS, gemma_layers)):
                h_gemma = g_resid[g_layer][g_pos].to(device=qwen_device, dtype=torch.float32)  # [A, d_gemma]
                h_qwen = q_resid[q_layer][q_pos].to(device=qwen_device, dtype=torch.float32)   # [A, d_qwen]
                pred = bridge(h_gemma, pair_idx)  # [A, d_qwen]
                loss_doc = loss_doc + cosine_loss(pred, h_qwen)
            loss_doc = loss_doc / len(QWEN_LAYERS)
            batch_loss = batch_loss + loss_doc
            contributing += 1

        if contributing == 0:
            continue
        batch_loss = batch_loss / contributing

        optim.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optim.step()

        step += 1
        history.append({"step": step, "docs": docs_consumed, "loss": float(batch_loss.item())})
        pbar.update(1)
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}", docs=docs_consumed)
        if wandb_run is not None:
            wandb_run.log({"train/loss": float(batch_loss.item()), "train/docs": docs_consumed}, step=step)
    pbar.close()

    # Per-layer held-out cosine similarity as a sanity check.
    print("\nHeld-out cosine similarity (higher = better; random ≈ 0):")
    bridge.eval()
    eval_cos = {pair_idx: [] for pair_idx in range(len(QWEN_LAYERS))}
    with torch.no_grad():
        n_eval = 0
        while n_eval < 16:
            text = next_usable_doc()
            if text is None:
                break
            aligned = align_doc(text, qwen_tokenizer, gemma_tokenizer, args.max_tokens, args.anchors_per_doc)
            if aligned is None:
                continue
            q_ids, q_attn, g_ids, g_attn, pairs = aligned
            q_resid = forward_residuals(qwen_model, q_ids.to(qwen_device), q_attn.to(qwen_device), QWEN_LAYERS)
            g_resid = forward_residuals(gemma_model, g_ids.to(gemma_device), g_attn.to(gemma_device), gemma_layers)
            q_pos = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            g_pos = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            for pair_idx, (q_layer, g_layer) in enumerate(zip(QWEN_LAYERS, gemma_layers)):
                h_g = g_resid[g_layer][g_pos].to(device=qwen_device, dtype=torch.float32)
                h_q = q_resid[q_layer][q_pos].to(device=qwen_device, dtype=torch.float32)
                pred = bridge(h_g, pair_idx)
                cs = F.cosine_similarity(pred, h_q, dim=-1).mean().item()
                eval_cos[pair_idx].append(cs)
            n_eval += 1
    eval_summary = {}
    for pair_idx, (q_layer, g_layer) in enumerate(zip(QWEN_LAYERS, gemma_layers)):
        if eval_cos[pair_idx]:
            mean = sum(eval_cos[pair_idx]) / len(eval_cos[pair_idx])
            print(f"  Gemma L{g_layer} → Qwen L{q_layer}: mean cos={mean:.3f}  (n={len(eval_cos[pair_idx])})")
            eval_summary[f"eval/cos_gemma_L{g_layer}_to_qwen_L{q_layer}"] = mean
    if wandb_run is not None and eval_summary:
        wandb_run.log(eval_summary)

    # Save in bf16 to keep the file small; user can load back in any dtype.
    bridge.to(dtype=torch.bfloat16)
    save_bridge(
        bridge,
        output_path,
        extra={
            "training": {
                "steps": step,
                "docs_consumed": docs_consumed,
                "lr": args.lr,
                "rank": args.rank,
                "fineweb_repo": args.fineweb_repo,
                "fineweb_subset": args.fineweb_subset,
                "seed": args.seed,
                "loss_history": history,
                "eval_cos": {f"pair{i}": eval_cos[i] for i in range(len(QWEN_LAYERS))},
            },
        },
    )
    if wandb_run is not None:
        artifact = wandb.Artifact(name="gemma_bridge", type="model")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    print(f"Checkpoint at {output_path}")


if __name__ == "__main__":
    main()
