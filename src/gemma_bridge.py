"""Cross-model bridge: Gemma-3-12B-it residuals → Qwen3-8B oracle activation space.

The oracle LoRA on Qwen3-8B was trained to read residuals from Qwen's own
L9/L18/L27 (25%/50%/75% of 36 layers). Gemma-3-12B-it has 48 layers, so the
proportionally matched layers are L12/L24/L36.

We train three rank-16 linear adapters (one per layer pair) on fineweb text
with a cosine loss, because the oracle's injection hook renormalises vectors
to unit norm before scaling them to Qwen's ambient residual magnitude
(see `src/core/ao.py::get_steering_hook`) — only direction matters.

At inference the bridge is applied to **text-token positions only** of Gemma's
residuals for a multimodal (image + text) prompt, then fed into the oracle's
existing activation-injection path (`query_trained_oracle`).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from core.ao import get_hf_submodule


QWEN_LAYERS = [9, 18, 27]
D_QWEN = 4096
DEFAULT_RANK = 16

# Known Gemma configs. Layers are proportional to Qwen3-8B's L9/L18/L27 (25/50/75%
# of 36 layers). Keys are HF model IDs; values are (d_gemma, gemma_layers).
GEMMA_CONFIGS: dict[str, tuple[int, list[int]]] = {
    "google/gemma-3-12b-it": (3840, [12, 24, 36]),   # 48 layers → 12/24/36
    "google/gemma-4-31B-it":  (5376, [15, 30, 45]),  # 60 layers → 15/30/45
}

# Defaults (backward compat with earlier code that imported GEMMA_MODEL_NAME etc.)
GEMMA_MODEL_NAME = "google/gemma-3-12b-it"
D_GEMMA, GEMMA_LAYERS = GEMMA_CONFIGS[GEMMA_MODEL_NAME]


def infer_gemma_config(model_name: str) -> tuple[int, list[int]]:
    """Return (d_gemma, gemma_layers) for a known Gemma multimodal model."""
    if model_name in GEMMA_CONFIGS:
        return GEMMA_CONFIGS[model_name]
    raise ValueError(f"Unknown Gemma model {model_name!r}. Known: {list(GEMMA_CONFIGS)}")


class GemmaBridge(nn.Module):
    """Three rank-`r` linear adapters, one per (Gemma layer, Qwen layer) pair."""

    def __init__(
        self,
        rank: int = DEFAULT_RANK,
        d_gemma: int = D_GEMMA,
        d_qwen: int = D_QWEN,
        n_pairs: int = 3,
        gemma_layers: list[int] | None = None,
        qwen_layers: list[int] | None = None,
        gemma_model_name: str = GEMMA_MODEL_NAME,
    ):
        super().__init__()
        self.rank = rank
        self.d_gemma = d_gemma
        self.d_qwen = d_qwen
        self.n_pairs = n_pairs
        self.gemma_layers = list(gemma_layers) if gemma_layers is not None else list(GEMMA_LAYERS)
        self.qwen_layers = list(qwen_layers) if qwen_layers is not None else list(QWEN_LAYERS)
        self.gemma_model_name = gemma_model_name
        self.down = nn.ModuleList([nn.Linear(d_gemma, rank, bias=False) for _ in range(n_pairs)])
        self.up = nn.ModuleList([nn.Linear(rank, d_qwen, bias=False) for _ in range(n_pairs)])
        # LoRA-style init: down ~ Kaiming, up = 0. Start from an all-zero projection
        # so the loss signal pulls the adapter off zero rather than fighting a bad init.
        for d in self.down:
            nn.init.kaiming_uniform_(d.weight, a=5 ** 0.5)
        for u in self.up:
            nn.init.zeros_(u.weight)

    def forward(self, h_gemma: torch.Tensor, pair_idx: int) -> torch.Tensor:
        """Project `h_gemma` of shape [..., d_gemma] to Qwen space for pair `pair_idx`."""
        return self.up[pair_idx](self.down[pair_idx](h_gemma))


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_gemma(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    quantization: str = "none",
    model_name: str = GEMMA_MODEL_NAME,
):
    """Load a Gemma multimodal model + processor.

    Args:
        model_name: HF model ID. Must be a key in GEMMA_CONFIGS.
        quantization: "none" (full bf16), "8bit" (LLM.int8), or "4bit" (NF4).
            Quantization introduces a distribution shift relative to the bf16
            training of the bridge adapters.

    Returns (model, processor). The model is set to eval mode.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    print(f"Loading {model_name} (dtype={dtype}, quantization={quantization})...")
    processor = AutoProcessor.from_pretrained(model_name)
    kwargs = {
        "device_map": device,
        "attn_implementation": "eager",  # Gemma 3 currently requires eager
        # Always force bf16 for non-quantized ops (layer norms, residual stream, etc.).
        # Without this, 4-bit Gemma falls back to fp16, whose ±65504 range overflows on
        # Gemma 3's late-layer residual norms (~hundreds) and produces NaNs — which
        # then propagate through the bridge into the oracle as garbage tokens.
        "torch_dtype": dtype,
    }
    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization != "none":
        raise ValueError(f"Unknown quantization {quantization!r}; expected none|8bit|4bit")
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, processor


# -----------------------------------------------------------------------------
# Residual extraction
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def _residual_capture(model, layers: list[int]):
    """Install forward hooks on specified transformer blocks and collect their residual outputs.

    Yields a dict that after the `with` body is exited will contain
    `{layer_idx: [seq, d_gemma]}` (single-batch; we only ever forward batch=1 here).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inputs, outputs):
            resid = outputs[0] if isinstance(outputs, tuple) else outputs
            # resid: [B, L, D]; we only use B=1
            captured[layer_idx] = resid[0].detach()
        return hook_fn

    for layer in layers:
        submodule = get_hf_submodule(model, layer)
        handles.append(submodule.register_forward_hook(make_hook(layer)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def extract_gemma_residuals(
    model,
    processor,
    text: str,
    image: Image.Image | None,
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict]:
    """Run Gemma on (text, image?) and return residuals at requested layers.

    Returns:
        (residuals, meta) where
            residuals: {layer_idx: [seq_len, d_gemma]} (on Gemma's device, bf16)
            meta: {"input_ids": [seq_len], "text_mask": [seq_len] bool, "image_mask": [seq_len] bool}
    """
    if image is not None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    with _residual_capture(model, layers) as captured:
        # We only need forward residuals; don't actually generate. Pass everything
        # the processor emitted (pixel_values, attention_mask, input_ids, ...).
        model(**inputs)

    input_ids = inputs["input_ids"][0].cpu()
    # Build image/text masks. Gemma 3 exposes the image-token id via the processor.
    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        # Fall back to the special token string used by Gemma 3 ("<image_soft_token>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    image_mask = (input_ids == image_token_id)
    text_mask = ~image_mask

    meta = {
        "input_ids": input_ids,
        "text_mask": text_mask,
        "image_mask": image_mask,
        "image_token_id": int(image_token_id),
    }
    return captured, meta


@torch.no_grad()
def extract_gemma_residuals_with_generation(
    model,
    processor,
    text: str,
    image: Image.Image | None,
    layers: list[int],
    max_new_tokens: int = 150,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Generate a response, then re-forward over (prompt + generated) to capture residuals.

    This is the "Gemma actually reasons about the image, and we read that reasoning"
    pathway — closer to how the oracle was trained (reading CoT residuals). Returns
    residuals and a metadata dict whose `text_mask` covers *all* non-image tokens in
    the concatenated sequence (prompt + generated), and additionally exposes a
    `generated_mask` selecting only positions from the generated span.
    """
    if image is not None:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    # Step 1: generate a response.
    gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # gen_out is [1, prompt_len + new_tokens]
    full_ids = gen_out  # already on device
    new_tokens = int(full_ids.shape[1] - prompt_len)

    # Step 2: re-forward the full sequence with hooks to capture residuals at the target layers.
    # We need to include pixel_values so the image branch is exercised identically to step 1
    # (otherwise the model falls back to treating image tokens as plain text embeddings).
    forward_kwargs = {k: v for k, v in inputs.items() if k != "input_ids" and k != "attention_mask"}
    forward_kwargs["input_ids"] = full_ids
    forward_kwargs["attention_mask"] = torch.ones_like(full_ids)
    with _residual_capture(model, layers) as captured:
        model(**forward_kwargs)

    full_ids_cpu = full_ids[0].cpu()
    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    image_mask = (full_ids_cpu == image_token_id)
    text_mask = ~image_mask
    generated_mask = torch.zeros_like(text_mask)
    generated_mask[prompt_len:] = True

    generated_text = processor.tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)

    meta = {
        "input_ids": full_ids_cpu,
        "text_mask": text_mask,
        "image_mask": image_mask,
        "generated_mask": generated_mask,
        "image_token_id": int(image_token_id),
        "prompt_len": prompt_len,
        "new_tokens": new_tokens,
        "generated_text": generated_text,
    }
    return captured, meta


# -----------------------------------------------------------------------------
# Projection to oracle input
# -----------------------------------------------------------------------------

def project_text_residuals_to_qwen(
    bridge: GemmaBridge,
    gemma_resid: dict[int, torch.Tensor],
    text_mask: torch.Tensor,
    k_positions: int,
    target_dtype: torch.dtype = torch.bfloat16,
    target_device: str | torch.device = "cuda",
    selection_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project Gemma text-token residuals into oracle injection format.

    Returns a tensor of shape `[3 * k_positions, d_qwen]` with ordering
    `[L_first_qwen; L_second_qwen; L_third_qwen]`, matching what
    `query_trained_oracle` expects (`selected_layers=QWEN_LAYERS`,
    `layer_counts=[k_positions]*3`).

    Args:
        selection_mask: Optional boolean mask over the sequence restricting which
            positions we're allowed to sample from. If None, uses `text_mask`
            (all non-image positions). Typical use: pass `generated_mask & text_mask`
            to sample only from Gemma's generated response tokens.
    """
    chosen_mask = selection_mask if selection_mask is not None else text_mask
    text_idx = torch.nonzero(chosen_mask, as_tuple=False).flatten()
    n_text = int(text_idx.numel())
    if n_text == 0:
        raise ValueError("No positions found after applying selection mask.")
    # Uniform-stride subsample down to k_positions (or tile if fewer than k).
    if n_text >= k_positions:
        stride = n_text / k_positions
        sample_idx = text_idx[[int(i * stride) for i in range(k_positions)]]
    else:
        # Fewer text tokens than requested K: repeat the last to pad.
        repeats = k_positions - n_text
        sample_idx = torch.cat([text_idx, text_idx[-1:].repeat(repeats)])

    bridge_device = next(bridge.parameters()).device
    projected_per_layer = []
    for pair_idx, gemma_layer in enumerate(bridge.gemma_layers):
        h = gemma_resid[gemma_layer][sample_idx].to(device=bridge_device, dtype=next(bridge.parameters()).dtype)  # [K, d_gemma]
        h_qwen = bridge(h, pair_idx)  # [K, d_qwen]
        projected_per_layer.append(h_qwen)
    stacked = torch.cat(projected_per_layer, dim=0)  # [3K, d_qwen]
    return stacked.to(device=target_device, dtype=target_dtype).detach()


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------

def save_bridge(bridge: GemmaBridge, path: str | Path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": bridge.state_dict(),
        "config": {
            "rank": bridge.rank,
            "d_gemma": bridge.d_gemma,
            "d_qwen": bridge.d_qwen,
            "n_pairs": bridge.n_pairs,
            "gemma_layers": bridge.gemma_layers,
            "qwen_layers": bridge.qwen_layers,
            "gemma_model_name": bridge.gemma_model_name,
        },
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"Saved GemmaBridge → {path}")


def load_bridge(path: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16) -> GemmaBridge:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    bridge = GemmaBridge(
        rank=cfg["rank"],
        d_gemma=cfg["d_gemma"],
        d_qwen=cfg["d_qwen"],
        n_pairs=cfg["n_pairs"],
        gemma_layers=cfg.get("gemma_layers"),  # older checkpoints default to Gemma 3 12B
        qwen_layers=cfg.get("qwen_layers"),
        gemma_model_name=cfg.get("gemma_model_name", GEMMA_MODEL_NAME),
    )
    bridge.load_state_dict(payload["state_dict"])
    bridge.to(device=device, dtype=dtype)
    bridge.eval()
    print(f"Loaded GemmaBridge ← {path} (rank={cfg['rank']}, gemma={bridge.gemma_model_name})")
    return bridge
