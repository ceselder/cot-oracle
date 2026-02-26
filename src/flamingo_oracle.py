"""
Flamingo-Style Gated Cross-Attention Oracle

Two modes of operation:

1. **Chimera (single-pass)**: [CoT | Oracle] concatenated in one sequence.
   At xattn layers, oracle positions cross-attend to CoT hidden states
   from the same layer. No separate activation extraction needed.
   CoT hidden states are detached — oracle monitors, doesn't shape CoT.

2. **Pre-extracted KV (legacy)**: Supervisee activations pre-extracted per layer,
   passed as supervisee_kvs dict.

Architecture per cross-attention layer:
    hidden = SelfAttn(hidden)       ← existing LoRA (via PEFT)
    hidden = MLP(hidden)            ← existing LoRA (via PEFT)
    [oracle only] hidden += tanh(gate) * CrossAttn(Q=oracle_hidden, KV=cot_hidden)
                                    ↑ frozen base weights + manual LoRA adapters

Gates start at 0 (identity at init). Per-layer cross-attention: block i
attends only to layer i's residual stream.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights + LoRA adapters.

    output = base(x) + (lora_B @ lora_A @ x) * scaling
    Base weights are frozen; only lora_A, lora_B are trainable.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 64, lora_alpha: int = 128, lora_dropout: float = 0.0):
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.base = nn.Linear(in_features, out_features, bias=base_linear.bias is not None)
        self.base.weight.data.copy_(base_linear.weight.data)
        if base_linear.bias is not None:
            self.base.bias.data.copy_(base_linear.bias.data)
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


class GatedCrossAttentionLayer(nn.Module):
    """Gated cross-attention: Q from oracle hidden states, KV from supervisee activations.

    Projections initialized as frozen copies of self-attention weights + LoRA adapters.
    Gate starts at 0 → tanh(0) = 0 → identity at initialization.
    No RoPE (supervisee positions aren't sequential text).
    No causal mask (oracle freely attends to all supervisee positions).
    """

    def __init__(self, config, source_layer: nn.Module, lora_r: int = 64, lora_alpha: int = 128):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim  # Qwen3 uses independent head_dim (128 for all sizes)
        self.q_dim = self.num_heads * self.head_dim  # may differ from hidden_size
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        sa = source_layer.self_attn

        self.q_proj = LoRALinear(sa.q_proj, r=lora_r, lora_alpha=lora_alpha)
        self.k_proj = LoRALinear(sa.k_proj, r=lora_r, lora_alpha=lora_alpha)
        self.v_proj = LoRALinear(sa.v_proj, r=lora_r, lora_alpha=lora_alpha)
        self.o_proj = LoRALinear(sa.o_proj, r=lora_r, lora_alpha=lora_alpha)

        self.q_norm = deepcopy(sa.q_norm)
        self.k_norm = deepcopy(sa.k_norm)

        # Pre-norm (query side)
        self.norm = deepcopy(source_layer.input_layernorm)

        # Gate initialized to 0 → tanh(0) = 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, kv: torch.Tensor, kv_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_q, D] oracle hidden states
            kv: [B, T_cot, D] supervisee activations at this layer
            kv_attention_mask: [B, T_cot] bool mask (True = attend, False = ignore)
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        B, L_q, _ = hidden_states.shape
        L_kv = kv.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L_kv, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L_kv, self.head_dim)

        # Pass None when all-True to enable flash attention
        attn_mask = None
        if kv_attention_mask is not None and not kv_attention_mask.all():
            attn_mask = kv_attention_mask[:, None, None, :].expand(-1, -1, L_q, -1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.q_dim)
        attn_output = self.o_proj(attn_output)

        return residual + torch.tanh(self.gate) * attn_output


class DecoderLayerWithCrossAttention(nn.Module):
    """Wraps an original decoder layer + appends gated cross-attention.

    Supports two modes (set by the wrapper's runtime state):
    - Chimera: splits hidden states at _cot_len boundary, oracle cross-attends to CoT
    - Pre-extracted KV: reads per-layer KVs from _current_kvs dict
    """

    def __init__(self, original_layer: nn.Module, xattn_layer: GatedCrossAttentionLayer, wrapper: "FlamingoOracleWrapper", layer_idx: int):
        super().__init__()
        self.original_layer = original_layer
        object.__setattr__(self, '_xattn_layer', xattn_layer)
        object.__setattr__(self, '_wrapper', wrapper)
        object.__setattr__(self, '_layer_idx', layer_idx)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        if isinstance(outputs, tuple):
            hs = outputs[0]
        else:
            hs = outputs

        cot_len = self._wrapper._cot_len
        if cot_len is not None and cot_len > 0:
            # Chimera mode: split at CoT/oracle boundary
            cot_hs = hs[:, :cot_len, :]
            oracle_hs = hs[:, cot_len:, :]
            cot_mask = self._wrapper._cot_mask
            # Detach CoT — oracle monitors, doesn't shape CoT representations
            oracle_hs = self._xattn_layer(oracle_hs, cot_hs.detach(), kv_attention_mask=cot_mask)
            hs = torch.cat([cot_hs, oracle_hs], dim=1)
        else:
            # Pre-extracted KV mode (legacy)
            kvs = self._wrapper._current_kvs
            if kvs is not None and self._layer_idx in kvs:
                kv = kvs[self._layer_idx]
                kv_mask = self._wrapper._current_kv_masks.get(self._layer_idx)
                hs = self._xattn_layer(hs, kv, kv_attention_mask=kv_mask)

        if isinstance(outputs, tuple):
            return (hs,) + outputs[1:]
        return hs


class FlamingoOracleWrapper(nn.Module):
    """Wraps a PeftModel with Flamingo-style gated cross-attention layers.

    Each cross-attention layer at block i attends only to supervisee layer i's
    residual stream. No layer embeddings needed.
    """

    def __init__(self, peft_model: nn.Module, config, xattn_interval: int = 4, lora_r: int = 64, lora_alpha: int = 128):
        super().__init__()
        self.base_model = peft_model
        self.config = config

        n_layers = config.num_hidden_layers

        # Cross-attention every xattn_interval blocks
        # For Qwen3-0.6B (28 layers), interval=4 → [3, 7, 11, 15, 19, 23, 27]
        self.xattn_layer_indices = [i for i in range(xattn_interval - 1, n_layers, xattn_interval)]

        layers = self._get_transformer_layers()

        self.xattn_layers = nn.ModuleDict()
        for idx in self.xattn_layer_indices:
            xattn = GatedCrossAttentionLayer(config, layers[idx], lora_r=lora_r, lora_alpha=lora_alpha)
            self.xattn_layers[str(idx)] = xattn

        # Move new modules to same device/dtype as the base model
        device = next(peft_model.parameters()).device
        dtype = next(peft_model.parameters()).dtype
        self.xattn_layers.to(device=device, dtype=dtype)

        # Monkey-patch transformer blocks at xattn positions
        for idx in self.xattn_layer_indices:
            original_layer = layers[idx]
            wrapped = DecoderLayerWithCrossAttention(original_layer, self.xattn_layers[str(idx)], self, idx)
            layers[idx] = wrapped

        # Runtime state (set/cleared within forward pass)
        self._cot_len: int | None = None
        self._cot_mask: torch.Tensor | None = None
        self._current_kvs: dict[int, torch.Tensor] | None = None
        self._current_kv_masks: dict[int, torch.Tensor] = {}

    def _get_transformer_layers(self):
        if hasattr(self.base_model, "base_model"):
            return self.base_model.base_model.model.model.layers
        return self.base_model.model.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        cot_len: int | None = None,
        cot_mask: torch.Tensor | None = None,
        supervisee_kvs: dict[int, torch.Tensor] | None = None,
        supervisee_kv_masks: dict[int, torch.Tensor] | None = None,
    ):
        """
        Two modes:

        Chimera mode (cot_len provided):
            input_ids: [B, max_cot + max_oracle] — [left_pad_cot | cot | oracle | right_pad_oracle]
            cot_len: int — split point (= max_cot_len), same for all items
            cot_mask: [B, cot_len] — True for real CoT tokens (not padding)
            labels: [B, L] — -100 for CoT and padding positions

        Pre-extracted KV mode (supervisee_kvs provided):
            input_ids: [B, L] oracle input tokens
            supervisee_kvs: {layer_idx: [B, T_cot, D]} per-layer supervisee activations
            supervisee_kv_masks: {layer_idx: [B, T_cot]} per-layer attention masks
        """
        if cot_len is not None:
            self._cot_len = cot_len
            self._cot_mask = cot_mask
        else:
            self._cot_len = None
            self._current_kvs = supervisee_kvs
            self._current_kv_masks = supervisee_kv_masks or {}

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self._cot_len = None
        self._cot_mask = None
        self._current_kvs = None
        self._current_kv_masks = {}

        return outputs

    def generate(self, **kwargs):
        cot_len = kwargs.pop("cot_len", None)
        cot_mask = kwargs.pop("cot_mask", None)
        supervisee_kvs = kwargs.pop("supervisee_kvs", None)
        supervisee_kv_masks = kwargs.pop("supervisee_kv_masks", None)

        if cot_len is not None:
            self._cot_len = cot_len
            self._cot_mask = cot_mask
        else:
            self._current_kvs = supervisee_kvs
            self._current_kv_masks = supervisee_kv_masks or {}

        outputs = self.base_model.generate(**kwargs)

        self._cot_len = None
        self._cot_mask = None
        self._current_kvs = None
        self._current_kv_masks = {}
        return outputs

    def save_flamingo_modules(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        state = {
            "xattn_layers": self.xattn_layers.state_dict(),
            "xattn_layer_indices": self.xattn_layer_indices,
        }
        torch.save(state, os.path.join(path, "flamingo_modules.pt"))

    def load_flamingo_modules(self, path: str):
        import os
        state = torch.load(os.path.join(path, "flamingo_modules.pt"), map_location="cpu", weights_only=False)
        self.xattn_layers.load_state_dict(state["xattn_layers"])

    def print_trainable_parameters(self):
        self.base_model.print_trainable_parameters()
        xattn_total = sum(p.numel() for p in self.xattn_layers.parameters())
        xattn_trainable = sum(p.numel() for p in self.xattn_layers.parameters() if p.requires_grad)
        xattn_frozen = xattn_total - xattn_trainable
        print(f"  Flamingo xattn: {xattn_trainable:,} trainable / {xattn_frozen:,} frozen / {xattn_total:,} total")
        print(f"  Gates: {len(self.xattn_layer_indices)} (one per xattn layer)")
        print(f"  Flamingo new trainable: {xattn_trainable:,}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def save_pretrained(self, path: str, **kwargs):
        self.base_model.save_pretrained(path, **kwargs)
        self.save_flamingo_modules(path)
