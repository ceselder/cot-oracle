"""
Flamingo-Style Gated Cross-Attention Oracle (Two-Pass)

Pass 1 (CoT): Forward through frozen base model, collect hidden states at xattn layers.
Pass 2 (Oracle): Forward with cross-attention to cached CoT hidden states.

The oracle CANNOT read CoT text — it only accesses CoT via cross-attention
to intermediate hidden states.

Architecture per cross-attention layer:
    hidden = SelfAttn(hidden)       ← frozen base weights
    hidden = MLP(hidden)            ← frozen base weights
    hidden += tanh(gate) * CrossAttn(Q=hidden, KV=cot_hs)
                                    ↑ frozen base weights + LoRA adapters (trainable)

Gates start at 0 (identity at init). Per-layer cross-attention: block i
attends only to layer i's residual stream. CoT hidden states are detached.
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
    """Gated cross-attention: Q from oracle hidden states, KV from CoT activations.

    Projections initialized as frozen copies of self-attention weights + LoRA adapters.
    Gate starts at 0 → tanh(0) = 0 → identity at initialization.
    No RoPE (CoT positions aren't shared with oracle sequence).
    No causal mask (oracle freely attends to all CoT positions).
    """

    def __init__(self, config, source_layer: nn.Module, lora_r: int = 64, lora_alpha: int = 128):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim  # Qwen3 uses independent head_dim (128 for all sizes)
        self.q_dim = self.num_heads * self.head_dim  # may differ from hidden_size

        sa = source_layer.self_attn

        self.q_proj = LoRALinear(sa.q_proj, r=lora_r, lora_alpha=lora_alpha)
        self.k_proj = LoRALinear(sa.k_proj, r=lora_r, lora_alpha=lora_alpha)
        self.v_proj = LoRALinear(sa.v_proj, r=lora_r, lora_alpha=lora_alpha)
        self.o_proj = LoRALinear(sa.o_proj, r=lora_r, lora_alpha=lora_alpha)

        self.q_norm = deepcopy(sa.q_norm)
        self.k_norm = deepcopy(sa.k_norm)

        # Pre-norm (query side)
        self.norm = deepcopy(source_layer.input_layernorm)

        # Ensure deepcopy'd norms are trainable (source may have been frozen)
        for module in [self.q_norm, self.k_norm, self.norm]:
            for p in module.parameters():
                p.requires_grad = True

        # Gate initialized to 0 → tanh(0) = 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, kv: torch.Tensor, kv_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_q, D] oracle hidden states
            kv: [B, T_cot, D] CoT activations at this layer
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

        # Pass None when all-True to enable flash attention
        attn_mask = None
        if kv_attention_mask is not None and not kv_attention_mask.all():
            attn_mask = kv_attention_mask[:, None, None, :].expand(-1, -1, L_q, -1)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.q_dim)
        attn_output = self.o_proj(attn_output)

        return residual + torch.tanh(self.gate) * attn_output


class DecoderLayerWithCrossAttention(nn.Module):
    """Wraps an original decoder layer + appends gated cross-attention.

    During oracle forward pass, cross-attends to cached CoT hidden states.
    During CoT collection pass (no kvs set), just runs the original layer.
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

        kvs = self._wrapper._current_kvs
        if kvs is not None and self._layer_idx in kvs:
            kv = kvs[self._layer_idx]
            kv_mask = self._wrapper._current_kv_mask
            hs = self._xattn_layer(hs, kv, kv_attention_mask=kv_mask)

        if isinstance(outputs, tuple):
            return (hs,) + outputs[1:]
        return hs


class FlamingoOracleWrapper(nn.Module):
    """Wraps a frozen base model with Flamingo-style gated cross-attention layers.

    Two-pass usage:
        1. collect_cot_hidden_states(cot_input_ids, cot_attention_mask)
           → {layer_idx: [B, L_cot, D]} (no grad, frozen model)
        2. forward(oracle_input_ids, oracle_attention_mask, labels, supervisee_kvs=...)
           → oracle outputs with cross-attention to CoT hidden states

    Only the cross-attention LoRA + gates are trainable. Base model is frozen.
    """

    def __init__(self, base_model: nn.Module, config, xattn_interval: int = 4, lora_r: int = 64, lora_alpha: int = 128):
        super().__init__()
        self.base_model = base_model
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
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.xattn_layers.to(device=device, dtype=dtype)

        # Monkey-patch transformer blocks at xattn positions
        for idx in self.xattn_layer_indices:
            original_layer = layers[idx]
            wrapped = DecoderLayerWithCrossAttention(original_layer, self.xattn_layers[str(idx)], self, idx)
            layers[idx] = wrapped

        # Runtime state (set/cleared within forward pass)
        self._current_kvs: dict[int, torch.Tensor] | None = None
        self._current_kv_mask: torch.Tensor | None = None

    def _get_transformer_layers(self):
        model = self.base_model
        # Raw HF model: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        # PeftModel: model.base_model.model.model.layers
        return model.base_model.model.model.layers

    def collect_cot_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass through frozen base model, collecting hidden states at xattn layer indices.

        Returns: {layer_idx: [B, L_cot, D]} detached hidden states.
        """
        hooks = []
        hidden_states = {}

        layers = self._get_transformer_layers()
        for idx in self.xattn_layer_indices:
            layer = layers[idx]
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    hidden_states[layer_idx] = hs.detach()
                return hook_fn
            target = layer.original_layer if hasattr(layer, 'original_layer') else layer
            hooks.append(target.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            if hasattr(self.base_model, 'disable_adapter'):
                with self.base_model.disable_adapter():
                    self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        supervisee_kvs: dict[int, torch.Tensor] | None = None,
        cot_attention_mask: torch.Tensor | None = None,
    ):
        """Oracle forward pass with cross-attention to cached CoT hidden states.

        Args:
            input_ids: [B, L_oracle] oracle input tokens
            attention_mask: [B, L_oracle] oracle attention mask
            labels: [B, L_oracle] oracle training labels
            supervisee_kvs: {layer_idx: [B, L_cot, D]} from collect_cot_hidden_states()
            cot_attention_mask: [B, L_cot] mask for CoT tokens (True = attend)
        """
        self._current_kvs = supervisee_kvs
        self._current_kv_mask = cot_attention_mask

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self._current_kvs = None
        self._current_kv_mask = None

        return outputs

    def generate(self, **kwargs):
        supervisee_kvs = kwargs.pop("supervisee_kvs", None)
        cot_attention_mask = kwargs.pop("cot_attention_mask", None)

        self._current_kvs = supervisee_kvs
        self._current_kv_mask = cot_attention_mask

        outputs = self.base_model.generate(**kwargs)

        self._current_kvs = None
        self._current_kv_mask = None
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
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}")
        xattn_total = sum(p.numel() for p in self.xattn_layers.parameters())
        xattn_trainable = sum(p.numel() for p in self.xattn_layers.parameters() if p.requires_grad)
        xattn_frozen = xattn_total - xattn_trainable
        print(f"  Flamingo xattn: {xattn_trainable:,} trainable / {xattn_frozen:,} frozen / {xattn_total:,} total")
        print(f"  Gates: {len(self.xattn_layer_indices)} (one per xattn layer)")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def save_pretrained(self, path: str, **kwargs):
        self.save_flamingo_modules(path)
