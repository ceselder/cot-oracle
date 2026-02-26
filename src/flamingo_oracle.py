"""
Flamingo-Style Gated Cross-Attention Oracle

Replaces additive steering injection with learned cross-attention layers.
The oracle (Qwen3-8B + LoRA) attends to the supervisee's full residual stream
(all 36 layers, all CoT tokens) via interleaved gated cross-attention.

Cross-attention projections are initialized from the base model's pretrained
self-attention weights with frozen base + LoRA adapters on top.
Gates start at 0 (identity at init).

Architecture per cross-attention layer:
    hidden = SelfAttn(hidden)       ← existing LoRA (via PEFT)
    hidden = MLP(hidden)            ← existing LoRA (via PEFT)
    hidden = hidden + tanh(gate) * CrossAttn(Q=hidden, KV=supervisee_acts)
                                    ↑ frozen base weights + manual LoRA adapters
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

    def __init__(self, base_linear: nn.Linear, r: int = 64, lora_alpha: int = 128, lora_dropout: float = 0.05):
        super().__init__()
        # Create a clean nn.Linear copy (strip any LoRA wrappers from PEFT)
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

        # Initialize: A ~ N(0, 1/r), B = 0 → LoRA output starts at 0
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
        """
        Args:
            config: model config (has hidden_size, num_attention_heads, num_key_value_heads)
            source_layer: the Qwen3DecoderLayer to copy weights from
            lora_r: LoRA rank for cross-attention projections
            lora_alpha: LoRA alpha scaling
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim  # Qwen3 uses independent head_dim (128 for all sizes)
        self.q_dim = self.num_heads * self.head_dim  # may differ from hidden_size
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        sa = source_layer.self_attn

        # Q/K/V/O as frozen base + LoRA
        self.q_proj = LoRALinear(sa.q_proj, r=lora_r, lora_alpha=lora_alpha)
        self.k_proj = LoRALinear(sa.k_proj, r=lora_r, lora_alpha=lora_alpha)
        self.v_proj = LoRALinear(sa.v_proj, r=lora_r, lora_alpha=lora_alpha)
        self.o_proj = LoRALinear(sa.o_proj, r=lora_r, lora_alpha=lora_alpha)

        # Copy QK norms (trainable, small)
        self.q_norm = deepcopy(sa.q_norm)
        self.k_norm = deepcopy(sa.k_norm)

        # Pre-norm (query side) — copy from input_layernorm (trainable, small)
        self.norm = deepcopy(source_layer.input_layernorm)

        # Gate initialized to 0 → tanh(0) = 0
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, kv: torch.Tensor, kv_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L_q, D] oracle hidden states
            kv: [B, L_kv, D] supervisee activations (with layer embeddings added)
            kv_attention_mask: [B, L_kv] bool mask (True = attend, False = ignore)

        Returns:
            [B, L_q, D] = hidden_states + tanh(gate) * cross_attn_output
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        B, L_q, _ = hidden_states.shape
        L_kv = kv.shape[1]

        # Project Q from oracle, K/V from supervisee
        q = self.q_proj(hidden_states)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Reshape to [B, n_heads, L, head_dim]
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK norms (per-head RMSNorm, matches Qwen3 convention)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L_kv, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, L_kv, self.head_dim)

        # Build attention mask for SDPA: [B, 1, L_q, L_kv]
        attn_mask = None
        if kv_attention_mask is not None:
            # kv_attention_mask: [B, L_kv] bool, True=attend
            attn_mask = kv_attention_mask[:, None, None, :].expand(-1, -1, L_q, -1)

        # Scaled dot-product attention (flash/memory-efficient via SDPA)
        # is_causal=False: oracle freely attends to all supervisee positions
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        # Reshape back and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.q_dim)
        attn_output = self.o_proj(attn_output)

        return residual + torch.tanh(self.gate) * attn_output


class DecoderLayerWithCrossAttention(nn.Module):
    """Wraps an original decoder layer + appends gated cross-attention.

    Forward: run original layer → extract hidden_states → cross-attend → repack output.
    """

    def __init__(self, original_layer: nn.Module, xattn_layer: GatedCrossAttentionLayer, wrapper: "FlamingoOracleWrapper"):
        super().__init__()
        self.original_layer = original_layer
        # Store as plain Python attributes (bypass nn.Module.__setattr__) to avoid
        # registering as submodules — prevents double-registration and circular refs
        object.__setattr__(self, '_xattn_layer', xattn_layer)
        object.__setattr__(self, '_wrapper', wrapper)

    def __getattr__(self, name: str):
        """Proxy attribute access to the original decoder layer for compatibility.

        Qwen3's forward loop accesses decoder_layer.attention_type, etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        # Run original decoder layer
        outputs = self.original_layer(hidden_states, *args, **kwargs)

        # Extract hidden_states from output tuple
        if isinstance(outputs, tuple):
            hs = outputs[0]
        else:
            hs = outputs

        # Apply cross-attention if supervisee activations are available
        kv = self._wrapper._current_kv
        if kv is not None:
            kv_mask = self._wrapper._current_kv_mask
            hs = self._xattn_layer(hs, kv, kv_attention_mask=kv_mask)

        # Repack into same tuple format
        if isinstance(outputs, tuple):
            return (hs,) + outputs[1:]
        return hs


class FlamingoOracleWrapper(nn.Module):
    """Wraps a PeftModel with Flamingo-style gated cross-attention layers.

    The wrapper:
    1. Adds learned layer embeddings to supervisee activations
    2. Monkey-patches transformer blocks at xattn positions with DecoderLayerWithCrossAttention
    3. During forward, sets _current_kv for patched layers to read

    All new parameters (xattn layers, layer embeddings, gates) are registered as
    proper submodules → DDP finds them automatically.
    """

    def __init__(self, peft_model: nn.Module, config, xattn_interval: int = 4, lora_r: int = 64, lora_alpha: int = 128):
        """
        Args:
            peft_model: PeftModel (already has LoRA on self-attn + MLP)
            config: model config (for hidden_size, num_layers, etc.)
            xattn_interval: insert cross-attention every N blocks
            lora_r: LoRA rank for cross-attention projections
            lora_alpha: LoRA alpha for cross-attention projections
        """
        super().__init__()
        self.base_model = peft_model
        self.config = config

        n_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        # Determine xattn layer positions: every xattn_interval blocks
        # For Qwen3-8B (36 layers), interval=4 → [3, 7, 11, 15, 19, 23, 27, 31, 35]
        self.xattn_layer_indices = [i for i in range(xattn_interval - 1, n_layers, xattn_interval)]

        # Learned layer embeddings: [n_supervisee_layers, hidden_size]
        self.layer_embedding = nn.Embedding(n_layers, hidden_size)
        nn.init.normal_(self.layer_embedding.weight, std=0.02)

        # Access the actual transformer layers (in the peft model)
        layers = self._get_transformer_layers()

        # Create cross-attention layers
        # LoRALinear copies .weight.data from the source (base weight even if PEFT-wrapped)
        self.xattn_layers = nn.ModuleDict()
        for idx in self.xattn_layer_indices:
            xattn = GatedCrossAttentionLayer(config, layers[idx], lora_r=lora_r, lora_alpha=lora_alpha)
            self.xattn_layers[str(idx)] = xattn

        # Move new modules to same device/dtype as the base model
        device = next(peft_model.parameters()).device
        dtype = next(peft_model.parameters()).dtype
        self.xattn_layers.to(device=device, dtype=dtype)
        self.layer_embedding.to(device=device, dtype=dtype)

        # Monkey-patch transformer blocks at xattn positions
        for idx in self.xattn_layer_indices:
            original_layer = layers[idx]
            wrapped = DecoderLayerWithCrossAttention(original_layer, self.xattn_layers[str(idx)], self)
            layers[idx] = wrapped

        # Runtime state (set/cleared within forward pass)
        self._current_kv: torch.Tensor | None = None
        self._current_kv_mask: torch.Tensor | None = None

    def _get_transformer_layers(self):
        """Get the nn.ModuleList of transformer layers from the (possibly wrapped) model."""
        # PeftModel wrapping: base_model.model.model.layers
        if hasattr(self.base_model, "base_model"):
            return self.base_model.base_model.model.model.layers
        # Plain model: model.model.layers
        return self.base_model.model.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        supervisee_activations: torch.Tensor | None = None,
        supervisee_layer_ids: torch.Tensor | None = None,
        supervisee_attention_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            input_ids: [B, L] oracle input tokens
            attention_mask: [B, L] oracle attention mask
            labels: [B, L] training labels (-100 for ignored)
            supervisee_activations: [B, L_kv, D] flattened supervisee residual stream
            supervisee_layer_ids: [B, L_kv] layer index for each activation position
            supervisee_attention_mask: [B, L_kv] bool mask for supervisee activations
        """
        # Add layer embeddings to supervisee activations
        if supervisee_activations is not None:
            layer_embs = self.layer_embedding(supervisee_layer_ids)  # [B, L_kv, D]
            self._current_kv = supervisee_activations + layer_embs
            self._current_kv_mask = supervisee_attention_mask
        else:
            self._current_kv = None
            self._current_kv_mask = None

        # Forward through the base model (patched layers will read _current_kv)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Clean up runtime state
        self._current_kv = None
        self._current_kv_mask = None

        return outputs

    def generate(self, **kwargs):
        """Proxy generate to base_model, with optional supervisee activations."""
        supervisee_activations = kwargs.pop("supervisee_activations", None)
        supervisee_layer_ids = kwargs.pop("supervisee_layer_ids", None)
        supervisee_attention_mask = kwargs.pop("supervisee_attention_mask", None)

        if supervisee_activations is not None:
            layer_embs = self.layer_embedding(supervisee_layer_ids)
            self._current_kv = supervisee_activations + layer_embs
            self._current_kv_mask = supervisee_attention_mask

        outputs = self.base_model.generate(**kwargs)

        self._current_kv = None
        self._current_kv_mask = None
        return outputs

    def save_flamingo_modules(self, path: str):
        """Save xattn layers + layer embeddings (base model LoRA saved separately by PEFT)."""
        import os
        os.makedirs(path, exist_ok=True)
        state = {
            "xattn_layers": self.xattn_layers.state_dict(),
            "layer_embedding": self.layer_embedding.state_dict(),
            "xattn_layer_indices": self.xattn_layer_indices,
        }
        torch.save(state, os.path.join(path, "flamingo_modules.pt"))

    def load_flamingo_modules(self, path: str):
        """Load xattn layers + layer embeddings from checkpoint."""
        import os
        state = torch.load(os.path.join(path, "flamingo_modules.pt"), map_location="cpu", weights_only=False)
        self.xattn_layers.load_state_dict(state["xattn_layers"])
        self.layer_embedding.load_state_dict(state["layer_embedding"])

    def print_trainable_parameters(self):
        """Print parameter summary including Flamingo modules."""
        # Base model (LoRA) params
        self.base_model.print_trainable_parameters()

        # Flamingo-specific params
        xattn_total = sum(p.numel() for p in self.xattn_layers.parameters())
        xattn_trainable = sum(p.numel() for p in self.xattn_layers.parameters() if p.requires_grad)
        xattn_frozen = xattn_total - xattn_trainable
        emb_params = sum(p.numel() for p in self.layer_embedding.parameters())

        print(f"  Flamingo xattn: {xattn_trainable:,} trainable / {xattn_frozen:,} frozen / {xattn_total:,} total")
        print(f"  Layer embeddings: {emb_params:,}")
        print(f"  Gates: {len(self.xattn_layer_indices)} (one per xattn layer)")
        print(f"  Flamingo new trainable: {xattn_trainable + emb_params:,}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # train() and eval() inherited from nn.Module — no override needed.
    # super().train(mode) recursively sets mode on all children including base_model.

    def save_pretrained(self, path: str, **kwargs):
        """Save both LoRA adapters and Flamingo modules."""
        self.base_model.save_pretrained(path, **kwargs)
        self.save_flamingo_modules(path)
