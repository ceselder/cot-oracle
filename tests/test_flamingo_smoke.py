"""Smoke tests for Flamingo-style cross-attention oracle.

Uses Qwen3-0.6B for speed (same architecture family as 8B).
Qwen3-0.6B: hidden=1024, heads=16, kv_heads=8, head_dim=64, layers=28.

Run: python tests/test_flamingo_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from flamingo_oracle import FlamingoOracleWrapper


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load_model():
    """Load Qwen3-0.6B with fresh LoRA."""
    print(f"Loading {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map={"": DEVICE},
        attn_implementation="sdpa",
    )
    base_model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, base_model, tokenizer


def test_forward_pass():
    """Test: FlamingoOracleWrapper forward pass produces correct output shape."""
    model, base_model, tokenizer = load_model()
    config = base_model.config
    n_layers = config.num_hidden_layers  # 28 for 0.6B
    hidden_size = config.hidden_size  # 1024 for 0.6B

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    wrapper.print_trainable_parameters()

    # Create dummy inputs
    B, L_oracle = 2, 32
    T_cot = 20  # CoT length per layer
    L_kv = n_layers * T_cot

    input_ids = torch.randint(0, tokenizer.vocab_size, (B, L_oracle), device=DEVICE)
    attention_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)
    labels = torch.randint(0, tokenizer.vocab_size, (B, L_oracle), device=DEVICE)
    labels[:, :10] = -100  # mask first 10 as prompt

    supervisee_acts = torch.randn(B, L_kv, hidden_size, dtype=DTYPE, device=DEVICE)
    layer_ids = torch.arange(n_layers, device=DEVICE).unsqueeze(1).expand(-1, T_cot).reshape(-1).unsqueeze(0).expand(B, -1)
    act_mask = torch.ones(B, L_kv, dtype=torch.bool, device=DEVICE)

    # Forward pass
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            supervisee_activations=supervisee_acts,
            supervisee_layer_ids=layer_ids,
            supervisee_attention_mask=act_mask,
        )

    vocab_size = base_model.config.vocab_size
    assert outputs.logits.shape == (B, L_oracle, vocab_size), \
        f"Expected ({B}, {L_oracle}, {vocab_size}), got {outputs.logits.shape}"
    assert outputs.loss is not None and outputs.loss.item() > 0

    print(f"  Forward pass: OK (logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f})")

    # Test backward
    outputs.loss.backward()
    # Check gates have gradients
    for idx in wrapper.xattn_layer_indices:
        gate = wrapper.xattn_layers[str(idx)].gate
        assert gate.grad is not None, f"Gate at layer {idx} has no gradient"
    print(f"  Backward pass: OK (all gates have gradients)")


def test_gate_zero_identity():
    """Test: With gates=0, wrapper output matches base model output (no cross-attn contribution)."""
    model, base_model, tokenizer = load_model()
    config = base_model.config
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    # Verify all gates are zero
    for idx in wrapper.xattn_layer_indices:
        assert wrapper.xattn_layers[str(idx)].gate.item() == 0.0

    B, L = 2, 32
    T_cot = 10
    L_kv = n_layers * T_cot

    input_ids = torch.randint(0, tokenizer.vocab_size, (B, L), device=DEVICE)
    attention_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)

    supervisee_acts = torch.randn(B, L_kv, hidden_size, dtype=DTYPE, device=DEVICE)
    layer_ids = torch.arange(n_layers, device=DEVICE).unsqueeze(1).expand(-1, T_cot).reshape(-1).unsqueeze(0).expand(B, -1)
    act_mask = torch.ones(B, L_kv, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        # With supervisee activations (gates=0 → should be identity)
        out_with = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            supervisee_activations=supervisee_acts,
            supervisee_layer_ids=layer_ids,
            supervisee_attention_mask=act_mask,
        )

        # Without supervisee activations (pure base model)
        out_without = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
        )

    diff = (out_with.logits - out_without.logits).abs().max().item()
    print(f"  Gate=0 identity test: max logit diff = {diff:.2e}")
    # With LoRA B=0 init, the LoRA output is 0, so xattn produces base_proj output only.
    # But tanh(0)=0 gates it to zero. So diff should be 0 (or very small due to float precision).
    assert diff < 1e-3, f"Gate=0 should produce identical outputs, but max diff = {diff}"
    print(f"  Gate=0 identity: PASSED")


def test_save_load_roundtrip():
    """Test: Save and load Flamingo modules, verify identical outputs."""
    import tempfile

    model, base_model, tokenizer = load_model()
    config = base_model.config
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    # Perturb a gate so it's non-zero
    wrapper.xattn_layers[str(wrapper.xattn_layer_indices[0])].gate.data.fill_(0.5)

    B, L = 1, 16
    T_cot = 5
    L_kv = n_layers * T_cot

    input_ids = torch.randint(0, tokenizer.vocab_size, (B, L), device=DEVICE)
    attention_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    supervisee_acts = torch.randn(B, L_kv, hidden_size, dtype=DTYPE, device=DEVICE)
    layer_ids = torch.arange(n_layers, device=DEVICE).unsqueeze(1).expand(-1, T_cot).reshape(-1).unsqueeze(0).expand(B, -1)
    act_mask = torch.ones(B, L_kv, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_before = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            supervisee_activations=supervisee_acts,
            supervisee_layer_ids=layer_ids,
            supervisee_attention_mask=act_mask,
        ).logits.clone()

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_flamingo_modules(tmpdir)

        # Create new wrapper and load
        model2, base_model2, _ = load_model()
        wrapper2 = FlamingoOracleWrapper(model2, config, xattn_interval=4, lora_r=8, lora_alpha=16)
        wrapper2.load_flamingo_modules(tmpdir)

    wrapper2.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_after = wrapper2(
            input_ids=input_ids.to(next(wrapper2.parameters()).device),
            attention_mask=attention_mask.to(next(wrapper2.parameters()).device),
            supervisee_activations=supervisee_acts.to(next(wrapper2.parameters()).device),
            supervisee_layer_ids=layer_ids.to(next(wrapper2.parameters()).device),
            supervisee_attention_mask=act_mask.to(next(wrapper2.parameters()).device),
        ).logits

    diff = (out_before - out_after.to(out_before.device)).abs().max().item()
    print(f"  Save/load roundtrip: max logit diff = {diff:.2e}")
    assert diff < 1e-3, f"Save/load should produce identical outputs, but max diff = {diff}"
    print(f"  Save/load roundtrip: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Flamingo Oracle Smoke Tests")
    print("=" * 60)

    print("\n1. Forward + backward pass")
    test_forward_pass()

    print("\n2. Gate=0 identity")
    test_gate_zero_identity()

    print("\n3. Save/load roundtrip")
    test_save_load_roundtrip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
