"""Smoke tests for Flamingo-style cross-attention oracle.

Uses Qwen3-0.6B for speed (same architecture family as 8B).
Qwen3-0.6B: hidden=1024, heads=16, kv_heads=8, head_dim=128, layers=28.

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


def make_per_layer_kvs(wrapper, B, T_cot, hidden_size):
    """Create dummy per-layer supervisee KVs for testing."""
    kvs = {}
    kv_masks = {}
    for idx in wrapper.xattn_layer_indices:
        kvs[idx] = torch.randn(B, T_cot, hidden_size, dtype=DTYPE, device=DEVICE)
        kv_masks[idx] = torch.ones(B, T_cot, dtype=torch.bool, device=DEVICE)
    return kvs, kv_masks


def test_forward_pass():
    """Test: FlamingoOracleWrapper forward pass produces correct output shape."""
    model, base_model, tokenizer = load_model()
    config = base_model.config
    hidden_size = config.hidden_size

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    wrapper.print_trainable_parameters()

    B, L_oracle = 2, 32
    T_cot = 20

    input_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    attention_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    labels[:, :10] = -100

    kvs, kv_masks = make_per_layer_kvs(wrapper, B, T_cot, hidden_size)

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = wrapper(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            supervisee_kvs=kvs, supervisee_kv_masks=kv_masks,
        )

    vocab_size = config.vocab_size
    assert outputs.logits.shape == (B, L_oracle, vocab_size), \
        f"Expected ({B}, {L_oracle}, {vocab_size}), got {outputs.logits.shape}"
    assert outputs.loss is not None and outputs.loss.item() > 0

    print(f"  Forward pass: OK (logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f})")

    outputs.loss.backward()
    for idx in wrapper.xattn_layer_indices:
        gate = wrapper.xattn_layers[str(idx)].gate
        assert gate.grad is not None, f"Gate at layer {idx} has no gradient"
    print(f"  Backward pass: OK (all gates have gradients)")


def test_gate_zero_identity():
    """Test: With gates=0, wrapper output matches base model output."""
    model, base_model, tokenizer = load_model()
    config = base_model.config
    hidden_size = config.hidden_size

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    for idx in wrapper.xattn_layer_indices:
        assert wrapper.xattn_layers[str(idx)].gate.item() == 0.0

    B, L = 2, 32
    T_cot = 10

    input_ids = torch.randint(0, config.vocab_size, (B, L), device=DEVICE)
    attention_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)

    kvs, kv_masks = make_per_layer_kvs(wrapper, B, T_cot, hidden_size)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_with = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            supervisee_kvs=kvs, supervisee_kv_masks=kv_masks,
        )
        out_without = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
        )

    diff = (out_with.logits - out_without.logits).abs().max().item()
    print(f"  Gate=0 identity test: max logit diff = {diff:.2e}")
    assert diff < 1e-3, f"Gate=0 should produce identical outputs, but max diff = {diff}"
    print(f"  Gate=0 identity: PASSED")


def test_save_load_roundtrip():
    """Test: Save and load Flamingo modules, verify identical outputs."""
    import tempfile

    model, base_model, tokenizer = load_model()
    config = base_model.config
    hidden_size = config.hidden_size

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    # Perturb a gate so it's non-zero
    wrapper.xattn_layers[str(wrapper.xattn_layer_indices[0])].gate.data.fill_(0.5)

    B, L = 1, 16
    T_cot = 5

    input_ids = torch.randint(0, config.vocab_size, (B, L), device=DEVICE)
    attention_mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    kvs, kv_masks = make_per_layer_kvs(wrapper, B, T_cot, hidden_size)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_before = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            supervisee_kvs=kvs, supervisee_kv_masks=kv_masks,
        ).logits.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_flamingo_modules(tmpdir)

        model2, base_model2, _ = load_model()
        wrapper2 = FlamingoOracleWrapper(model2, config, xattn_interval=4, lora_r=8, lora_alpha=16)
        wrapper2.load_flamingo_modules(tmpdir)

    wrapper2.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_after = wrapper2(
            input_ids=input_ids.to(next(wrapper2.parameters()).device),
            attention_mask=attention_mask.to(next(wrapper2.parameters()).device),
            supervisee_kvs={k: v.to(next(wrapper2.parameters()).device) for k, v in kvs.items()},
            supervisee_kv_masks={k: v.to(next(wrapper2.parameters()).device) for k, v in kv_masks.items()},
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
