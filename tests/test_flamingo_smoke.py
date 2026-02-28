"""Smoke tests for Flamingo-style cross-attention oracle (two-pass).

Uses Qwen3-0.6B for speed (same architecture family as 8B).
Qwen3-0.6B: hidden=1024, heads=16, kv_heads=8, head_dim=128, layers=28.

Run: python tests/test_flamingo_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flamingo_oracle import FlamingoOracleWrapper


MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load_model():
    """Load Qwen3-0.6B with frozen weights (no LoRA)."""
    print(f"Loading {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map={"": DEVICE},
        attn_implementation="sdpa",
    )
    base_model.enable_input_require_grads()
    for p in base_model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return base_model, base_model.config, tokenizer


def test_two_pass_forward_backward():
    """Test: Two-pass — collect CoT hidden states, then oracle forward + backward."""
    base_model, config, tokenizer = load_model()

    wrapper = FlamingoOracleWrapper(base_model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    wrapper.print_trainable_parameters()

    B = 2
    L_cot, L_oracle = 40, 20

    cot_ids = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_mask = torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)
    oracle_labels = torch.full((B, L_oracle), -100, dtype=torch.long, device=DEVICE)
    oracle_labels[:, 5:] = torch.randint(0, config.vocab_size, (B, L_oracle - 5), device=DEVICE)

    # Pass 1: Collect CoT hidden states
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        cot_hs = wrapper.collect_cot_hidden_states(cot_ids, cot_mask)

    assert len(cot_hs) == len(wrapper.xattn_layer_indices)
    for idx in wrapper.xattn_layer_indices:
        assert cot_hs[idx].shape == (B, L_cot, config.hidden_size), \
            f"Layer {idx}: expected ({B}, {L_cot}, {config.hidden_size}), got {cot_hs[idx].shape}"
    print(f"  Pass 1 (CoT): OK — collected {len(cot_hs)} layer hidden states")

    # Pass 2: Oracle forward with cross-attention
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = wrapper(
            input_ids=oracle_ids, attention_mask=oracle_mask, labels=oracle_labels,
            supervisee_kvs=cot_hs, cot_attention_mask=cot_mask,
        )

    assert outputs.logits.shape == (B, L_oracle, config.vocab_size), \
        f"Expected ({B}, {L_oracle}, {config.vocab_size}), got {outputs.logits.shape}"
    assert outputs.loss is not None and outputs.loss.item() > 0
    print(f"  Pass 2 (Oracle): OK (logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f})")

    outputs.loss.backward()
    for idx in wrapper.xattn_layer_indices:
        gate = wrapper.xattn_layers[str(idx)].gate
        assert gate.grad is not None, f"Gate at layer {idx} has no gradient"
    # Base model should have NO gradients
    for p in base_model.parameters():
        assert p.grad is None, "Base model should have no gradients"
    print(f"  Backward: OK (all gates have gradients, base model frozen)")


def test_gate_zero_identity():
    """Test: gate=0 → oracle output is invariant to CoT content.

    Since tanh(0)=0, cross-attention output is zeroed. Different CoTs should
    produce identical oracle logits.
    """
    base_model, config, tokenizer = load_model()

    wrapper = FlamingoOracleWrapper(base_model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    for idx in wrapper.xattn_layer_indices:
        assert wrapper.xattn_layers[str(idx)].gate.item() == 0.0

    B = 1
    L_cot, L_oracle = 30, 20

    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)

    # Two different CoTs
    cot_ids_a = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_ids_b = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_mask = torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        cot_hs_a = wrapper.collect_cot_hidden_states(cot_ids_a, cot_mask)
        cot_hs_b = wrapper.collect_cot_hidden_states(cot_ids_b, cot_mask)

        out_a = wrapper(input_ids=oracle_ids, attention_mask=oracle_mask,
                        supervisee_kvs=cot_hs_a, cot_attention_mask=cot_mask)
        out_b = wrapper(input_ids=oracle_ids, attention_mask=oracle_mask,
                        supervisee_kvs=cot_hs_b, cot_attention_mask=cot_mask)

    diff = (out_a.logits - out_b.logits).abs().max().item()
    print(f"  Gate=0 identity: max oracle logit diff (different CoTs) = {diff:.2e}")
    assert diff < 1e-3, f"Gate=0 should make oracle invariant to CoT content, but max diff = {diff}"
    print(f"  Gate=0 identity: PASSED")


def test_different_sequence_lengths():
    """Test: CoT and oracle can have different sequence lengths (the whole point)."""
    base_model, config, tokenizer = load_model()

    wrapper = FlamingoOracleWrapper(base_model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    B = 2
    L_cot, L_oracle = 100, 20  # CoT much longer than oracle

    cot_ids = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_mask = torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)
    oracle_labels = torch.full((B, L_oracle), -100, dtype=torch.long, device=DEVICE)
    oracle_labels[:, 5:] = torch.randint(0, config.vocab_size, (B, L_oracle - 5), device=DEVICE)

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        cot_hs = wrapper.collect_cot_hidden_states(cot_ids, cot_mask)
        outputs = wrapper(
            input_ids=oracle_ids, attention_mask=oracle_mask, labels=oracle_labels,
            supervisee_kvs=cot_hs, cot_attention_mask=cot_mask,
        )

    assert outputs.logits.shape == (B, L_oracle, config.vocab_size)
    assert outputs.loss is not None and outputs.loss.item() > 0
    outputs.loss.backward()
    print(f"  Different lengths (CoT={L_cot}, Oracle={L_oracle}): OK (loss={outputs.loss.item():.4f})")


def test_save_load_roundtrip():
    """Test: Save and load Flamingo modules, verify identical outputs."""
    import tempfile

    base_model, config, tokenizer = load_model()

    wrapper = FlamingoOracleWrapper(base_model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    # Perturb a gate so it's non-zero
    wrapper.xattn_layers[str(wrapper.xattn_layer_indices[0])].gate.data.fill_(0.5)

    B, L_cot, L_oracle = 1, 20, 16

    cot_ids = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_mask = torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_mask = torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        cot_hs = wrapper.collect_cot_hidden_states(cot_ids, cot_mask)
        out_before = wrapper(
            input_ids=oracle_ids, attention_mask=oracle_mask,
            supervisee_kvs=cot_hs, cot_attention_mask=cot_mask,
        ).logits.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_flamingo_modules(tmpdir)

        # Reload into fresh wrapper (same base model — weights are frozen/shared)
        wrapper2 = FlamingoOracleWrapper(base_model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
        wrapper2.load_flamingo_modules(tmpdir)

    wrapper2.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        cot_hs2 = wrapper2.collect_cot_hidden_states(cot_ids, cot_mask)
        out_after = wrapper2(
            input_ids=oracle_ids, attention_mask=oracle_mask,
            supervisee_kvs=cot_hs2, cot_attention_mask=cot_mask,
        ).logits

    diff = (out_before - out_after).abs().max().item()
    print(f"  Save/load roundtrip: max logit diff = {diff:.2e}")
    assert diff < 1e-3, f"Save/load should produce identical outputs, but max diff = {diff}"
    print(f"  Save/load roundtrip: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Flamingo Oracle Smoke Tests (Two-Pass)")
    print("=" * 60)

    print("\n1. Two-pass forward + backward")
    test_two_pass_forward_backward()

    print("\n2. Gate=0 identity")
    test_gate_zero_identity()

    print("\n3. Different sequence lengths")
    test_different_sequence_lengths()

    print("\n4. Save/load roundtrip")
    test_save_load_roundtrip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
