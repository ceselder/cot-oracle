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


def test_chimera_forward_backward():
    """Test: Chimera (single-pass) mode — [CoT | Oracle] concatenated."""
    model, base_model, tokenizer = load_model()
    config = base_model.config

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    B = 2
    T_cot = 20  # CoT length
    L_oracle = 32  # Oracle length

    # Simulate [cot_pad | cot | oracle | oracle_pad] layout
    # Item 0: full cot (20) + full oracle (32)
    # Item 1: shorter cot (15) + shorter oracle (28), padded
    cot_lens = [T_cot, 15]
    oracle_lens = [L_oracle, 28]
    max_cot = T_cot
    max_oracle = L_oracle

    all_ids = []
    all_labels = []
    all_mask = []
    all_cot_mask = []

    for i in range(B):
        cot_pad = max_cot - cot_lens[i]
        oracle_pad = max_oracle - oracle_lens[i]
        ids = [tokenizer.pad_token_id] * cot_pad + \
              torch.randint(0, config.vocab_size, (cot_lens[i],)).tolist() + \
              torch.randint(0, config.vocab_size, (oracle_lens[i],)).tolist() + \
              [tokenizer.pad_token_id] * oracle_pad
        labs = [-100] * max_cot + \
               [-100] * 5 + torch.randint(0, config.vocab_size, (oracle_lens[i] - 5,)).tolist() + \
               [-100] * oracle_pad
        mask = [False] * cot_pad + [True] * (cot_lens[i] + oracle_lens[i]) + [False] * oracle_pad
        cmask = [False] * cot_pad + [True] * cot_lens[i]

        all_ids.append(torch.tensor(ids, dtype=torch.long, device=DEVICE))
        all_labels.append(torch.tensor(labs, dtype=torch.long, device=DEVICE))
        all_mask.append(torch.tensor(mask, dtype=torch.bool, device=DEVICE))
        all_cot_mask.append(torch.tensor(cmask, dtype=torch.bool, device=DEVICE))

    input_ids = torch.stack(all_ids)
    attention_mask = torch.stack(all_mask)
    labels = torch.stack(all_labels)
    cot_mask = torch.stack(all_cot_mask)

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = wrapper(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            cot_len=max_cot, cot_mask=cot_mask,
        )

    total_len = max_cot + max_oracle
    assert outputs.logits.shape == (B, total_len, config.vocab_size), \
        f"Expected ({B}, {total_len}, {config.vocab_size}), got {outputs.logits.shape}"
    assert outputs.loss is not None and outputs.loss.item() > 0
    print(f"  Chimera forward: OK (logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f})")

    outputs.loss.backward()
    for idx in wrapper.xattn_layer_indices:
        gate = wrapper.xattn_layers[str(idx)].gate
        assert gate.grad is not None, f"Gate at layer {idx} has no gradient"
    print(f"  Chimera backward: OK (all gates have gradients)")


def test_chimera_gate_zero_identity():
    """Test: In chimera mode, gate=0 should give same oracle logits as no cross-attention."""
    model, base_model, tokenizer = load_model()
    config = base_model.config

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    for idx in wrapper.xattn_layer_indices:
        assert wrapper.xattn_layers[str(idx)].gate.item() == 0.0

    B, T_cot, L_oracle = 1, 15, 20
    max_cot = T_cot

    # Build chimera input
    cot_ids = torch.randint(0, config.vocab_size, (B, T_cot), device=DEVICE)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    input_ids = torch.cat([cot_ids, oracle_ids], dim=1)
    attention_mask = torch.ones(B, T_cot + L_oracle, dtype=torch.bool, device=DEVICE)
    cot_mask = torch.ones(B, T_cot, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        # Chimera mode
        out_chimera = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            cot_len=max_cot, cot_mask=cot_mask,
        )
        # No cross-attention (same input, no cot_len)
        out_plain = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
        )

    # Compare oracle portion only (CoT portion is identical since xattn only modifies oracle)
    chimera_oracle_logits = out_chimera.logits[:, max_cot:, :]
    plain_oracle_logits = out_plain.logits[:, max_cot:, :]
    diff = (chimera_oracle_logits - plain_oracle_logits).abs().max().item()
    print(f"  Chimera gate=0 identity: max oracle logit diff = {diff:.2e}")
    assert diff < 1e-3, f"Gate=0 should produce identical oracle outputs, but max diff = {diff}"
    print(f"  Chimera gate=0 identity: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Flamingo Oracle Smoke Tests")
    print("=" * 60)

    print("\n1. Forward + backward pass (pre-extracted KV)")
    test_forward_pass()

    print("\n2. Gate=0 identity (pre-extracted KV)")
    test_gate_zero_identity()

    print("\n3. Save/load roundtrip")
    test_save_load_roundtrip()

    print("\n4. Chimera forward + backward")
    test_chimera_forward_backward()

    print("\n5. Chimera gate=0 identity")
    test_chimera_gate_zero_identity()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
