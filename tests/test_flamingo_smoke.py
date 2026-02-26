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


def test_parallel_forward_backward():
    """Test: Parallel mode — CoT and oracle as separate batch items."""
    model, base_model, tokenizer = load_model()
    config = base_model.config

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    wrapper.print_trainable_parameters()

    B = 2
    L_cot, L_oracle = 40, 20
    L_max = max(L_cot, L_oracle)

    # CoT items: left-padded to L_max
    cot_ids = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_pad = L_max - L_cot
    if cot_pad > 0:
        cot_ids = torch.cat([torch.full((B, cot_pad), tokenizer.pad_token_id, device=DEVICE, dtype=torch.long), cot_ids], dim=1)
    cot_mask = torch.cat([torch.zeros(B, cot_pad, dtype=torch.bool, device=DEVICE), torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)], dim=1) if cot_pad > 0 else torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)

    # Oracle items: left-padded to L_max
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_pad = L_max - L_oracle
    oracle_ids = torch.cat([torch.full((B, oracle_pad), tokenizer.pad_token_id, device=DEVICE, dtype=torch.long), oracle_ids], dim=1)
    oracle_mask = torch.cat([torch.zeros(B, oracle_pad, dtype=torch.bool, device=DEVICE), torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)], dim=1)

    # Labels: CoT = -100, oracle = random targets
    cot_labels = torch.full((B, L_max), -100, dtype=torch.long, device=DEVICE)
    oracle_labels = torch.full((B, L_max), -100, dtype=torch.long, device=DEVICE)
    oracle_labels[:, oracle_pad + 5:] = torch.randint(0, config.vocab_size, (B, L_oracle - 5), device=DEVICE)

    # Stack [CoT; Oracle] in batch dim
    input_ids = torch.cat([cot_ids, oracle_ids], dim=0)
    attention_mask = torch.cat([cot_mask, oracle_mask], dim=0)
    labels = torch.cat([cot_labels, oracle_labels], dim=0)
    cot_kv_mask = cot_mask  # [B, L_max]

    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = wrapper(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            parallel_B=B, cot_mask=cot_kv_mask,
        )

    # Logits should be oracle-only [B, L_max, V]
    assert outputs.logits.shape == (B, L_max, config.vocab_size), \
        f"Expected ({B}, {L_max}, {config.vocab_size}), got {outputs.logits.shape}"
    assert outputs.loss is not None and outputs.loss.item() > 0
    print(f"  Parallel forward: OK (logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f})")

    outputs.loss.backward()
    for idx in wrapper.xattn_layer_indices:
        gate = wrapper.xattn_layers[str(idx)].gate
        assert gate.grad is not None, f"Gate at layer {idx} has no gradient"
    print(f"  Parallel backward: OK (all gates have gradients)")


def test_parallel_gate_zero_identity():
    """Test: In parallel mode, gate=0 should give same oracle logits regardless of CoT content.

    Both calls use [2B, L] batch structure to avoid SDPA kernel differences from batch size.
    Call 1: real CoT + oracle (with xattn, gates=0)
    Call 2: different CoT + same oracle (with xattn, gates=0)
    Since gates=0, CoT content shouldn't affect oracle output.
    """
    model, base_model, tokenizer = load_model()
    config = base_model.config

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)
    for idx in wrapper.xattn_layer_indices:
        assert wrapper.xattn_layers[str(idx)].gate.item() == 0.0

    B = 1
    L_cot, L_oracle = 30, 20
    L_max = max(L_cot, L_oracle)

    # Oracle input (same for both calls)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_oracle), device=DEVICE)
    oracle_pad = L_max - L_oracle
    oracle_ids_padded = torch.cat([torch.full((B, oracle_pad), tokenizer.pad_token_id, device=DEVICE, dtype=torch.long), oracle_ids], dim=1)
    oracle_mask_padded = torch.cat([torch.zeros(B, oracle_pad, dtype=torch.bool, device=DEVICE), torch.ones(B, L_oracle, dtype=torch.bool, device=DEVICE)], dim=1)

    # Two different CoTs
    cot_ids_a = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_ids_b = torch.randint(0, config.vocab_size, (B, L_cot), device=DEVICE)
    cot_mask = torch.ones(B, L_cot, dtype=torch.bool, device=DEVICE)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_a = wrapper(
            input_ids=torch.cat([cot_ids_a, oracle_ids_padded], dim=0),
            attention_mask=torch.cat([cot_mask, oracle_mask_padded], dim=0),
            parallel_B=B, cot_mask=cot_mask,
        )
        out_b = wrapper(
            input_ids=torch.cat([cot_ids_b, oracle_ids_padded], dim=0),
            attention_mask=torch.cat([cot_mask, oracle_mask_padded], dim=0),
            parallel_B=B, cot_mask=cot_mask,
        )

    diff = (out_a.logits - out_b.logits).abs().max().item()
    print(f"  Parallel gate=0 identity: max oracle logit diff (different CoTs) = {diff:.2e}")
    assert diff < 1e-3, f"Gate=0 should make oracle invariant to CoT content, but max diff = {diff}"
    print(f"  Parallel gate=0 identity: PASSED")


def test_save_load_roundtrip():
    """Test: Save and load Flamingo modules, verify identical outputs."""
    import tempfile

    model, base_model, tokenizer = load_model()
    config = base_model.config

    wrapper = FlamingoOracleWrapper(model, config, xattn_interval=4, lora_r=8, lora_alpha=16)

    # Perturb a gate so it's non-zero
    wrapper.xattn_layers[str(wrapper.xattn_layer_indices[0])].gate.data.fill_(0.5)

    B, L_cot, L_oracle = 1, 20, 16
    L_max = max(L_cot, L_oracle)

    cot_ids = torch.randint(0, config.vocab_size, (B, L_max), device=DEVICE)
    cot_mask = torch.ones(B, L_max, dtype=torch.bool, device=DEVICE)
    oracle_ids = torch.randint(0, config.vocab_size, (B, L_max), device=DEVICE)
    oracle_mask = torch.ones(B, L_max, dtype=torch.bool, device=DEVICE)
    input_ids = torch.cat([cot_ids, oracle_ids], dim=0)
    attention_mask = torch.cat([cot_mask, oracle_mask], dim=0)

    wrapper.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_before = wrapper(
            input_ids=input_ids, attention_mask=attention_mask,
            parallel_B=B, cot_mask=cot_mask,
        ).logits.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_flamingo_modules(tmpdir)

        model2, base_model2, _ = load_model()
        wrapper2 = FlamingoOracleWrapper(model2, config, xattn_interval=4, lora_r=8, lora_alpha=16)
        wrapper2.load_flamingo_modules(tmpdir)

    dev2 = next(wrapper2.parameters()).device
    wrapper2.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        out_after = wrapper2(
            input_ids=input_ids.to(dev2), attention_mask=attention_mask.to(dev2),
            parallel_B=B, cot_mask=cot_mask.to(dev2),
        ).logits

    diff = (out_before - out_after.to(out_before.device)).abs().max().item()
    print(f"  Save/load roundtrip: max logit diff = {diff:.2e}")
    assert diff < 1e-3, f"Save/load should produce identical outputs, but max diff = {diff}"
    print(f"  Save/load roundtrip: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Flamingo Oracle Smoke Tests")
    print("=" * 60)

    print("\n1. Parallel forward + backward")
    test_parallel_forward_backward()

    print("\n2. Parallel gate=0 identity")
    test_parallel_gate_zero_identity()

    print("\n3. Save/load roundtrip")
    test_save_load_roundtrip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
