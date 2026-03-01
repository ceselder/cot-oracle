"""Fast unit tests for Flamingo wrapper invariants.

Run: python -m unittest tests.test_flamingo_unit
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flamingo_oracle import FlamingoOracleWrapper


class _HeadNorm(nn.Module):
    def forward(self, x):
        return x


class _FakeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
        self.q_norm = _HeadNorm()
        self.k_norm = _HeadNorm()


class _FakeDecoderLayer(nn.Module):
    def __init__(self, config, scale):
        super().__init__()
        self.self_attn = _FakeSelfAttention(config)
        self.input_layernorm = nn.Identity()
        self.scale = scale

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + self.scale


class _FakeBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer(config, 0.05 * (idx + 1)) for idx in range(config.num_hidden_layers)])


class _FakeCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _FakeBackbone(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.raise_after_layers = False

    def forward(self, input_ids, attention_mask, labels=None):
        del attention_mask
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        if self.raise_after_layers:
            raise RuntimeError("boom")
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        return input_ids


class FlamingoUnitTests(unittest.TestCase):
    def setUp(self):
        config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=2,
            vocab_size=32,
        )
        self.base_model = _FakeCausalLM(config)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.wrapper = FlamingoOracleWrapper(self.base_model, config, xattn_interval=1, lora_r=2, lora_alpha=4)

    def test_forward_clears_runtime_state_on_exception(self):
        self.base_model.raise_after_layers = True
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        supervisee_kvs = {
            idx: torch.zeros(1, input_ids.shape[1], self.base_model.config.hidden_size)
            for idx in self.wrapper.xattn_layer_indices
        }

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                supervisee_kvs=supervisee_kvs,
                cot_attention_mask=attention_mask,
            )

        self.assertIsNone(self.wrapper._current_kvs)
        self.assertIsNone(self.wrapper._current_kv_mask)

    def test_collect_cot_hidden_states_restores_mode_and_hooks_on_exception(self):
        self.base_model.train()
        self.base_model.raise_after_layers = True
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        target_layer = self.wrapper._get_transformer_layers()[0].original_layer

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.wrapper.collect_cot_hidden_states(input_ids, attention_mask)

        self.assertTrue(self.base_model.training)
        self.assertEqual(len(target_layer._forward_hooks), 0)

    def test_xattn_state_dict_stays_lora_only(self):
        xattn_layer = self.wrapper.xattn_layers["0"]
        state_keys = set(xattn_layer.state_dict())

        self.assertIs(xattn_layer.q_proj._base, self.wrapper._get_transformer_layers()[0].original_layer.self_attn.q_proj)
        self.assertIn("q_proj.lora_A.weight", state_keys)
        self.assertNotIn("q_proj.base.weight", state_keys)


if __name__ == "__main__":
    unittest.main()
