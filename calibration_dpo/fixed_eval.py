"""Fixed eval: run the same probes on the same CoT positions every N steps.

Inspired by Arya's "Current Activation Oracles Are Hard to Use" —
tests the probes that AOs historically struggle with:
uncertainty detection, backtracking, specifics, next-step prediction.

Logs oracle responses as a wandb table so you can track qualitative
improvement over training.
"""

from __future__ import annotations

import torch
from peft import PeftModel

from core.ao import (
    TRAINED_PLACEHOLDER,
    add_hook,
    get_batched_steering_hook,
    get_hf_submodule,
)

from rollouts import _build_manual_prefix_token_ids, _build_oracle_prefix


# Fixed probes — the kinds of questions Arya found AOs bad at
FIXED_PROBES = [
    # Uncertainty detection (AO got 1/54 on Arya's test)
    "Is the model uncertain about something? If yes, what specifically? If no, say not sure.",
    # Specifics — AO confabulated numbers regardless of problem
    "What numbers or values is the model working with? Be specific.",
    # Backtracking — AO got ~5% accuracy
    "Is the model backtracking or reconsidering? If yes, describe what changed and why.",
    # Next step prediction
    "What will the model do next? Be specific.",
    # Computational state — the core use case
    "Describe the model's computational state right now. What operation is being performed and why?",
]


class FixedEval:
    """Runs fixed probes on fixed corpus examples every N steps."""

    def __init__(
        self,
        sampler,
        tokenizer,
        layers: list[int],
        n_examples: int = 5,
        seed: int = 12345,
    ):
        """Sample and freeze n_examples from the corpus.

        These examples never change — same CoT, same positions, every eval.
        """
        import random
        rng = random.Random(seed)

        self.tokenizer = tokenizer
        self.layers = layers
        self.examples = []

        # Sample diverse examples via the sampler's own sampling logic
        for _ in range(n_examples * 3):
            if len(self.examples) >= n_examples:
                break
            try:
                item = rng.choice(sampler.corpus)
                from data_sampler import prepare_example
                ex = prepare_example(
                    item, tokenizer, layers,
                    sampler.stride, sampler.min_positions, sampler.max_positions, rng,
                )
                if ex is not None:
                    cot_len = ex["cot_end"] - ex["cot_start"]
                    if cot_len >= 50:
                        self.examples.append(ex)
            except Exception:
                continue

        if len(self.examples) < n_examples:
            print(f"  [fixed_eval] Warning: only got {len(self.examples)}/{n_examples} examples")

        print(f"  [fixed_eval] Initialized with {len(self.examples)} fixed examples")
        for i, ex in enumerate(self.examples):
            q_trunc = ex["question"][:80].replace("\n", " ")
            n_pos = len(ex["base_positions"])
            print(f"    {i+1}. ({n_pos} pos) {q_trunc}...")

    def run(
        self,
        model: PeftModel,
        injection_layer: int,
        device: torch.device,
        step: int,
    ) -> list[dict]:
        """Run all probes on all fixed examples. Returns list of result dicts."""
        from activations import extract_activations

        dtype = torch.bfloat16
        ph_token = TRAINED_PLACEHOLDER
        ph_id_list = self.tokenizer.encode(ph_token, add_special_tokens=False)
        ph_id = ph_id_list[0]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        injection_submodule = get_hf_submodule(model, injection_layer)

        # Extract activations for all examples
        acts_list = extract_activations(
            model, self.tokenizer, self.examples, self.layers, device,
        )

        model.set_adapter("policy")
        was_training = model.training
        model.eval()

        results = []

        try:
            for ex_idx, (ex, activations) in enumerate(zip(self.examples, acts_list)):
                num_positions = activations.shape[0]

                for probe in FIXED_PROBES:
                    # Build oracle input
                    prefix = _build_oracle_prefix(num_positions, self.layers, ph_token)
                    full_prompt = prefix + probe

                    messages = [{"role": "user", "content": full_prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    prefix_idx = formatted.find(prefix)
                    assert prefix_idx >= 0
                    before_ids = self.tokenizer.encode(formatted[:prefix_idx], add_special_tokens=False)
                    after_ids = self.tokenizer.encode(formatted[prefix_idx + len(prefix):], add_special_tokens=False)
                    prefix_ids, rel_positions = _build_manual_prefix_token_ids(
                        self.tokenizer, num_positions, self.layers, ph_id,
                    )
                    input_ids = before_ids + prefix_ids + after_ids
                    positions = [len(before_ids) + p for p in rel_positions]

                    input_tensor = torch.tensor([input_ids], device=device)
                    attn_mask = torch.ones_like(input_tensor)

                    hook_fn = get_batched_steering_hook(
                        vectors=[activations],
                        positions=[positions],
                        device=device,
                        dtype=dtype,
                    )

                    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
                        outputs = model.generate(
                            input_ids=input_tensor,
                            attention_mask=attn_mask,
                            max_new_tokens=200,
                            do_sample=False,  # greedy for reproducibility
                            pad_token_id=pad_id,
                        )

                    generated = outputs[0][len(input_ids):]
                    response = self.tokenizer.decode(generated, skip_special_tokens=True)

                    results.append({
                        "step": step,
                        "example_idx": ex_idx,
                        "question": ex["question"][:200],
                        "cot_excerpt": ex["cot_response"][:200],
                        "n_positions": len(ex["base_positions"]),
                        "probe": probe,
                        "response": response,
                    })

        finally:
            if was_training:
                model.train()

        return results

    def run_and_log(
        self,
        model: PeftModel,
        injection_layer: int,
        device: torch.device,
        step: int,
    ) -> None:
        """Run eval and log to wandb as a table."""
        results = self.run(model, injection_layer, device, step)

        try:
            import wandb
            if wandb.run is None:
                return

            table = wandb.Table(columns=[
                "step", "example", "question", "cot_excerpt",
                "n_pos", "probe", "response",
            ])
            for r in results:
                table.add_data(
                    r["step"],
                    r["example_idx"],
                    r["question"],
                    r["cot_excerpt"],
                    r["n_positions"],
                    r["probe"],
                    r["response"],
                )
            wandb.log({"eval/fixed_probes": table}, step=step)
            print(f"  [fixed_eval] Logged {len(results)} probe responses at step {step}")

            # Also print a couple to console
            for r in results[:3]:
                probe_short = r["probe"][:50]
                resp_short = r["response"][:120].replace("\n", " ")
                print(f"    [{r['example_idx']}] {probe_short}")
                print(f"        → {resp_short}")

        except Exception as e:
            print(f"  [fixed_eval] Wandb logging failed: {e}")
