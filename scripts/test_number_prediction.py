#!/usr/bin/env python3
"""
Test: Can the oracle predict what number the model is computing?

From the LessWrong post: Adam's AO confabulates "10" and "12" regardless of problem.
We test both Adam's checkpoint and our final-sprint checkpoint.

Flow:
1. Give Qwen3-8B arithmetic problems
2. Generate CoT responses (greedy)
3. Use an LLM to find where a numeric answer appears in the CoT
4. Extract activations around that position
5. Ask both oracles: "What number is the model computing/about to produce?"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ao_reference"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAM_CHECKPOINT = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
FINAL_SPRINT_CHECKPOINT = "ceselder/cot-oracle-qwen3-8b-final-sprint-checkpoint-no-DPO"
BASE_MODEL = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
INJECTION_LAYER = 1

# ── Arithmetic problems (from the post + extras) ──
PROBLEMS = [
    {"problem": "What is 1 + 1?", "answer": 2},
    {"problem": "What is 7 * 8?", "answer": 56},
    {"problem": "What is 123 + 456?", "answer": 579},
    {"problem": "What is 99 * 3?", "answer": 297},
    {"problem": "What is 1000 - 387?", "answer": 613},
    {"problem": "What is 15 * 17?", "answer": 255},
    {"problem": "What is 2048 / 8?", "answer": 256},
    {"problem": "What is 37 + 89?", "answer": 126},
    {"problem": "What is 144 / 12?", "answer": 12},
    {"problem": "What is 25 * 25?", "answer": 625},
    # From the post (simplified versions of their expressions)
    {"problem": "What is -93 + 42?", "answer": -51},
    {"problem": "What is 87 - 38 + 79?", "answer": 128},
]

# Oracle questions to ask about the activations
ORACLE_QUESTIONS = [
    "What number is the model thinking about?",
    "What number is the model about to produce?",
    "What is the numerical result the model has computed?",
]


def load_model_and_adapters():
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Loading Adam's adapter...")
    model = PeftModel.from_pretrained(
        base, ADAM_CHECKPOINT, adapter_name="adam",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"Loading final-sprint adapter...")
    model.load_adapter(
        FINAL_SPRINT_CHECKPOINT, adapter_name="trained",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


def generate_cot(model, tokenizer, problem_text, max_new_tokens=512):
    """Generate a CoT response for an arithmetic problem."""
    messages = [{"role": "user", "content": problem_text + " Think step by step."}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to("cuda")

    with model.disable_adapter(), torch.no_grad():
        model.eval()
        outputs = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response, formatted, outputs[0].tolist()


def find_number_positions(tokenizer, full_ids, answer, response_text):
    """Find token positions where the answer number appears in the response.

    Returns a list of (position, token_text) tuples for tokens containing the answer.
    """
    answer_str = str(answer)
    positions = []

    # Decode each token and look for the answer
    for i, tid in enumerate(full_ids):
        token_text = tokenizer.decode([tid])
        if answer_str in token_text.strip():
            positions.append((i, token_text))

    return positions


def extract_activations_at_positions(model, tokenizer, full_ids, positions, layers):
    """Extract activations at specific positions across multiple layers."""
    from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

    input_tensor = torch.tensor([full_ids], device="cuda")
    attn_mask = torch.ones_like(input_tensor)

    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers}

    with model.disable_adapter(), torch.no_grad():
        model.eval()
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules=submodules,
            inputs_BL={"input_ids": input_tensor, "attention_mask": attn_mask},
            min_offset=None, max_offset=None,
        )

    # Collect vectors at the specified positions for each layer
    vectors = []
    for layer in layers:
        layer_acts = acts_by_layer[layer]  # [1, seq_len, D]
        for pos in positions:
            vectors.append(layer_acts[0, pos, :].detach())

    if not vectors:
        return None
    return torch.stack(vectors, dim=0)  # [K*n_layers, D]


def query_oracle(model, tokenizer, activations, question, adapter_name, layers):
    """Query the oracle with given activations and question."""
    from core.ao import get_batched_steering_hook, TRAINED_PLACEHOLDER
    from nl_probes.utils.activation_utils import get_hf_submodule
    from nl_probes.utils.steering_hooks import add_hook
    from eval_loop import _build_manual_prefix_token_ids

    ph_token = TRAINED_PLACEHOLDER
    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    num_positions = activations.shape[0]
    n_layers = len(layers)
    k_per_layer = num_positions // n_layers

    # Build prefix: "L9:¶¶¶ L18:¶¶¶ L27:¶¶¶.\n{question}"
    prefix_ids, rel_positions = _build_manual_prefix_token_ids(
        tokenizer, num_positions, layers, ph_id,
    )

    # Build full input
    messages = [{"role": "user", "content": "PLACEHOLDER"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # Find where PLACEHOLDER is and replace
    ph_idx = formatted.find("PLACEHOLDER")
    before_text = formatted[:ph_idx]
    after_text = formatted[ph_idx + len("PLACEHOLDER"):]

    before_ids = tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)

    # Insert prefix_ids then question
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    input_ids = before_ids + prefix_ids + question_ids + after_ids
    positions = [len(before_ids) + p for p in rel_positions]

    input_tensor = torch.tensor([input_ids], device="cuda")
    attn_mask = torch.ones_like(input_tensor)

    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER, use_lora=True)

    hook_fn = get_batched_steering_hook(
        vectors=[activations.to("cuda")],
        positions=[positions],
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        model.eval()
        outputs = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=64, do_sample=False, pad_token_id=pad_id,
        )

    response_ids = outputs[0][len(input_ids):]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def main():
    model, tokenizer = load_model_and_adapters()

    results = []

    print("\n" + "=" * 80)
    print("NUMBER PREDICTION TEST")
    print("Can the oracle tell what number the model is computing?")
    print("=" * 80)

    for prob in PROBLEMS:
        problem_text = prob["problem"]
        true_answer = prob["answer"]

        print(f"\n{'─' * 70}")
        print(f"Problem: {problem_text}  (true answer: {true_answer})")

        # Generate CoT
        response, formatted_prompt, full_ids = generate_cot(model, tokenizer, problem_text)
        print(f"Model response: {response[:200]}...")

        # Find where the answer number appears
        prompt_len = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))
        answer_positions = find_number_positions(tokenizer, full_ids, true_answer, response)

        # Filter to only positions in the response (after prompt)
        answer_positions = [(p, t) for p, t in answer_positions if p >= prompt_len]

        if not answer_positions:
            print(f"  WARNING: Could not find answer '{true_answer}' in response tokens")
            # Use positions near the end of the response instead
            resp_len = len(full_ids) - prompt_len
            if resp_len > 5:
                end_positions = list(range(len(full_ids) - 5, len(full_ids)))
            else:
                end_positions = list(range(prompt_len, len(full_ids)))
            positions_to_use = end_positions
            print(f"  Using last {len(positions_to_use)} response positions instead")
        else:
            # Use positions around the FIRST occurrence of the answer
            first_pos = answer_positions[0][0]
            # Take a window of 5 tokens centered on the answer
            window_start = max(prompt_len, first_pos - 2)
            window_end = min(len(full_ids), first_pos + 3)
            positions_to_use = list(range(window_start, window_end))
            pos_tokens = [tokenizer.decode([full_ids[p]]) for p in positions_to_use]
            print(f"  Answer found at position {first_pos}, using window [{window_start}:{window_end}]")
            print(f"  Tokens at positions: {pos_tokens}")

        # Extract activations
        activations = extract_activations_at_positions(
            model, tokenizer, full_ids, positions_to_use, LAYERS,
        )
        if activations is None:
            print("  ERROR: No activations extracted")
            continue

        print(f"  Activations shape: {activations.shape}")

        # Query both oracles with each question
        result_entry = {
            "problem": problem_text,
            "true_answer": true_answer,
            "model_response": response[:300],
            "positions_used": positions_to_use,
            "oracle_responses": {},
        }

        for q in ORACLE_QUESTIONS:
            for adapter_name in ["adam", "trained"]:
                label = f"{adapter_name}"
                resp = query_oracle(model, tokenizer, activations, q, adapter_name, LAYERS)
                key = f"{adapter_name}:{q}"
                result_entry["oracle_responses"][key] = resp
                print(f"  [{adapter_name:>7}] Q: {q}")
                print(f"           A: {resp[:150]}")

        results.append(result_entry)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY: Number Prediction")
    print("=" * 80)
    print(f"{'Problem':<30} {'True':>6}  {'Adam says':>30}  {'Ours says':>30}")
    print("-" * 100)

    q = ORACLE_QUESTIONS[0]  # "What number is the model thinking about?"
    for r in results:
        adam_resp = r["oracle_responses"].get(f"adam:{q}", "?")[:30]
        ours_resp = r["oracle_responses"].get(f"trained:{q}", "?")[:30]
        print(f"{r['problem']:<30} {r['true_answer']:>6}  {adam_resp:>30}  {ours_resp:>30}")

    # Save
    output_path = "data/number_prediction_test.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
