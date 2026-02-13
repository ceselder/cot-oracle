"""
Batch comparison: Original AO vs Trained CoT Oracle.

Takes problems from the corpus, collects activations at a few sentence
boundaries (not all — stay closer to distribution), asks both oracles
a set of questions, prints side-by-side.

Usage:
    python3 src/batch_compare.py \
        --checkpoint checkpoints/cot_oracle_8b_ctx_pred/step_5000 \
        --corpus data/cot_corpus_diverse/corpus.jsonl \
        --n-problems 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))

from signs_of_life.ao_lib import (
    layer_percent_to_layer,
    get_hf_submodule,
    get_steering_hook,
    add_hook,
    find_sentence_boundary_positions,
    split_cot_into_sentences,
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    EarlyStopException,
)

QUESTIONS = [
    "What is the model reasoning about?",
    "Summarize what computation is happening in this reasoning trace.",
    "Is this reasoning step important for reaching the final answer?",
    "Was this reasoning influenced by anything other than the problem itself?",
    "What type of reasoning is this? (e.g. calculation, planning, checking, etc.)",
]


def load_dual_model(model_name, checkpoint_path, device="cuda"):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"device_map": device, "torch_dtype": dtype, "attn_implementation": "sdpa"}

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    print(f"Loading trained LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name="trained", is_trainable=False)

    ao_repo = AO_CHECKPOINTS[model_name]
    from huggingface_hub import snapshot_download
    ao_path = snapshot_download(ao_repo)
    print(f"Loading original AO from {ao_path}...")
    model.load_adapter(ao_path, adapter_name="original_ao", is_trainable=False)

    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer


def collect_activations_base(model, tokenizer, text, layer, positions, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    activations = None
    submodule = get_hf_submodule(model, layer)

    def hook_fn(module, inp, out):
        nonlocal activations
        activations = out[0] if isinstance(out, tuple) else out
        raise EarlyStopException()

    with model.disable_adapter():
        handle = submodule.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        except EarlyStopException:
            pass
        finally:
            handle.remove()

    valid_positions = [p for p in positions if p < activations.shape[1]]
    return activations[0, valid_positions, :].detach()


def query_oracle(model, tokenizer, activations, prompt, model_name, adapter_name,
                 act_layer=None, injection_layer=1, max_new_tokens=150, device="cuda"):
    dtype = torch.bfloat16
    num_positions = activations.shape[0]
    if act_layer is None:
        act_layer = layer_percent_to_layer(model_name, 50)

    prefix = f"Layer: {act_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
    full_prompt = prefix + prompt
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_id) == 1
    special_id = special_id[0]

    positions = []
    for i, tid in enumerate(input_ids):
        if tid == special_id and len(positions) < num_positions:
            positions.append(i)
    assert len(positions) == num_positions, f"Expected {num_positions} positions, found {len(positions)}"

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    model.set_adapter(adapter_name)

    hook_fn = get_steering_hook(
        vectors=activations, positions=positions,
        device=device, dtype=dtype,
    )
    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(
            input_ids=input_tensor, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )

    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--n-positions", type=int, default=5,
                        help="Number of activation positions to inject (fewer = closer to training distribution)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Custom questions to ask (default: use built-in set)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load corpus
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    print(f"Loaded {len(corpus)} corpus entries")

    # Sample diverse problems
    by_source = {}
    for entry in corpus:
        src = entry.get("source", "unknown")
        by_source.setdefault(src, []).append(entry)

    sampled = []
    sources = list(by_source.keys())
    random.shuffle(sources)
    per_source = max(1, args.n_problems // len(sources))
    for src in sources:
        pool = by_source[src]
        sampled.extend(random.sample(pool, min(per_source, len(pool))))
    sampled = sampled[:args.n_problems]
    print(f"Selected {len(sampled)} problems from {len(set(e['source'] for e in sampled))} sources")

    questions = args.questions or QUESTIONS

    # Load model
    model, tokenizer = load_dual_model(args.model, args.checkpoint, args.device)
    act_layer = layer_percent_to_layer(args.model, 50)
    print(f"Activation layer: {act_layer}\n")

    results = []

    for idx, entry in enumerate(sampled):
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}/{len(sampled)} [{entry['source']}]")
        print(f"Q: {entry['question'][:200]}")
        print(f"Category: {entry.get('category', '?')}")
        print(f"CoT correct: {entry.get('cot_correct', '?')}")

        # Get sentence boundaries
        sentences = entry.get("sentences", [])
        if not sentences:
            sentences = split_cot_into_sentences(entry["cot_response"])

        if len(sentences) < 2:
            print("  Skipping (too few sentences)")
            continue

        boundary_positions = entry.get("boundary_positions", [])
        if not boundary_positions:
            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + entry["cot_response"]
            boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
        else:
            messages = [{"role": "user", "content": entry["question"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = formatted + entry["cot_response"]

        if len(boundary_positions) < 2:
            print("  Skipping (no boundary positions)")
            continue

        # Pick a few positions (stay close to training distribution)
        n_pos = min(args.n_positions, len(boundary_positions))
        if n_pos < len(boundary_positions):
            # Evenly spaced through the trajectory
            indices = [int(i * (len(boundary_positions) - 1) / (n_pos - 1)) for i in range(n_pos)]
            selected_positions = [boundary_positions[i] for i in indices]
        else:
            selected_positions = boundary_positions

        print(f"  {len(sentences)} sentences, using {len(selected_positions)} activation positions")

        # Show a few sentences
        for i, s in enumerate(sentences[:5]):
            print(f"    [{i+1}] {s[:100]}{'...' if len(s) > 100 else ''}")
        if len(sentences) > 5:
            print(f"    ... and {len(sentences) - 5} more")

        # Collect activations (base model, no adapter)
        try:
            activations = collect_activations_base(
                model, tokenizer, full_text, act_layer, selected_positions, device=args.device,
            )
        except Exception as e:
            print(f"  Activation collection failed: {e}")
            continue

        print(f"  Activations shape: {activations.shape}")

        # Ask each question to both oracles
        for q_idx, question in enumerate(questions):
            print(f"\n  Q{q_idx+1}: {question}")

            try:
                resp_orig = query_oracle(
                    model, tokenizer, activations, question,
                    model_name=args.model, adapter_name="original_ao",
                    act_layer=act_layer, max_new_tokens=150, device=args.device,
                )
            except Exception as e:
                resp_orig = f"ERROR: {e}"

            try:
                resp_trained = query_oracle(
                    model, tokenizer, activations, question,
                    model_name=args.model, adapter_name="trained",
                    act_layer=act_layer, max_new_tokens=150, device=args.device,
                )
            except Exception as e:
                resp_trained = f"ERROR: {e}"

            print(f"  Original AO: {resp_orig[:300]}")
            print(f"  Trained:     {resp_trained[:300]}")

            results.append({
                "problem_idx": idx,
                "source": entry["source"],
                "question_text": entry["question"][:200],
                "category": entry.get("category"),
                "oracle_question": question,
                "original_ao": resp_orig,
                "trained": resp_trained,
            })

    # Save results
    output_path = Path(args.checkpoint).parent / "batch_compare_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} comparisons to {output_path}")


if __name__ == "__main__":
    main()
