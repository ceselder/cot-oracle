"""
Experiment B: Do logit lens trajectories show structure across CoT?

For math problems where CoT is load-bearing:
- At each sentence boundary, project residual stream through unembedding
- Track P(correct answer token) across sentences
- Plot trajectories

Pass: S-curve visible (prob starts low, jumps at key computation sentences).
Fail: flat/random trajectory.
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ao_lib import (
    load_model_with_ao,
    generate_cot,
    generate_direct_answer,
    split_cot_into_sentences,
    collect_activations,
    get_hf_submodule,
    layer_percent_to_layer,
    LAYER_COUNTS,
    MATH_PROBLEMS,
)


def get_answer_token_ids(tokenizer, answer_text: str) -> list[int]:
    """Get token IDs for the answer text (may be multiple tokens)."""
    # Try the answer as-is
    ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if ids:
        return ids
    # Try with leading space
    ids = tokenizer.encode(" " + answer_text, add_special_tokens=False)
    return ids


def _find_final_norm(model):
    """Find the final RMSNorm before lm_head by walking named_modules."""
    candidates = []
    for name, module in model.named_modules():
        if name.endswith('.norm') and 'layers' not in name:
            candidates.append((name, module))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0]


def _find_lm_head(model):
    """Find the lm_head linear layer by walking named_modules."""
    for name, module in model.named_modules():
        if name.endswith('lm_head') and hasattr(module, 'weight'):
            return name, module
    return None, None


# Cache these after first discovery
_cached_norm = None
_cached_lm_head = None


def logit_lens_at_position(
    model,
    layer: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Get logit distribution at a specific position by projecting through unembedding."""
    global _cached_norm, _cached_lm_head

    acts = collect_activations(model, layer, input_ids, attention_mask)
    hidden = acts[0, position, :]  # [d_model]

    if _cached_norm is None:
        _, _cached_norm = _find_final_norm(model)
        _, _cached_lm_head = _find_lm_head(model)
        assert _cached_norm is not None, "Could not find final norm"
        assert _cached_lm_head is not None, "Could not find lm_head"

    with torch.no_grad():
        h = hidden.unsqueeze(0)
        h = _cached_norm(h)
        logits = _cached_lm_head(h)

    return logits[0]


def layer_scan(model, tokenizer, question, correct, base_formatted, model_name, device):
    """Quick scan: try logit lens at multiple layers on the full CoT to find where signal emerges."""
    cot_text = generate_cot(model, tokenizer, question, max_new_tokens=512, device=device)
    sentences = split_cot_into_sentences(cot_text)
    if len(sentences) < 3:
        return None, cot_text

    answer_token_ids = get_answer_token_ids(tokenizer, correct)
    if not answer_token_ids:
        return None, cot_text

    # Build full CoT text
    full_text = base_formatted + "<think>\n" + " ".join(sentences) + " "
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    seq_len = inputs["input_ids"].shape[1]

    n_layers = LAYER_COUNTS[model_name]
    # Test layers at 50%, 75%, 85%, 93%, 100% (last layer)
    test_percents = [50, 75, 85, 93]
    test_layers = [layer_percent_to_layer(model_name, p) for p in test_percents]
    test_layers.append(n_layers - 1)  # Last layer

    model.disable_adapters()
    print(f"\n  Layer scan (checking P('{correct}') at last token of full CoT):")
    best_layer = None
    best_prob = 0.0

    for layer in test_layers:
        logits = logit_lens_at_position(
            model, layer,
            inputs["input_ids"], inputs["attention_mask"],
            position=seq_len - 1,
        )
        probs = F.softmax(logits, dim=-1)
        answer_prob = probs[answer_token_ids[0]].item()
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

        pct = int(100 * layer / n_layers)
        print(f"    Layer {layer:2d} ({pct:2d}%): P('{correct}')={answer_prob:.4f}, "
              f"top: {top5_tokens[0]}({top5_probs[0]:.3f}) {top5_tokens[1]}({top5_probs[1]:.3f})")

        if answer_prob > best_prob:
            best_prob = answer_prob
            best_layer = layer

    model.enable_adapters()
    return best_layer, cot_text


def run_experiment_b(
    model_name: str = "Qwen/Qwen3-1.7B",
    n_problems: int = 10,
    output_path: str = "results/signs_of_life/experiment_b.json",
    device: str = "cuda",
):
    print("=" * 60)
    print("EXPERIMENT B: Logit Lens Trajectories across CoT")
    print("=" * 60)

    model, tokenizer = load_model_with_ao(model_name, use_8bit=True, device=device)

    # Verify norm/lm_head
    norm_name, norm_mod = _find_final_norm(model)
    head_name, head_mod = _find_lm_head(model)
    print(f"Norm: '{norm_name}' ({type(norm_mod).__name__}), lm_head: '{head_name}'")

    correct_answers = {
        "What is 17 * 24?": "408",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?": "150",
        "What is the sum of the first 10 positive integers?": "55",
        "A rectangle has a length of 12 cm and width of 8 cm. What is its area?": "96",
        "If 3x + 7 = 22, what is x?": "5",
        "What is 15% of 200?": "30",
        "How many ways can you arrange the letters in the word MATH?": "24",
        "What is the greatest common divisor of 48 and 36?": "12",
        "If a pizza is cut into 8 equal slices and you eat 3, what fraction remains?": "5/8",
        "What is 2^10?": "1024",
        "A bag has 3 red, 4 blue, and 5 green marbles. What is the probability of drawing a red marble?": "1/4",
        "What is the perimeter of a square with side length 7?": "28",
        "If f(x) = 2x + 3, what is f(5)?": "13",
        "What is 144 divided by 12?": "12",
        "How many prime numbers are less than 20?": "8",
        "What is the average of 15, 20, 25, 30, and 35?": "25",
        "A car depreciates by 15% each year. If it costs $20000, what is it worth after 1 year?": "17000",
        "What is the volume of a cube with side length 5?": "125",
        "If you flip a coin 3 times, how many possible outcomes are there?": "8",
        "What is the square root of 169?": "13",
    }

    # --- Phase 1: Layer scan on first problem ---
    first_q = MATH_PROBLEMS[0]
    first_correct = correct_answers[first_q]
    messages = [{"role": "user", "content": first_q}]
    base_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )

    print(f"\n--- Layer scan on: {first_q} (answer: {first_correct}) ---")
    # Generate direct answer first
    direct_answer = generate_direct_answer(model, tokenizer, first_q, device=device)
    print(f"  Direct: {direct_answer[:80]} (has '{first_correct}': {first_correct in direct_answer})")

    best_layer, first_cot_text = layer_scan(
        model, tokenizer, first_q, first_correct, base_formatted, model_name, device,
    )

    n_layers = LAYER_COUNTS[model_name]
    if best_layer is None:
        # Fallback to 90% depth
        best_layer = layer_percent_to_layer(model_name, 90)
        print(f"\n  Layer scan inconclusive, using layer {best_layer} (90%)")
    else:
        pct = int(100 * best_layer / n_layers)
        print(f"\n  Best layer: {best_layer} ({pct}%)")

    # Use the best layer we found, but also try the last layer if best was too deep
    act_layer = best_layer
    print(f"\n  Running full experiment at layer {act_layer}")

    # --- Phase 2: Full trajectory tracking ---
    results = []
    problems = MATH_PROBLEMS[:n_problems]

    for i, question in enumerate(problems):
        correct = correct_answers.get(question, "")
        if not correct:
            continue

        print(f"\n--- Problem {i+1}/{n_problems}: {question} (answer: {correct}) ---")

        # Check if model gets it wrong without CoT
        direct_answer = generate_direct_answer(model, tokenizer, question, device=device)
        direct_has_answer = correct in direct_answer
        print(f"  Direct answer: {direct_answer[:80]} (contains '{correct}': {direct_has_answer})")

        # Generate with CoT (reuse first problem's CoT)
        if i == 0 and first_cot_text:
            cot_text = first_cot_text
        else:
            cot_text = generate_cot(model, tokenizer, question, max_new_tokens=512, device=device)
        cot_has_answer = correct in cot_text
        print(f"  CoT contains correct answer: {cot_has_answer}")

        sentences = split_cot_into_sentences(cot_text)
        if len(sentences) < 3:
            print("  Skipping: too few sentences")
            continue

        messages = [{"role": "user", "content": question}]
        base_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        answer_token_ids = get_answer_token_ids(tokenizer, correct)
        if not answer_token_ids:
            print(f"  Skipping: couldn't tokenize answer '{correct}'")
            continue

        # Track probability at each sentence boundary
        trajectory = []
        cot_prefix = ""
        model.disable_adapters()

        for j, sentence in enumerate(sentences):
            cot_prefix += sentence + " "
            current_text = base_formatted + "<think>\n" + cot_prefix

            inputs = tokenizer(current_text, return_tensors="pt", add_special_tokens=False).to(device)
            seq_len = inputs["input_ids"].shape[1]

            logits = logit_lens_at_position(
                model, act_layer,
                inputs["input_ids"], inputs["attention_mask"],
                position=seq_len - 1,
            )

            probs = F.softmax(logits, dim=-1)
            answer_prob = probs[answer_token_ids[0]].item()

            top5_probs, top5_ids = torch.topk(probs, 5)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]

            trajectory.append({
                "sentence_idx": j,
                "sentence": sentence[:80],
                "answer_prob": answer_prob,
                "top5_tokens": top5_tokens,
                "top5_probs": top5_probs.tolist(),
            })

            if j < 3 or j == len(sentences) - 1 or answer_prob > 0.01:
                print(f"  Sentence {j}: P('{correct}')={answer_prob:.4f}, top: {top5_tokens[0]}({top5_probs[0]:.3f})")

        model.enable_adapters()

        probs_list = [t["answer_prob"] for t in trajectory]
        max_prob = max(probs_list) if probs_list else 0
        final_prob = probs_list[-1] if probs_list else 0
        max_jump = max(
            probs_list[k] - probs_list[k-1]
            for k in range(1, len(probs_list))
        ) if len(probs_list) > 1 else 0

        results.append({
            "question": question,
            "correct_answer": correct,
            "direct_has_answer": direct_has_answer,
            "cot_has_answer": cot_has_answer,
            "is_load_bearing": cot_has_answer and not direct_has_answer,
            "num_sentences": len(sentences),
            "layer": act_layer,
            "trajectory": trajectory,
            "max_prob": max_prob,
            "final_prob": final_prob,
            "max_jump": max_jump,
        })

        print(f"  Trajectory: max={max_prob:.4f}, final={final_prob:.4f}, max_jump={max_jump:.4f}")
        print(f"  Load-bearing CoT: {cot_has_answer and not direct_has_answer}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layer used: {act_layer} ({int(100*act_layer/n_layers)}%)")
    load_bearing = [r for r in results if r["is_load_bearing"]]
    decorative = [r for r in results if not r["is_load_bearing"]]
    print(f"Total problems: {len(results)}")
    print(f"Load-bearing CoTs: {len(load_bearing)}")
    print(f"Decorative CoTs: {len(decorative)}")
    if load_bearing:
        avg_max = sum(r["max_prob"] for r in load_bearing) / len(load_bearing)
        avg_jump = sum(r["max_jump"] for r in load_bearing) / len(load_bearing)
        print(f"Avg max P(answer) (load-bearing): {avg_max:.4f}")
        print(f"Avg max jump (load-bearing): {avg_jump:.4f}")
    if decorative:
        avg_max = sum(r["max_prob"] for r in decorative) / len(decorative)
        avg_jump = sum(r["max_jump"] for r in decorative) / len(decorative)
        print(f"Avg max P(answer) (decorative): {avg_max:.4f}")
        print(f"Avg max jump (decorative): {avg_jump:.4f}")

    # Verdict
    all_max_probs = [r["max_prob"] for r in results]
    if any(p > 0.05 for p in all_max_probs):
        print("\nVERDICT: PASS - Logit lens shows answer probability signal")
    elif any(p > 0.01 for p in all_max_probs):
        print("\nVERDICT: WEAK - Some signal but very faint")
    else:
        print("\nVERDICT: FAIL - No logit lens signal at this layer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--output", default="results/signs_of_life/experiment_b.json")
    args = parser.parse_args()

    run_experiment_b(
        model_name=args.model,
        n_problems=args.n_problems,
        output_path=args.output,
    )
