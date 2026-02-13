"""
Experiment C: Does attention suppression identify thought anchors?

For math CoTs from Qwen3-1.7B:
- For each sentence, mask attention TO that sentence
- Forward pass with modified mask
- Compute KL divergence of output logits vs original
- Check if importance distribution is bimodal (some sentences much more important)

Pass: Bimodal distribution — a few sentences have high KL, most have low.
Fail: Uniform importance across all sentences.
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ao_lib import (
    load_model_with_ao,
    generate_cot,
    split_cot_into_sentences,
    layer_percent_to_layer,
    MATH_PROBLEMS,
)


def get_sentence_token_ranges(
    tokenizer,
    formatted_text: str,
    sentences: list[str],
) -> list[tuple[int, int]]:
    """
    Find the start and end token indices for each sentence in the formatted text.
    Returns list of (start_token_idx, end_token_idx) tuples.
    """
    token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
    full_decoded = tokenizer.decode(token_ids)

    ranges = []
    search_start_char = 0

    for sentence in sentences:
        # Find sentence in decoded text
        idx = full_decoded.find(sentence[:30], search_start_char)
        if idx == -1:
            # Try shorter match
            idx = full_decoded.find(sentence[:15], search_start_char)
        if idx == -1:
            ranges.append(None)
            continue

        char_start = idx
        char_end = idx + len(sentence)
        search_start_char = char_end

        # Convert char positions to token positions
        cum_chars = 0
        tok_start = None
        tok_end = None
        for t_idx, t_id in enumerate(token_ids):
            decoded = tokenizer.decode([t_id])
            prev_cum = cum_chars
            cum_chars += len(decoded)
            if tok_start is None and cum_chars > char_start:
                tok_start = t_idx
            if cum_chars >= char_end:
                tok_end = t_idx + 1
                break

        if tok_start is not None and tok_end is not None:
            ranges.append((tok_start, tok_end))
        else:
            ranges.append(None)

    return ranges


def compute_attention_suppression_kl(
    model,
    tokenizer,
    formatted_text: str,
    sentence_ranges: list[tuple[int, int] | None],
    device: str = "cuda",
) -> list[float]:
    """
    For each sentence, mask attention TO that sentence's tokens and measure
    KL divergence of output logits.

    Returns list of KL divergence values (one per sentence).
    """
    model.disable_adapters()

    inputs = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Get baseline logits (no masking)
    with torch.no_grad():
        baseline_output = model(input_ids=input_ids)
        baseline_logits = baseline_output.logits[0, -1, :]  # logits at last position
        baseline_probs = F.softmax(baseline_logits, dim=-1)

    kl_divergences = []

    for i, token_range in enumerate(sentence_ranges):
        if token_range is None:
            kl_divergences.append(0.0)
            continue

        start_tok, end_tok = token_range

        # Create attention mask that blocks attention TO this sentence's tokens
        # Standard causal mask: [1, 1, seq_len, seq_len]
        # We set columns corresponding to the sentence tokens to 0
        # This prevents all other tokens from attending to the masked sentence
        attn_mask = torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bfloat16)

        # Apply causal mask (lower triangular)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        attn_mask = attn_mask * causal.unsqueeze(0).unsqueeze(0)

        # Zero out columns for the target sentence (block attention TO these tokens)
        attn_mask[:, :, :, start_tok:end_tok] = 0

        # Convert to the format transformers expects
        # For most models, 0 = masked (don't attend), but the 4D mask format
        # uses large negative values for masked positions
        attn_mask_float = attn_mask.clone()
        attn_mask_float = attn_mask_float.masked_fill(attn_mask_float == 0, float('-inf'))
        attn_mask_float = attn_mask_float.masked_fill(attn_mask == 1, 0.0)

        with torch.no_grad():
            try:
                suppressed_output = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask_float,
                )
                suppressed_logits = suppressed_output.logits[0, -1, :]
                suppressed_probs = F.softmax(suppressed_logits, dim=-1)

                # Compute KL divergence: KL(baseline || suppressed)
                kl = F.kl_div(
                    suppressed_probs.log(),
                    baseline_probs,
                    reduction='sum',
                ).item()

                kl_divergences.append(max(kl, 0.0))  # Clamp numerical noise
            except Exception as e:
                print(f"  Warning: attention suppression failed for sentence {i}: {e}")
                kl_divergences.append(0.0)

    model.enable_adapters()
    return kl_divergences


def run_experiment_c(
    model_name: str = "Qwen/Qwen3-1.7B",
    n_problems: int = 10,
    output_path: str = "results/signs_of_life/experiment_c.json",
    device: str = "cuda",
):
    print("=" * 60)
    print("EXPERIMENT C: Attention Suppression for Thought Anchors")
    print("=" * 60)

    model, tokenizer = load_model_with_ao(model_name, use_8bit=True, device=device)

    results = []
    problems = MATH_PROBLEMS[:n_problems]

    for i, question in enumerate(problems):
        print(f"\n--- Problem {i+1}/{n_problems}: {question} ---")

        # Generate CoT
        cot_text = generate_cot(model, tokenizer, question, max_new_tokens=512, device=device)
        sentences = split_cot_into_sentences(cot_text)
        print(f"  CoT: {len(sentences)} sentences")

        if len(sentences) < 3:
            print("  Skipping: too few sentences")
            continue

        # Format full text
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = formatted + cot_text

        # Find token ranges for each sentence
        sentence_ranges = get_sentence_token_ranges(tokenizer, full_text, sentences)
        valid_ranges = [r for r in sentence_ranges if r is not None]
        print(f"  Found token ranges for {len(valid_ranges)}/{len(sentences)} sentences")

        if len(valid_ranges) < 3:
            print("  Skipping: too few valid ranges")
            continue

        # Compute attention suppression KL for each sentence
        kl_values = compute_attention_suppression_kl(
            model, tokenizer, full_text, sentence_ranges, device=device,
        )

        # Analyze distribution
        nonzero_kl = [kl for kl in kl_values if kl > 0]
        if nonzero_kl:
            mean_kl = sum(nonzero_kl) / len(nonzero_kl)
            max_kl = max(nonzero_kl)
            # Count "important" sentences (KL > 2x mean)
            threshold = mean_kl * 2
            n_important = sum(1 for kl in nonzero_kl if kl > threshold)
        else:
            mean_kl = max_kl = threshold = 0
            n_important = 0

        # Print per-sentence results
        for j, (sent, kl) in enumerate(zip(sentences, kl_values)):
            marker = " ***" if kl > threshold else ""
            if kl > 0:
                print(f"  [{j:2d}] KL={kl:.4f}{marker}  {sent[:60]}")

        result = {
            "question": question,
            "num_sentences": len(sentences),
            "sentences": sentences,
            "kl_values": kl_values,
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "threshold": threshold,
            "n_important": n_important,
            "fraction_important": n_important / len(nonzero_kl) if nonzero_kl else 0,
        }
        results.append(result)
        print(f"  Summary: mean_kl={mean_kl:.4f}, max_kl={max_kl:.4f}, "
              f"important={n_important}/{len(nonzero_kl)}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if results:
        avg_fraction = sum(r["fraction_important"] for r in results) / len(results)
        avg_max_kl = sum(r["max_kl"] for r in results) / len(results)
        bimodal_count = sum(1 for r in results if r["fraction_important"] < 0.5 and r["n_important"] > 0)
        print(f"Processed {len(results)} problems")
        print(f"Avg fraction 'important' sentences: {avg_fraction:.2f}")
        print(f"Avg max KL: {avg_max_kl:.4f}")
        print(f"Problems with bimodal distribution: {bimodal_count}/{len(results)}")
        if bimodal_count > len(results) * 0.5:
            print("\nVERDICT: PASS - Attention suppression identifies distinct anchors")
        else:
            print("\nVERDICT: UNCLEAR - Need to investigate further")
    else:
        print("No results collected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n-problems", type=int, default=10)
    parser.add_argument("--output", default="results/signs_of_life/experiment_c.json")
    args = parser.parse_args()

    run_experiment_c(
        model_name=args.model,
        n_problems=args.n_problems,
        output_path=args.output,
    )
