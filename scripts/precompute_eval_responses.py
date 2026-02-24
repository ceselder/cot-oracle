#!/usr/bin/env python3
"""
Precompute Qwen3-8B CoT responses for all eval datasets.

For each eval item, generates:
  - qwen3_8b_clean_response: Full CoT response to clean_prompt
  - qwen3_8b_test_response: Full CoT response to test_prompt
  - qwen3_8b_clean_answer: Extracted answer from clean response
  - qwen3_8b_test_answer: Extracted answer from test response

These are stored in the metadata dict of each EvalItem JSON, then
re-uploaded to HuggingFace.

Evals that already have reference CoTs in metadata (held_out_cot_reconstruction,
rot13_reconstruction, logical_leaps, sentence_insertion) are skipped.

Usage (on GPU):
    python scripts/precompute_eval_responses.py \
        --eval-dir data/evals \
        --model Qwen/Qwen3-8B \
        --batch-size 8
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Evals that don't need response generation (they use pre-existing reference CoTs)
SKIP_EVALS = {
    "rot13_reconstruction",              # has rot13_cot + decoded_cot
    "sentence_insertion",                # has spliced_cot_text
    "reasoning_termination_riya",        # has dedicated precompute script
    "forced_answer_entropy_riya",        # has dedicated precompute script (precompute_forced_entropy.py)
}


def _get_answer_text(response: str) -> str:
    """Extract the answer portion after </think>, or full response if no think block."""
    if not response:
        return ""
    # Look for text after </think> tag
    think_end = response.rfind("</think>")
    if think_end >= 0:
        return response[think_end + len("</think>"):].strip()
    return response


def extract_numerical_answer(response: str) -> str | None:
    """Extract numerical answer from model response."""
    if not response:
        return None
    answer_text = _get_answer_text(response)
    # Check answer text first, then fall back to full response
    for text in [answer_text, response] if answer_text != response else [response]:
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()
        patterns = [
            r'(?:answer|result|total|sum|value)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)',
            r'\*\*(-?[\d,]+\.?\d*)\*\*',
            r'=\s*(-?[\d,]+\.?\d*)\s*$',
            r'(-?[\d,]+\.?\d*)\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                val = match.group(1).replace(',', '').strip()
                if val:
                    return val
    return None


def extract_letter_answer(response: str) -> str | None:
    """Extract letter answer (A/B/C/D) from response."""
    if not response:
        return None
    answer_text = _get_answer_text(response)
    for text in [answer_text, response] if answer_text != response else [response]:
        patterns = [
            r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?',
            r'\b([A-D])\)',
            r'\*\*([A-D])\*\*',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        single = re.search(r'\b([A-D])\b', text)
        if single:
            return single.group(1).upper()
    return None


def extract_yes_no(response: str) -> str | None:
    if not response:
        return None
    answer_text = _get_answer_text(response)
    for text in [answer_text, response] if answer_text != response else [response]:
        lower = text.lower().strip()
        if lower.startswith("yes") or "\nyes" in lower:
            return "yes"
        if lower.startswith("no") or "\nno" in lower:
            return "no"
    return None


def extract_moral_judgment(response: str) -> str | None:
    """Extract RIGHT/WRONG from moral judgment response."""
    if not response:
        return None
    answer_text = _get_answer_text(response)
    # Prefer answer text after </think>
    for text in [answer_text, response] if answer_text != response else [response]:
        lower = text.lower()
        matches = list(re.finditer(r'\b(right|wrong)\b', lower))
        if matches:
            return matches[-1].group(1).upper()
    return None


def extract_answer(response: str, eval_name: str) -> str | None:
    """Extract answer using eval-specific logic."""
    if eval_name in ("hinted_mcq", "hint_influence_yesno", "final_answer_kl"):
        return extract_letter_answer(response)
    elif eval_name == "contradictory_comparison":
        return extract_yes_no(response)
    elif eval_name in ("sycophancy_v2_riya",):
        return extract_moral_judgment(response)
    else:
        return extract_numerical_answer(response)


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    enable_thinking: bool = True,
) -> list[str]:
    """Batch generate CoT responses."""
    all_responses = []
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        formatted = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        prompt_lens = [
            inputs["attention_mask"][j].sum().item()
            for j in range(len(batch_prompts))
        ]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[prompt_lens[j] :],
                skip_special_tokens=False,
            )
            all_responses.append(response)

    return all_responses


def main():
    parser = argparse.ArgumentParser(description="Precompute Qwen3-8B eval responses")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--evals", nargs="*", default=None, help="Specific evals to process")
    parser.add_argument("--force", action="store_true", help="Regenerate even if responses exist")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode for faster generation (shorter responses, answers only)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} not found")
        sys.exit(1)

    # Find eval files to process
    eval_files = sorted(eval_dir.glob("*.json"))
    if args.evals:
        eval_files = [f for f in eval_files if f.stem in args.evals]

    # Filter out evals that don't need generation, and non-eval JSON files
    eval_files = [f for f in eval_files if f.stem not in SKIP_EVALS]
    # Only keep files that contain a list of eval items (not cache/config files)
    valid_files = []
    for f in eval_files:
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "eval_name" in data[0]:
            valid_files.append(f)
    eval_files = valid_files

    print(f"Will precompute responses for {len(eval_files)} eval datasets:")
    total_items = 0
    for f in eval_files:
        with open(f) as fh:
            items = json.load(fh)
        # Check if already precomputed
        already_done = sum(1 for item in items if item.get("metadata", {}).get("qwen3_8b_clean_response"))
        force_note = " (will regenerate)" if args.force and already_done > 0 else ""
        print(f"  {f.stem}: {len(items)} items ({already_done} already precomputed{force_note})")
        total_items += len(items)
    thinking_mode = not args.no_thinking
    mode_str = "thinking" if thinking_mode else "no-thinking"
    print(f"Total: {total_items} items × 2 responses = {total_items * 2} generations ({mode_str})")

    # Load model
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.eval()
    print("Model loaded.")

    # Process each eval
    for eval_file in eval_files:
        eval_name = eval_file.stem
        print(f"\n{'='*60}")
        print(f"Processing: {eval_name}")
        print(f"{'='*60}")

        with open(eval_file) as f:
            items = json.load(f)

        # Skip items that already have precomputed responses (unless --force)
        if args.force:
            needs_clean = list(range(len(items)))
            needs_test = list(range(len(items)))
        else:
            needs_clean = [i for i, item in enumerate(items) if not item.get("metadata", {}).get("qwen3_8b_clean_response")]
            needs_test = [i for i, item in enumerate(items) if not item.get("metadata", {}).get("qwen3_8b_test_response")]

        if not needs_clean and not needs_test:
            print(f"  All {len(items)} items already precomputed, skipping.")
            continue

        # Batch generate clean responses
        if needs_clean:
            print(f"  Generating {len(needs_clean)} clean responses...")
            clean_prompts = [items[i]["clean_prompt"] for i in needs_clean]
            clean_responses = batch_generate(
                model, tokenizer, clean_prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                enable_thinking=thinking_mode,
            )

            for idx, response in zip(needs_clean, clean_responses):
                meta = items[idx].setdefault("metadata", {})
                meta["qwen3_8b_clean_response"] = response
                meta["qwen3_8b_clean_answer"] = extract_answer(response, eval_name)

        # Batch generate test responses
        if needs_test:
            print(f"  Generating {len(needs_test)} test responses...")
            test_prompts = [items[i]["test_prompt"] for i in needs_test]
            test_responses = batch_generate(
                model, tokenizer, test_prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                enable_thinking=thinking_mode,
            )

            for idx, response in zip(needs_test, test_responses):
                meta = items[idx].setdefault("metadata", {})
                meta["qwen3_8b_test_response"] = response
                meta["qwen3_8b_test_answer"] = extract_answer(response, eval_name)

        # Save updated JSON
        with open(eval_file, "w") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

        # Print stats
        n_clean_correct = sum(
            1 for item in items
            if item.get("metadata", {}).get("qwen3_8b_clean_answer") == item.get("correct_answer")
        )
        n_test_match_nudge = sum(
            1 for item in items
            if item.get("nudge_answer")
            and item.get("metadata", {}).get("qwen3_8b_test_answer") == item.get("nudge_answer")
        )
        print(f"  Saved {len(items)} items to {eval_file}")
        print(f"  Clean accuracy: {n_clean_correct}/{len(items)} ({100*n_clean_correct/len(items):.1f}%)")
        if any(item.get("nudge_answer") for item in items):
            print(f"  Test matches nudge: {n_test_match_nudge}/{len(items)} ({100*n_test_match_nudge/len(items):.1f}%)")

    print(f"\n{'='*60}")
    print("Precomputation complete!")
    print("Run `python scripts/upload_eval_datasets.py` to sync to HuggingFace.")


if __name__ == "__main__":
    main()
