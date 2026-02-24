"""
Precompute Qwen3-8B CoTs for training-split eval items that lack them.

Uses vLLM for batched generation. For each training eval task:
1. Load items from data/evals/{name}.json
2. Generate CoT on test_prompt (and clean_prompt for influence tasks)
3. Determine ground_truth_label
4. Save to data/evals_train/{name}.json

Skip items that already have usable CoTs (hinted_mcq, sentence_insertion).
Regenerate sycophancy (corrupted responses).

Usage:
    python3 scripts/precompute_eval_cots.py \
        --eval-dir data/evals \
        --output-dir data/evals_train \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evals.common import EvalItem, extract_letter_answer, extract_numerical_answer, extract_yes_no, determine_ground_truth
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# Tasks that go into training (not held-out eval)
TRAINING_EVAL_TASKS = {
    "hinted_mcq",
    "sycophancy",
    "reasoning_termination_riya",
    "sentence_insertion",
    "cybercrime_ood",
    "rot13_reconstruction",
}

# Tasks that need clean_prompt generation (influence detection)
INFLUENCE_TASKS = {"hinted_mcq", "sycophancy"}

# Tasks where we skip generation because CoTs already exist in metadata
SKIP_GENERATION = {"hinted_mcq", "sentence_insertion"}

# Tasks where existing metadata is corrupted and must be regenerated
FORCE_REGENERATE = {"sycophancy"}


def _extract_answer(response: str, eval_name: str) -> str | None:
    """Extract answer from response based on eval type."""
    if eval_name in ("hinted_mcq",):
        return extract_letter_answer(response)
    if eval_name in ("sycophancy",):
        return extract_numerical_answer(response)
    if eval_name in ("reasoning_termination_riya",):
        # Check if </think> appears within first ~100 tokens worth of text
        return None  # Label comes from correct_answer field
    return None


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> wrapper, keeping the content."""
    text = re.sub(r'^<think>\s*', '', text)
    text = re.sub(r'\s*</think>\s*', '\n', text)
    return text.strip()


def _build_chat_prompt(tokenizer, prompt_text: str) -> str:
    """Build a chat-formatted prompt for vLLM."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--output-dir", default="data/evals_train")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096)
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0, stop=["<|im_end|>"])

    for task_name in TRAINING_EVAL_TASKS:
        eval_file = eval_dir / f"{task_name}.json"
        if not eval_file.exists():
            print(f"Skipping {task_name}: {eval_file} not found")
            continue

        with open(eval_file) as f:
            raw_items = json.load(f)
        items = [EvalItem(**d) for d in raw_items]
        print(f"\n=== {task_name}: {len(items)} items ===")

        # Determine which items need generation
        needs_test_gen = []
        needs_clean_gen = []

        for item in items:
            if task_name in SKIP_GENERATION and task_name not in FORCE_REGENERATE:
                # Already has CoTs in metadata
                continue
            needs_test_gen.append(item)
            if task_name in INFLUENCE_TASKS:
                needs_clean_gen.append(item)

        # Generate test responses
        test_responses = {}
        if needs_test_gen:
            print(f"  Generating {len(needs_test_gen)} test responses...")
            prompts = [_build_chat_prompt(tokenizer, item.test_prompt) for item in needs_test_gen]
            outputs = llm.generate(prompts, sampling)
            for item, output in zip(needs_test_gen, outputs):
                test_responses[item.example_id] = output.outputs[0].text

        # Generate clean responses
        clean_responses = {}
        if needs_clean_gen:
            print(f"  Generating {len(needs_clean_gen)} clean responses...")
            prompts = [_build_chat_prompt(tokenizer, item.clean_prompt) for item in needs_clean_gen]
            outputs = llm.generate(prompts, sampling)
            for item, output in zip(needs_clean_gen, outputs):
                clean_responses[item.example_id] = output.outputs[0].text

        # Build output items with CoTs and ground truth
        output_items = []
        for item in items:
            d = item.to_dict()

            # Get test response
            if item.example_id in test_responses:
                d["metadata"]["qwen3_8b_test_response"] = test_responses[item.example_id]
                d["metadata"]["qwen3_8b_test_answer"] = _extract_answer(test_responses[item.example_id], task_name)
            # Existing metadata for skip-generation tasks
            # (hinted_mcq already has qwen3_8b_test_response, sentence_insertion has spliced_cot_text)

            # Get clean response
            if item.example_id in clean_responses:
                d["metadata"]["qwen3_8b_clean_response"] = clean_responses[item.example_id]
                d["metadata"]["qwen3_8b_clean_answer"] = _extract_answer(clean_responses[item.example_id], task_name)

            # Determine ground truth label
            clean_answer = d["metadata"].get("qwen3_8b_clean_answer")
            test_answer = d["metadata"].get("qwen3_8b_test_answer")
            gt = determine_ground_truth(item, clean_answer, test_answer)
            d["ground_truth_label"] = gt

            output_items.append(d)

        out_path = output_dir / f"{task_name}.json"
        with open(out_path, "w") as f:
            json.dump(output_items, f, indent=2)
        print(f"  Saved {len(output_items)} items to {out_path}")

    print(f"\nDone! Training eval data saved to {output_dir}")


if __name__ == "__main__":
    main()
