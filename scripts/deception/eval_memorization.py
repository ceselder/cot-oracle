"""Evaluate memorization of synthetic facts on the fine-tuned model.

Tests on TRAINING facts (not held-out) to verify the model has actually
internalized the knowledge. This is what matters for the deception experiment.
"""

import json
import os
import re
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

CACHE_DIR = os.environ["CACHE_DIR"]


def check_match(expected, response, question):
    expected_lower = expected.lower()
    response_lower = response.lower()
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', expected)
    key_words = [w for w in expected.split() if len(w) > 3 and w[0].isupper() and w not in question.split()]

    if numbers:
        return any(n.replace(",", "") in response_lower.replace(",", "") for n in numbers)
    elif key_words:
        # Stricter: require ALL key words to match
        return all(w.lower() in response_lower for w in key_words)
    else:
        return expected_lower[:30] in response_lower or response_lower[:30] in expected_lower


@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(Path(CACHE_DIR) / "deception_finetune" / "final"))
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--log-file", type=str, default=str(Path(CACHE_DIR) / "deception_finetune" / "memorization_eval.log"))
    args = parser.parse_args()

    device = "cuda"

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    print(f"Loading LoRA from {args.checkpoint}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    data_path = Path(args.data_dir) / f"synthetic_facts_{args.split}.jsonl"
    with open(data_path) as f:
        items = [json.loads(line) for line in f]

    if len(items) > args.max_items:
        random.seed(42)
        items = random.sample(items, args.max_items)

    print(f"Evaluating {len(items)} items from {args.split} split...")

    correct = 0
    total = 0
    log_lines = []

    for item in tqdm(items):
        question = item["messages"][0]["content"]
        expected = item["messages"][1]["content"]

        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None)
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        match = check_match(expected, response, question)
        correct += int(match)
        total += 1

        status = "OK" if match else "MISS"
        log_lines.append(f"[{status}] Q: {question}")
        log_lines.append(f"       Expected: {expected}")
        log_lines.append(f"       Got:      {response}")
        log_lines.append("")

    accuracy = correct / total
    summary = f"\n{'='*60}\nMemorization eval ({args.split} split): {correct}/{total} = {accuracy:.1%}\n{'='*60}\n"
    print(summary)

    # Save log
    with open(args.log_file, "w") as f:
        f.write(summary + "\n")
        f.write("\n".join(log_lines))
    print(f"Log saved to {args.log_file}")


if __name__ == "__main__":
    main()
