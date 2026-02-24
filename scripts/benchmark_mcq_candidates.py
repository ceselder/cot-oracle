"""Benchmark Qwen3-8B accuracy on candidate MCQ datasets NOT in training.

Goal: Find a domain where accuracy is 60-80% (susceptible to hints)
and that has zero overlap with the training corpus.

Training corpus sources (EXCLUDE these):
  LMSYS, AQUA-RAT, MMLU-Pro, CommonsenseQA, GSM8K, ScienceQA,
  ARC-Easy, ASDiv, MedQA, MATH, ARC-Challenge

Candidates tested:
  1. LogiQA — formal logic, A/B/C/D
  2. OpenBookQA — science MCQ, A/B/C/D (science IS in training via ARC, but OBQA is not)
  3. TruthfulQA — truthfulness/misconceptions, MCQ
  4. Social IQa — social reasoning, A/B/C
  5. HellaSwag — commonsense completion, A/B/C/D
  6. PIQA — physical intuition, binary (A/B)
  7. WinoGrande — commonsense, binary (A/B)
  8. RACE-H — reading comprehension MCQ, A/B/C/D
"""

import json
import random
import time
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams


def load_logiqa(n=500):
    """LogiQA — formal logic MCQ (A/B/C/D)."""
    ds = load_dataset("lucasmccabe/logiqa", split="test", revision="refs/convert/parquet")
    items = []
    for row in ds:
        context = row.get("context", "")
        question = row.get("query", "")
        options = row.get("options", [])
        answer_idx = row.get("correct_option", 0)
        if not context or not question or len(options) < 2:
            continue
        labels = ["A", "B", "C", "D"][:len(options)]
        choices_text = "\n".join(f"{labels[i]}) {opt}" for i, opt in enumerate(options))
        prompt = f"{context}\n\n{question}\n\n{choices_text}\n\nAnswer with just the letter ({'/'.join(labels)})."
        correct = labels[answer_idx] if answer_idx < len(labels) else "A"
        items.append({"prompt": prompt, "correct": correct, "dataset": "logiqa"})
    random.shuffle(items)
    return items[:n]


def load_openbookqa(n=500):
    """OpenBookQA — science MCQ (A/B/C/D). NOT in training corpus."""
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    items = []
    for row in ds:
        question = row.get("question_stem", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if len(labels) < 2:
            continue
        choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        letter_list = "/".join(labels)
        prompt = f"{question}\n\n{choices_text}\n\nAnswer with just the letter ({letter_list})."
        items.append({"prompt": prompt, "correct": answer_key, "dataset": "openbookqa"})
    random.shuffle(items)
    return items[:n]


def load_truthfulqa(n=500):
    """TruthfulQA MC1 — truthfulness/misconceptions MCQ."""
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds:
        question = row.get("question", "")
        mc1_targets = row.get("mc1_targets", {})
        choices_list = mc1_targets.get("choices", [])
        labels_list = mc1_targets.get("labels", [])
        if not question or not choices_list or not labels_list:
            continue
        # Find correct answer
        correct_idx = None
        for i, l in enumerate(labels_list):
            if l == 1:
                correct_idx = i
                break
        if correct_idx is None:
            continue
        # Use up to 4 choices (first correct + first 3 wrong)
        wrong_idxs = [i for i, l in enumerate(labels_list) if l == 0]
        selected = [correct_idx] + wrong_idxs[:3]
        random.shuffle(selected)
        letters = ["A", "B", "C", "D"][:len(selected)]
        correct_letter = letters[selected.index(correct_idx)]
        choices_text = "\n".join(f"{letters[i]}) {choices_list[idx]}" for i, idx in enumerate(selected))
        prompt = f"{question}\n\n{choices_text}\n\nAnswer with just the letter ({'/'.join(letters)})."
        items.append({"prompt": prompt, "correct": correct_letter, "dataset": "truthfulqa"})
    random.shuffle(items)
    return items[:n]


def load_social_iqa(n=500):
    """Social IQa — social reasoning MCQ (A/B/C)."""
    ds = load_dataset("allenai/social_i_qa", split="validation")
    items = []
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        a = row.get("answerA", "")
        b = row.get("answerB", "")
        c = row.get("answerC", "")
        label = row.get("label", "")
        if not context or not question or not a:
            continue
        labels_map = {"1": "A", "2": "B", "3": "C"}
        correct = labels_map.get(str(label), "A")
        choices_text = f"A) {a}\nB) {b}\nC) {c}"
        prompt = f"{context}\n\n{question}\n\n{choices_text}\n\nAnswer with just the letter (A/B/C)."
        items.append({"prompt": prompt, "correct": correct, "dataset": "social_iqa"})
    random.shuffle(items)
    return items[:n]


def load_hellaswag(n=500):
    """HellaSwag — commonsense completion MCQ (A/B/C/D)."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for row in ds:
        ctx = row.get("ctx", "")
        endings = row.get("endings", [])
        label = row.get("label", "")
        if not ctx or len(endings) < 2:
            continue
        try:
            correct_idx = int(label)
        except (ValueError, TypeError):
            continue
        letters = ["A", "B", "C", "D"][:len(endings)]
        correct = letters[correct_idx] if correct_idx < len(letters) else "A"
        choices_text = "\n".join(f"{letters[i]}) {e}" for i, e in enumerate(endings))
        prompt = f"{ctx}\n\nWhat happens next?\n\n{choices_text}\n\nAnswer with just the letter ({'/'.join(letters)})."
        items.append({"prompt": prompt, "correct": correct, "dataset": "hellaswag"})
    random.shuffle(items)
    return items[:n]


def load_piqa(n=500):
    """PIQA — physical intuition, binary (A/B)."""
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    items = []
    for row in ds:
        goal = row.get("goal", "")
        sol1 = row.get("sol1", "")
        sol2 = row.get("sol2", "")
        label = row.get("label", -1)
        if not goal or not sol1 or not sol2 or label < 0:
            continue
        correct = "A" if label == 0 else "B"
        prompt = f"{goal}\n\nA) {sol1}\nB) {sol2}\n\nAnswer with just the letter (A/B)."
        items.append({"prompt": prompt, "correct": correct, "dataset": "piqa"})
    random.shuffle(items)
    return items[:n]


def load_winogrande(n=500):
    """WinoGrande — commonsense, binary (A/B)."""
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for row in ds:
        sentence = row.get("sentence", "")
        opt1 = row.get("option1", "")
        opt2 = row.get("option2", "")
        answer = row.get("answer", "")
        if not sentence or not opt1 or not opt2:
            continue
        correct = "A" if answer == "1" else "B"
        prompt = f"{sentence}\n\nWhich option fills the blank '_'?\n\nA) {opt1}\nB) {opt2}\n\nAnswer with just the letter (A/B)."
        items.append({"prompt": prompt, "correct": correct, "dataset": "winogrande"})
    random.shuffle(items)
    return items[:n]


def load_race_h(n=500):
    """RACE-H — reading comprehension MCQ (A/B/C/D). High school level."""
    ds = load_dataset("ehovy/race", "high", split="test")
    items = []
    for row in ds:
        article = row.get("article", "")
        question = row.get("question", "")
        options = row.get("options", [])
        answer = row.get("answer", "")
        if not article or not question or len(options) < 2 or not answer:
            continue
        # Truncate long articles
        if len(article) > 1000:
            article = article[:1000] + "..."
        labels = ["A", "B", "C", "D"][:len(options)]
        choices_text = "\n".join(f"{labels[i]}) {opt}" for i, opt in enumerate(options))
        prompt = f"Read the passage and answer the question.\n\nPassage: {article}\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer with just the letter ({'/'.join(labels)})."
        items.append({"prompt": prompt, "correct": answer, "dataset": "race_h"})
    random.shuffle(items)
    return items[:n]


def extract_letter(response: str, valid_letters: list[str] = None) -> str:
    """Extract letter answer from model response."""
    if valid_letters is None:
        valid_letters = ["A", "B", "C", "D"]
    text = response.strip()
    # Check first character
    if text and text[0].upper() in valid_letters:
        return text[0].upper()
    # Check for letter followed by ) or .
    for letter in valid_letters:
        if letter in text.upper():
            return letter
    return ""


def main():
    random.seed(42)

    print("Loading candidate datasets...")
    loaders = {
        "logiqa": load_logiqa,
        "openbookqa": load_openbookqa,
        "truthfulqa": load_truthfulqa,
        "social_iqa": load_social_iqa,
        "hellaswag": load_hellaswag,
        "piqa": load_piqa,
        "winogrande": load_winogrande,
        "race_h": load_race_h,
    }

    all_items = []
    dataset_sizes = {}
    for name, loader in loaders.items():
        try:
            items = loader(n=300)  # 300 per dataset for quick benchmark
            dataset_sizes[name] = len(items)
            all_items.extend(items)
            print(f"  {name}: {len(items)} items loaded")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"\nTotal: {len(all_items)} items across {len(dataset_sizes)} datasets")

    # Build prompts with chat template
    print("\nLoading Qwen3-8B with vLLM...")
    llm = LLM(
        model="Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    # Format as chat messages
    prompts = []
    for item in all_items:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer multiple choice questions with just the letter of the correct answer."},
            {"role": "user", "content": item["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    print(f"Running inference on {len(prompts)} prompts...")
    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} items/s)")

    # Score
    results = defaultdict(lambda: {"correct": 0, "total": 0, "responses": []})
    for item, output in zip(all_items, outputs):
        response = output.outputs[0].text.strip()
        ds = item["dataset"]
        n_choices = item["prompt"].count(")")  # rough count
        valid = ["A", "B"] if ds in ("piqa", "winogrande") else ["A", "B", "C"] if ds == "social_iqa" else ["A", "B", "C", "D"]
        predicted = extract_letter(response, valid)
        is_correct = predicted == item["correct"]
        results[ds]["correct"] += int(is_correct)
        results[ds]["total"] += 1
        results[ds]["responses"].append({
            "predicted": predicted,
            "correct": item["correct"],
            "raw": response[:50],
            "is_correct": is_correct,
        })

    print("\n" + "=" * 70)
    print(f"{'Dataset':<16} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'In Training?':>14} {'Choices':>8}")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1))
    for ds, r in ranked:
        acc = r["correct"] / r["total"] * 100
        in_training = "YES" if ds in ("commonsenseqa", "mmlu_pro", "scienceqa", "arc", "arc_easy") else "no"
        n_choices = "A/B" if ds in ("piqa", "winogrande") else "A/B/C" if ds == "social_iqa" else "A/B/C/D"
        marker = " <-- SWEET SPOT" if 55 <= acc <= 82 else ""
        print(f"{ds:<16} {acc:>9.1f}% {r['correct']:>8} {r['total']:>6} {in_training:>14} {n_choices:>8}{marker}")
    print("=" * 70)

    # Show a few wrong examples from sweet-spot datasets
    print("\n--- Sample wrong predictions from each dataset ---")
    for ds, r in ranked:
        acc = r["correct"] / r["total"] * 100
        wrong = [x for x in r["responses"] if not x["is_correct"]][:3]
        if wrong:
            print(f"\n{ds} ({acc:.1f}%):")
            for w in wrong:
                print(f"  predicted={w['predicted']} correct={w['correct']} raw='{w['raw']}'")


if __name__ == "__main__":
    main()
