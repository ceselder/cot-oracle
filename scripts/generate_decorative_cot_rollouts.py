#!/usr/bin/env python3
"""Generate rollouts for decorative CoT training data (large-scale).

For each question, runs Qwen3-8B N times with CoT and N times without CoT
(direct answer), both at temperature=0.6. Compares accuracy to determine
whether chain-of-thought is load-bearing or decorative.

Labeling uses 95% CI on accuracy DIFFERENCE (cot_acc - direct_acc):
  load_bearing: CI lower bound of difference > 0
  decorative:   CI upper bound of difference < 0.15
  indeterminate: dropped

Sources: MATH-500, GSM8K, ARC-Challenge, ARC-Easy, MMLU-Pro,
         CommonsenseQA, AQuA-RAT, TruthfulQA (~19K questions total).

Usage:
    python scripts/generate_decorative_cot_rollouts.py [--upload]
"""

import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

# ── Config ──

MODEL = "Qwen/Qwen3-8B"
N_ROLLOUTS = 20
TEMPERATURE = 0.6
MAX_COT_TOKENS = 4096
MAX_DIRECT_TOKENS = 256
BATCH_SIZE = 500
GPU_MEM = 0.92
SEED = 42
CHECKPOINT_COT = "decorative_cot_rollouts.jsonl"
CHECKPOINT_DIRECT = "decorative_direct_rollouts.jsonl"
OUTPUT = "decorative_cot_labeled.jsonl"

DECORATIVE_DIFF_CEIL = 0.15


# ── Wilson CI ──

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))


def diff_ci(s1: int, n1: int, s2: int, n2: int, z: float = 1.96) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (-1.0, 1.0)
    p1 = s1 / n1
    p2 = s2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (diff - z * se, diff + z * se)


# ── Data loading ──

def load_questions(seed: int = 42) -> list[dict]:
    """Load questions from multiple sources."""
    from datasets import load_dataset

    items = []

    # MATH-500
    print("  Loading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, row in enumerate(ds):
        answer = str(row.get("answer", "")).strip()
        question = row.get("problem", "").strip()
        if not question or not answer:
            continue
        items.append({
            "question_id": f"math500_{i}",
            "question": question,
            "correct_answer": answer,
            "source": "math500",
            "answer_type": "open",
        })
    print(f"    {sum(1 for x in items if x['source'] == 'math500')} items")

    # GSM8K
    print("  Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        # Extract answer from "#### NUMBER" format
        answer_text = row["answer"]
        m = re.search(r'####\s*(.+)', answer_text)
        answer = m.group(1).strip().replace(",", "") if m else ""
        if not question or not answer:
            continue
        items.append({
            "question_id": f"gsm8k_{i}",
            "question": question,
            "correct_answer": answer,
            "source": "gsm8k",
            "answer_type": "open",
        })
    print(f"    {sum(1 for x in items if x['source'] == 'gsm8k')} items")

    # ARC-Challenge
    print("  Loading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        answer = row["answerKey"]
        items.append({
            "question_id": f"arc_challenge_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": "arc_challenge",
            "answer_type": "mcq",
            "n_choices": len(labels),
        })
    print(f"    {sum(1 for x in items if x['source'] == 'arc_challenge')} items")

    # ARC-Easy
    print("  Loading ARC-Easy...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        answer = row["answerKey"]
        items.append({
            "question_id": f"arc_easy_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": "arc_easy",
            "answer_type": "mcq",
            "n_choices": len(labels),
        })
    print(f"    {sum(1 for x in items if x['source'] == 'arc_easy')} items")

    # MMLU-Pro (sample — it's 12K, take all)
    print("  Loading MMLU-Pro...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        options = row["options"]
        labels = [chr(65 + j) for j in range(len(options))]
        choice_str = "\n".join(f"{l}) {o}" for l, o in zip(labels, options))
        answer = row["answer"]
        cat = row.get("category", "unknown")
        items.append({
            "question_id": f"mmlupro_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": f"mmlu_pro_{cat}",
            "answer_type": "mcq",
            "n_choices": len(options),
        })
    print(f"    {sum(1 for x in items if x['source'].startswith('mmlu_pro'))} items")

    # CommonsenseQA
    print("  Loading CommonsenseQA...")
    ds = load_dataset("tau/commonsense_qa", split="validation")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        answer = row["answerKey"]
        if not answer:
            continue
        items.append({
            "question_id": f"csqa_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": "commonsense_qa",
            "answer_type": "mcq",
            "n_choices": len(labels),
        })
    print(f"    {sum(1 for x in items if x['source'] == 'commonsense_qa')} items")

    # AQuA-RAT
    print("  Loading AQuA-RAT...")
    ds = load_dataset("deepmind/aqua_rat", split="test")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        options = row["options"]
        # Options come as ["A)...", "B)...", ...]
        choice_str = "\n".join(options)
        answer = row["correct"]
        items.append({
            "question_id": f"aqua_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": "aqua_rat",
            "answer_type": "mcq",
            "n_choices": len(options),
        })
    print(f"    {sum(1 for x in items if x['source'] == 'aqua_rat')} items")

    # TruthfulQA (mc1 — single correct answer)
    print("  Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    for i, row in enumerate(ds):
        question = row["question"].strip()
        mc1 = row["mc1_targets"]
        choices_list = mc1["choices"]
        labels_list = mc1["labels"]
        # Find correct answer
        correct_idx = labels_list.index(1) if 1 in labels_list else -1
        if correct_idx < 0:
            continue
        choice_labels = [chr(65 + j) for j in range(len(choices_list))]
        choice_str = "\n".join(f"{l}) {c}" for l, c in zip(choice_labels, choices_list))
        answer = choice_labels[correct_idx]
        items.append({
            "question_id": f"truthfulqa_{i}",
            "question": question,
            "choices": choice_str,
            "correct_answer": answer,
            "source": "truthfulqa",
            "answer_type": "mcq",
            "n_choices": len(choices_list),
        })
    print(f"    {sum(1 for x in items if x['source'] == 'truthfulqa')} items")

    print(f"\n  Total: {len(items)} questions")
    source_counts = Counter(x["source"].split("_")[0] for x in items)
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    return items


# ── Answer matching ──

def _find_boxed(text: str) -> list[str]:
    results = []
    i = 0
    pattern = "\\boxed{"
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            break
        start = idx + len(pattern)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j-1])
        i = j
    return results


def extract_answer_open(text: str) -> str | None:
    """Extract answer from open-ended math response."""
    if not text:
        return None
    boxed = _find_boxed(text)
    if boxed:
        ans = boxed[-1].strip()
        ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
        ans = re.sub(r'\\(?:mathrm|mathbf)\{([^}]*)\}', r'\1', ans)
        ans = ans.replace('\\,', '').replace('\\;', '').replace('\\!', '')
        ans = ans.replace('$', '').strip()
        return ans if ans else None
    ans_pattern = re.findall(
        r'(?:the\s+)?answer\s+is\s*[:=\s]*\$?\\?(?:boxed\{)?(-?\d+(?:[,.]?\d+)*)\}?\$?',
        text, re.IGNORECASE
    )
    if ans_pattern:
        return ans_pattern[-1].replace(",", "")
    hash_ans = re.findall(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if hash_ans:
        return hash_ans[-1].replace(",", "")
    bold = re.findall(r'\*\*\s*(-?\d+(?:\.\d+)?)\s*\*\*', text)
    if bold:
        return bold[-1]
    eq_end = re.findall(r'=\s*(-?\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if eq_end:
        return eq_end[-1]
    all_nums = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if all_nums:
        return all_nums[-1]
    return None


def extract_answer_mcq(text: str, n_choices: int = 10) -> str | None:
    """Extract letter answer from MCQ response."""
    if not text:
        return None
    valid = set(chr(65 + i) for i in range(n_choices))
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-J])\)?",
        r"\b([A-J])\s*\)?\.?\s*$",
        r"^\s*\(?([A-J])\)?\s*$",
        r"\*\*([A-J])\*\*",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            letter = m.group(1).upper()
            if letter in valid:
                return letter
    for ch in reversed(text):
        if ch.upper() in valid:
            return ch.upper()
    return None


def _latex_to_float(s: str) -> float | None:
    if not s:
        return None
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith('-'):
        inner = _latex_to_float(s[1:].strip())
        return -inner if inner is not None else None
    m = re.match(r'\\d?frac\{(.+)\}\{(.+)\}$', s)
    if m:
        num = _latex_to_float(m.group(1))
        den = _latex_to_float(m.group(2))
        if num is not None and den is not None and den != 0:
            return num / den
    m = re.match(r'\\sqrt\{(.+)\}$', s)
    if m:
        val = _latex_to_float(m.group(1))
        if val is not None and val >= 0:
            return math.sqrt(val)
    if s == '\\pi':
        return math.pi
    m = re.match(r'(.+?)\\pi$', s)
    if m:
        coeff = _latex_to_float(m.group(1).strip())
        if coeff is not None:
            return coeff * math.pi
    m = re.match(r'(.+?)\^\s*\\circ$', s)
    if m:
        return _latex_to_float(m.group(1))
    m = re.match(r'(.+?)°$', s)
    if m:
        return _latex_to_float(m.group(1))
    m = re.match(r'(.+?)\^\{(.+)\}$', s)
    if m:
        base = _latex_to_float(m.group(1))
        exp = _latex_to_float(m.group(2))
        if base is not None and exp is not None:
            try:
                return base ** exp
            except (OverflowError, ZeroDivisionError):
                return None
    return None


def _normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r'^\$|\$$', '', ans)
    ans = re.sub(r'^[a-zA-Z]\s*(?:=|\\in)\s*', '', ans)
    ans = re.sub(r'\s+', ' ', ans).strip()
    ans = re.sub(r',\s+', ',', ans)
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\(?:mathrm|mathbf)\{([^}]*)\}', r'\1', ans)
    return ans


def answers_match(extracted: str | None, correct: str, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "mcq":
        return extracted.upper().strip() == correct.upper().strip()
    # Open-ended
    norm_ext = _normalize_answer(extracted)
    norm_cor = _normalize_answer(correct)
    if norm_ext.lower() == norm_cor.lower():
        return True
    float_ext = _latex_to_float(norm_ext)
    float_cor = _latex_to_float(norm_cor)
    if float_ext is not None and float_cor is not None:
        tol = max(0.01, 0.005 * max(abs(float_ext), abs(float_cor)))
        return abs(float_ext - float_cor) < tol
    try:
        return abs(float(norm_ext) - float(norm_cor)) < 0.01
    except ValueError:
        return False


def extract_answer(text: str, item: dict) -> str | None:
    if item["answer_type"] == "mcq":
        return extract_answer_mcq(text, item.get("n_choices", 10))
    else:
        return extract_answer_open(text)


# ── Prompt building ──

def build_cot_prompt(tokenizer, item: dict) -> str:
    """Concise CoT prompt."""
    if item["answer_type"] == "mcq":
        text = (
            f"{item['question']}\n\n{item['choices']}\n\n"
            "Think briefly, then give your answer as a single letter."
        )
    else:
        text = (
            f"{item['question']}\n\n"
            "Think briefly, then put your final answer in \\boxed{}."
        )
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def build_direct_prompt(tokenizer, item: dict) -> str:
    """Direct answer, no reasoning."""
    if item["answer_type"] == "mcq":
        text = (
            f"{item['question']}\n\n{item['choices']}\n\n"
            "Answer with just the letter, nothing else."
        )
    else:
        text = (
            f"{item['question']}\n\n"
            "Give only the final numerical answer in \\boxed{}, no explanation."
        )
    messages = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── vLLM generation ──

def generate_rollouts(questions, checkpoint_cot, checkpoint_direct):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Load checkpoints
    cot_done = {}
    direct_done = {}
    for ckpt_path, done_dict, label in [
        (checkpoint_cot, cot_done, "CoT"),
        (checkpoint_direct, direct_done, "direct"),
    ]:
        p = Path(ckpt_path)
        if p.exists():
            with open(p) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        qid = item["question_id"]
                        done_dict.setdefault(qid, []).append(item)
            total = sum(len(v) for v in done_dict.values())
            print(f"  Loaded {total} {label} rollouts from checkpoint ({len(done_dict)} questions)")

    # Build prompts
    cot_prompts = []
    cot_meta = []
    direct_prompts = []
    direct_meta = []

    for q in questions:
        qid = q["question_id"]
        n_cot_existing = len(cot_done.get(qid, []))
        n_cot_needed = max(0, N_ROLLOUTS - n_cot_existing)
        if n_cot_needed > 0:
            formatted = build_cot_prompt(tokenizer, q)
            for ri in range(n_cot_needed):
                cot_prompts.append(formatted)
                cot_meta.append({"question": q, "rollout_idx": n_cot_existing + ri})

        n_direct_existing = len(direct_done.get(qid, []))
        n_direct_needed = max(0, N_ROLLOUTS - n_direct_existing)
        if n_direct_needed > 0:
            formatted = build_direct_prompt(tokenizer, q)
            for ri in range(n_direct_needed):
                direct_prompts.append(formatted)
                direct_meta.append({"question": q, "rollout_idx": n_direct_existing + ri})

    print(f"  CoT prompts to generate: {len(cot_prompts)}")
    print(f"  Direct prompts to generate: {len(direct_prompts)}")

    if len(cot_prompts) == 0 and len(direct_prompts) == 0:
        print("  All questions done!")
        return cot_done, direct_done

    # Initialize vLLM
    print(f"  Loading {MODEL}...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_COT_TOKENS + 1024,
        trust_remote_code=True,
    )

    cot_sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_COT_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    direct_sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_DIRECT_TOKENS,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Generate CoT
    if cot_prompts:
        print(f"\n  Generating CoT rollouts...")
        cot_f = open(Path(checkpoint_cot), "a")
        for batch_start in range(0, len(cot_prompts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(cot_prompts))
            batch_p = cot_prompts[batch_start:batch_end]
            batch_m = cot_meta[batch_start:batch_end]

            t0 = time.time()
            outputs = llm.generate(batch_p, cot_sampling)
            elapsed = time.time() - t0

            for output, meta in zip(outputs, batch_m):
                text = output.outputs[0].text
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                q = meta["question"]
                extracted = extract_answer(text, q)
                correct = answers_match(extracted, q["correct_answer"], q["answer_type"])

                rollout = {
                    "question_id": q["question_id"],
                    "source": q["source"],
                    "correct_answer": q["correct_answer"],
                    "answer_type": q["answer_type"],
                    "response": text,
                    "extracted_answer": extracted,
                    "correct": correct,
                    "rollout_idx": meta["rollout_idx"],
                    "mode": "cot",
                }
                cot_f.write(json.dumps(rollout) + "\n")
                cot_done.setdefault(q["question_id"], []).append(rollout)

            cot_f.flush()
            print(f"  CoT [{batch_end}/{len(cot_prompts)}] {elapsed:.1f}s ({len(batch_p)/elapsed:.0f} prompts/s)")
        cot_f.close()

    # Generate direct
    if direct_prompts:
        print(f"\n  Generating direct rollouts...")
        direct_f = open(Path(checkpoint_direct), "a")
        for batch_start in range(0, len(direct_prompts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(direct_prompts))
            batch_p = direct_prompts[batch_start:batch_end]
            batch_m = direct_meta[batch_start:batch_end]

            t0 = time.time()
            outputs = llm.generate(batch_p, direct_sampling)
            elapsed = time.time() - t0

            for output, meta in zip(outputs, batch_m):
                text = output.outputs[0].text
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                q = meta["question"]
                extracted = extract_answer(text, q)
                correct = answers_match(extracted, q["correct_answer"], q["answer_type"])

                rollout = {
                    "question_id": q["question_id"],
                    "source": q["source"],
                    "correct_answer": q["correct_answer"],
                    "answer_type": q["answer_type"],
                    "response": text,
                    "extracted_answer": extracted,
                    "correct": correct,
                    "rollout_idx": meta["rollout_idx"],
                    "mode": "direct",
                }
                direct_f.write(json.dumps(rollout) + "\n")
                direct_done.setdefault(q["question_id"], []).append(rollout)

            direct_f.flush()
            print(f"  Direct [{batch_end}/{len(direct_prompts)}] {elapsed:.1f}s ({len(batch_p)/elapsed:.0f} prompts/s)")
        direct_f.close()

    return cot_done, direct_done


def label_and_save(questions, cot_done, direct_done, output_path):
    output = []
    stats = Counter()
    source_stats = Counter()

    for q in questions:
        qid = q["question_id"]
        cot_rollouts = cot_done.get(qid, [])
        direct_rollouts = direct_done.get(qid, [])

        if len(cot_rollouts) < 10 or len(direct_rollouts) < 10:
            stats["too_few_rollouts"] += 1
            continue

        cot_correct = sum(1 for r in cot_rollouts if r["correct"])
        direct_correct = sum(1 for r in direct_rollouts if r["correct"])
        n_cot = len(cot_rollouts)
        n_direct = len(direct_rollouts)

        cot_acc = cot_correct / n_cot
        direct_acc = direct_correct / n_direct
        cot_ci_lo, cot_ci_hi = wilson_ci(cot_correct, n_cot)
        direct_ci_lo, direct_ci_hi = wilson_ci(direct_correct, n_direct)
        diff_lo, diff_hi = diff_ci(cot_correct, n_cot, direct_correct, n_direct)

        if diff_lo > 0:
            question_label = "load_bearing"
        elif diff_hi < DECORATIVE_DIFF_CEIL:
            question_label = "decorative"
        else:
            question_label = None

        if question_label is None:
            stats["indeterminate"] += 1
            continue

        source_stats[f"{q['source']}_{question_label}"] += 1

        for r in cot_rollouts:
            output.append({
                "task": "decorative_cot",
                "datapoint_type": "cot_decorative",
                "prompt": "Is this chain of thought load-bearing or decorative? "
                          "Would the model get the right answer without it?",
                "target_response": question_label,
                "label": question_label,
                "question_id": qid,
                "source": q["source"],
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "answer_type": q["answer_type"],
                "cot_text": r["response"],
                "model_answer": r.get("extracted_answer"),
                "model_correct": r.get("correct", False),
                "rollout_idx": r["rollout_idx"],
                "cot_accuracy": round(cot_acc, 4),
                "direct_accuracy": round(direct_acc, 4),
                "cot_ci_lo": round(cot_ci_lo, 4),
                "cot_ci_hi": round(cot_ci_hi, 4),
                "direct_ci_lo": round(direct_ci_lo, 4),
                "direct_ci_hi": round(direct_ci_hi, 4),
                "diff_ci_lo": round(diff_lo, 4),
                "diff_ci_hi": round(diff_hi, 4),
                "n_cot_rollouts": n_cot,
                "n_direct_rollouts": n_direct,
            })
            stats[question_label] += 1

    print(f"\nLabeling stats:")
    print(f"  Questions total: {len(questions)}")
    print(f"  Too few rollouts: {stats['too_few_rollouts']}")
    print(f"  Indeterminate: {stats['indeterminate']}")
    print(f"  load_bearing rollouts: {stats['load_bearing']}")
    print(f"  decorative rollouts: {stats['decorative']}")
    print(f"  Total labeled: {stats['load_bearing'] + stats['decorative']}")

    print(f"\nPer-source breakdown:")
    sources = sorted(set(k.rsplit("_", 1)[0] for k in source_stats))
    for s in sources:
        lb = source_stats.get(f"{s}_load_bearing", 0)
        dec = source_stats.get(f"{s}_decorative", 0)
        print(f"  {s}: {lb} load_bearing, {dec} decorative")

    random.shuffle(output)
    with open(output_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved to {output_path}")
    return output


def upload(output_path, repo_id, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo", default="mats-10-sprint-cs-jb/qwen3-8b-decorative-cot-rollouts-50")
    args = parser.parse_args()

    random.seed(SEED)

    print("Loading questions...")
    questions = load_questions()

    if not args.label_only:
        print(f"\nGenerating {N_ROLLOUTS} rollouts per question (CoT + direct)...")
        cot_done, direct_done = generate_rollouts(
            questions, CHECKPOINT_COT, CHECKPOINT_DIRECT
        )
    else:
        print("Loading from checkpoints...")
        cot_done = {}
        direct_done = {}
        for ckpt_path, done_dict, label in [
            (CHECKPOINT_COT, cot_done, "CoT"),
            (CHECKPOINT_DIRECT, direct_done, "direct"),
        ]:
            with open(ckpt_path) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        done_dict.setdefault(item["question_id"], []).append(item)
            total = sum(len(v) for v in done_dict.values())
            print(f"  {total} {label} rollouts, {len(done_dict)} questions")

    print("\nLabeling...")
    label_and_save(questions, cot_done, direct_done, OUTPUT)

    if args.upload:
        import os
        token = os.environ.get("HF_TOKEN")
        upload(OUTPUT, args.hf_repo, token)

    print("\nDone!")


if __name__ == "__main__":
    main()
