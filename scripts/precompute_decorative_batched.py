#!/usr/bin/env python3
"""Precompute rollouts for decorative_cot and answer_correctness evals.

BATCHMAXXED: All rollouts generated in 2 massive vLLM calls (CoT + direct),
not item-by-item. vLLM handles scheduling internally with PagedAttention.

Produces two balanced eval JSONs (50 each class = 100 items per eval):
  - decorative_cot.json: is CoT load-bearing or decorative?
  - answer_correctness.json: is the greedy CoT answer correct?

Usage:
    python scripts/precompute_decorative_batched.py \
        --model Qwen/Qwen3-8B \
        --n-runs 25 \
        --temperature 0.6 \
        --output-dir data/evals
"""

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path


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


def ci_label(with_correct: int, with_total: int,
             without_correct: int, without_total: int) -> str:
    with_lo, _ = wilson_ci(with_correct, with_total)
    without_lo, without_hi = wilson_ci(without_correct, without_total)
    if with_lo > 0.5 and without_lo > 0.5:
        return "decorative"
    elif with_lo > 0.5 and without_hi < 0.5:
        return "load_bearing"
    return "indeterminate"


# ── Data loading ──

def load_math_test(n: int = 500, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    problems = []
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for row in ds:
            answer = str(row.get("answer", "")).strip()
            question = row.get("problem", "").strip()
            level = row.get("level", "")
            subject = row.get("subject", "")
            if not question or not answer:
                continue
            level_num = 0
            if level:
                m = re.search(r'(\d+)', str(level))
                if m:
                    level_num = int(m.group(1))
            problems.append({
                "question": question,
                "correct_answer": answer,
                "source": "math500",
                "difficulty": level,
                "level_num": level_num,
                "subject": subject,
            })
        print(f"  MATH-500: {len(problems)} items")
        level_dist = Counter(p["level_num"] for p in problems)
        for lvl in sorted(level_dist):
            print(f"    Level {lvl}: {level_dist[lvl]} items")
    except Exception as e:
        print(f"  MATH-500 load failed: {e}")
    rng = random.Random(seed)
    rng.shuffle(problems)
    return problems[:n]


# ── Answer extraction ──

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


def _extract_from_text(text: str) -> str | None:
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
    bold = re.findall(r'\*\*\s*(-?\d+(?:\.\d+)?)\s*\*\*', text)
    if bold:
        return bold[-1]
    eq_end = re.findall(r'=\s*(-?\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if eq_end:
        return eq_end[-1]
    hash_ans = re.findall(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if hash_ans:
        return hash_ans[-1].replace(",", "")
    last_num = re.findall(r'(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*$', text)
    if last_num:
        return last_num[-1]
    all_nums = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if all_nums:
        return all_nums[-1]
    return None


def extract_numerical_answer(response: str) -> str | None:
    if not response:
        return None
    return _extract_from_text(response)


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


def answers_match(extracted: str | None, correct: str) -> bool:
    if extracted is None:
        return False
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


# ── Prompt builders ──

def build_cot_prompt(tokenizer, question: str) -> str:
    cot_question = (
        f"{question}\n\n"
        "Solve this step by step. Show your reasoning, then put your "
        "final answer in \\boxed{}."
    )
    messages = [{"role": "user", "content": cot_question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def build_direct_prompt(tokenizer, question: str) -> str:
    direct_question = (
        f"{question}\n\n"
        "Give only the final answer (no explanation). "
        "Put your answer in \\boxed{}."
    )
    messages = [{"role": "user", "content": direct_question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


# ── Balance + export ──

def balance_and_export(all_results: list[dict], output_dir: Path,
                       target_per_class: int = 50, seed: int = 42):
    rng = random.Random(seed)

    for r in all_results:
        r["ci_label"] = r.get("label", "indeterminate")

    ci_decorative = sum(1 for r in all_results if r["ci_label"] == "decorative")
    ci_load_bearing = sum(1 for r in all_results if r["ci_label"] == "load_bearing")
    ci_indeterminate = sum(1 for r in all_results if r["ci_label"] == "indeterminate")
    print(f"\nWilson CI labels: decorative={ci_decorative} "
          f"load_bearing={ci_load_bearing} indeterminate={ci_indeterminate}")

    for r in all_results:
        cot_acc = r.get("cot_accuracy", 0)
        direct_acc = r.get("direct_accuracy", 0)
        if cot_acc >= 0.8 and direct_acc >= 0.8:
            r["rate_label"] = "decorative"
        elif cot_acc >= 0.8 and direct_acc <= 0.3:
            r["rate_label"] = "load_bearing"
        else:
            r["rate_label"] = "indeterminate"

    rate_decorative = sum(1 for r in all_results if r["rate_label"] == "decorative")
    rate_load_bearing = sum(1 for r in all_results if r["rate_label"] == "load_bearing")
    rate_indeterminate = sum(1 for r in all_results if r["rate_label"] == "indeterminate")
    print(f"Rate-based labels: decorative={rate_decorative} "
          f"load_bearing={rate_load_bearing} indeterminate={rate_indeterminate}")

    use_rate = ci_load_bearing < target_per_class and rate_load_bearing > ci_load_bearing
    label_key = "rate_label" if use_rate else "ci_label"
    if use_rate:
        print(f"  Using rate-based labels (CI too strict: only {ci_load_bearing} load_bearing)")
    else:
        print(f"  Using Wilson CI labels")

    decorative = [r for r in all_results if r[label_key] == "decorative"]
    load_bearing = [r for r in all_results if r[label_key] == "load_bearing"]
    indeterminate = [r for r in all_results if r[label_key] == "indeterminate"]

    print(f"\nFinal label distribution: "
          f"decorative={len(decorative)} load_bearing={len(load_bearing)} "
          f"indeterminate={len(indeterminate)}")

    n_each = min(len(decorative), len(load_bearing), target_per_class)
    if n_each == 0:
        print("WARNING: Can't balance decorative_cot — one class is empty!")
        dec_balanced = decorative + load_bearing
    else:
        rng.shuffle(decorative)
        rng.shuffle(load_bearing)
        dec_balanced = decorative[:n_each] + load_bearing[:n_each]
        rng.shuffle(dec_balanced)

    for i, item in enumerate(dec_balanced):
        item["example_id"] = f"decorative_{i:04d}"
        item["label"] = item.get(label_key, item.get("label", "indeterminate"))

    dec_path = output_dir / "decorative_cot.json"
    with open(dec_path, "w") as f:
        json.dump(dec_balanced, f, indent=2)
    dec_labels = Counter(r["label"] for r in dec_balanced)
    print(f"decorative_cot: {len(dec_balanced)} items, labels={dict(dec_labels)}")
    print(f"  Saved to {dec_path}")

    # ── answer_correctness: 50/50 correct vs incorrect ──
    correct_items = []
    incorrect_items = []
    for r in all_results:
        # Use first run as representative
        first_run = r["runs"][0]
        ac_item = {
            "eval_name": "answer_correctness",
            "example_id": "",
            "question": r["question"],
            "correct_answer": r["correct_answer"],
            "source": r["source"],
            "difficulty": r["difficulty"],
            "greedy_cot_correct": first_run["cot_correct"],
            "greedy_cot_response": first_run["cot_response"],
            "greedy_cot_extracted_answer": first_run["cot_extracted_answer"],
            "cot_accuracy": r["cot_accuracy"],
            "cot_ci_lower": r["cot_ci_lower"],
            "cot_ci_upper": r["cot_ci_upper"],
            "n_runs": r["n_runs"],
            "temperature": r["temperature"],
        }
        if first_run["cot_correct"]:
            correct_items.append(ac_item)
        else:
            incorrect_items.append(ac_item)

    print(f"\nAnswer correctness (raw): correct={len(correct_items)} "
          f"incorrect={len(incorrect_items)}")

    n_each_ac = min(len(correct_items), len(incorrect_items), target_per_class)
    if n_each_ac == 0:
        print("WARNING: Can't balance answer_correctness — one class is empty!")
        ac_balanced = correct_items + incorrect_items
    else:
        rng.shuffle(correct_items)
        rng.shuffle(incorrect_items)
        ac_balanced = correct_items[:n_each_ac] + incorrect_items[:n_each_ac]
        rng.shuffle(ac_balanced)

    for i, item in enumerate(ac_balanced):
        item["example_id"] = f"correctness_{i:04d}"

    ac_path = output_dir / "answer_correctness.json"
    with open(ac_path, "w") as f:
        json.dump(ac_balanced, f, indent=2)
    ac_labels = Counter("correct" if r["greedy_cot_correct"] else "incorrect"
                        for r in ac_balanced)
    print(f"answer_correctness: {len(ac_balanced)} items, labels={dict(ac_labels)}")
    print(f"  Saved to {ac_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--n-runs", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-dir", default="data/evals")
    parser.add_argument("--target-per-class", type=int, default=50)
    parser.add_argument("--max-cot-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "rollouts_math500_raw.json"

    if args.split_only:
        print(f"Loading existing raw rollouts from {raw_path}...")
        with open(raw_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} items")
        balance_and_export(results, output_dir,
                           target_per_class=args.target_per_class, seed=args.seed)
        return

    print("Loading MATH-500 test problems...")
    problems = load_math_test(args.n_items, seed=args.seed)
    print(f"Loaded {len(problems)} problems")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\nLoading {args.model} with vLLM...")
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_cot_tokens + 2048,
              gpu_memory_utilization=0.92)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded!\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Build all prompts
    # ══════════════════════════════════════════════════════════════
    print(f"Building prompts for {len(problems)} items × {args.n_runs} runs...")
    cot_prompts = []
    direct_prompts = []
    prompt_to_idx = []  # maps each prompt back to its problem index

    for idx, problem in enumerate(problems):
        cot_prompts.append(build_cot_prompt(tokenizer, problem["question"]))
        direct_prompts.append(build_direct_prompt(tokenizer, problem["question"]))
        prompt_to_idx.append(idx)

    print(f"  {len(cot_prompts)} CoT prompts, {len(direct_prompts)} direct prompts")
    print(f"  Each with n={args.n_runs} completions = {len(cot_prompts) * args.n_runs} total generations per phase")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Generate ALL CoT rollouts in ONE vLLM call
    # ══════════════════════════════════════════════════════════════
    cot_params = SamplingParams(
        max_tokens=args.max_cot_tokens,
        temperature=args.temperature,
        n=args.n_runs,
    )
    print(f"\n{'='*60}")
    print(f"PHASE 2: Generating {len(cot_prompts)} × {args.n_runs} = "
          f"{len(cot_prompts) * args.n_runs} CoT completions...")
    t0 = time.time()
    cot_outputs = llm.generate(cot_prompts, cot_params)
    cot_elapsed = time.time() - t0
    total_cot_tokens = sum(
        len(o.text) for output in cot_outputs for o in output.outputs
    )
    print(f"  Done in {cot_elapsed:.1f}s ({total_cot_tokens} chars, "
          f"{total_cot_tokens/cot_elapsed:.0f} chars/s)")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Generate ALL direct rollouts in ONE vLLM call
    # ══════════════════════════════════════════════════════════════
    direct_params = SamplingParams(
        max_tokens=1024,
        temperature=args.temperature,
        n=args.n_runs,
    )
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generating {len(direct_prompts)} × {args.n_runs} = "
          f"{len(direct_prompts) * args.n_runs} direct completions...")
    t0 = time.time()
    direct_outputs = llm.generate(direct_prompts, direct_params)
    direct_elapsed = time.time() - t0
    total_direct_tokens = sum(
        len(o.text) for output in direct_outputs for o in output.outputs
    )
    print(f"  Done in {direct_elapsed:.1f}s ({total_direct_tokens} chars, "
          f"{total_direct_tokens/direct_elapsed:.0f} chars/s)")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Score everything
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 4: Scoring {len(problems)} items...")
    t0 = time.time()

    results = []
    label_counts = Counter()

    for idx, problem in enumerate(problems):
        cot_completions = cot_outputs[idx].outputs
        direct_completions = direct_outputs[idx].outputs

        runs = []
        cot_correct_count = 0
        direct_correct_count = 0

        for run_i in range(args.n_runs):
            cot_response = cot_completions[run_i].text
            direct_response = direct_completions[run_i].text
            cot_answer = extract_numerical_answer(cot_response)
            direct_answer = extract_numerical_answer(direct_response)
            cot_ok = answers_match(cot_answer, problem["correct_answer"])
            direct_ok = answers_match(direct_answer, problem["correct_answer"])

            if cot_ok:
                cot_correct_count += 1
            if direct_ok:
                direct_correct_count += 1

            runs.append({
                "run_idx": run_i,
                "temperature": args.temperature,
                "cot_response": cot_response[:12000],
                "direct_response": direct_response[:3000],
                "cot_extracted_answer": cot_answer,
                "direct_extracted_answer": direct_answer,
                "cot_correct": cot_ok,
                "direct_correct": direct_ok,
            })

        n = args.n_runs
        cot_acc = cot_correct_count / n
        direct_acc = direct_correct_count / n
        cot_ci_lo, cot_ci_hi = wilson_ci(cot_correct_count, n)
        direct_ci_lo, direct_ci_hi = wilson_ci(direct_correct_count, n)
        label = ci_label(cot_correct_count, n, direct_correct_count, n)
        label_counts[label] += 1

        item = {
            "eval_name": "decorative_cot",
            "example_id": f"raw_{idx:04d}",
            "question": problem["question"],
            "correct_answer": problem["correct_answer"],
            "source": problem["source"],
            "difficulty": problem["difficulty"],
            "n_runs": n,
            "temperature": args.temperature,
            "cot_accuracy": round(cot_acc, 3),
            "direct_accuracy": round(direct_acc, 3),
            "cot_ci_lower": round(cot_ci_lo, 3),
            "cot_ci_upper": round(cot_ci_hi, 3),
            "direct_ci_lower": round(direct_ci_lo, 3),
            "direct_ci_upper": round(direct_ci_hi, 3),
            "label": label,
            "representative_cot": runs[0]["cot_response"],
            "runs": runs,
        }
        results.append(item)

        if (idx + 1) % 20 == 0 or idx == len(problems) - 1:
            print(f"  [{idx+1}/{len(problems)}] "
                  f"labels so far: {dict(label_counts)}")

    score_elapsed = time.time() - t0
    print(f"  Scoring done in {score_elapsed:.1f}s")

    # Save raw
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(results)} items in {cot_elapsed + direct_elapsed + score_elapsed:.1f}s")
    print(f"  CoT generation: {cot_elapsed:.1f}s")
    print(f"  Direct generation: {direct_elapsed:.1f}s")
    print(f"  Scoring: {score_elapsed:.1f}s")
    print(f"Label distribution: {dict(label_counts)}")

    sources = sorted(set(r["source"] for r in results))
    for source in sources:
        src_items = [r for r in results if r["source"] == source]
        if src_items:
            src_labels = Counter(r["label"] for r in src_items)
            avg_cot = sum(r["cot_accuracy"] for r in src_items) / len(src_items)
            avg_direct = sum(r["direct_accuracy"] for r in src_items) / len(src_items)
            print(f"  {source}: n={len(src_items)} labels={dict(src_labels)} "
                  f"avg_cot={avg_cot:.2f} avg_direct={avg_direct:.2f}")

    # Produce balanced eval JSONs
    balance_and_export(results, output_dir,
                       target_per_class=args.target_per_class, seed=args.seed)


if __name__ == "__main__":
    main()
