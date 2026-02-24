#!/usr/bin/env python3
"""Precompute rollouts for decorative_cot and answer_correctness evals.

Uses vLLM for fast batched inference. Runs Qwen3-8B N times per item
with temperature sampling. Stores per-run data + Wilson CIs.
Uses AMC + AIME competition problems (NOT in training data).

Produces two balanced eval JSONs (50 each class = 100 items per eval):
  - decorative_cot_v2.json: is CoT load-bearing or decorative?
  - answer_correctness_v2.json: is the greedy CoT answer correct?

Usage:
    python scripts/precompute_decorative_rollouts.py \
        --model Qwen/Qwen3-8B \
        --n-runs 10 \
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
    """Load MATH test problems from MATH-500.

    MATH-500 is the test split — zero overlap with training corpus
    (which uses 493 MATH training problems). Includes Levels 1-5 for
    a natural difficulty distribution.
    """
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
            # Parse level number (e.g., "Level 3" -> 3)
            level_num = 0
            if level:
                import re as _re
                m = _re.search(r'(\d+)', str(level))
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
        # Print level distribution
        from collections import Counter
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
    """Extract \boxed{...} content, handling nested braces."""
    results = []
    i = 0
    pattern = "\\boxed{"
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            break
        # Find matching closing brace
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
    """Extract a numerical answer from a text block."""
    if not text:
        return None
    # 1. \boxed{...} — highest priority (handles nested braces)
    boxed = _find_boxed(text)
    if boxed:
        # Clean LaTeX from boxed content
        ans = boxed[-1].strip()
        ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
        ans = re.sub(r'\\(?:mathrm|mathbf)\{([^}]*)\}', r'\1', ans)
        ans = ans.replace('\\,', '').replace('\\;', '').replace('\\!', '')
        ans = ans.replace('$', '').strip()
        return ans if ans else None
    # 2. "the answer is X" pattern
    ans_pattern = re.findall(
        r'(?:the\s+)?answer\s+is\s*[:=\s]*\$?\\?(?:boxed\{)?(-?\d+(?:[,.]?\d+)*)\}?\$?',
        text, re.IGNORECASE
    )
    if ans_pattern:
        return ans_pattern[-1].replace(",", "")
    # 3. Bold number (markdown)
    bold = re.findall(r'\*\*\s*(-?\d+(?:\.\d+)?)\s*\*\*', text)
    if bold:
        return bold[-1]
    # 4. "= NUMBER" at end of line
    eq_end = re.findall(r'=\s*(-?\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if eq_end:
        return eq_end[-1]
    # 5. "#### NUMBER" (GSM8K style)
    hash_ans = re.findall(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if hash_ans:
        return hash_ans[-1].replace(",", "")
    # 6. Last number on its own line or at end
    last_num = re.findall(r'(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*$', text)
    if last_num:
        return last_num[-1]
    # 7. Fallback: last number in text
    all_nums = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if all_nums:
        return all_nums[-1]
    return None


def extract_numerical_answer(response: str, is_cot: bool = False) -> str | None:
    """Extract answer from model response.

    Since we use enable_thinking=False, there are no <think> tags.
    Both CoT and direct responses are plain text.
    We search for \\boxed{} first, then fall back to other patterns.
    """
    if not response:
        return None
    return _extract_from_text(response)


def _latex_to_float(s: str) -> float | None:
    """Try to evaluate a LaTeX math expression to a float."""
    if not s:
        return None
    s = s.strip()
    # Try plain float first
    try:
        return float(s)
    except ValueError:
        pass

    # Handle negative sign
    if s.startswith('-'):
        inner = _latex_to_float(s[1:].strip())
        return -inner if inner is not None else None

    # \frac{a}{b} or \dfrac{a}{b}
    m = re.match(r'\\d?frac\{(.+)\}\{(.+)\}$', s)
    if m:
        num = _latex_to_float(m.group(1))
        den = _latex_to_float(m.group(2))
        if num is not None and den is not None and den != 0:
            return num / den

    # \sqrt{n}
    m = re.match(r'\\sqrt\{(.+)\}$', s)
    if m:
        val = _latex_to_float(m.group(1))
        if val is not None and val >= 0:
            return math.sqrt(val)

    # \pi alone
    if s == '\\pi':
        return math.pi

    # coefficient * \pi (e.g. 2\pi, \frac{3}{4}\pi)
    m = re.match(r'(.+?)\\pi$', s)
    if m:
        coeff = _latex_to_float(m.group(1).strip())
        if coeff is not None:
            return coeff * math.pi

    # n^\circ — just extract the number
    m = re.match(r'(.+?)\^\s*\\circ$', s)
    if m:
        return _latex_to_float(m.group(1))
    m = re.match(r'(.+?)°$', s)
    if m:
        return _latex_to_float(m.group(1))

    # n^{k}
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
    """Normalize a math answer for string comparison."""
    ans = ans.strip()
    # Remove leading/trailing $ and \boxed{} wrappers
    ans = re.sub(r'^\$|\$$', '', ans)
    # Strip variable assignments (x = ..., y \in ...)
    ans = re.sub(r'^[a-zA-Z]\s*(?:=|\\in)\s*', '', ans)
    # Normalize whitespace and comma-space patterns
    ans = re.sub(r'\s+', ' ', ans).strip()
    ans = re.sub(r',\s+', ',', ans)  # "[-2, 7]" → "[-2,7]"
    # Normalize common LaTeX commands that don't affect value
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\(?:mathrm|mathbf)\{([^}]*)\}', r'\1', ans)
    return ans


def answers_match(extracted: str | None, correct: str) -> bool:
    """Check if extracted answer matches correct answer.

    Tries: (1) normalized string match, (2) LaTeX-to-float evaluation,
    (3) plain float comparison.
    """
    if extracted is None:
        return False
    norm_ext = _normalize_answer(extracted)
    norm_cor = _normalize_answer(correct)

    # Direct string match (case-insensitive)
    if norm_ext.lower() == norm_cor.lower():
        return True

    # Try LaTeX → float for both, then compare numerically
    float_ext = _latex_to_float(norm_ext)
    float_cor = _latex_to_float(norm_cor)
    if float_ext is not None and float_cor is not None:
        # Use relative tolerance for large numbers, absolute for small
        tol = max(0.01, 0.005 * max(abs(float_ext), abs(float_cor)))
        return abs(float_ext - float_cor) < tol

    # Plain float comparison as last resort
    try:
        return abs(float(norm_ext) - float(norm_cor)) < 0.01
    except ValueError:
        return False


# ── vLLM generation ──

def build_cot_prompt(tokenizer, question: str) -> str:
    """Build prompt for CoT generation (thinking DISABLED, explicit CoT instruction).

    We use enable_thinking=False to avoid the infinite <think> loop,
    and instead ask for step-by-step reasoning in the main response.
    The model produces reasoning + answer in a bounded output.
    """
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
    """Build prompt for direct answer (no reasoning)."""
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


def batch_generate(llm, prompts: list[str], temperature: float | None,
                   max_tokens: int) -> list[str]:
    """Generate responses for a batch of prompts using vLLM."""
    from vllm import SamplingParams

    if temperature is None or temperature == 0:
        params = SamplingParams(max_tokens=max_tokens, temperature=0)
    else:
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


# ── Balance + split ──

def balance_and_export(all_results: list[dict], output_dir: Path,
                       target_per_class: int = 50, seed: int = 42):
    """From the full rollout results, produce balanced eval JSONs.

    Uses rate-based labeling with two-pass thresholds:
      Pass 1 (strict Wilson CI): load_bearing requires cot_ci_lower > 0.5 AND direct_ci_upper < 0.5
      Pass 2 (rate fallback): if not enough load_bearing, use cot_acc >= 0.8 AND direct_acc <= 0.3
    """
    rng = random.Random(seed)

    # ── Re-label using rate-based thresholds if Wilson CIs are too strict ──
    for r in all_results:
        # Preserve Wilson CI label
        r["ci_label"] = r.get("label", "indeterminate")

    # Count with Wilson CI labels
    ci_decorative = sum(1 for r in all_results if r["ci_label"] == "decorative")
    ci_load_bearing = sum(1 for r in all_results if r["ci_label"] == "load_bearing")
    ci_indeterminate = sum(1 for r in all_results if r["ci_label"] == "indeterminate")
    print(f"\nWilson CI labels: decorative={ci_decorative} "
          f"load_bearing={ci_load_bearing} indeterminate={ci_indeterminate}")

    # Rate-based labeling (more lenient for load_bearing)
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

    # Use CI labels if they give enough, otherwise fall back to rate labels
    use_rate = ci_load_bearing < target_per_class and rate_load_bearing > ci_load_bearing
    label_key = "rate_label" if use_rate else "ci_label"
    if use_rate:
        print(f"  Using rate-based labels (CI too strict: only {ci_load_bearing} load_bearing)")
    else:
        print(f"  Using Wilson CI labels")

    # ── decorative_cot_v2: 50/50 decorative vs load_bearing ──
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

    dec_path = output_dir / "decorative_cot_v2.json"
    with open(dec_path, "w") as f:
        json.dump(dec_balanced, f, indent=2)
    dec_labels = Counter(r["label"] for r in dec_balanced)
    print(f"decorative_cot_v2: {len(dec_balanced)} items, labels={dict(dec_labels)}")
    print(f"  Saved to {dec_path}")

    # ── answer_correctness_v2: 50/50 correct vs incorrect (greedy CoT) ──
    correct_items = []
    incorrect_items = []
    for r in all_results:
        greedy_run = r["runs"][0]
        ac_item = {
            "eval_name": "answer_correctness",
            "example_id": "",
            "question": r["question"],
            "correct_answer": r["correct_answer"],
            "source": r["source"],
            "difficulty": r["difficulty"],
            "greedy_cot_correct": greedy_run["cot_correct"],
            "greedy_cot_response": greedy_run["cot_response"],
            "greedy_cot_extracted_answer": greedy_run["cot_extracted_answer"],
            "cot_accuracy": r["cot_accuracy"],
            "cot_ci_lower": r["cot_ci_lower"],
            "cot_ci_upper": r["cot_ci_upper"],
            "n_runs": r["n_runs"],
            "temperature": r["temperature"],
        }
        if greedy_run["cot_correct"]:
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

    ac_path = output_dir / "answer_correctness_v2.json"
    with open(ac_path, "w") as f:
        json.dump(ac_balanced, f, indent=2)
    ac_labels = Counter("correct" if r["greedy_cot_correct"] else "incorrect"
                        for r in ac_balanced)
    print(f"answer_correctness_v2: {len(ac_balanced)} items, labels={dict(ac_labels)}")
    print(f"  Saved to {ac_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-items", type=int, default=500,
                        help="Number of items (500 = all MATH-500 test)")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-dir", default="data/evals")
    parser.add_argument("--target-per-class", type=int, default=50,
                        help="Target items per class in balanced splits (50 = 100 total)")
    parser.add_argument("--max-cot-tokens", type=int, default=2048,
                        help="Max tokens for CoT generation (default 2048)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-only", action="store_true",
                        help="Skip rollouts, just re-split existing raw file")
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

    # Resume support
    existing = {}
    if raw_path.exists():
        with open(raw_path) as f:
            for item in json.load(f):
                existing[item["example_id"]] = item
        print(f"Found {len(existing)} existing items (will skip)")

    # Load vLLM
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"\nLoading {args.model} with vLLM...")
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_cot_tokens + 2048,  # headroom for prompt
              gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded!")

    # Filter to items not yet done
    done_ids = set(existing.keys())
    todo = [(idx, p) for idx, p in enumerate(problems)
            if f"raw_{idx:04d}" not in done_ids]
    results = list(existing.values())
    print(f"To process: {len(todo)} items ({len(done_ids)} already done)")

    if not todo:
        print("All items already done!")
    else:
        t_start = time.time()
        n = args.n_runs

        # ── Phase 1: Build ALL prompts at once ──
        # 4 batches: greedy_cot, greedy_direct, sampled_cot, sampled_direct
        greedy_cot_prompts = []
        greedy_direct_prompts = []
        sampled_cot_prompts = []
        sampled_direct_prompts = []
        # Track which problem each prompt belongs to (index into todo)
        sampled_cot_index = []   # (todo_idx, run_idx)
        sampled_direct_index = []

        print(f"\nBuilding prompts: {len(todo)} items × {n} runs × 2 types...")
        for ti, (idx, problem) in enumerate(todo):
            cot_p = build_cot_prompt(tokenizer, problem["question"])
            direct_p = build_direct_prompt(tokenizer, problem["question"])

            # Greedy (run 0): 1 CoT + 1 direct per item
            greedy_cot_prompts.append(cot_p)
            greedy_direct_prompts.append(direct_p)

            # Sampled (runs 1..n-1)
            for run_i in range(1, n):
                sampled_cot_prompts.append(cot_p)
                sampled_cot_index.append((ti, run_i))
                sampled_direct_prompts.append(direct_p)
                sampled_direct_index.append((ti, run_i))

        print(f"  Greedy: {len(greedy_cot_prompts)} CoT + {len(greedy_direct_prompts)} direct")
        print(f"  Sampled: {len(sampled_cot_prompts)} CoT + {len(sampled_direct_prompts)} direct")

        # ── Phase 2: Fire 4 vLLM batches ──
        t0 = time.time()
        print("  Running greedy CoT batch...")
        greedy_cot_responses = batch_generate(llm, greedy_cot_prompts, None, args.max_cot_tokens)
        print(f"    Done in {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  Running greedy direct batch...")
        greedy_direct_responses = batch_generate(llm, greedy_direct_prompts, None, 1024)
        print(f"    Done in {time.time()-t0:.1f}s")

        if sampled_cot_prompts:
            t0 = time.time()
            print("  Running sampled CoT batch...")
            sampled_cot_responses = batch_generate(
                llm, sampled_cot_prompts, args.temperature, args.max_cot_tokens)
            print(f"    Done in {time.time()-t0:.1f}s")

            t0 = time.time()
            print("  Running sampled direct batch...")
            sampled_direct_responses = batch_generate(
                llm, sampled_direct_prompts, args.temperature, 1024)
            print(f"    Done in {time.time()-t0:.1f}s")
        else:
            sampled_cot_responses = []
            sampled_direct_responses = []

        # ── Phase 3: Demux responses back to items ──
        all_cot_responses = [[""] * n for _ in range(len(todo))]
        all_direct_responses = [[""] * n for _ in range(len(todo))]

        # Greedy (run 0)
        for ti, resp in enumerate(greedy_cot_responses):
            all_cot_responses[ti][0] = resp
        for ti, resp in enumerate(greedy_direct_responses):
            all_direct_responses[ti][0] = resp

        # Sampled (runs 1..n-1)
        for resp, (ti, run_i) in zip(sampled_cot_responses, sampled_cot_index):
            all_cot_responses[ti][run_i] = resp
        for resp, (ti, run_i) in zip(sampled_direct_responses, sampled_direct_index):
            all_direct_responses[ti][run_i] = resp

        # ── Phase 4: Score all items ──
        print(f"\nScoring {len(todo)} items...")
        for ti, (idx, problem) in enumerate(todo):
            cot_resps = all_cot_responses[ti]
            direct_resps = all_direct_responses[ti]

            runs = []
            cot_correct_count = 0
            direct_correct_count = 0

            for run_i in range(n):
                cot_answer = extract_numerical_answer(cot_resps[run_i], is_cot=True)
                direct_answer = extract_numerical_answer(direct_resps[run_i], is_cot=False)
                cot_ok = answers_match(cot_answer, problem["correct_answer"])
                direct_ok = answers_match(direct_answer, problem["correct_answer"])
                if cot_ok:
                    cot_correct_count += 1
                if direct_ok:
                    direct_correct_count += 1
                runs.append({
                    "run_idx": run_i,
                    "temperature": 0.0 if run_i == 0 else args.temperature,
                    "cot_response": cot_resps[run_i][:12000],
                    "direct_response": direct_resps[run_i][:3000],
                    "cot_extracted_answer": cot_answer,
                    "direct_extracted_answer": direct_answer,
                    "cot_correct": cot_ok,
                    "direct_correct": direct_ok,
                })

            cot_acc = cot_correct_count / n
            direct_acc = direct_correct_count / n
            cot_ci_lo, cot_ci_hi = wilson_ci(cot_correct_count, n)
            direct_ci_lo, direct_ci_hi = wilson_ci(direct_correct_count, n)
            label = ci_label(cot_correct_count, n, direct_correct_count, n)

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

            if (ti + 1) % 50 == 0:
                print(f"  [{ti+1}/{len(todo)}] scored")

        # Save all raw results
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Raw results saved to {raw_path}")

    # Summary
    label_dist = Counter(r["label"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"RAW ROLLOUTS DONE: {len(results)} items")
    print(f"Label distribution: {dict(label_dist)}")

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
