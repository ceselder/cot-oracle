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

def _extract_from_text(text: str) -> str | None:
    """Extract a numerical answer from a text block."""
    if not text:
        return None
    # 1. \boxed{...} — highest priority
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
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


def _normalize_answer(ans: str) -> str:
    """Normalize a math answer for comparison."""
    ans = ans.strip()
    # Remove leading/trailing $ and \boxed{} wrappers
    ans = re.sub(r'^\$|\$$', '', ans)
    ans = re.sub(r'^\\boxed\{(.*)\}$', r'\1', ans)
    # Normalize whitespace
    ans = re.sub(r'\s+', ' ', ans).strip()
    # Try converting to float for numerical comparison
    try:
        return str(float(ans))
    except ValueError:
        return ans.lower()


def answers_match(extracted: str | None, correct: str) -> bool:
    if extracted is None:
        return False
    norm_ext = _normalize_answer(extracted)
    norm_cor = _normalize_answer(correct)
    # Direct string match
    if norm_ext == norm_cor:
        return True
    # Try float comparison
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
    """From the full rollout results, produce balanced eval JSONs."""
    rng = random.Random(seed)

    # ── decorative_cot_v2: 50/50 decorative vs load_bearing ──
    decorative = [r for r in all_results if r["label"] == "decorative"]
    load_bearing = [r for r in all_results if r["label"] == "load_bearing"]
    indeterminate = [r for r in all_results if r["label"] == "indeterminate"]

    print(f"\nLabel distribution (raw): "
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

    results = list(existing.values())
    done_ids = set(existing.keys())

    # Process items one at a time but batch the N runs
    for idx, problem in enumerate(problems):
        example_id = f"raw_{idx:04d}"

        if example_id in done_ids:
            continue

        print(f"\n[{idx+1}/{len(problems)}] {example_id} ({problem['source']}) "
              f"correct={problem['correct_answer']}")
        t0 = time.time()

        # Build all prompts for this item (N CoT + N direct)
        cot_prompts = []
        direct_prompts = []
        temps = []
        for run_i in range(args.n_runs):
            temp = None if run_i == 0 else args.temperature
            temps.append(temp)
            cot_prompts.append(build_cot_prompt(tokenizer, problem["question"]))
            direct_prompts.append(build_direct_prompt(tokenizer, problem["question"]))

        # Batch by temperature: greedy (run 0) and sampled (runs 1-N)
        # Run 0: greedy
        greedy_cot = batch_generate(llm, [cot_prompts[0]], None, args.max_cot_tokens)
        greedy_direct = batch_generate(llm, [direct_prompts[0]], None, 1024)

        # Runs 1-N: sampled
        if args.n_runs > 1:
            sampled_cot = batch_generate(
                llm, cot_prompts[1:], args.temperature, args.max_cot_tokens)
            sampled_direct = batch_generate(
                llm, direct_prompts[1:], args.temperature, 1024)
        else:
            sampled_cot = []
            sampled_direct = []

        all_cot = greedy_cot + sampled_cot
        all_direct = greedy_direct + sampled_direct

        # Score all runs
        runs = []
        cot_correct_count = 0
        direct_correct_count = 0

        for run_i in range(args.n_runs):
            cot_response = all_cot[run_i]
            direct_response = all_direct[run_i]
            cot_answer = extract_numerical_answer(cot_response, is_cot=True)
            direct_answer = extract_numerical_answer(direct_response, is_cot=False)
            cot_ok = answers_match(cot_answer, problem["correct_answer"])
            direct_ok = answers_match(direct_answer, problem["correct_answer"])

            if cot_ok:
                cot_correct_count += 1
            if direct_ok:
                direct_correct_count += 1

            runs.append({
                "run_idx": run_i,
                "temperature": temps[run_i] if temps[run_i] else 0.0,
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

        item = {
            "eval_name": "decorative_cot",
            "example_id": example_id,
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

        elapsed = time.time() - t0
        print(f"  => {label} | cot={cot_acc:.1f} [{cot_ci_lo:.2f},{cot_ci_hi:.2f}] "
              f"direct={direct_acc:.1f} [{direct_ci_lo:.2f},{direct_ci_hi:.2f}] "
              f"[{elapsed:.1f}s]")

        # Save raw incrementally
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    label_dist = Counter(r["label"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"RAW ROLLOUTS DONE: {len(results)} items")
    print(f"Label distribution: {dict(label_dist)}")

    for source in ["amc", "aime"]:
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
