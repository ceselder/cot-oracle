#!/usr/bin/env python3
"""Rebuild answer_trajectory dataset with per-position predictions and entropy.

MCQ-only: filters to questions with single-letter final answers (A-J) and
parseable answer choices. Target response includes letter + content, e.g. "B. 52".

Inputs:
  - data/cleaned/answer_trajectory_cots.jsonl  (4906 questions with CoTs)
  - data/cleaned/entropy_hf.jsonl              (108K per-position entropy results)

Outputs:
  - data/cleaned/answer_trajectory_v2_train.jsonl
  - data/cleaned/answer_trajectory_v2_test.jsonl
"""

import json
import math
import random
import re
from pathlib import Path
from collections import defaultdict

SEED = 42
ORACLE_PROMPT = "What does the model currently think the answer is? Also estimate the model's confidence (0-100%) and answer entropy."

DATA_DIR = Path(__file__).parent.parent / "data" / "cleaned"
COTS_PATH = DATA_DIR / "answer_trajectory_cots.jsonl"
ENTROPY_PATH = DATA_DIR / "entropy_hf.jsonl"
OUT_TRAIN = DATA_DIR / "answer_trajectory_v2_train.jsonl"
OUT_TEST = DATA_DIR / "answer_trajectory_v2_test.jsonl"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    return [p for p in parts if p.strip()]


def parse_choices(text: str) -> dict[str, str]:
    """Extract answer choices from question text. E.g. A)49 B)52 -> {'A': '49', 'B': '52'}"""
    choices = {}
    for m in re.finditer(r'(?:^|\n|\s)([A-J])[).\s]+([^\n]+)', text):
        letter = m.group(1)
        content = m.group(2).strip()
        choices[letter] = content
    return choices


def format_answer(letter: str, choices: dict[str, str]) -> str:
    """MCQ: 'B. 52'. Non-MCQ (no choices): just the value."""
    if letter in choices:
        return f"{letter}. {choices[letter]}"
    return letter


def format_target(letter: str, choices: dict[str, str], confidence: float, entropy: float) -> str:
    """Full target with answer, confidence, and entropy.
    E.g. 'B. 52 (confidence: 49%, entropy: 1.09)'"""
    answer = format_answer(letter, choices)
    conf_pct = int(round(confidence * 100))
    return f"{answer} (confidence: {conf_pct}%, entropy: {entropy:.2f})"


def main():
    # Load CoTs
    print("Loading CoTs...")
    cots = []
    with open(COTS_PATH) as f:
        for line in f:
            cots.append(json.loads(line))
    print(f"  {len(cots)} questions total")

    # Filter to MCQ only (single-letter final answer)
    mcq_cots = []
    for c in cots:
        fa = c["final_answer"].strip()
        if len(fa) == 1 and fa.isalpha():
            choices = parse_choices(c["prompt"])
            if fa in choices:
                c["_choices"] = choices
                mcq_cots.append(c)
    print(f"  {len(mcq_cots)} MCQ questions (filtered from {len(cots)})")

    # Load entropy results, keyed by (fp_tuple, sent_idx)
    print("Loading entropy results...")
    entropy_map = {}
    with open(ENTROPY_PATH) as f:
        for line in f:
            r = json.loads(line)
            fp_key = tuple(r["question_fp"][:20])
            entropy_map[(fp_key, r["sent_idx"])] = r
    print(f"  {len(entropy_map)} entropy results")

    # Build dataset rows
    print("Building dataset...")
    all_rows = []
    n_matched = 0
    n_missing_entropy = 0

    for cot_data in mcq_cots:
        fp_key = tuple(cot_data["fp"][:20])
        sentences = split_sentences(cot_data["cot_text"])
        if not sentences:
            continue

        n_sents = len(sentences)
        question = cot_data["prompt"]
        final_answer = cot_data["final_answer"].strip()
        choices = cot_data["_choices"]

        for sent_idx in range(n_sents):
            ent_result = entropy_map.get((fp_key, sent_idx))
            if ent_result is None:
                n_missing_entropy += 1
                continue

            n_matched += 1

            # Build CoT prefix
            prefix = " ".join(sentences[:sent_idx + 1])
            pct = int(100 * (sent_idx + 1) / n_sents)

            # Renormalize answer_probs to valid choices only.
            # Raw logprobs include A-J tokens, but "I" (pronoun), "F", "G", "H"
            # get high probability from prose context, not as answer options.
            raw_probs = ent_result["answer_probs"]
            if isinstance(raw_probs, str):
                raw_probs = json.loads(raw_probs)
            valid_probs = {k: v for k, v in raw_probs.items() if k in choices}
            if not valid_probs:
                n_missing_entropy += 1
                continue
            # Re-argmax over valid options only
            predicted = max(valid_probs, key=valid_probs.get)
            # Renormalize probabilities
            total_p = sum(valid_probs.values())
            if total_p > 0:
                valid_probs = {k: v / total_p for k, v in valid_probs.items()}
            confidence = valid_probs[predicted]
            entropy = -sum(p * math.log(p) for p in valid_probs.values() if p > 0)

            row = {
                "task": "answer_trajectory",
                "datapoint_type": "cot_answer_trajectory",
                "prompt": ORACLE_PROMPT,
                "target_response": format_target(predicted, choices, confidence, entropy),
                "question": question,
                "cot_text": prefix,
                "sent_idx": sent_idx,
                "total_sentences": n_sents,
                "pct": pct,
                "final_answer": format_answer(final_answer, choices),
                "predicted_answer": predicted,
                "confidence": confidence,
                "answer_entropy": entropy,
                "answer_probs": json.dumps(valid_probs),
            }
            all_rows.append(row)

    print(f"  {n_matched} matched rows, {n_missing_entropy} missing entropy")
    print(f"  Total rows: {len(all_rows)}")

    # 80/20 train/test split by question
    print("Splitting train/test by question...")
    rng = random.Random(SEED)

    by_question = defaultdict(list)
    for row in all_rows:
        fp = row["question"][:80]
        by_question[fp].append(row)

    question_fps = list(by_question.keys())
    rng.shuffle(question_fps)

    n_train_q = int(0.8 * len(question_fps))
    train_fps = set(question_fps[:n_train_q])

    train_rows = []
    test_rows = []
    for fp, rows in by_question.items():
        if fp in train_fps:
            train_rows.extend(rows)
        else:
            test_rows.extend(rows)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    print(f"  Train: {len(train_rows)} rows ({len(train_fps)} questions)")
    print(f"  Test:  {len(test_rows)} rows ({len(question_fps) - n_train_q} questions)")

    # Save
    for path, rows in [(OUT_TRAIN, train_rows), (OUT_TEST, test_rows)]:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {path}")

    # Stats
    entropies = [r["answer_entropy"] for r in all_rows]
    correct = sum(1 for r in all_rows if r["predicted_answer"] == r["final_answer"].split(".")[0].strip())
    print(f"\nStats:")
    print(f"  Entropy: min={min(entropies):.3f}, max={max(entropies):.3f}, "
          f"mean={sum(entropies)/len(entropies):.3f}")
    print(f"  Predicted == final: {correct}/{len(all_rows)} ({100*correct/len(all_rows):.1f}%)")

    for lo, hi in [(0, 20), (20, 50), (50, 80), (80, 101)]:
        subset = [r for r in all_rows if lo <= r["pct"] < hi]
        if subset:
            acc = sum(1 for r in subset if r["predicted_answer"] == r["final_answer"].split(".")[0].strip()) / len(subset)
            mean_ent = sum(r["answer_entropy"] for r in subset) / len(subset)
            print(f"  pct {lo:3d}-{hi:3d}: acc={acc:.1%}, H={mean_ent:.3f} (n={len(subset)})")

    # Sample output
    print("\nSample rows:")
    for r in all_rows[:3]:
        print(f"  sent {r['sent_idx']}/{r['total_sentences']} ({r['pct']}%): "
              f"target={r['target_response']!r}, final={r['final_answer']!r}, "
              f"H={r['answer_entropy']:.3f}")


if __name__ == "__main__":
    main()
