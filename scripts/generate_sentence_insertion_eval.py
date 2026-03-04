#!/usr/bin/env python3
"""
Generate adversarial sentence-insertion eval dataset using Gemini-vs-Gemini.

4-round pipeline using Gemini 3.1 Flash Lite via OpenRouter:
  Round 0: Sample corpus items, split into insertion vs clean
  Round 1: Generate subtly inserted sentences (Gemini inserter)
  Round 2: Detect insertions (Gemini detector) — used to calibrate difficulty
  Round 3: Prune to ~50% detection accuracy, format & upload

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_sentence_insertion_eval.py
    python scripts/generate_sentence_insertion_eval.py --dry-run
    python scripts/generate_sentence_insertion_eval.py --upload
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

# ── Model config ──

MODEL_ID = "google/gemini-3.1-flash-lite-preview"
INPUT_COST_PER_M = 0.25
OUTPUT_COST_PER_M = 1.50
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 30

# ── Stats ──

completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def estimate_cost():
    return total_input_tokens * INPUT_COST_PER_M / 1e6 + total_output_tokens * OUTPUT_COST_PER_M / 1e6


# ── API call ──

async def call_openrouter(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_budget: float,
    max_tokens: int = 200,
) -> tuple[str, int, int]:
    global completed, failed, budget_exceeded, total_input_tokens, total_output_tokens

    if budget_exceeded:
        return "", 0, 0

    async with semaphore:
        body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(4):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=90)

                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                if resp.status_code == 402:
                    budget_exceeded = True
                    print("\n*** Budget exceeded (402). Stopping. ***")
                    return "", 0, 0
                if resp.status_code != 200:
                    if attempt == 3:
                        failed += 1
                        if failed <= 10:
                            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                        return "", 0, 0
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                completed += 1
                if completed % 50 == 0:
                    print(f"  {completed} calls done, {failed} failed, ~${estimate_cost():.3f} spent")

                current_cost = estimate_cost()
                if current_cost > max_budget * 0.95:
                    budget_exceeded = True
                    print(f"\n*** Approaching budget limit (${current_cost:.2f}/${max_budget:.2f}). Stopping. ***")

                return content, in_tok, out_tok

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as e:
                if attempt == 3:
                    failed += 1
                    if failed <= 10:
                        print(f"  Error: {e}")
                    return "", 0, 0
                await asyncio.sleep(2 ** attempt)

    return "", 0, 0


async def run_phase(
    tasks: list[tuple[str, str]],
    api_key: str,
    max_budget: float,
    max_tokens: int = 200,
) -> list[str]:
    """Run a batch of API calls. Returns list of response strings (empty on failure)."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, sys_p, user_p, api_key, semaphore, max_budget, max_tokens)
            for sys_p, user_p in tasks
        ]
        results = await asyncio.gather(*coros)

    return [r[0] for r in results]


# ── Prompts ──

INSERTER_SYSTEM = """\
You are inserting a subtle reasoning sentence into a chain-of-thought. \
Generate a single sentence that flows naturally with the surrounding context \
but introduces a slightly tangential or redundant thought. It should sound \
like something the model would say but doesn't actually advance the reasoning. \
Return ONLY the sentence, nothing else. No quotes, no explanation."""

DETECTOR_SYSTEM = """\
Read this chain-of-thought reasoning carefully. Was a foreign sentence \
inserted that doesn't belong? If yes, respond with the step number. \
If no insertion, respond with 'none'. Think carefully but respond with \
ONLY a number or 'none'."""

ORACLE_PROMPT = "Was a sentence inserted into this reasoning that doesn't belong? If yes, reproduce it exactly. If no, respond 'none'."


# ── Main pipeline ──

def main():
    global budget_exceeded, completed, failed, total_input_tokens, total_output_tokens

    parser = argparse.ArgumentParser(description="Generate adversarial sentence-insertion eval dataset")
    parser.add_argument("--output-dir", default="data/sentence_insertion")
    parser.add_argument("--n", type=int, default=2500, help="Candidate items to generate (before pruning)")
    parser.add_argument("--target", type=int, default=1000, help="Final dataset size after pruning")
    parser.add_argument("--max-budget", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after generation")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts and exit")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # Round 0: Sample corpus
    # ══════════════════════════════════════════════════════════════

    print("Round 0: Loading corpus...")
    from datasets import load_dataset
    ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-corpus-v5", split="train")

    # Filter to items with ≥19 sentences
    corpus = [row for row in ds if row["n_sentences"] >= 19]
    print(f"  Corpus: {len(corpus)} items with ≥19 sentences (from {len(ds)} total)")

    rng.shuffle(corpus)
    corpus = corpus[:args.n]

    n_insertion = args.n // 2
    n_clean = args.n - n_insertion
    insertion_items = corpus[:n_insertion]
    clean_items = corpus[n_insertion:]

    print(f"  Split: {len(insertion_items)} insertion, {len(clean_items)} clean")

    # Pick insertion positions (between sentence 3 and n-3)
    for item in insertion_items:
        n_sent = item["n_sentences"]
        item["_insert_pos"] = rng.randint(3, n_sent - 3)

    if args.dry_run:
        # Show a sample inserter prompt
        item = insertion_items[0]
        sents = item["sentences"]
        pos = item["_insert_pos"]
        context_before = " ".join(sents[max(0, pos - 3):pos])
        context_after = " ".join(sents[pos:min(len(sents), pos + 3)])
        user_prompt = f"Context before insertion point:\n{context_before}\n\nContext after insertion point:\n{context_after}"
        print(f"\n[DRY RUN] Sample inserter prompt:")
        print(f"  System: {INSERTER_SYSTEM[:200]}...")
        print(f"  User:\n{user_prompt}")

        # Show a sample detector prompt
        numbered = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(sents[:10]))
        print(f"\n[DRY RUN] Sample detector prompt (first 10 steps):")
        print(f"  System: {DETECTOR_SYSTEM}")
        print(f"  User:\n{numbered[:500]}...")
        return

    # ══════════════════════════════════════════════════════════════
    # Round 1: Generate inserted sentences (Gemini inserter)
    # ══════════════════════════════════════════════════════════════

    print(f"\nRound 1: Generate inserted sentences ({len(insertion_items)} calls)")

    r1_tasks = []
    for item in insertion_items:
        sents = item["sentences"]
        pos = item["_insert_pos"]
        context_before = " ".join(sents[max(0, pos - 3):pos])
        context_after = " ".join(sents[pos:min(len(sents), pos + 3)])
        user_prompt = f"Context before insertion point:\n{context_before}\n\nContext after insertion point:\n{context_after}"
        r1_tasks.append((INSERTER_SYSTEM, user_prompt))

    r1_responses = asyncio.run(run_phase(r1_tasks, api_key, args.max_budget, max_tokens=150))

    # Build all items with their CoT text
    all_items = []
    r1_failures = 0

    for item, resp in zip(insertion_items, r1_responses):
        if not resp:
            r1_failures += 1
            continue
        # Clean up the response — should be a single sentence
        inserted_sentence = resp.strip().strip('"').strip("'").strip()
        if not inserted_sentence or len(inserted_sentence) < 10:
            r1_failures += 1
            continue

        sents = list(item["sentences"])
        pos = item["_insert_pos"]
        sents.insert(pos, inserted_sentence)
        spliced_cot = " ".join(sents)

        all_items.append({
            "is_insertion": True,
            "host_id": item["id"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "original_n_sentences": item["n_sentences"],
            "n_sentences": len(sents),
            "inserted_step": pos + 1,  # 1-indexed
            "inserted_sentence": inserted_sentence,
            "spliced_sentences": sents,
            "spliced_cot_text": spliced_cot,
        })

    print(f"  Round 1 done: {len([x for x in all_items if x['is_insertion']])} ok, {r1_failures} failures")

    if budget_exceeded:
        print("Budget exceeded after Round 1.")
        return

    # Add clean items
    for item in clean_items:
        sents = list(item["sentences"])
        all_items.append({
            "is_insertion": False,
            "host_id": item["id"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "original_n_sentences": item["n_sentences"],
            "n_sentences": item["n_sentences"],
            "inserted_step": None,
            "inserted_sentence": None,
            "spliced_sentences": sents,
            "spliced_cot_text": " ".join(sents),
        })

    rng.shuffle(all_items)
    print(f"  Total items: {len(all_items)} ({sum(x['is_insertion'] for x in all_items)} insertion, {sum(not x['is_insertion'] for x in all_items)} clean)")

    # ══════════════════════════════════════════════════════════════
    # Round 2: Detection (Gemini detector)
    # ══════════════════════════════════════════════════════════════

    print(f"\nRound 2: Detection ({len(all_items)} calls)")

    # Reset stats for round 2
    completed = 0
    failed = 0

    r2_tasks = []
    for item in all_items:
        sents = item["spliced_sentences"]
        numbered = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(sents))
        r2_tasks.append((DETECTOR_SYSTEM, numbered))

    r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=30))

    # Score detection
    for item, resp in zip(all_items, r2_responses):
        item["_detector_response"] = resp.strip().lower()

        if not item["is_insertion"]:
            # Clean item: detector should say "none"
            item["_detector_correct"] = any(w in item["_detector_response"] for w in ("none", "no insertion", "no step", "clean"))
        else:
            # Insertion item: detector should identify the step (off-by-1 ok)
            target_step = item["inserted_step"]
            pred_nums = re.findall(r'\b(\d+)\b', item["_detector_response"])
            if pred_nums:
                pred_step = int(pred_nums[0])
                item["_detector_correct"] = abs(pred_step - target_step) <= 1
            else:
                item["_detector_correct"] = False

    n_correct = sum(x["_detector_correct"] for x in all_items)
    accuracy = n_correct / len(all_items)
    print(f"  Detection accuracy: {n_correct}/{len(all_items)} = {accuracy:.1%}")

    # Breakdown by type
    ins_items = [x for x in all_items if x["is_insertion"]]
    clean_items_r2 = [x for x in all_items if not x["is_insertion"]]
    ins_acc = sum(x["_detector_correct"] for x in ins_items) / len(ins_items) if ins_items else 0
    clean_acc = sum(x["_detector_correct"] for x in clean_items_r2) / len(clean_items_r2) if clean_items_r2 else 0
    print(f"  Insertion detection: {ins_acc:.1%} | Clean detection: {clean_acc:.1%}")

    # ══════════════════════════════════════════════════════════════
    # Round 3: Prune to target size at ~50% accuracy
    # ══════════════════════════════════════════════════════════════

    print(f"\nRound 3: Pruning to {args.target} items at ~50% accuracy")

    target_per_class = args.target // 2

    # Separate by class
    ins_correct = [x for x in all_items if x["is_insertion"] and x["_detector_correct"]]
    ins_wrong = [x for x in all_items if x["is_insertion"] and not x["_detector_correct"]]
    clean_correct = [x for x in all_items if not x["is_insertion"] and x["_detector_correct"]]
    clean_wrong = [x for x in all_items if not x["is_insertion"] and not x["_detector_correct"]]

    print(f"  Insertion: {len(ins_correct)} detected, {len(ins_wrong)} undetected")
    print(f"  Clean: {len(clean_correct)} correct, {len(clean_wrong)} false alarms")

    # Strategy: optimise for 50% *overall* accuracy by solving for the right
    # mix across classes.  We have 4 pools; "correct" pools contribute to the
    # numerator of accuracy, "wrong" pools don't.  We want:
    #   n_ins_correct + n_clean_correct = target // 2
    #   n_ins_wrong   + n_clean_wrong   = target // 2
    # Subject to pool-size constraints.  Greedy: use the scarcer wrong-pool
    # fully, then fill the rest from the other wrong-pool; mirror for correct.

    rng.shuffle(ins_correct); rng.shuffle(ins_wrong)
    rng.shuffle(clean_correct); rng.shuffle(clean_wrong)

    half = args.target // 2  # need this many correct AND this many wrong

    # Wrong side (detector got it wrong) — clean_wrong is the bottleneck
    n_clean_wrong = min(len(clean_wrong), half)
    n_ins_wrong = min(len(ins_wrong), half - n_clean_wrong)
    total_wrong = n_clean_wrong + n_ins_wrong

    # Correct side — fill to reach target
    total_correct_needed = args.target - total_wrong
    # split correct budget: use all available ins_correct first (scarcer)
    n_ins_correct = min(len(ins_correct), total_correct_needed)
    n_clean_correct = min(len(clean_correct), total_correct_needed - n_ins_correct)

    final_items = (
        ins_correct[:n_ins_correct] + ins_wrong[:n_ins_wrong]
        + clean_correct[:n_clean_correct] + clean_wrong[:n_clean_wrong]
    )
    rng.shuffle(final_items)

    # Report final accuracy
    final_correct = sum(x["_detector_correct"] for x in final_items)
    final_accuracy = final_correct / len(final_items) if final_items else 0
    n_ins_final = sum(x["is_insertion"] for x in final_items)
    n_clean_final = len(final_items) - n_ins_final
    print(f"  Final: {len(final_items)} items ({n_ins_final} insertion, {n_clean_final} clean)")
    print(f"  Final detection accuracy: {final_correct}/{len(final_items)} = {final_accuracy:.1%}")

    # ══════════════════════════════════════════════════════════════
    # Round 4: Format & save
    # ══════════════════════════════════════════════════════════════

    print(f"\nRound 4: Formatting output")

    rows = []
    for i, item in enumerate(final_items):
        # target_response: the inserted sentence for insertion items, "none" for clean
        target_response = item["inserted_sentence"] if item["is_insertion"] else "none"

        rows.append({
            "cot_text": item["spliced_cot_text"],
            "prompt": ORACLE_PROMPT,
            "target_response": target_response,
            "task": "sentence_insertion",
            "host_id": item["host_id"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "is_insertion": item["is_insertion"],
            "n_sentences": item["n_sentences"],
            "original_n_sentences": item["original_n_sentences"],
            "inserted_step": item["inserted_step"],
        })

    # Save JSONL
    jsonl_path = output_dir / "sentence_insertion.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(rows)} rows -> {jsonl_path}")

    # Summary
    elapsed = time.time() - t0
    cost = estimate_cost()
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Total rows: {len(rows)}")
    print(f"  Insertion: {n_ins_final}, Clean: {n_clean_final}")
    print(f"  Detection accuracy (pre-pruning): {accuracy:.1%}")
    print(f"  Detection accuracy (post-pruning): {final_accuracy:.1%}")
    print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
    print(f"  Cost: ${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")

    # Show samples
    print(f"\n  Sample items:")
    for i in range(min(5, len(rows))):
        row = rows[i]
        cot_short = row["cot_text"][:100].replace("\n", " ")
        print(f"    [{i}] is_insertion={row['is_insertion']} host={row['host_id']}")
        print(f"        target: {row['target_response'][:100]}")
        print(f"        cot: {cot_short}...")

    # Upload
    if args.upload and rows:
        _upload_to_hf(jsonl_path, rows)


def _upload_to_hf(jsonl_path: Path, rows: list[dict]):
    """Upload dataset to HuggingFace as parquet."""
    import pandas as pd
    from huggingface_hub import HfApi

    repo_id = "mats-10-sprint-cs-jb/cot-oracle-eval-sentence-insertion"
    print(f"\nUploading to {repo_id}...")

    df = pd.DataFrame(rows)
    parquet_path = jsonl_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded {len(rows)} rows to {repo_id}")


if __name__ == "__main__":
    main()
