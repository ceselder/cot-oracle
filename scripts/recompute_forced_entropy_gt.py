#!/usr/bin/env python3
"""Recompute forced-answer entropy ground truth with exact logprobs.

Fixes the vLLM logprobs=20 bug: uses HF forward pass to get exact logprobs
for A/B/C/D answer tokens at each sentence boundary.

Two-phase approach:
  Phase 1 (vLLM): Fast batched CoT generation + sentence detection
  Phase 2 (HF):   Forward pass at each boundary for exact logprobs

Usage:
    python scripts/recompute_forced_entropy_gt.py --eval-path data/evals/forced_answer_entropy_riya.json
    # Resume after interruption:
    python scripts/recompute_forced_entropy_gt.py --eval-path data/evals/forced_answer_entropy_riya.json --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
from pathlib import Path


def split_cot(text: str) -> list[str]:
    text = re.sub(r"<think>|</think>", "", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/evals/forced_answer_entropy_riya.json")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-cot-tokens", type=int, default=2048)
    parser.add_argument("--max-boundaries", type=int, default=15)
    parser.add_argument("--target-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase2-batch", type=int, default=1,
                        help="Batch size for HF forward pass (1=safest)")
    args = parser.parse_args()

    random.seed(args.seed)

    eval_path = Path(args.eval_path)
    output_path = Path(args.output or str(eval_path).replace(".json", "_v2.json"))
    ckpt_phase1 = Path(str(output_path) + ".phase1.json")
    ckpt_phase2 = Path(str(output_path) + ".phase2.json")

    # --- Load eval items, deduplicate to unique questions ---
    with open(eval_path) as f:
        items = json.load(f)

    unique_qs: dict[str, dict] = {}
    for item in items:
        pid = item["metadata"].get("parent_example_id", item["example_id"])
        if pid not in unique_qs:
            unique_qs[pid] = item
    questions = list(unique_qs.values())
    print(f"Loaded {len(items)} items -> {len(questions)} unique questions")

    # =================================================================
    # Phase 1: vLLM CoT generation
    # =================================================================
    phase1_data: dict[str, list] = {}
    if args.resume and ckpt_phase1.exists():
        with open(ckpt_phase1) as f:
            phase1_data = json.load(f)
        print(f"Phase 1 checkpoint: {len(phase1_data)} questions done")

    pending_q = []
    for q in questions:
        pid = q["metadata"].get("parent_example_id", q["example_id"])
        if pid not in phase1_data:
            pending_q.append((pid, q))

    if pending_q:
        print(f"\nPhase 1: Generating CoTs for {len(pending_q)} questions...")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        tok = llm.get_tokenizer()

        gen_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_cot_tokens,
            n=args.n_rollouts,
        )

        # Build prompts
        prompts = []
        pids = []
        for pid, q in pending_q:
            messages = [{"role": "user", "content": q["clean_prompt"]}]
            formatted = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted)
            pids.append(pid)

        print(f"  Generating {len(prompts)} × {args.n_rollouts} rollouts...")
        outputs = llm.generate(prompts, gen_params)

        for pid, output in zip(pids, outputs):
            rollouts = []
            for comp in output.outputs:
                text = comp.text
                # Extract thinking text or use raw
                m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                cot = m.group(1).strip() if m else text.strip()
                sentences = split_cot(cot)
                if len(sentences) >= 2:
                    rollouts.append(sentences)
            phase1_data[pid] = rollouts

        with open(ckpt_phase1, "w") as f:
            json.dump(phase1_data, f)
        print(f"Phase 1 done: {len(phase1_data)} questions, saved to {ckpt_phase1}")

        # Free vLLM
        del llm, tok, outputs
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    total_rollouts = sum(len(r) for r in phase1_data.values())
    total_boundaries = sum(
        min(len(s), args.max_boundaries)
        for rollouts in phase1_data.values()
        for s in rollouts
    )
    print(f"\nPhase 1 totals: {len(phase1_data)} questions, {total_rollouts} rollouts, "
          f"~{total_boundaries} boundaries to process")

    # =================================================================
    # Phase 2: HF forward pass for exact logprobs
    # =================================================================
    print("\nPhase 2: Loading HF model for logprob extraction...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Answer token IDs (space-prefixed, matching generation behavior)
    answer_tids: dict[str, int] = {}
    for letter in "ABCD":
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        answer_tids[letter] = ids[-1]
        print(f"  {letter} -> token_id={ids[-1]} ({repr(tokenizer.decode([ids[-1]]))})")

    # Load phase 2 checkpoint
    phase2_data: dict[str, dict] = {}
    if args.resume and ckpt_phase2.exists():
        with open(ckpt_phase2) as f:
            phase2_data = json.load(f)
        print(f"Phase 2 checkpoint: {len(phase2_data)} questions done")

    # Build question map
    q_map: dict[str, dict] = {}
    for q in questions:
        pid = q["metadata"].get("parent_example_id", q["example_id"])
        q_map[pid] = q

    pending_pids = [pid for pid in phase1_data if pid not in phase2_data]
    print(f"Phase 2: {len(pending_pids)} questions to process")

    for qi, pid in enumerate(pending_pids):
        q = q_map.get(pid, {})
        rollouts = phase1_data[pid]

        messages = [{"role": "user", "content": q.get("clean_prompt", "")}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        all_boundaries = []

        for ri, sentences in enumerate(rollouts):
            boundary_indices = list(range(1, len(sentences) + 1))
            if len(boundary_indices) > args.max_boundaries:
                step = len(boundary_indices) / args.max_boundaries
                boundary_indices = [
                    boundary_indices[int(i * step)]
                    for i in range(args.max_boundaries)
                ]

            for bi in boundary_indices:
                partial = " ".join(sentences[:bi])
                prefix = f"{formatted}{partial}\n\nSo, the answer is:"

                input_ids = tokenizer(
                    prefix, return_tensors="pt"
                ).input_ids.to(model.device)

                with torch.no_grad():
                    logits = model(input_ids).logits[0, -1, :]

                log_probs = torch.log_softmax(logits.float(), dim=-1)

                lps = {l: log_probs[tid].item() for l, tid in answer_tids.items()}

                # Softmax over just ABCD
                vals = [lps[l] for l in "ABCD"]
                mx = max(vals)
                exps = [math.exp(v - mx) for v in vals]
                tot = sum(exps)
                probs = [e / tot for e in exps]
                entropy = compute_entropy(probs)

                all_boundaries.append({
                    "rollout_idx": ri,
                    "boundary_idx": bi,
                    "n_sentences_so_far": bi,
                    "n_sentences_total": len(sentences),
                    "fraction_complete": bi / len(sentences),
                    "entropy": entropy,
                    "probs": {l: p for l, p in zip("ABCD", probs)},
                    "raw_logprobs": lps,
                })

        phase2_data[pid] = {
            "n_rollouts": len(rollouts),
            "n_boundaries": len(all_boundaries),
            "boundary_data": all_boundaries,
        }

        # Save checkpoint after each question
        with open(ckpt_phase2, "w") as f:
            json.dump(phase2_data, f)

        if all_boundaries:
            ents = [b["entropy"] for b in all_boundaries]
            print(
                f"  [{qi+1}/{len(pending_pids)}] {pid}: "
                f"{len(all_boundaries)} boundaries, "
                f"entropy [{min(ents):.3f}, {max(ents):.3f}], "
                f"mean {sum(ents)/len(ents):.3f}"
            )
        else:
            print(f"  [{qi+1}/{len(pending_pids)}] {pid}: no valid boundaries")

    # =================================================================
    # Merge: downsample to target_items, write final JSON
    # =================================================================
    print("\nMerging results...")
    all_dp = []
    for pid, fd in phase2_data.items():
        q = q_map.get(pid, {})
        for bd in fd.get("boundary_data", []):
            all_dp.append({"q": q, "bd": bd, "pid": pid})

    print(f"Total datapoints: {len(all_dp)}")

    rng = random.Random(args.seed)
    if len(all_dp) > args.target_items:
        rng.shuffle(all_dp)
        all_dp = all_dp[:args.target_items]
        print(f"Downsampled to {args.target_items}")

    final_items = []
    for i, dp in enumerate(all_dp):
        q = dp["q"]
        bd = dp["bd"]
        pid = dp["pid"]
        final_items.append({
            "eval_name": "forced_answer_entropy_riya",
            "example_id": f"forced_entropy_{i:04d}",
            "clean_prompt": q.get("clean_prompt", ""),
            "test_prompt": q.get("test_prompt", q.get("clean_prompt", "")),
            "correct_answer": f"{bd['entropy']:.4f}",
            "nudge_answer": None,
            "metadata": {
                "parent_example_id": pid,
                "choices": q.get("metadata", {}).get("choices", {}),
                "correct_letter": q.get("metadata", {}).get("correct_letter", ""),
                "source": q.get("metadata", {}).get("source", "arc_challenge"),
                "answer_tokens": ["A", "B", "C", "D"],
                "metric": "r_squared",
                "task_type": "regression",
                "entropy": bd["entropy"],
                "answer_probs": bd["probs"],
                "raw_logprobs": bd["raw_logprobs"],
                "rollout_idx": bd["rollout_idx"],
                "boundary_idx": bd["boundary_idx"],
                "n_sentences_so_far": bd["n_sentences_so_far"],
                "n_sentences_total": bd["n_sentences_total"],
                "fraction_complete": bd["fraction_complete"],
            },
        })

    with open(output_path, "w") as f:
        json.dump(final_items, f, indent=2)

    # --- Diagnostics ---
    ents = [dp["bd"]["entropy"] for dp in all_dp]
    n_fallback = sum(
        1 for dp in all_dp
        if any(v <= -19.0 for v in dp["bd"]["raw_logprobs"].values())
    )
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(final_items)} items saved to {output_path}")
    print(f"Entropy: min={min(ents):.4f}, max={max(ents):.4f}, mean={sum(ents)/len(ents):.4f}")
    print(f"Items with any logprob <= -19: {n_fallback}/{len(all_dp)} "
          f"(should be 0 with HF forward pass)")

    # Entropy histogram
    import collections
    buckets = collections.Counter()
    for e in ents:
        if e < 0.01:
            buckets["<0.01"] += 1
        elif e < 0.1:
            buckets["0.01-0.1"] += 1
        elif e < 0.5:
            buckets["0.1-0.5"] += 1
        elif e < 1.0:
            buckets["0.5-1.0"] += 1
        elif e < 1.35:
            buckets["1.0-1.35"] += 1
        else:
            buckets[">=1.35"] += 1
    print("Entropy distribution:")
    for k in ["<0.01", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1.0-1.35", ">=1.35"]:
        bar = "#" * buckets.get(k, 0)
        print(f"  {k:>10}: {buckets.get(k, 0):3d} {bar}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
