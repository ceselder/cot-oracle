#!/usr/bin/env python3
"""
All-in-one: generate on-policy Qwen3-8B CoTs + resampling importance.
Data-parallel across all available GPUs. Designed for 8x H100 (gpu-xd670-30).

Pipeline:
  Phase 1: Generate on-policy CoTs from diverse datasets (vLLM, data-parallel)
  Phase 2: Importance++ resampling (thought-anchors methodology, vLLM, data-parallel)
  Phase 3: Merge results + compute per-sentence importance++ scores

Graceful early termination: Ctrl+C or kill sends SIGTERM to workers.
Workers finish their current vLLM batch, save progress, and exit.
Re-run the same command with --output-dir to resume from where it stopped.

--frugal flag: Uses sparse sampling + fill-in strategy from frugal-thought-anchors
  to reduce resampling cost by ~50-80%. Instead of testing every truncation point,
  samples every Nth sentence, fills in gaps where accuracy jumps, and interpolates
  the rest. Two vLLM rounds per batch instead of one, but far fewer total prompts.

Importance++ methodology (from thought-anchors/thought-branches papers):
  - Full rollout continuations: truncate CoT, let model continue reasoning (not just answer)
  - Answer distributions: collect all answers across rollouts per truncation point
  - KL divergence: binary (P(correct)) and categorical (answer distribution)
  - importance_accuracy = accuracy(with sentence) - accuracy(without sentence)
  - importance_kl = KL divergence between adjacent truncation point distributions

Usage:
    ssh gpu-xd670-30
    cd ~/cot-oracle
    set -a; source ~/.env; set +a
    source "${CACHE_DIR}/venvs/cot-oracle/bin/activate"

    python scripts/run_pipeline_h100as.py --preset mini      # smoke test (~650 problems)
    python scripts/run_pipeline_h100as.py --preset medium     # ~50K problems
    python scripts/run_pipeline_h100as.py --preset full       # ~125K problems

    # Frugal mode: ~50-80% fewer resampling prompts
    python scripts/run_pipeline_h100as.py --preset medium --frugal

    # Resume a stopped run (point to the same output dir)
    python scripts/run_pipeline_h100as.py --preset medium --output-dir data/pipeline_medium_20260223_...

    # Resume from existing corpus (skip Phase 1)
    python scripts/run_pipeline_h100as.py --skip-generate --corpus data/pipeline_.../corpus.jsonl

    # Just generate CoTs, skip resampling
    python scripts/run_pipeline_h100as.py --preset mini --skip-resample
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data_pipeline"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "Qwen/Qwen3-8B"
NO_GROUND_TRUTH = {"LMSYS"}
RESAMPLE_BATCH = 25  # entries per vLLM batch in resample phase

# ── Graceful stop flag for workers ──
_STOP = False

def _worker_stop_handler(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n  Signal received — finishing current batch and saving...", flush=True)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="All-in-one CoT + resampling pipeline")
    p.add_argument("--preset", choices=["mini", "medium", "full", "resample"], default="mini")
    p.add_argument("--num-gpus", type=int, default=None, help="Auto-detect if not set")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-rollouts", type=int, default=1, help="CoT rollouts per problem")
    p.add_argument("--n-resamples", type=int, default=20, help="Rollouts per truncation point (paper uses 50)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-resample", action="store_true")
    p.add_argument("--corpus", default=None, help="Existing corpus (with --skip-generate)")
    p.add_argument("--include-incorrect", action="store_true", help="Resample incorrect CoTs too")
    p.add_argument("--resample-batch", type=int, default=RESAMPLE_BATCH, help="Entries per resample batch")
    # Frugal mode (from frugal-thought-anchors)
    p.add_argument("--frugal", action="store_true", help="Sparse sampling + fill-in (50-80%% fewer prompts)")
    p.add_argument("--sparse-step", type=int, default=3, help="Frugal: sample every Nth sentence (default 3)")
    p.add_argument("--jump-threshold", type=float, default=0.10, help="Frugal: fill-in trigger threshold (default 0.10)")
    p.add_argument("--resample-limit", type=int, default=None, help="Subsample corpus to N entries for resampling (random)")
    p.add_argument("--exclude-resampled", type=str, default="auto", help="Path to prior resampling.jsonl to skip already-done IDs. 'auto' = same output-dir, 'none' = no exclusion")
    # Internal worker flags
    p.add_argument("--_phase", choices=["generate", "resample"])
    p.add_argument("--_shard", type=int)
    p.add_argument("--_nshards", type=int)
    p.add_argument("--_input", type=str)
    p.add_argument("--_output", type=str)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Multi-GPU launcher
# ═══════════════════════════════════════════════════════════════════

def launch_workers(phase, num_gpus, args, input_file, shard_dir):
    """Launch one worker per GPU. On interrupt, SIGTERM workers so they save."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    procs = []

    def cleanup(signum=None, frame=None):
        print(f"\n  Sending SIGTERM to {len(procs)} workers (saving progress)...")
        for proc, _, _ in procs:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 60
        for proc, fh, gpu in procs:
            remaining = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(f"  GPU {gpu}: saved and exited (rc={proc.returncode})")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  GPU {gpu}: killed (didn't exit in 60s)")
            fh.close()
        if signum:
            sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    for gpu in range(num_gpus):
        out_file = f"{shard_dir}/{phase}_{gpu}.jsonl"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--_phase", phase,
            "--_shard", str(gpu),
            "--_nshards", str(num_gpus),
            "--_input", input_file,
            "--_output", out_file,
            "--model", args.model,
            "--n-rollouts", str(args.n_rollouts),
            "--n-resamples", str(args.n_resamples),
            "--resample-batch", str(args.resample_batch),
            "--sparse-step", str(args.sparse_step),
            "--jump-threshold", str(args.jump_threshold),
        ]
        if args.include_incorrect:
            cmd.append("--include-incorrect")
        if args.frugal:
            cmd.append("--frugal")

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TORCHINDUCTOR_CACHE_DIR": f"/tmp/torchinductor_{os.environ.get('USER', 'user')}_gpu{gpu}",
        }
        log_path = log_dir / f"{phase}_{gpu}.log"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((proc, log_fh, gpu))
        print(f"  GPU {gpu}: PID {proc.pid}  log={log_path}")

    # Poll output files for progress while waiting
    from tqdm.auto import tqdm
    t0 = time.time()
    pbar = tqdm(desc=f"{phase}", unit="entry")
    while True:
        # Check if any workers still running
        still_running = [gpu for proc, _, gpu in procs if proc.poll() is None]
        # Count output lines across all shards
        n = 0
        for gpu in range(num_gpus):
            shard = Path(f"{shard_dir}/{phase}_{gpu}.jsonl")
            if shard.exists():
                with open(shard) as f:
                    n += sum(1 for line in f if line.strip())
        pbar.n = n
        pbar.refresh()
        if not still_running:
            break
        time.sleep(10)
    pbar.close()

    failed = []
    for proc, log_fh, gpu in procs:
        log_fh.close()
        if proc.returncode != 0:
            failed.append(gpu)
            print(f"  GPU {gpu} FAILED (exit {proc.returncode}). Tail of log:")
            _print_log_tail(log_dir / f"{phase}_{gpu}.log")
        else:
            print(f"  GPU {gpu} done")

    elapsed = time.time() - t0
    print(f"  Phase '{phase}' took {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if failed:
        print(f"  WARNING: GPUs {failed} failed — merging partial results from others")
    return failed


def _print_log_tail(path, n=15):
    try:
        with open(path) as f:
            lines = f.readlines()
        for line in lines[-n:]:
            print(f"    {line.rstrip()}")
    except Exception:
        pass


def merge_shards(shard_dir, phase, num_gpus, merged_path):
    """Merge per-shard JSONL files into one."""
    n = 0
    with open(merged_path, "w") as out:
        for gpu in range(num_gpus):
            shard = Path(f"{shard_dir}/{phase}_{gpu}.jsonl")
            if not shard.exists():
                continue
            with open(shard) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        n += 1
    print(f"  Merged {num_gpus} shards -> {merged_path} ({n} entries)")
    return n


# ═══════════════════════════════════════════════════════════════════
# Phase 1 worker: generate CoTs
# ═══════════════════════════════════════════════════════════════════

def worker_generate(args):
    signal.signal(signal.SIGINT, _worker_stop_handler)
    signal.signal(signal.SIGTERM, _worker_stop_handler)

    import re
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from cot_utils import split_cot_into_sentences
    from generate_cots import extract_answer, answers_match, SOURCE_TO_DOMAIN

    with open(args._input) as f:
        all_problems = json.load(f)

    sz = math.ceil(len(all_problems) / args._nshards)
    start = args._shard * sz
    end = min(start + sz, len(all_problems))
    problems = all_problems[start:end]
    if not problems:
        print(f"Shard {args._shard}: empty, nothing to do")
        Path(args._output).touch()
        return
    print(f"Shard {args._shard}: {len(problems)} problems (indices {start}-{end-1})")

    # Resume: check which rollout indices are already fully done
    done_rollouts = set()
    out_path = Path(args._output)
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_rollouts.add(json.loads(line).get("rollout_idx", 0))
        if done_rollouts:
            print(f"  Resuming: rollouts {sorted(done_rollouts)} already saved")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_prefix_caching=True,
    )
    cot_sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
    direct_sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=200)

    # Format prompts
    cot_prompts, direct_prompts, direct_idx = [], [], []
    for i, p in enumerate(problems):
        msgs = [{"role": "user", "content": p["question"]}]
        cot_prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        if p["source"] not in NO_GROUND_TRUTH:
            direct_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            direct_idx.append(i)

    # Generate direct responses (small, fast — only once)
    direct_map = {}
    if direct_prompts and 0 not in done_rollouts:
        t0 = time.time()
        print(f"  Direct: {len(direct_prompts)} prompts...")
        direct_out = llm.generate(direct_prompts, direct_sp)
        for j, idx in enumerate(direct_idx):
            direct_map[idx] = direct_out[j].outputs[0].text.strip()
        print(f"    Done in {time.time()-t0:.0f}s")

    # Generate CoT rollouts — write results after each rollout (resumable)
    for r_idx in range(args.n_rollouts):
        if _STOP:
            print(f"  Stopping early after {r_idx}/{args.n_rollouts} rollouts")
            break
        if r_idx in done_rollouts:
            print(f"  Rollout {r_idx+1}/{args.n_rollouts}: already done, skipping")
            continue

        t0 = time.time()
        print(f"  Rollout {r_idx+1}/{args.n_rollouts}: {len(cot_prompts)} prompts...")
        outputs = llm.generate(cot_prompts, cot_sp)
        print(f"    Generated in {time.time()-t0:.0f}s, processing...")

        with open(args._output, "a") as f:
            for i, (p, out) in enumerate(zip(problems, outputs)):
                text = out.outputs[0].text
                if "</think>" in text:
                    parts = text.split("</think>", 1)
                    reasoning = re.sub(r'^<think>\s*', '', parts[0]).strip()
                    content = parts[1].strip()
                else:
                    reasoning, content = text.strip(), ""

                if not reasoning:
                    continue
                sentences = split_cot_into_sentences(reasoning)
                if len(sentences) < 2:
                    continue

                has_gt = p["correct_answer"] is not None
                cot_answer = extract_answer(content) if content else extract_answer(reasoning)
                direct_text = direct_map.get(i, "")
                direct_answer = extract_answer(direct_text) if direct_text else None
                cot_correct = answers_match(cot_answer, p["correct_answer"]) if has_gt and cot_answer else None
                direct_correct = answers_match(direct_answer, p["correct_answer"]) if has_gt and direct_answer else None

                if has_gt and cot_correct is not None and direct_correct is not None:
                    if cot_correct and not direct_correct: cat = "load_bearing"
                    elif cot_correct and direct_correct: cat = "both_correct"
                    elif not cot_correct and not direct_correct: cat = "both_wrong"
                    else: cat = "cot_hurt"
                else:
                    cat = None

                domain = SOURCE_TO_DOMAIN.get(p["source"], "unknown")
                entry = {
                    "id": f"{p['source'].lower()}_{start+i:04d}_r{r_idx}",
                    "source": p["source"], "domain": domain,
                    "question": p["question"],
                    "correct_answer": p["correct_answer"],
                    "subject": p.get("subject", ""), "level": p.get("level", ""),
                    "cot_response": reasoning, "cot_content": content,
                    "direct_response": direct_text if r_idx == 0 else None,
                    "cot_answer": cot_answer,
                    "direct_answer": direct_answer if r_idx == 0 else None,
                    "cot_correct": cot_correct,
                    "direct_correct": direct_correct if r_idx == 0 else None,
                    "category": cat,
                    "sentences": sentences, "n_sentences": len(sentences),
                    "rollout_idx": r_idx,
                }
                f.write(json.dumps(entry) + "\n")

        print(f"    Rollout {r_idx+1} saved ({time.time()-t0:.0f}s total)")

    print(f"  Shard {args._shard} generate done")


# ═══════════════════════════════════════════════════════════════════
# Phase 2 worker: resampling importance (batched, resumable)
# ═══════════════════════════════════════════════════════════════════

def _build_trunc_prefix(sentences, t):
    """Build a truncated CoT prefix for truncation point t (open-ended, no </think>).

    The model continues reasoning from this prefix, generating a full rollout.
    This is the proper thought-anchors methodology: remove chunk t onwards,
    let the model regenerate from that point.
    """
    return "<think>\n" + "\n".join(sentences[:t]) + "\n"


def _kl_divergence_binary(sols1, sols2, alpha=1e-9):
    """KL(P||Q) over binary P(correct) distributions (thought anchors methodology)."""
    correct1 = sum(1 for s in sols1 if s.get("is_correct") is True)
    total1 = sum(1 for s in sols1 if s.get("is_correct") is not None)
    correct2 = sum(1 for s in sols2 if s.get("is_correct") is True)
    total2 = sum(1 for s in sols2 if s.get("is_correct") is not None)
    if total1 == 0 or total2 == 0:
        return 0.0
    p = (correct1 + alpha) / (total1 + 2 * alpha)
    q = (correct2 + alpha) / (total2 + 2 * alpha)
    return max(0.0, p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


def _kl_divergence_categorical(sols1, sols2, alpha=1e-9):
    """KL(P||Q) over full answer distributions (thought anchors methodology)."""
    from collections import defaultdict
    counts1, counts2 = defaultdict(int), defaultdict(int)
    for s in sols1:
        a = (s.get("answer") or "").strip().lower().replace(",", "").replace(" ", "")
        if a:
            counts1[a] += 1
    for s in sols2:
        a = (s.get("answer") or "").strip().lower().replace(",", "").replace(" ", "")
        if a:
            counts2[a] += 1
    if not counts1 or not counts2:
        return 0.0
    all_answers = set(counts1) | set(counts2)
    V = len(all_answers)
    total1, total2 = sum(counts1.values()), sum(counts2.values())
    st1, st2 = total1 + alpha * V, total2 + alpha * V
    kl = 0.0
    for a in all_answers:
        p_smooth = (counts1[a] + alpha) / st1
        q_smooth = (counts2[a] + alpha) / st2
        p_raw = counts1[a] / total1
        kl += p_raw * math.log(p_smooth / q_smooth)
    return max(0.0, kl)


def _parse_rollout(text, extract_answer_fn):
    """Parse a full rollout continuation into (resampled_chunk, answer, full_text)."""
    from cot_utils import split_cot_into_sentences
    # The model continues from the truncation prefix, generating more reasoning + answer
    if "</think>" in text:
        parts = text.split("</think>", 1)
        reasoning = parts[0].strip()
        content = parts[1].strip()
    else:
        reasoning, content = text.strip(), ""
    chunk_resampled = ""
    if reasoning:
        sents = split_cot_into_sentences(reasoning)
        chunk_resampled = sents[0] if sents else reasoning[:200]
    answer = extract_answer_fn(content) if content else extract_answer_fn(reasoning)
    return chunk_resampled, answer, text


def _resample_batch_brute(batch, llm, tokenizer, args):
    """Brute force resampling with proper thought-anchors methodology.

    For each truncation point t, generates full rollout continuations (not just answers).
    Collects answer distributions and computes KL divergence between adjacent points.
    """
    from generate_cots import extract_answer, answers_match

    SamplingParams = __import__("vllm", fromlist=["SamplingParams"]).SamplingParams
    rollout_sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)

    all_prompts, all_meta = [], []

    for pi, entry in enumerate(batch):
        sentences = entry["sentences"]
        n = len(sentences)
        msgs = [{"role": "user", "content": entry["question"]}]
        base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

        # t=0: baseline (no CoT prefix, model generates from scratch)
        for r in range(args.n_resamples):
            all_prompts.append(base)
            all_meta.append({"pi": pi, "t": 0, "r": r})

        # t=1..n: truncate after sentence t, model continues from there
        for t in range(1, n + 1):
            prefix = _build_trunc_prefix(sentences, t)
            for r in range(args.n_resamples):
                all_prompts.append(base + prefix)
                all_meta.append({"pi": pi, "t": t, "r": r})

    print(f"    {len(all_prompts)} rollout prompts (brute force)")
    outputs = llm.generate(all_prompts, rollout_sp)

    # Collect solutions per (entry, truncation_point)
    solutions = {}  # (pi, t) -> list of {answer, is_correct, chunk_resampled}
    for output, m in zip(outputs, all_meta):
        chunk_resampled, answer, _ = _parse_rollout(output.outputs[0].text, extract_answer)
        pi, t = m["pi"], m["t"]
        entry = batch[pi]
        gt = entry.get("correct_answer")
        is_correct = answers_match(answer, gt) if (answer and gt) else None
        solutions.setdefault((pi, t), []).append({"answer": answer, "is_correct": is_correct, "chunk_resampled": chunk_resampled})

    return _compute_importance_pp(batch, solutions, args)


def _resample_batch_frugal(batch, llm, tokenizer, args):
    """Frugal resampling with proper thought-anchors methodology.

    Round 1: Baseline + every sparse_step-th truncation point (full rollouts)
    Analyze: Find gaps where accuracy jumps > jump_threshold
    Round 2: Fill-in rollouts for those gaps
    Final: Linearly interpolate remaining truncation points
    """
    from generate_cots import extract_answer, answers_match

    SamplingParams = __import__("vllm", fromlist=["SamplingParams"]).SamplingParams
    rollout_sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)

    sparse_step = args.sparse_step
    jump_threshold = args.jump_threshold

    # ── Round 1: Baseline + sparse truncation points ──
    r1_prompts, r1_meta = [], []
    entry_sparse_points = []

    for pi, entry in enumerate(batch):
        sentences = entry["sentences"]
        n = len(sentences)
        msgs = [{"role": "user", "content": entry["question"]}]
        base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

        # Baseline (t=0)
        for r in range(args.n_resamples):
            r1_prompts.append(base)
            r1_meta.append({"pi": pi, "t": 0, "r": r})

        # Sparse points: every sparse_step-th + always include last
        sparse_pts = list(range(sparse_step, n + 1, sparse_step))
        if n not in sparse_pts:
            sparse_pts.append(n)
        sparse_pts.sort()
        entry_sparse_points.append(sparse_pts)

        for t in sparse_pts:
            prefix = _build_trunc_prefix(sentences, t)
            for r in range(args.n_resamples):
                r1_prompts.append(base + prefix)
                r1_meta.append({"pi": pi, "t": t, "r": r})

    print(f"    Round 1 (sparse): {len(r1_prompts)} rollout prompts")
    r1_out = llm.generate(r1_prompts, rollout_sp)

    solutions = {}  # (pi, t) -> list of solution dicts
    for output, m in zip(r1_out, r1_meta):
        chunk_resampled, answer, _ = _parse_rollout(output.outputs[0].text, extract_answer)
        pi, t = m["pi"], m["t"]
        entry = batch[pi]
        gt = entry.get("correct_answer")
        is_correct = answers_match(answer, gt) if (answer and gt) else None
        solutions.setdefault((pi, t), []).append({"answer": answer, "is_correct": is_correct, "chunk_resampled": chunk_resampled})

    # ── Analyze: compute accuracy at sparse points, find jumps ──
    fill_prompts, fill_meta = [], []

    for pi, entry in enumerate(batch):
        sentences = entry["sentences"]
        n = len(sentences)
        sparse_pts = [0] + entry_sparse_points[pi]
        msgs = [{"role": "user", "content": entry["question"]}]
        base = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

        # Compute accuracy at each sparse point
        accuracies = {}
        for t in sparse_pts:
            sols = solutions.get((pi, t), [])
            n_with_label = sum(1 for s in sols if s["is_correct"] is not None)
            n_correct = sum(1 for s in sols if s["is_correct"] is True)
            accuracies[t] = n_correct / max(n_with_label, 1)

        # Find gaps with large accuracy jumps → fill in
        for i in range(len(sparse_pts) - 1):
            t_a, t_b = sparse_pts[i], sparse_pts[i + 1]
            if abs(accuracies.get(t_b, 0) - accuracies.get(t_a, 0)) >= jump_threshold:
                for t in range(t_a + 1, t_b):
                    if t == 0:
                        continue
                    prefix = _build_trunc_prefix(sentences, t)
                    for r in range(args.n_resamples):
                        fill_prompts.append(base + prefix)
                        fill_meta.append({"pi": pi, "t": t, "r": r})

    # ── Round 2: Fill-in ──
    if fill_prompts:
        print(f"    Round 2 (fill-in): {len(fill_prompts)} rollout prompts")
        fill_out = llm.generate(fill_prompts, rollout_sp)
        for output, m in zip(fill_out, fill_meta):
            chunk_resampled, answer, _ = _parse_rollout(output.outputs[0].text, extract_answer)
            pi, t = m["pi"], m["t"]
            entry = batch[pi]
            gt = entry.get("correct_answer")
            is_correct = answers_match(answer, gt) if (answer and gt) else None
            solutions.setdefault((pi, t), []).append({"answer": answer, "is_correct": is_correct, "chunk_resampled": chunk_resampled})
    else:
        print(f"    Round 2 (fill-in): 0 prompts (no jumps detected)")

    print(f"    Total: {len(r1_prompts) + len(fill_prompts)} rollout prompts")

    return _compute_importance_pp(batch, solutions, args, frugal=True)


def _compute_importance_pp(batch, solutions, args, frugal=False):
    """Compute importance++ metrics from rollout solutions.

    For each sentence i, computes:
    - resampling_importance_accuracy: accuracy(t=i+1) - accuracy(t=i)
    - resampling_importance_kl: KL divergence between answer distributions at t=i and t=i+1
    - resampling_importance_kl_binary: KL divergence over P(correct) at t=i vs t=i+1
    - For frugal mode: interpolates metrics for unsampled truncation points
    """
    results = []
    for pi, entry in enumerate(batch):
        sentences = entry["sentences"]
        n = len(sentences)

        # Compute per-truncation-point stats from solutions
        sampled_stats = {}  # t -> {accuracy, sols, n_total}
        for t in range(n + 1):
            sols = solutions.get((pi, t), [])
            if sols:
                n_with_label = sum(1 for s in sols if s["is_correct"] is not None)
                n_correct = sum(1 for s in sols if s["is_correct"] is True)
                sampled_stats[t] = {
                    "accuracy": n_correct / max(n_with_label, 1),
                    "sols": sols,
                    "n_total": len(sols),
                }

        # For frugal: interpolate accuracy for missing truncation points
        all_accuracy = {}
        for t in range(n + 1):
            if t in sampled_stats:
                all_accuracy[t] = sampled_stats[t]["accuracy"]
            elif frugal:
                below = max((k for k in sampled_stats if k < t), default=None)
                above = min((k for k in sampled_stats if k > t), default=None)
                if below is not None and above is not None:
                    frac = (t - below) / (above - below)
                    all_accuracy[t] = sampled_stats[below]["accuracy"] * (1 - frac) + sampled_stats[above]["accuracy"] * frac
                elif below is not None:
                    all_accuracy[t] = sampled_stats[below]["accuracy"]
                elif above is not None:
                    all_accuracy[t] = sampled_stats[above]["accuracy"]
                else:
                    all_accuracy[t] = 0.0
            else:
                all_accuracy[t] = 0.0

        # Build truncation info
        truncations = []
        for t in range(n + 1):
            sols = solutions.get((pi, t), [])
            truncations.append({
                "truncate_at": t,
                "accuracy": all_accuracy.get(t, 0.0),
                "n_rollouts": len(sols),
                "interpolated": t not in sampled_stats,
            })

        # Per-sentence importance
        importance = []
        for i in range(n):
            sols_before = solutions.get((pi, i), [])
            sols_after = solutions.get((pi, i + 1), [])
            has_both = bool(sols_before) and bool(sols_after)

            acc_before = all_accuracy.get(i, 0.0)
            acc_after = all_accuracy.get(i + 1, 0.0)
            imp_acc = acc_after - acc_before

            # KL divergence (only if both endpoints were actually sampled)
            kl_binary = _kl_divergence_binary(sols_before, sols_after) if has_both else 0.0
            kl_cat = _kl_divergence_categorical(sols_before, sols_after) if has_both else 0.0

            is_sampled = (i in sampled_stats) and (i + 1 in sampled_stats)
            is_partial = (i in sampled_stats) or (i + 1 in sampled_stats)
            source = "sampled" if is_sampled else ("partial" if is_partial else "interpolated")

            importance.append({
                "sentence_idx": i,
                "sentence_text": sentences[i][:200],
                "resampling_importance_accuracy": imp_acc,
                "resampling_importance_kl_binary": kl_binary,
                "resampling_importance_kl_categorical": kl_cat,
                "accuracy_without": acc_before,
                "accuracy_with": acc_after,
                "n_rollouts_before": len(sols_before),
                "n_rollouts_after": len(sols_after),
                "important": imp_acc > 0.3 or kl_binary > 0.1,
                "source": source,
            })

        results.append({
            "id": entry["id"],
            "source": entry.get("source", ""),
            "question": entry["question"][:300],
            "correct_answer": entry.get("correct_answer", ""),
            "n_sentences": n,
            "truncations": truncations,
            "sentence_importance": importance,
        })
    return results


def worker_resample(args):
    signal.signal(signal.SIGINT, _worker_stop_handler)
    signal.signal(signal.SIGTERM, _worker_stop_handler)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load corpus and take shard
    corpus = []
    with open(args._input) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if not entry.get("sentences") or len(entry["sentences"]) < 3:
                continue
            if entry.get("rollout_idx", 0) != 0:
                continue
            if not args.include_incorrect and entry.get("cot_correct") is False:
                continue
            corpus.append(entry)

    # Shuffle so heavy entries (long CoTs) are spread across shards
    import random as _rng
    _rng.seed(42)
    _rng.shuffle(corpus)

    sz = math.ceil(len(corpus) / args._nshards)
    start = args._shard * sz
    selected = corpus[start : min(start + sz, len(corpus))]
    if not selected:
        print(f"Shard {args._shard}: empty, nothing to do")
        Path(args._output).touch()
        return

    # Resume: load already-processed IDs
    done_ids = set()
    out_path = Path(args._output)
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])

    todo = [e for e in selected if e["id"] not in done_ids]
    mode_str = "frugal" if args.frugal else "brute-force"
    print(f"Shard {args._shard}: {len(selected)} total, {len(done_ids)} done, {len(todo)} remaining ({mode_str})")
    if args.frugal:
        print(f"  Frugal config: sparse_step={args.sparse_step}, jump_threshold={args.jump_threshold}")

    if not todo:
        print(f"  All done!")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_prefix_caching=True,
    )

    from tqdm.auto import tqdm

    batch_size = args.resample_batch
    n_batches = math.ceil(len(todo) / batch_size)
    process_fn = _resample_batch_frugal if args.frugal else _resample_batch_brute

    pbar = tqdm(total=len(selected), initial=len(done_ids), desc=f"shard {args._shard}", unit="entry")
    for bi in range(n_batches):
        if _STOP:
            tqdm.write(f"  Stopping early at batch {bi}/{n_batches} ({pbar.n} entries saved)")
            break

        batch = todo[bi * batch_size : (bi + 1) * batch_size]
        t0 = time.time()
        tqdm.write(f"  Batch {bi+1}/{n_batches}: {len(batch)} entries...")

        batch_results = process_fn(batch, llm, tokenizer, args)

        with open(args._output, "a") as f:
            for result in batch_results:
                f.write(json.dumps(result) + "\n")

        pbar.update(len(batch))
        elapsed = time.time() - t0
        tqdm.write(f"    Batch {bi+1} done in {elapsed:.0f}s")
    pbar.close()

    print(f"  Shard {args._shard} resample done")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: merge importance labels into corpus
# ═══════════════════════════════════════════════════════════════════

def merge_importance_labels(corpus_path, resampling_path, output_path):
    by_id = {}
    with open(resampling_path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                by_id[e["id"]] = e

    total, labeled = 0, 0
    n_important, n_total_sent = 0, 0
    n_sampled, n_partial, n_interpolated = 0, 0, 0
    with open(corpus_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            total += 1
            imp = by_id.get(entry["id"])
            if imp:
                entry["sentence_importance"] = imp["sentence_importance"]
                labeled += 1
                for si in imp["sentence_importance"]:
                    n_total_sent += 1
                    if si["important"]:
                        n_important += 1
                    src = si.get("source", "sampled")
                    if src == "sampled": n_sampled += 1
                    elif src == "partial": n_partial += 1
                    elif src == "interpolated": n_interpolated += 1
            fout.write(json.dumps(entry) + "\n")

    pct = n_important / max(n_total_sent, 1) * 100
    print(f"  Labeled {labeled}/{total} entries")
    print(f"  Important sentences: {n_important}/{n_total_sent} ({pct:.1f}%)")
    if n_partial or n_interpolated:
        print(f"  Source breakdown: {n_sampled} sampled, {n_partial} partial, {n_interpolated} interpolated")
    # Log KL stats if available
    kl_vals = []
    for imp in by_id.values():
        for si in imp["sentence_importance"]:
            kl = si.get("resampling_importance_kl_binary", 0.0)
            if kl > 0:
                kl_vals.append(kl)
    if kl_vals:
        kl_vals.sort()
        print(f"  KL binary stats: median={kl_vals[len(kl_vals)//2]:.4f}, max={kl_vals[-1]:.4f}, >0.1: {sum(1 for k in kl_vals if k > 0.1)}/{len(kl_vals)}")


# ═══════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Worker mode — dispatched by orchestrator
    if args._phase == "generate":
        worker_generate(args)
        return
    if args._phase == "resample":
        worker_resample(args)
        return

    # ── Orchestrator mode ──
    # Prevent orchestrator from grabbing GPU memory (workers handle their own GPUs)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.num_gpus:
        num_gpus = args.num_gpus
    else:
        # Briefly detect GPU count then release
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        num_gpus = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"data/pipeline_{args.preset}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    corpus_path = out_dir / "corpus.jsonl"
    resampling_path = out_dir / "resampling.jsonl"
    labeled_path = out_dir / "labeled_corpus.jsonl"

    print(f"{'='*60}")
    print(f"CoT + Resampling Pipeline")
    print(f"{'='*60}")
    print(f"  Preset:      {args.preset}")
    print(f"  GPUs:        {num_gpus}")
    print(f"  Model:       {args.model}")
    print(f"  Rollouts:    {args.n_rollouts}")
    print(f"  Resamples:   {args.n_resamples}")
    print(f"  Batch size:  {args.resample_batch} entries/batch (resample phase)")
    if args.frugal:
        print(f"  Mode:        FRUGAL (sparse_step={args.sparse_step}, jump_threshold={args.jump_threshold})")
    else:
        print(f"  Mode:        brute-force (all truncation points)")
    print(f"  Output:      {out_dir}")
    print()
    t_start = time.time()

    # ── Phase 1: Generate CoTs ──
    if not args.skip_generate:
        print(f"{'='*60}")
        print("Phase 1: Generate on-policy CoTs")
        print(f"{'='*60}")

        from generate_cots import load_all_problems, MINI_SPLIT, MEDIUM_SPLIT, FULL_SPLIT, RESAMPLE_SPLIT
        splits = {"mini": MINI_SPLIT, "medium": MEDIUM_SPLIT, "full": FULL_SPLIT, "resample": RESAMPLE_SPLIT}
        split = splits[args.preset]

        print("Loading problems...")
        problems = load_all_problems(list(split.keys()), split)

        problems_file = out_dir / "problems.json"
        with open(problems_file, "w") as f:
            json.dump(problems, f)
        print(f"  {len(problems)} problems saved to {problems_file}\n")

        launch_workers("generate", num_gpus, args, str(problems_file), str(shard_dir))
        n = merge_shards(str(shard_dir), "generate", num_gpus, str(corpus_path))
        print(f"  Phase 1 complete: {n} CoT entries\n")
    else:
        if args.corpus:
            corpus_path = Path(args.corpus)
        print(f"Skipping Phase 1, using corpus: {corpus_path}\n")

    # ── Phase 2: Resampling importance ──
    if not args.skip_resample:
        # Resolve exclusion set from prior resampling runs
        exclude_ids = set()
        exclude_path = args.exclude_resampled
        if exclude_path == "auto":
            exclude_path = str(resampling_path)
        if exclude_path != "none" and Path(exclude_path).exists():
            with open(exclude_path) as f:
                for line in f:
                    if line.strip():
                        exclude_ids.add(json.loads(line)["id"])
            print(f"  Excluding {len(exclude_ids)} already-resampled entries from {exclude_path}")

        resample_input = corpus_path
        with open(corpus_path) as f:
            all_lines = f.readlines()
        if exclude_ids:
            all_lines = [l for l in all_lines if json.loads(l)["id"] not in exclude_ids]
            print(f"  After exclusion: {len(all_lines)} entries remaining")
        if args.resample_limit:
            import random as _rng
            _rng.shuffle(all_lines)
            all_lines = all_lines[:args.resample_limit]
        resample_input = out_dir / f"corpus_subset_{len(all_lines)}.jsonl"
        with open(resample_input, "w") as f:
            f.writelines(all_lines)
        print(f"  Resampling input: {len(all_lines)} entries → {resample_input}")

        print(f"{'='*60}")
        print("Phase 2: Resampling importance")
        print(f"{'='*60}")
        launch_workers("resample", num_gpus, args, str(resample_input), str(shard_dir))
        n = merge_shards(str(shard_dir), "resample", num_gpus, str(resampling_path))

        # ── Phase 3: Merge labels ──
        print(f"\n{'='*60}")
        print("Phase 3: Merge importance labels")
        print(f"{'='*60}")
        merge_importance_labels(str(corpus_path), str(resampling_path), str(labeled_path))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*60}")
    print(f"  Corpus:      {corpus_path}")
    if not args.skip_resample:
        print(f"  Resampling:  {resampling_path}")
        print(f"  Labeled:     {labeled_path}")
    print(f"  Logs:        logs/generate_*.log, logs/resample_*.log")


if __name__ == "__main__":
    main()
