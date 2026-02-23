"""
Collect resampling importance scores for Qwen3-8B on MATH problems.

Replicates the thought-anchors resampling pipeline using vLLM for fast batch generation.
For each CoT chunk, removes it and generates N continuations, then measures how the
answer distribution shifts (KL divergence).

Multi-GPU via data parallelism: spawns independent subprocesses, one vLLM instance per GPU.

Usage:
    # Smoke test (2 problems, 5 rollouts, 1 GPU)
    python scripts/collect_resampling_importance.py --num-gpus 1 --problems 2 --num-rollouts 5

    # Full run (4 GPUs)
    python scripts/collect_resampling_importance.py --num-gpus 4

    # Resume from existing partial results
    python scripts/collect_resampling_importance.py --num-gpus 4 --resume

    # Single-shard worker (called by launcher)
    python scripts/collect_resampling_importance.py --shard 0 --num-shards 4
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

REPO_ID = "uzaymacar/math-rollouts"
MODEL_DIR = "deepseek-r1-distill-qwen-14b"  # source of problems + deepseek base solutions
CONDITION = "correct_base_solution"
SAMPLING = "temperature_0.6_top_p_0.95"
QWEN_MODEL = "Qwen/Qwen3-8B"

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "qwen3_rollouts"

PROMPT_TEMPLATE = "Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n"


# ---------------------------------------------------------------------------
# Utilities inlined from thought-anchors to avoid submodule import issues
# ---------------------------------------------------------------------------

def split_solution_into_chunks(solution_text: str) -> list[str]:
    """Split a solution into sentence-level chunks. From thought-anchors/utils.py."""
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks = []
    current_chunk = ""
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        is_paragraph_end = any(
            i + len(p) <= len(solution_text) and solution_text[i : i + len(p)] == p
            for p in paragraph_ending_patterns
        )
        is_sentence_end = (
            i < len(solution_text) - 1
            and solution_text[i] in sentence_ending_tokens
            and solution_text[i + 1] in (" ", "\n")
        )
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        i += 1

    # Merge small chunks (<10 chars)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1
    return chunks


def extract_boxed_answers(text: str) -> list[str]:
    """Extract answers from \\boxed{}. From thought-anchors/utils.py."""
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_starts:
        return [""]
    answers = []
    for start_idx in boxed_starts:
        idx = start_idx + 7
        brace_count = 1
        answer = ""
        while idx < len(text) and brace_count > 0:
            char = text[idx]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            if brace_count > 0:
                answer += char
            idx += 1
        if answer:
            answers.append(answer)
    return answers if answers else [""]


def normalize_latex(latex_str: str) -> str:
    """Normalize LaTeX for answer comparison. From thought-anchors/utils.py."""
    n = latex_str.strip().lower()
    n = n.replace("dfrac", "frac").replace("tfrac", "frac")
    n = re.sub(r"\s+", "", n)
    n = n.replace("\\%", "").replace("{,}", "")
    n = n.replace("\\times", "*").replace("\\cdot", "*")
    n = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", n)
    n = re.sub(r"{([^{}]+)}", r"\1", n)
    n = n.replace("\\pi", "pi")
    n = re.sub(r"\\text\{([^{}]+)\}", r"\1", n)
    n = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", n)
    n = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", n)
    n = n.replace("\\text", "")
    return n


def normalize_answer(answer: str) -> str:
    return normalize_latex(answer)


def check_answer(answer: str, gt_answer: str) -> bool:
    return normalize_latex(answer) == normalize_latex(gt_answer)


def calculate_kl_divergence(sols1: list[dict], sols2: list[dict], alpha: float = 1e-9) -> float:
    """KL(P_1 || P_2) over answer distributions. From thought-anchors/analyze_rollouts.py."""
    counts1: dict[str, int] = defaultdict(int)
    counts2: dict[str, int] = defaultdict(int)
    for sol in sols1:
        a = normalize_answer(sol.get("answer", ""))
        if a:
            counts1[a] += 1
    for sol in sols2:
        a = normalize_answer(sol.get("answer", ""))
        if a:
            counts2[a] += 1
    if not counts1 or not counts2:
        return 0.0
    all_answers = set(counts1) | set(counts2)
    V = len(all_answers)
    total1, total2 = sum(counts1.values()), sum(counts2.values())
    if total1 == 0 or total2 == 0:
        return 0.0
    st1, st2 = total1 + alpha * V, total2 + alpha * V
    kl = 0.0
    for ans in all_answers:
        c1, c2 = counts1[ans], counts2[ans]
        p = (c1 + alpha) / st1
        q = (c2 + alpha) / st2
        p_raw = c1 / total1
        kl += p_raw * math.log(p / q)
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_problems(output_dir: Path, max_problems: int | None = None) -> list[dict]:
    """Download problem.json + base_solution.json + chunks from HF for each problem."""
    api = HfApi()
    base_path = f"{MODEL_DIR}/{SAMPLING}/{CONDITION}"
    entries = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=base_path))
    problem_dirs = sorted([e.path for e in entries if "problem_" in e.path])

    if max_problems is not None:
        problem_dirs = problem_dirs[:max_problems]

    print(f"Downloading {len(problem_dirs)} problems from {REPO_ID}...")
    problems = []
    for problem_path in tqdm(problem_dirs, desc="Downloading"):
        problem_name = problem_path.split("/")[-1]
        problem_dir = output_dir / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        files_to_get = ["problem.json", "base_solution.json", "chunks.json"]
        data = {}
        for fname in files_to_get:
            local_file = problem_dir / fname
            if local_file.exists():
                with open(local_file) as f:
                    data[fname] = json.load(f)
            else:
                file_path = f"{problem_path}/{fname}"
                downloaded = hf_hub_download(REPO_ID, file_path, repo_type="dataset")
                with open(downloaded) as f:
                    data[fname] = json.load(f)
                with open(local_file, "w") as f:
                    json.dump(data[fname], f, indent=2)

        problems.append({
            "problem_name": problem_name,
            "problem": data["problem.json"],
            "base_solution": data["base_solution.json"],
            "chunks": data["chunks.json"]["chunks"],
        })

    return problems


# ---------------------------------------------------------------------------
# Base solution generation with Qwen3-8B
# ---------------------------------------------------------------------------

def generate_base_solutions(problems: list[dict], output_dir: Path) -> list[dict]:
    """Generate Qwen3-8B base solutions for problems that don't have one yet."""
    from vllm import LLM, SamplingParams

    needs_generation = []
    for p in problems:
        sol_file = output_dir / p["problem_name"] / "qwen3_base_solution.json"
        if not sol_file.exists():
            needs_generation.append(p)

    if not needs_generation:
        print("All base solutions already exist, skipping generation.")
        # Load existing and rechunk
        for p in problems:
            sol_file = output_dir / p["problem_name"] / "qwen3_base_solution.json"
            with open(sol_file) as f:
                sol = json.load(f)
            p["qwen3_solution"] = sol
            chunks_file = output_dir / p["problem_name"] / "qwen3_chunks.json"
            if chunks_file.exists():
                with open(chunks_file) as f:
                    p["qwen3_chunks"] = json.load(f)["chunks"]
            else:
                p["qwen3_chunks"] = split_solution_into_chunks(sol["solution"])
        return problems

    print(f"Generating base solutions for {len(needs_generation)} problems...")

    llm = LLM(
        model=QWEN_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384)

    # Try up to 3 times per problem to get a correct solution
    max_attempts = 3
    for attempt in range(max_attempts):
        prompts, prompt_indices = [], []
        for i, p in enumerate(needs_generation):
            sol_file = output_dir / p["problem_name"] / "qwen3_base_solution.json"
            if not sol_file.exists():
                prompts.append(PROMPT_TEMPLATE.format(problem=p["problem"]["problem"]))
                prompt_indices.append(i)

        if not prompts:
            break

        print(f"  Attempt {attempt + 1}: generating {len(prompts)} solutions...")
        outputs = llm.generate(prompts, params)

        for prompt_pos, (idx, output) in enumerate(zip(prompt_indices, outputs)):
            p = needs_generation[idx]
            text = output.outputs[0].text
            answers = extract_boxed_answers(text)
            answer = answers[0] if answers else ""
            is_correct = check_answer(answer, p["problem"]["gt_answer"]) if answer else False

            if is_correct or attempt == max_attempts - 1:
                sol = {"prompt": prompts[prompt_pos], "solution": text, "answer": answer, "is_correct": is_correct}
                sol_file = output_dir / p["problem_name"] / "qwen3_base_solution.json"
                with open(sol_file, "w") as f:
                    json.dump(sol, f, indent=2)

    # Load all solutions and chunk them
    for p in problems:
        sol_file = output_dir / p["problem_name"] / "qwen3_base_solution.json"
        if sol_file.exists():
            with open(sol_file) as f:
                sol = json.load(f)
            p["qwen3_solution"] = sol
            chunks = split_solution_into_chunks(sol["solution"])
            p["qwen3_chunks"] = chunks
            chunks_file = output_dir / p["problem_name"] / "qwen3_chunks.json"
            with open(chunks_file, "w") as f:
                json.dump({"solution_text": sol["solution"], "chunks": chunks}, f, indent=2)
        else:
            # Fall back to deepseek chunks
            print(f"WARNING: No Qwen3 solution for {p['problem_name']}, using deepseek chunks")
            p["qwen3_solution"] = p["base_solution"]
            p["qwen3_chunks"] = p["chunks"]

    return problems


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def generate_rollouts_for_shard(problems: list[dict], output_dir: Path, num_rollouts: int, resume: bool):
    """Generate rollouts for a shard of problems using vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=QWEN_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        max_model_len=16384,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384)

    for p in tqdm(problems, desc="Problems"):
        chunks = p["qwen3_chunks"]
        problem_dir = output_dir / p["problem_name"]
        n_chunks = len(chunks)

        # Build cumulative prefixes
        cumulative = []
        current = ""
        for chunk in chunks:
            current += chunk + " "
            cumulative.append(current.strip())

        for chunk_idx in tqdm(range(n_chunks), desc=f"  {p['problem_name']} chunks", leave=False):
            chunk_dir = problem_dir / f"chunk_{chunk_idx}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            solutions_file = chunk_dir / "solutions.json"

            # Check existing solutions
            existing = []
            if resume and solutions_file.exists():
                with open(solutions_file) as f:
                    existing = json.load(f)
                valid = [s for s in existing if "answer" in s and "error" not in s]
                if len(valid) >= num_rollouts:
                    continue
                needed = num_rollouts - len(valid)
            else:
                needed = num_rollouts

            # Build prompts: remove chunk_idx from cumulative prefix up to chunk_idx
            prefix = cumulative[chunk_idx]
            prefix_without_chunk = prefix.replace(chunks[chunk_idx], "").strip()
            prompt = PROMPT_TEMPLATE.format(problem=p["problem"]["problem"]) + prefix_without_chunk

            prompts = [prompt] * needed
            outputs = llm.generate(prompts, params)

            new_solutions = []
            for output in outputs:
                text = output.outputs[0].text
                answers = extract_boxed_answers(text)
                answer = answers[0] if answers else ""
                is_correct = check_answer(answer, p["problem"]["gt_answer"]) if answer else False
                new_solutions.append({
                    "chunk_removed": chunks[chunk_idx],
                    "prefix_without_chunk": prefix_without_chunk,
                    "rollout": text,
                    "answer": answer,
                    "is_correct": is_correct,
                })

            all_solutions = existing + new_solutions
            with open(solutions_file, "w") as f:
                json.dump(all_solutions, f, indent=2)


# ---------------------------------------------------------------------------
# KL computation + labeling
# ---------------------------------------------------------------------------

def compute_importance_scores(problems: list[dict], output_dir: Path):
    """Compute resampling_importance_kl for each chunk from saved solutions."""
    for p in tqdm(problems, desc="Computing importance"):
        problem_dir = output_dir / p["problem_name"]
        chunks = p["qwen3_chunks"]
        n_chunks = len(chunks)

        # Load all solutions per chunk
        chunk_solutions = {}
        for chunk_idx in range(n_chunks):
            sol_file = problem_dir / f"chunk_{chunk_idx}" / "solutions.json"
            if sol_file.exists():
                with open(sol_file) as f:
                    sols = json.load(f)
                chunk_solutions[chunk_idx] = [s for s in sols if "answer" in s and "error" not in s]

        # Compute KL between consecutive chunks
        chunks_labeled = []
        for chunk_idx in range(n_chunks):
            sols_i = chunk_solutions.get(chunk_idx, [])
            # Find next chunk that has solutions
            next_idx = None
            for j in range(chunk_idx + 1, n_chunks):
                if j in chunk_solutions and chunk_solutions[j]:
                    next_idx = j
                    break

            if next_idx is not None:
                sols_next = chunk_solutions[next_idx]
                kl = calculate_kl_divergence(sols_i, sols_next)
            else:
                kl = 0.0

            # Accuracy for this chunk's rollouts
            if sols_i:
                accuracy = sum(1 for s in sols_i if s.get("is_correct", False)) / len(sols_i)
            else:
                accuracy = 0.0

            chunks_labeled.append({
                "chunk_idx": chunk_idx,
                "chunk": chunks[chunk_idx],
                "resampling_importance_kl": kl,
                "resampling_accuracy": accuracy,
                "n_rollouts": len(sols_i),
            })

        with open(problem_dir / "qwen3_chunks_labeled.json", "w") as f:
            json.dump(chunks_labeled, f, indent=2)


# ---------------------------------------------------------------------------
# Export to eval format
# ---------------------------------------------------------------------------

def export_to_eval_format(problems: list[dict], output_dir: Path, eval_output: Path):
    """Convert to the format expected by download_math_rollouts.py / step_importance eval."""
    items = []
    for p in problems:
        labeled_file = output_dir / p["problem_name"] / "qwen3_chunks_labeled.json"
        if not labeled_file.exists():
            continue
        with open(labeled_file) as f:
            chunks_labeled = json.load(f)

        cot_chunks = [c["chunk"] for c in chunks_labeled]
        importance_scores = [c["resampling_importance_kl"] for c in chunks_labeled]

        scored = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in scored[:3]]

        mean = sum(importance_scores) / len(importance_scores) if importance_scores else 0
        var = sum((v - mean) ** 2 for v in importance_scores) / len(importance_scores) if len(importance_scores) > 1 else 0

        items.append({
            "problem_idx": p["problem_name"],
            "problem": p["problem"]["problem"],
            "gt_answer": p["problem"]["gt_answer"],
            "level": p["problem"].get("level", ""),
            "math_type": p["problem"].get("type", ""),
            "model": "Qwen/Qwen3-8B",
            "source": "qwen3_resampling",
            "cot_chunks": cot_chunks,
            "importance_scores": importance_scores,
            "function_tags": [[] for _ in cot_chunks],
            "top_k_indices": top_k_indices,
            "score_variance": var,
            "n_chunks": len(cot_chunks),
            "n_high_importance": sum(1 for s in importance_scores if s > 0.1),
            "chunks_labeled": chunks_labeled,
        })

    items.sort(key=lambda x: x["score_variance"], reverse=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_output, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} items to {eval_output}")


# ---------------------------------------------------------------------------
# Multi-GPU launcher
# ---------------------------------------------------------------------------

def launch_shards(args):
    """Launch N independent worker processes, one per GPU."""
    num_gpus = args.num_gpus
    print(f"Launching {num_gpus} shard workers...")
    procs = []
    for gpu_id in range(num_gpus):
        cmd = [
            sys.executable, __file__,
            "--shard", str(gpu_id),
            "--num-shards", str(num_gpus),
            "--num-rollouts", str(args.num_rollouts),
            "--output-dir", str(args.output_dir),
        ]
        if args.problems is not None:
            cmd += ["--problems", str(args.problems)]
        if args.resume:
            cmd.append("--resume")
        if args.skip_base:
            cmd.append("--skip-base")

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout = open(log_dir / f"resampling_shard_{gpu_id}.log", "w")
        stderr = subprocess.STDOUT

        proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        procs.append((proc, stdout, gpu_id))
        print(f"  Shard {gpu_id} (GPU {gpu_id}): PID {proc.pid}")

    # Wait for all
    failed = []
    for proc, stdout, gpu_id in procs:
        proc.wait()
        stdout.close()
        if proc.returncode != 0:
            failed.append(gpu_id)
            print(f"  Shard {gpu_id} FAILED (exit code {proc.returncode})")
        else:
            print(f"  Shard {gpu_id} done.")

    if failed:
        print(f"WARNING: Shards {failed} failed. Check logs/resampling_shard_*.log")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect resampling importance scores via vLLM")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for data parallelism")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (worker mode)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards (worker mode)")
    parser.add_argument("--problems", type=int, default=None, help="Max number of problems (default: all 40)")
    parser.add_argument("--num-rollouts", type=int, default=100, help="Rollouts per chunk")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true", help="Skip chunks with enough existing rollouts")
    parser.add_argument("--skip-base", action="store_true", help="Skip base solution generation (use deepseek chunks)")
    parser.add_argument("--skip-rollouts", action="store_true", help="Skip rollout generation, only compute KL")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Launcher mode: spawn workers
    if args.shard is None and args.num_gpus > 1 and not args.skip_rollouts:
        # Phase 0: Download problems (single process)
        problems = download_problems(output_dir, max_problems=args.problems)
        print(f"Downloaded {len(problems)} problems")

        # Phase 1: Generate base solutions (single GPU)
        if not args.skip_base:
            problems = generate_base_solutions(problems, output_dir)

        # Phase 2: Launch parallel rollout workers
        success = launch_shards(args)

        # Phase 3: Compute importance scores (single process)
        if success:
            # Reload problems with qwen3_chunks
            for p in problems:
                chunks_file = output_dir / p["problem_name"] / "qwen3_chunks.json"
                if chunks_file.exists():
                    with open(chunks_file) as f:
                        p["qwen3_chunks"] = json.load(f)["chunks"]
                else:
                    p["qwen3_chunks"] = p["chunks"]

            compute_importance_scores(problems, output_dir)
            export_to_eval_format(problems, output_dir, output_dir.parent / "evals" / "step_importance_qwen3_raw.json")
        return

    # Worker mode (single shard) or single-GPU mode
    problems = download_problems(output_dir, max_problems=args.problems)

    if args.shard is not None:
        # Shard the problems
        shard_size = math.ceil(len(problems) / args.num_shards)
        start = args.shard * shard_size
        end = min(start + shard_size, len(problems))
        problems = problems[start:end]
        print(f"Shard {args.shard}/{args.num_shards}: problems {start}-{end-1} ({len(problems)} problems)")

    # Load qwen3 chunks (from base solution generation phase)
    for p in problems:
        chunks_file = output_dir / p["problem_name"] / "qwen3_chunks.json"
        if chunks_file.exists():
            with open(chunks_file) as f:
                p["qwen3_chunks"] = json.load(f)["chunks"]
        elif not args.skip_base:
            # Need to generate base solutions first (single GPU mode only)
            problems = generate_base_solutions(problems, output_dir)
            break
        else:
            p["qwen3_chunks"] = p["chunks"]

    if not args.skip_rollouts:
        generate_rollouts_for_shard(problems, output_dir, args.num_rollouts, args.resume)

    # Only compute KL in single-GPU mode or with --skip-rollouts
    if args.shard is None:
        compute_importance_scores(problems, output_dir)
        export_to_eval_format(problems, output_dir, output_dir.parent / "evals" / "step_importance_qwen3_raw.json")


if __name__ == "__main__":
    main()
