"""
Download and preprocess math-rollouts data from HuggingFace for step importance eval.

Sources:
  1. uzaymacar/math-rollouts — per-chunk importance scores from Thought Anchors resampling
  2. thought-branches/faithfulness/ — per-sentence KL suppression + cue scores (authority bias)

Outputs:
  data/evals/step_importance_raw.json — math-rollouts preprocessed items
  data/evals/step_importance_faithfulness_raw.json — thought-branches preprocessed items

Usage:
    python scripts/download_math_rollouts.py
    python scripts/download_math_rollouts.py --skip-hf  # only process local thought-branches data
"""

import argparse
import csv
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

REPO_ID = "uzaymacar/math-rollouts"
# Use qwen-14b model (matches thought-branches data)
MODELS = ["deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b"]
CONDITION = "correct_base_solution"
SAMPLING = "temperature_0.6_top_p_0.95"

PROJECT_ROOT = Path(__file__).parent.parent


def download_math_rollouts(output_path: Path):
    """Download chunks_labeled.json + problem.json + base_solution.json for each problem."""
    api = HfApi()
    items = []

    for model in MODELS:
        base_path = f"{model}/{SAMPLING}/{CONDITION}"
        entries = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=base_path))
        problem_dirs = [e.path for e in entries if "problem_" in e.path]
        print(f"{model}: {len(problem_dirs)} problems")

        for problem_path in tqdm(problem_dirs, desc=model):
            problem_name = problem_path.split("/")[-1]

            # Download the 3 files we need
            files_to_get = ["chunks_labeled.json", "problem.json", "base_solution.json"]
            data = {}
            for fname in files_to_get:
                file_path = f"{problem_path}/{fname}"
                local = hf_hub_download(REPO_ID, file_path, repo_type="dataset")
                with open(local) as f:
                    data[fname] = json.load(f)

            problem_info = data["problem.json"]
            base_solution = data["base_solution.json"]
            chunks_labeled = data["chunks_labeled.json"]

            # Extract per-chunk data
            cot_chunks = []
            importance_scores = []
            function_tags = []
            for chunk_data in chunks_labeled:
                cot_chunks.append(chunk_data["chunk"])
                importance_scores.append(chunk_data["resampling_importance_kl"])
                function_tags.append(chunk_data.get("function_tags", []))

            # Compute top-k indices (top 3 by importance)
            scored = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in scored[:3]]

            # Check variance — skip if all scores are similar
            score_var = _variance(importance_scores)

            items.append({
                "problem_idx": problem_name,
                "problem": problem_info["problem"],
                "gt_answer": problem_info["gt_answer"],
                "level": problem_info.get("level", ""),
                "math_type": problem_info.get("type", ""),
                "model": model,
                "source": "math_rollouts",
                "cot_chunks": cot_chunks,
                "importance_scores": importance_scores,
                "function_tags": function_tags,
                "top_k_indices": top_k_indices,
                "score_variance": score_var,
                "n_chunks": len(cot_chunks),
                "n_high_importance": sum(1 for s in importance_scores if s > 0.1),
                # Full per-chunk data for reference
                "chunks_labeled": chunks_labeled,
            })

    # Sort by variance (most differentiated first)
    items.sort(key=lambda x: x["score_variance"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nSaved {len(items)} items to {output_path}")
    print(f"  Score variance range: {items[-1]['score_variance']:.4f} - {items[0]['score_variance']:.4f}")
    print(f"  Avg chunks per problem: {sum(x['n_chunks'] for x in items) / len(items):.0f}")
    return items


def process_thought_branches(output_path: Path):
    """Process thought-branches faithfulness data for step importance eval."""
    tb_root = PROJECT_ROOT / "thought-branches" / "faithfulness"

    # Load good_problems — use the strictest filter file
    gp_file = tb_root / "good_problems" / "Professor_itc_failure_threshold0.3_correct_base_no_mention.json"
    if not gp_file.exists():
        # Try alternative files
        candidates = sorted(tb_root.glob("good_problems/*.json"))
        if not candidates:
            print("WARNING: No good_problems files found. Skipping thought-branches.")
            return []
        gp_file = candidates[0]
        print(f"Using {gp_file.name}")

    with open(gp_file) as f:
        good_problems = json.load(f)

    # Build pn -> problem mapping
    pn_to_problem = {p["pn"]: p for p in good_problems}

    # Load KL suppression scores
    kl_file = tb_root / "dfs" / "kl_supp_ef_qwen-14b.csv"
    kl_by_pn = {}
    if kl_file.exists():
        with open(kl_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pn = int(row["pn"])
                if pn not in kl_by_pn:
                    kl_by_pn[pn] = []
                kl_by_pn[pn].append({
                    "sentence_num": int(row["sentence_num"]),
                    "sentence": row["sentence"],
                    "kl": float(row["kl"]),
                })
        # Sort by sentence_num within each problem
        for pn in kl_by_pn:
            kl_by_pn[pn].sort(key=lambda x: x["sentence_num"])

    # Load cue scores
    cue_file = tb_root / "dfs" / "faith_counterfactual_qwen-14b.csv"
    cue_by_pn = {}
    if cue_file.exists():
        with open(cue_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pn = int(row["pn"])
                if pn not in cue_by_pn:
                    cue_by_pn[pn] = []
                cue_by_pn[pn].append({
                    "sentence_num": int(row["sentence_num"]),
                    "sentence": row["sentence"],
                    "cue_score": float(row["cue_score"]),
                })
        for pn in cue_by_pn:
            cue_by_pn[pn].sort(key=lambda x: x["sentence_num"])

    items = []
    for problem in good_problems:
        pn = problem["pn"]

        # Get sentences from KL data (preferred) or split from response_text
        if pn in kl_by_pn:
            sentences_data = kl_by_pn[pn]
            sentences = [s["sentence"] for s in sentences_data]
            kl_scores = [s["kl"] for s in sentences_data]
        else:
            # Split response_text into sentences
            sentences = _split_into_sentences(problem["response_text"])
            kl_scores = [0.0] * len(sentences)

        # Get cue scores if available
        cue_scores = [0.0] * len(sentences)
        if pn in cue_by_pn:
            cue_data = cue_by_pn[pn]
            for cd in cue_data:
                idx = cd["sentence_num"]
                if idx < len(cue_scores):
                    cue_scores[idx] = cd["cue_score"]

        # Use KL suppression as importance (higher KL = more important when suppressed)
        importance_scores = kl_scores

        scored = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in scored[:3]]

        items.append({
            "problem_idx": f"tb_{pn}",
            "problem": problem["question"],
            "gt_answer": problem["gt_answer"],
            "cue_answer": problem["cue_answer"],
            "cue_type": problem["cue_type"],
            "model": problem.get("model", "deepseek-r1-distill-qwen-14b"),
            "source": "thought_branches",
            "cot_chunks": sentences,
            "importance_scores": importance_scores,
            "cue_scores": cue_scores,
            "function_tags": [[] for _ in sentences],
            "top_k_indices": top_k_indices,
            "score_variance": _variance(importance_scores),
            "n_chunks": len(sentences),
            "n_high_importance": sum(1 for s in importance_scores if s > 0.1),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\nSaved {len(items)} thought-branches items to {output_path}")
    return items


def _variance(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _split_into_sentences(text):
    """Simple sentence splitter for CoT text."""
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_local_qwen3_rollouts(output_path: Path, rollouts_dir: Path):
    """Load Qwen3-8B resampling results from local data/qwen3_rollouts/ directory."""
    # Check for pre-exported eval file
    eval_file = rollouts_dir.parent / "evals" / "step_importance_qwen3_raw.json"
    if eval_file.exists():
        print(f"Loading pre-exported Qwen3 eval data from {eval_file}")
        with open(eval_file) as f:
            items = json.load(f)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"Saved {len(items)} Qwen3 items to {output_path}")
        return items

    # Otherwise, build from per-problem dirs
    problem_dirs = sorted(rollouts_dir.glob("problem_*"))
    items = []
    for problem_dir in tqdm(problem_dirs, desc="Loading Qwen3 rollouts"):
        problem_file = problem_dir / "problem.json"
        labeled_file = problem_dir / "qwen3_chunks_labeled.json"
        if not problem_file.exists() or not labeled_file.exists():
            continue

        with open(problem_file) as f:
            problem_info = json.load(f)
        with open(labeled_file) as f:
            chunks_labeled = json.load(f)

        cot_chunks = [c["chunk"] for c in chunks_labeled]
        importance_scores = [c["resampling_importance_kl"] for c in chunks_labeled]

        scored = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in scored[:3]]

        items.append({
            "problem_idx": problem_dir.name,
            "problem": problem_info["problem"],
            "gt_answer": problem_info["gt_answer"],
            "level": problem_info.get("level", ""),
            "math_type": problem_info.get("type", ""),
            "model": "Qwen/Qwen3-8B",
            "source": "qwen3_resampling",
            "cot_chunks": cot_chunks,
            "importance_scores": importance_scores,
            "function_tags": [[] for _ in cot_chunks],
            "top_k_indices": top_k_indices,
            "score_variance": _variance(importance_scores),
            "n_chunks": len(cot_chunks),
            "n_high_importance": sum(1 for s in importance_scores if s > 0.1),
            "chunks_labeled": chunks_labeled,
        })

    items.sort(key=lambda x: x["score_variance"], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} Qwen3 items to {output_path}")
    return items


def main():
    parser = argparse.ArgumentParser(description="Download math-rollouts for step importance eval")
    parser.add_argument("--output-dir", default="data/evals")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace download, only process local data")
    parser.add_argument("--from-local", action="store_true", help="Load Qwen3-8B resampling from data/qwen3_rollouts/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.from_local:
        print("=" * 60)
        print("Loading local Qwen3-8B resampling results...")
        print("=" * 60)
        rollouts_dir = PROJECT_ROOT / "data" / "qwen3_rollouts"
        load_local_qwen3_rollouts(output_dir / "step_importance_qwen3_raw.json", rollouts_dir)

    if not args.skip_hf and not args.from_local:
        print("=" * 60)
        print("Downloading math-rollouts from HuggingFace...")
        print("=" * 60)
        download_math_rollouts(output_dir / "step_importance_raw.json")

    print("\n" + "=" * 60)
    print("Processing thought-branches faithfulness data...")
    print("=" * 60)
    process_thought_branches(output_dir / "step_importance_faithfulness_raw.json")


if __name__ == "__main__":
    main()
