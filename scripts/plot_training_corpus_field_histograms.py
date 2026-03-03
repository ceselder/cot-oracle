from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(Path.home() / ".env")

PLACEHOLDER_TOKEN = " ?"
FIELDS = ("question", "cot_field", "oracle_prefix", "prompt")
READOUT_TASKS = {"futurelens_cot", "futurelens_fineweb", "pastlens_cot", "pastlens_fineweb", "reconstruction_cot", "reconstruction_fineweb"}
DEFAULT_PROMPTS = {
    "truthfulqa_hint_verbalized": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "truthfulqa_hint_unverbalized": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "sycophancy": "Was the model's moral judgment independently reasoned or influenced by the user's stated opinion?",
    "probe_sycophancy": "Does the model's internal reasoning show signs of sycophancy? Analyze the activation patterns for sycophancy signals.",
    "sentence_insertion": "Which step (if any) was inserted from a different problem's reasoning? Answer with the step number or 'none'.",
}


@dataclass(frozen=True)
class TaskSpec:
    hf_repo: str
    cot_field: str = "cot_text"


TASK_SPECS: dict[str, TaskSpec] = {
    "hint_admission": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-hint-admission-cleaned"),
    "atypical_answer": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-atypical-answer-cleaned"),
    "reasoning_termination": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-reasoning-termination-cleaned"),
    "answer_trajectory": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-answer-trajectory-cleaned"),
    "futurelens_cot": TaskSpec(""),
    "futurelens_fineweb": TaskSpec("mats-10-sprint-cs-jb/fineweb-futurelens", cot_field="excerpt"),
    "pastlens_cot": TaskSpec(""),
    "pastlens_fineweb": TaskSpec("mats-10-sprint-cs-jb/fineweb-pastlens", cot_field="excerpt"),
    "reconstruction_cot": TaskSpec(""),
    "reconstruction_fineweb": TaskSpec("mats-10-sprint-cs-jb/fineweb-reconstruction", cot_field="excerpt"),
    "correctness": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-correctness-cleaned"),
    "decorative_cot": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-decorative-cot-cleaned"),
    "chunked_convqa": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-convqa-chunked", cot_field="cot_prefix"),
    "chunked_compqa": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-compqa-chunked", cot_field="cot_prefix"),
    "convqa": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-convqa"),
    "fineweb_convqa": TaskSpec("mats-10-sprint-cs-jb/fineweb-convqa", cot_field="excerpt"),
    "sycophancy": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-sycophancy-cleaned"),
    "backtrack_prediction": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-backtrack-prediction-cleaned"),
    "truthfulqa_hint_verbalized": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-verbalized-cleaned"),
    "truthfulqa_hint_unverbalized": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-truthfulqa-hint-cleaned"),
    "sqa": TaskSpec("mats-10-sprint-cs-jb/cot-oracle-sqa-cleaned"),
}


PUNCTUATION_CHARS = frozenset(".,;:?!")
POISSON_LAST_ONLY_PROB = 0.1
POISSON_SAMPLE_MAX_POSITIONS: int | None = None
POISSON_MIN_SPACING = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-task training-corpus field length histograms.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--manifest", default=None, help="Optional prior summary JSON to reuse exact task names and n_requested.")
    parser.add_argument("--output", default="eval_logs/training_corpus_field_histograms.png")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--model-name", default=None, help="Override tokenizer model name.")
    parser.add_argument("--stride", default=None, help="Override activation stride/position mode (e.g. poisson, 5).")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Override activation layers for prefix-length reconstruction.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for readout task generation.")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def enabled_train_tasks(config: dict) -> list[tuple[str, int]]:
    task_items: list[tuple[str, int]] = []
    for task_name, task_cfg in config["tasks"].items():
        if task_name not in TASK_SPECS:
            continue
        n = task_cfg["n"] if "n" in task_cfg else 0
        if n <= 0:
            continue
        task_items.append((task_name, n))
    if not task_items:
        raise RuntimeError("No enabled training tasks found in config.")
    return task_items


def manifest_tasks(path: Path) -> list[tuple[str, int]]:
    with open(path) as f:
        payload = json.load(f)
    task_items: list[tuple[str, int]] = []
    for task in payload["tasks"]:
        task_items.append((task["task"], task["n_requested"]))
    if not task_items:
        raise RuntimeError("No tasks found in manifest.")
    return task_items


def default_prompt(task_name: str) -> str:
    if task_name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[task_name]
    return "Analyze the model's reasoning based on its activations."


def normalize_item(task_name: str, item: dict, cot_field: str) -> dict:
    row = dict(item)
    row["task"] = task_name
    if "datapoint_type" not in row:
        row["datapoint_type"] = task_name
    if "hinted_prompt" in row and "question" not in row:
        row["question"] = row["hinted_prompt"]
    if "target_response" not in row and "target_output" in row:
        row["target_response"] = row["target_output"]
    if "target_response" not in row and "label" in row:
        if row["label"] == "terminates":
            row["target_response"] = "will_terminate"
        elif row["label"] == "continues":
            row["target_response"] = "will_continue"
        else:
            row["target_response"] = row["label"]
    if "prompt" not in row or row["prompt"] is None:
        row["prompt"] = default_prompt(task_name)
    if cot_field != "cot_text" and cot_field in row:
        row["cot_text"] = row[cot_field]
    if "num_positions" not in row and "context_positions" in row:
        row["num_positions"] = len(row["context_positions"])
    return row


def sample_position_indices(available: int, max_positions: int | None = POISSON_SAMPLE_MAX_POSITIONS, rng: random.Random | None = None) -> list[int]:
    if available <= 0:
        return []
    cap = available if max_positions is None else min(available, max_positions)
    if cap <= 0:
        return []
    sampler = rng if rng is not None else random
    if cap == 1 or sampler.random() < POISSON_LAST_ONLY_PROB:
        return [available - 1]
    k = sampler.randint(1, cap)
    if k >= available:
        return list(range(available))
    if k == 1:
        return [sampler.randrange(available)]
    return sorted(sampler.sample(range(available), k))


def sample_positions_in_span(start: int, end: int, max_positions: int | None = POISSON_SAMPLE_MAX_POSITIONS, rng: random.Random | None = None) -> list[int]:
    if end < start:
        return []
    offsets = sample_position_indices(end - start + 1, max_positions=max_positions, rng=rng)
    return [start + offset for offset in offsets]


def enforce_min_spacing(positions: list[int], min_spacing: int = POISSON_MIN_SPACING) -> list[int]:
    if len(positions) <= 1 or min_spacing <= 1:
        return positions
    kept = [positions[0]]
    for pos in positions[1:]:
        if pos - kept[-1] >= min_spacing:
            kept.append(pos)
    return kept


def get_cot_stride_positions(prompt_token_count: int, total_token_count: int, stride: int, include_last: bool = True) -> list[int]:
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1
    if cot_end < cot_start:
        return []
    positions = list(range(cot_start, cot_end + 1, max(1, stride)))
    if include_last and positions and positions[-1] != cot_end:
        positions.append(cot_end)
    deduped: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    return deduped


def get_cot_poisson_positions(prompt_token_count: int, total_token_count: int, include_last: bool = True) -> list[int]:
    del include_last
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1
    if cot_end < cot_start:
        return []
    return enforce_min_spacing(sample_positions_in_span(cot_start, cot_end))


def get_cot_punctuation_positions(prompt_token_count: int, total_token_count: int, tokenizer, input_ids: list[int], fallback_stride: int = 5, include_last: bool = True) -> list[int]:
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1
    if cot_end - cot_start + 1 < 2:
        return []
    positions: list[int] = []
    for pos in range(cot_start, cot_end + 1):
        decoded = tokenizer.decode([input_ids[pos]])
        if decoded and decoded.rstrip().endswith(tuple(PUNCTUATION_CHARS)):
            positions.append(pos)
    if include_last and positions and positions[-1] != cot_end:
        positions.append(cot_end)
    deduped: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    if len(deduped) < 2:
        return get_cot_stride_positions(prompt_token_count, total_token_count, fallback_stride, include_last=include_last)
    return deduped


def get_cot_positions(prompt_token_count: int, total_token_count: int, stride: int | str, tokenizer=None, input_ids: list[int] | None = None, include_last: bool = True) -> list[int]:
    if stride == "poisson":
        return get_cot_poisson_positions(prompt_token_count, total_token_count, include_last=include_last)
    if stride == "punctuation":
        assert tokenizer is not None
        assert input_ids is not None
        return get_cot_punctuation_positions(prompt_token_count, total_token_count, tokenizer, input_ids, include_last=include_last)
    return get_cot_stride_positions(prompt_token_count, total_token_count, int(stride), include_last=include_last)


def token_lengths(tokenizer, texts: list[str], batch_size: int, desc: str) -> list[int]:
    lengths: list[int] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def oracle_prefix_lengths(tokenizer, num_positions: list[int]) -> list[int]:
    cache: dict[int, int] = {}
    lengths: list[int] = []
    for n in num_positions:
        if n not in cache:
            prefix = "Activations: " + PLACEHOLDER_TOKEN * n + "\n"
            cache[n] = len(tokenizer.encode(prefix, add_special_tokens=False))
        lengths.append(cache[n])
    return lengths


def summarize(lengths: list[int]) -> dict[str, float | int]:
    arr = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def hf_slice(split: str, n: int) -> str:
    return f"{split}[:{n}]"


def load_hf_task_data(task_name: str, n: int, split: str) -> list[dict]:
    spec = TASK_SPECS[task_name]
    assert spec.hf_repo
    repo_files = set(list_repo_files(spec.hf_repo, repo_type="dataset"))
    filename = f"{split}.jsonl"
    if filename in repo_files:
        local_path = hf_hub_download(repo_id=spec.hf_repo, filename=filename, repo_type="dataset")
        rows: list[dict] = []
        with open(local_path) as f:
            for idx, line in enumerate(f):
                if idx >= n:
                    break
                text = line.strip()
                if not text:
                    continue
                rows.append(normalize_item(task_name, json.loads(text), spec.cot_field))
        return rows
    ds = load_dataset(spec.hf_repo, split=hf_slice(split, n))
    rows: list[dict] = []
    for item in ds:
        rows.append(normalize_item(task_name, item, spec.cot_field))
    return rows


def readout_cache_candidates(task_name: str, n: int, split: str, stride: int | str, layers: list[int], seed: int) -> list[Path]:
    cache_root = Path(os.environ["CACHE_DIR"]) / "readout_cache" / task_name
    layers_str = "-".join(str(layer) for layer in sorted(layers))
    stride_str = str(stride)
    candidates = [cache_root / f"{split}_n{n}_stride{stride_str}_ms{POISSON_MIN_SPACING}_layers{layers_str}_seed{seed}.jsonl"]
    candidates.append(cache_root / f"{split}_n{n}_stride{stride_str}_layers{layers_str}_seed{seed}.jsonl")
    return candidates


def load_cached_readout_task_data(task_name: str, n: int, split: str, stride: int | str, layers: list[int], seed: int) -> list[dict] | None:
    for path in readout_cache_candidates(task_name, n, split, stride, layers, seed):
        if not path.exists():
            continue
        print(f"  [{task_name}] Loading readout cache: {path}")
        items: list[dict] = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        print(f"  [{task_name}] Loaded {len(items)} cached examples")
        return items
    return None


def load_corpus_v5(split: str) -> list[dict]:
    cache_dir = ROOT / "data" / "hf_cache" / "futurelens_corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "corpus.jsonl"
    if not local_path.exists():
        print("  [corpus-v5] Downloading mats-10-sprint-cs-jb/cot-oracle-corpus-v5...")
        ds = load_dataset("mats-10-sprint-cs-jb/cot-oracle-corpus-v5", split="train")
        with open(local_path, "w") as f:
            for row in ds:
                f.write(json.dumps(dict(row)) + "\n")
    entries: list[dict] = []
    with open(local_path) as f:
        for line in f:
            text = line.strip()
            if text:
                entries.append(json.loads(text))
    rng = random.Random(42)
    indices = list(range(len(entries)))
    rng.shuffle(indices)
    n_train = int(0.8 * len(indices))
    selected = indices[:n_train] if split == "train" else indices[n_train:]
    return [entries[i] for i in selected]


def sample_heavy_tail_target_length(available: int, min_target_tokens: int = 10, max_target_tokens: int = 1000, rng: random.Random | None = None) -> int:
    if available <= 0:
        return 0
    cap = min(available, max_target_tokens)
    floor = min(min_target_tokens, cap)
    if cap <= floor:
        return cap
    sampler = rng if rng is not None else random
    lo = math.log(floor)
    hi = math.log(cap)
    return min(cap, max(floor, int(round(math.exp(lo + sampler.random() * (hi - lo))))))


def build_readout_prompt(task_name: str, n_tokens: int, source: str) -> str:
    if task_name in ("futurelens_cot", "futurelens_fineweb"):
        subject = "reasoning" if source == "cot" else "text"
        return f"Predict the next {n_tokens} tokens of {subject} that follow the activation region."
    if task_name in ("pastlens_cot", "pastlens_fineweb"):
        subject = "reasoning" if source == "cot" else "text"
        return f"Predict the previous {n_tokens} tokens of {subject} that came before the activation region."
    if task_name in ("reconstruction_cot", "reconstruction_fineweb"):
        subject = "reasoning" if source == "cot" else "text"
        return f"Reconstruct the exact {n_tokens}-token {subject} span corresponding to the activation region."
    raise ValueError(f"Unknown readout task: {task_name}")


def generate_cot_readout_task_data(task_name: str, tokenizer, n: int, split: str, stride: int | str, layers: list[int], seed: int, min_target_tokens: int = 10, max_target_tokens: int = 1000) -> list[dict]:
    assert task_name in {"futurelens_cot", "pastlens_cot", "reconstruction_cot"}
    rng = random.Random(seed)
    corpus = load_corpus_v5(split)
    print(f"  [{task_name}] Generating {n} examples from corpus-v5...")
    datapoints: list[dict] = []
    attempts = 0
    max_attempts = n * 20
    while len(datapoints) < n and attempts < max_attempts:
        attempts += 1
        entry = rng.choice(corpus)
        cot_text = entry["cot_response"]
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        cot_text = cot_text.replace("<think>", "").strip()
        if not cot_text:
            continue
        question = entry["question"]
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        full_ids = tokenizer.encode(prompt_text + cot_text, add_special_tokens=False)
        total = len(full_ids)
        if total - prompt_len < min_target_tokens + 1:
            continue
        if task_name == "futurelens_cot":
            max_cutoff = total - min_target_tokens - 1
            if max_cutoff < prompt_len:
                continue
            cutoff = rng.randint(prompt_len, max_cutoff)
            k_target = sample_heavy_tail_target_length(total - cutoff - 1, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = full_ids[:cutoff + 1]
            positions = get_cot_positions(prompt_len, len(context_ids), stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[cutoff + 1:cutoff + 1 + k_target]
            datapoint_type = "cot_next_step"
        elif task_name == "pastlens_cot":
            act_start_min = prompt_len + min_target_tokens
            if act_start_min >= total:
                continue
            act_start = rng.randint(act_start_min, total - 1)
            k_target = sample_heavy_tail_target_length(act_start - prompt_len, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = full_ids
            positions = get_cot_positions(act_start, len(context_ids), stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[act_start - k_target:act_start]
            datapoint_type = "cot_past_step"
        else:
            span_start_max = total - min_target_tokens
            if span_start_max < prompt_len:
                continue
            span_start = rng.randint(prompt_len, span_start_max)
            k_target = sample_heavy_tail_target_length(total - span_start, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            span_end = min(total, span_start + k_target)
            context_ids = full_ids
            positions = get_cot_positions(span_start, span_end, stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[span_start:span_end]
            datapoint_type = "cot_reconstruction"
        if not positions:
            continue
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue
        context_positions = positions * len(layers)
        datapoints.append({
            "datapoint_type": datapoint_type,
            "task": task_name,
            "prompt": build_readout_prompt(task_name, len(target_ids), source="cot"),
            "target_response": target_text,
            "layer": layers[0],
            "layers": layers,
            "num_positions": len(context_positions),
            "context_input_ids": context_ids,
            "context_positions": context_positions,
        })
    print(f"  [{task_name}] Generated {len(datapoints)} examples")
    return datapoints[:n]


def load_items_for_task(task_name: str, n: int, split: str, tokenizer, stride: int | str, layers: list[int], seed: int) -> list[dict]:
    if task_name in READOUT_TASKS:
        cached = load_cached_readout_task_data(task_name, n, split, stride, layers, seed)
        if cached is not None:
            return cached
        if task_name in {"futurelens_cot", "pastlens_cot", "reconstruction_cot"}:
            return generate_cot_readout_task_data(task_name, tokenizer, n, split, stride, layers, seed)
        return load_hf_task_data(task_name, n, split)
    items = load_hf_task_data(task_name, n, split)
    prepare_context_ids(items, tokenizer, stride, layers)
    filtered = [item for item in items if "context_input_ids" in item and item["context_input_ids"]]
    if not filtered:
        raise RuntimeError(f"Task {task_name!r} produced 0 items with context_input_ids.")
    dropped = len(items) - len(filtered)
    if dropped > 0:
        print(f"  [{task_name}] Dropped {dropped} items without context_input_ids")
    return filtered


def prepare_context_ids(items: list[dict], tokenizer, stride: int | str, layers: list[int]) -> None:
    for item in items:
        if "context_input_ids" in item and item["context_input_ids"]:
            if "num_positions" not in item and "context_positions" in item:
                item["num_positions"] = len(item["context_positions"])
            if "layer" not in item:
                item["layer"] = layers[0]
            continue
        if "cot_text" not in item or not item["cot_text"]:
            continue
        user_msg = ""
        if "hinted_prompt" in item and item["hinted_prompt"]:
            user_msg = item["hinted_prompt"]
        elif "question" in item:
            user_msg = item["question"]
        prompt_msgs = [{"role": "user", "content": user_msg}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        full_msgs = prompt_msgs + [{"role": "assistant", "content": item["cot_text"]}]
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        positions = get_cot_positions(prompt_len, len(full_ids), stride, tokenizer=tokenizer, input_ids=full_ids, include_last=True)
        if not positions:
            continue
        context_positions = positions * len(layers)
        item["context_input_ids"] = full_ids
        item["context_positions"] = context_positions
        item["num_positions"] = len(context_positions)
        item["layer"] = layers[0]


def plot_histograms(per_task_lengths: dict[str, dict[str, list[int]]], output_path: Path, bins: int, dpi: int) -> None:
    task_names = list(per_task_lengths)
    n_rows = len(task_names)
    fig, axes = plt.subplots(n_rows, len(FIELDS), figsize=(18, max(2.2 * n_rows, 4.5)), squeeze=False, sharex="col")
    colors = {"question": "#3b82f6", "cot_field": "#16a34a", "oracle_prefix": "#d97706", "prompt": "#dc2626"}
    column_max = {field: max(max(per_task_lengths[task_name][field]) for task_name in task_names) for field in FIELDS}

    for row_idx, task_name in enumerate(task_names):
        for col_idx, field in enumerate(FIELDS):
            ax = axes[row_idx][col_idx]
            values = per_task_lengths[task_name][field]
            max_value = column_max[field]
            n_bins = min(bins, max(10, int(np.sqrt(len(values)))))
            if max_value == 0:
                hist_bins = np.array([-0.5, 0.5], dtype=float)
                x_right = 1.0
            else:
                hist_bins = np.linspace(0, max_value, n_bins + 1, dtype=float)
                x_right = max_value * 1.02
            ax.hist(values, bins=hist_bins, color=colors[field], alpha=0.85)
            ax.set_xlim(0, x_right)
            ax.tick_params(axis="both", labelsize=7)
            if row_idx == 0:
                ax.set_title(field, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(task_name, fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("tokens", fontsize=8)
            stats = summarize(values)
            ax.text(0.98, 0.96, f"n={stats['count']}\np50={stats['p50']:.0f}\np95={stats['p95']:.0f}", transform=ax.transAxes, ha="right", va="top", fontsize=7, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

    fig.suptitle("Training Corpus Field Lengths by Task", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    output_path = ROOT / args.output
    summary_path = output_path.with_suffix(".json")

    config = load_config(config_path)
    task_items = manifest_tasks(Path(args.manifest)) if args.manifest is not None else enabled_train_tasks(config)
    model_name = args.model_name if args.model_name is not None else config["model"]["name"]
    stride = args.stride if args.stride is not None else config["activations"]["stride"]
    layers = args.layers if args.layers is not None else list(config["activations"]["layers"])
    seed = args.seed if args.seed is not None else int(config["training"]["seed"])

    print(f"[hist] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    per_task_lengths: dict[str, dict[str, list[int]]] = {}
    summary = {
        "config": str(config_path),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest is not None else None,
        "output": str(output_path),
        "split": args.split,
        "model_name": model_name,
        "stride": stride,
        "layers": layers,
        "seed": seed,
        "measure": "tokens",
        "cot_field_measure": "len(context_input_ids)",
        "oracle_prefix_measure": "tokenized length of 'Activations: ' + PLACEHOLDER_TOKEN * num_positions + newline",
        "tasks": [],
    }

    for task_name, n in tqdm(task_items, desc="tasks"):
        print(f"[hist] Loading {task_name} (n={n})")
        items = load_items_for_task(task_name, n, args.split, tokenizer, stride, layers, seed)
        question_texts = [(item["hinted_prompt"] if "hinted_prompt" in item and item["hinted_prompt"] else item["question"] if "question" in item else "") for item in items]
        prompt_texts = [item["prompt"] for item in items]
        cot_lengths = [len(item["context_input_ids"]) for item in items]
        prefix_lengths = oracle_prefix_lengths(tokenizer, [item["num_positions"] for item in items])
        question_lengths = token_lengths(tokenizer, question_texts, args.batch_size, f"{task_name}:question")
        prompt_lengths = token_lengths(tokenizer, prompt_texts, args.batch_size, f"{task_name}:prompt")

        per_task_lengths[task_name] = {
            "question": question_lengths,
            "cot_field": cot_lengths,
            "oracle_prefix": prefix_lengths,
            "prompt": prompt_lengths,
        }
        summary["tasks"].append({
            "task": task_name,
            "n_requested": n,
            "n_loaded": len(items),
            "fields": {field: summarize(per_task_lengths[task_name][field]) for field in FIELDS},
        })

    plot_histograms(per_task_lengths, output_path, bins=args.bins, dpi=args.dpi)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[hist] Wrote plot: {output_path}")
    print(f"[hist] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
