"""
Unified data loading for the CoT Oracle.

Downloads precomputed JSONL from HuggingFace, caches locally, and returns
lists of dicts in the standard format expected by dicts_to_training_data().

Also provides load_fineweb_data() for PastLens-style context prediction
from FineWeb + LMSYS streaming (togglable auxiliary training data).
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

from tasks import TASKS, TaskDef, HF_ORG, get_trainable_tasks


_HF_CACHE_DIR = Path(os.environ.get("COT_ORACLE_CACHE_DIR", "data/hf_cache"))

# Readout task sets — these generate data from corpora, not precomputed HF JSONL
_COT_READOUT_TASKS = {"futurelens_cot", "pastlens_cot", "reconstruction_cot"}
_FINEWEB_READOUT_TASKS = {"futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb"}
_ALL_READOUT_TASKS = _COT_READOUT_TASKS | _FINEWEB_READOUT_TASKS

# Default prompts for tasks whose HF data doesn't include a 'prompt' field
_DEFAULT_PROMPTS: dict[str, str] = {
    "truthfulqa_hint_verbalized": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "truthfulqa_hint_unverbalized": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "sycophancy": "Was the model's moral judgment independently reasoned or influenced by the user's stated opinion?",
    "probe_sycophancy": "Does the model's internal reasoning show signs of sycophancy? Analyze the activation patterns for sycophancy signals.",
    "sentence_insertion": "Which step (if any) was inserted from a different problem's reasoning? Answer with the step number or 'none'.",
}

_READOUT_MIN_TARGET_TOKENS = 10
_READOUT_MAX_TARGET_TOKENS = 1000


def _default_prompt(task_name: str) -> str:
    """Return a default oracle prompt for tasks whose HF data lacks a prompt field."""
    return _DEFAULT_PROMPTS.get(task_name, "Analyze the model's reasoning based on its activations.")


def load_task_data(
    task_name: str,
    split: str = "train",
    n: int | None = None,
    shuffle: bool = True,
) -> list[dict]:
    """Download {split}.jsonl from task's HF repo, return list of dicts.

    Args:
        task_name: Key in TASKS dict.
        split: "train" or "test".
        n: Max examples to return. None = all.

    Returns:
        List of dicts with keys: task, prompt, target_response,
        context_input_ids, context_positions, layers, (+ datapoint_type).
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {sorted(TASKS.keys())}")

    task_def = TASKS[task_name]
    local_path = _download_from_hf(task_def, split)

    data = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Normalize: ensure 'task' field exists
            if "task" not in item and "datapoint_type" in item:
                item["task"] = item["datapoint_type"]
            elif "task" not in item:
                item["task"] = task_name
            # Ensure datapoint_type exists (needed by dicts_to_training_data)
            if "datapoint_type" not in item:
                item["datapoint_type"] = task_def.legacy_datapoint_type or task_name
            # Ensure num_positions exists
            if "num_positions" not in item:
                item["num_positions"] = len(item.get("context_positions", []))
            # Ensure layer exists
            if "layer" not in item:
                layers = item.get("layers", [9, 18, 27])
                item["layer"] = layers[0] if layers else 9
            # Map label -> target_response for reasoning_termination new format
            if "target_response" not in item and "label" in item:
                label_map = {"terminates": "will_terminate", "continues": "will_continue"}
                item["target_response"] = label_map.get(item["label"], item["label"])
            # Map hinted_prompt -> question for truthfulqa datasets (used by prepare_context_ids)
            if "hinted_prompt" in item and "question" not in item:
                item["question"] = item["hinted_prompt"]
            # Inject default prompt if missing (e.g. truthfulqa eval datasets)
            if "prompt" not in item or item["prompt"] is None:
                item["prompt"] = _default_prompt(task_name)
            # Map cot_field → cot_text for tasks that use a different field
            # (e.g. chunked_convqa/compqa use cot_prefix for activations)
            if task_def.cot_field != "cot_text" and task_def.cot_field in item:
                item["cot_text"] = item[task_def.cot_field]
            # Recompute context_input_ids from text for standard tasks.
            # Readout tasks (futurelens/pastlens/reconstruction) keep precomputed
            # positions since they encode task-specific span boundaries.
            if "layers" not in item:
                item.pop("context_input_ids", None)
                item.pop("context_positions", None)
            data.append(item)

    if n is not None and len(data) > n:
        if shuffle:
            random.shuffle(data)
        data = data[:n]

    return data


def load_all_training_data(
    task_config: dict[str, dict],
    tokenizer=None,
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    model_name: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load train splits for all enabled tasks.

    Args:
        task_config: Maps task_name -> {"n": int, ...}.
            Set n: 0 to disable a task.
        tokenizer: Required for readout tasks (futurelens/pastlens/reconstruction).
        stride: Position mode for readout task generation.
        layers: Layer indices for readout tasks.
        model_name: Model name for FineWeb readout tasks.
        seed: Random seed for readout task generation.

    Returns:
        Combined list of dicts from all enabled tasks, shuffled.
    """
    trainable = get_trainable_tasks()
    all_data: list[dict] = []

    for task_name, cfg in task_config.items():
        n = cfg.get("n", 0)
        if n <= 0:
            continue

        if task_name not in trainable:
            raise ValueError(
                f"Task {task_name!r} is not trainable. "
                f"Trainable tasks: {sorted(trainable.keys())}"
            )

        print(f"  [data] Loading {task_name} (n={n})...")

        if task_name in _ALL_READOUT_TASKS:
            assert tokenizer is not None, f"tokenizer required for readout task {task_name}"
            items = load_readout_task_data(
                task_name=task_name, tokenizer=tokenizer, n=n,
                split="train", stride=stride, layers=layers,
                seed=seed, model_name=model_name,
            )
        else:
            items = load_task_data(task_name, split="train", n=n)

        if not items:
            raise RuntimeError(
                f"Task {task_name!r} is enabled (n={n}) but returned 0 items. "
                f"Check HF repo: {TASKS[task_name].hf_repo}"
            )

        print(f"  [data]   -> {len(items)} examples")
        all_data.extend(items)

    random.shuffle(all_data)
    if all_data:
        print(f"  [data] Total task examples: {len(all_data)}")
    return all_data


def prepare_context_ids(
    items: list[dict],
    tokenizer,
    stride: int | str = "poisson",
    layers: list[int] | None = None,
) -> list[dict]:
    """Compute context_input_ids and context_positions for items with cot_text.

    Items that already have context_input_ids (e.g. answer_trajectory) are
    left unchanged. For all others, builds the chat-templated input from
    cot_text + question/hinted_prompt, tokenizes it, and computes activation
    positions in the CoT region.

    Args:
        items: Raw dicts from load_task_data().
        tokenizer: HuggingFace tokenizer.
        stride: Position mode for CoT region: int, "poisson", or "punctuation".
        layers: Layer indices (positions are repeated per layer).

    Returns:
        Same list, mutated in place (items gain context_input_ids,
        context_positions, num_positions, layer fields).
    """
    from cot_utils import get_cot_positions

    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)

    # Find items that need tokenization
    need_tokenize = [i for i, item in enumerate(items) if not item.get("context_input_ids")]
    if need_tokenize:
        print(f"  [data] Tokenizing {len(need_tokenize)}/{len(items)} items...")
    _log_interval = max(1, len(need_tokenize) // 20)

    prepared = 0
    for count, idx in enumerate(need_tokenize):
        item = items[idx]
        cot_text = item.get("cot_text", "")
        if not cot_text:
            raise ValueError(f"Item missing cot_text and context_input_ids: task={item.get('task', '?')}")

        if (count + 1) % _log_interval == 0:
            print(f"  [data]   {count + 1}/{len(need_tokenize)} tokenized...")

        # Build user message: hinted_prompt for hint tasks, question otherwise
        user_msg = item.get("hinted_prompt") or item.get("question", "")

        # Tokenize with chat template (use tokenize=False + encode to avoid
        # transformers 5.x returning Encoding objects instead of flat ID lists)
        prompt_msgs = [{"role": "user", "content": user_msg}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        full_msgs = prompt_msgs + [{"role": "assistant", "content": cot_text}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Activation positions in the CoT region
        cot_positions = get_cot_positions(
            prompt_len, len(full_ids), stride=stride,
            tokenizer=tokenizer, input_ids=full_ids, include_last=True,
        )
        if not cot_positions:
            continue

        # Repeat CoT positions for each layer
        all_positions = cot_positions * n_layers

        item["context_input_ids"] = full_ids
        item["context_positions"] = all_positions
        item["num_positions"] = len(all_positions)
        item["layer"] = layers[0]
        prepared += 1

    if prepared > 0:
        print(f"  [data] Tokenized cot_text → context_input_ids for {prepared} items "
              f"(position_mode={stride}, {n_layers} layers)")

    return items


def _download_from_hf(task_def: TaskDef, split: str) -> Path:
    """Download a split from the task's HF repo, cache locally. Return local path."""
    cache_dir = _HF_CACHE_DIR / task_def.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{split}.jsonl"

    if local_path.exists():
        return local_path

    print(f"  [data] Downloading {task_def.hf_repo}/{split}.jsonl ...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=task_def.hf_repo,
            filename=f"{split}.jsonl",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        # hf_hub_download may place it in a subdirectory — symlink if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path and downloaded_path.exists():
            if not local_path.exists():
                local_path.symlink_to(downloaded_path)
    except Exception as e:
        # Try loading via HuggingFace datasets library as fallback
        try:
            _download_via_datasets_lib(task_def, split, local_path)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download {task_def.hf_repo}/{split}.jsonl: {e}\n"
                f"Fallback also failed: {e2}"
            ) from e2

    if not local_path.exists():
        raise FileNotFoundError(
            f"Download completed but file not found at {local_path}"
        )

    return local_path


def _download_via_datasets_lib(task_def: TaskDef, split: str, local_path: Path) -> None:
    """Fallback: use HuggingFace `datasets` library to load and save as JSONL."""
    from datasets import load_dataset

    # Try the requested split, then alternatives
    ds = None
    tried = [split]
    try:
        ds = load_dataset(task_def.hf_repo, split=split)
    except (ValueError, KeyError):
        pass

    if ds is None and split == "test":
        tried.append("eval")
        try:
            ds = load_dataset(task_def.hf_repo, split="eval")
        except (ValueError, KeyError):
            pass

    if ds is None and split == "train":
        # Try combining id_train + ood_train
        tried.append("id_train+ood_train")
        try:
            from datasets import concatenate_datasets
            id_ds = load_dataset(task_def.hf_repo, split="id_train")
            ood_ds = load_dataset(task_def.hf_repo, split="ood_train")
            ds = concatenate_datasets([id_ds, ood_ds])
        except (ValueError, KeyError):
            pass
        if ds is None:
            for alt in ["id_train", "ood_train"]:
                tried.append(alt)
                try:
                    ds = load_dataset(task_def.hf_repo, split=alt)
                    break
                except (ValueError, KeyError):
                    pass

    if ds is None:
        raise ValueError(f"Could not find split for {task_def.hf_repo}, tried: {tried}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")


def _sample_heavy_tail_target_length(
    available: int,
    min_target_tokens: int = _READOUT_MIN_TARGET_TOKENS,
    max_target_tokens: int = _READOUT_MAX_TARGET_TOKENS,
    rng: random.Random | None = None,
) -> int:
    """Sample a heavy-tailed target length in [min_target_tokens, max_target_tokens]."""
    if available <= 0:
        return 0

    cap = min(available, max_target_tokens)
    floor = min(min_target_tokens, cap)
    if cap <= floor:
        return cap

    sampler = rng or random
    lo = math.log(floor)
    hi = math.log(cap)
    return min(cap, max(floor, int(round(math.exp(lo + sampler.random() * (hi - lo))))))


def _build_readout_prompt(task_name: str, n_tokens: int, source: str) -> str:
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


# ── Readout task caching ──


def _readout_cache_path(task_name: str, n: int, split: str, stride, layers: list[int], seed: int) -> Path:
    """Return the cache file path for a readout task's processed data."""
    from cot_utils import POISSON_MIN_SPACING
    cache_dir = Path(os.environ["CACHE_DIR"]) / "readout_cache"
    layers_str = "-".join(str(l) for l in sorted(layers))
    stride_str = str(stride)
    if stride == "poisson" and POISSON_MIN_SPACING > 1:
        stride_str += f"_ms{POISSON_MIN_SPACING}"
    filename = f"{split}_n{n}_stride{stride_str}_layers{layers_str}_seed{seed}.jsonl"
    return cache_dir / task_name / filename


def load_readout_task_data(
    task_name: str,
    tokenizer,
    n: int,
    split: str = "train",
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    seed: int = 42,
    model_name: str | None = None,
) -> list[dict]:
    """Load readout task data with disk caching to $CACHE_DIR/readout_cache/.

    On first call, generates data via load_cot_readout_task_data or
    load_fineweb_readout_task_data and saves to disk. On subsequent calls
    with the same params, loads from cache instantly.
    """
    if layers is None:
        layers = [9, 18, 27]

    cache_path = _readout_cache_path(task_name, n, split, stride, layers, seed)

    if cache_path.exists():
        print(f"  [{task_name}] Loading from cache: {cache_path}")
        items = []
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        print(f"  [{task_name}] Loaded {len(items)} cached examples")
        return items

    # Generate fresh data
    if task_name in _COT_READOUT_TASKS:
        items = load_cot_readout_task_data(
            task_name=task_name, tokenizer=tokenizer, n=n,
            split=split, stride=stride, layers=layers, seed=seed,
        )
    elif task_name in _FINEWEB_READOUT_TASKS:
        items = load_fineweb_readout_task_data(
            task_name=task_name, tokenizer=tokenizer, model_name=model_name,
            n=n, stride=stride, layers=layers, seed=seed,
        )
    else:
        raise ValueError(f"Not a readout task: {task_name}")

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  [{task_name}] Saved {len(items)} examples to cache: {cache_path}")

    return items


# ── Readout tasks from corpus-v5 / FineWeb ──


def load_cot_readout_task_data(
    task_name: str,
    tokenizer,
    n: int = 30000,
    split: str = "train",
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    seed: int = 42,
    min_target_tokens: int = _READOUT_MIN_TARGET_TOKENS,
    max_target_tokens: int = _READOUT_MAX_TARGET_TOKENS,
) -> list[dict]:
    """Generate FutureLens/PastLens/Reconstruction data from corpus-v5."""
    from cot_utils import get_cot_positions

    if task_name not in {"futurelens_cot", "pastlens_cot", "reconstruction_cot"}:
        raise ValueError(f"Unsupported CoT readout task: {task_name}")

    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)
    rng = random.Random(seed)

    corpus = _load_corpus_v5(split)
    if not corpus:
        raise RuntimeError(f"No entries found in corpus-v5 ({split} split)")

    print(f"  [{task_name}] Loaded {len(corpus)} corpus entries ({split} split)")
    print(f"  [{task_name}] Generating {n} examples (target_tokens={min_target_tokens}-{max_target_tokens})...")

    datapoints: list[dict] = []
    attempts = 0
    max_attempts = n * 20

    while len(datapoints) < n and attempts < max_attempts:
        attempts += 1
        entry = rng.choice(corpus)

        cot_text = entry.get("cot_response", "")
        if not cot_text:
            continue
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        cot_text = cot_text.replace("<think>", "").strip()
        if not cot_text:
            continue

        question = entry.get("question", "")
        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
            k_target = _sample_heavy_tail_target_length(total - cutoff - 1, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = full_ids[:cutoff + 1]
            positions = get_cot_positions(prompt_len, len(context_ids), stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[cutoff + 1: cutoff + 1 + k_target]
            datapoint_type = "cot_next_step"
        elif task_name == "pastlens_cot":
            act_start_min = prompt_len + min_target_tokens
            if act_start_min >= total:
                continue
            act_start = rng.randint(act_start_min, total - 1)
            k_target = _sample_heavy_tail_target_length(act_start - prompt_len, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = full_ids
            positions = get_cot_positions(act_start, len(context_ids), stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[act_start - k_target: act_start]
            datapoint_type = "cot_past_step"
        else:
            span_start_max = total - min_target_tokens
            if span_start_max < prompt_len:
                continue
            span_start = rng.randint(prompt_len, span_start_max)
            k_target = _sample_heavy_tail_target_length(total - span_start, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            span_end = min(total, span_start + k_target)
            context_ids = full_ids
            positions = get_cot_positions(span_start, span_end, stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = full_ids[span_start:span_end]
            datapoint_type = "cot_reconstruction"

        if not positions:
            continue
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            continue

        context_positions = positions * n_layers
        datapoints.append({
            "datapoint_type": datapoint_type,
            "task": task_name,
            "prompt": _build_readout_prompt(task_name, len(target_ids), source="cot"),
            "target_response": target_text,
            "layer": layers[0],
            "layers": layers,
            "num_positions": len(context_positions),
            "context_input_ids": context_ids,
            "context_positions": context_positions,
        })

        if len(datapoints) % 10000 == 0 and len(datapoints) > 0:
            print(f"  [{task_name}] {len(datapoints)}/{n} examples...")

    print(f"  [{task_name}] Generated {len(datapoints)} examples (position_mode={stride}, from {attempts} attempts)")
    return datapoints[:n]


def load_futurelens_data(
    tokenizer,
    n: int = 30000,
    split: str = "train",
    predict_tokens: int = 50,
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Backward-compatible wrapper for corpus-v5 FutureLens generation."""
    _ = predict_tokens
    return load_cot_readout_task_data(
        task_name="futurelens_cot",
        tokenizer=tokenizer,
        n=n,
        split=split,
        stride=stride,
        layers=layers,
        seed=seed,
    )


def _load_corpus_v5(split: str = "train") -> list[dict]:
    """Load corpus-v5 from HF and split 80/20 for train/test.

    Corpus-v5 has only a train split on HF, so we deterministically
    split by entry index (seed=42).
    """
    corpus_repo = f"{HF_ORG}/cot-oracle-corpus-v5"
    cache_dir = _HF_CACHE_DIR / "futurelens_corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "corpus.jsonl"

    if not local_path.exists():
        print(f"  [corpus-v5] Downloading corpus from {corpus_repo}...")
        try:
            from datasets import load_dataset
            ds = load_dataset(corpus_repo, split="train")
            with open(local_path, "w") as f:
                for row in ds:
                    f.write(json.dumps(dict(row)) + "\n")
            print(f"  [futurelens] Saved {len(ds)} entries to {local_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download corpus-v5: {e}"
            ) from e

    # Load all entries
    entries = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Deterministic 80/20 split
    rng = random.Random(42)
    indices = list(range(len(entries)))
    rng.shuffle(indices)

    n_train = int(0.8 * len(indices))
    if split == "train":
        selected = indices[:n_train]
    else:
        selected = indices[n_train:]

    return [entries[i] for i in selected]


# ── FineWeb corpus loading ──


def _load_fineweb_corpus(split: str = "train") -> list[str]:
    """Load FineWeb corpus subset from HF, cache locally to $CACHE_DIR."""
    cache_dir = Path(os.environ["CACHE_DIR"]) / "fineweb-corpus"
    cache_file = cache_dir / f"{split}.jsonl"

    if cache_file.exists():
        texts = []
        with open(cache_file) as f:
            for line in f:
                if line.strip():
                    texts.append(json.loads(line))
        print(f"  [fineweb-corpus] Loaded {len(texts)} texts from {cache_file}")
        return texts

    from datasets import load_dataset
    repo = f"{HF_ORG}/fineweb-corpus"
    print(f"  [fineweb-corpus] Downloading {repo}/{split}...")
    ds = load_dataset(repo, split=split)
    texts = [row["text"] for row in ds]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        for text in texts:
            f.write(json.dumps(text) + "\n")
    print(f"  [fineweb-corpus] Saved {len(texts)} texts to {cache_file}")
    return texts


# ── FineWeb context prediction (PastLens-style) ──


def _fineweb_text_generator(tokenizer) -> Generator[str, None, None]:
    """Stream a 50/50 mix of FineWeb pretrain and LMSYS chat text."""
    from datasets import load_dataset

    pretrain_ds = iter(load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    ))
    chat_ds = iter(load_dataset(
        "lmsys/lmsys-chat-1m", split="train", streaming=True,
    ))

    bos = tokenizer.bos_token or ""

    while True:
        # FineWeb sample
        try:
            yield bos + next(pretrain_ds)["text"]
        except StopIteration:
            pretrain_ds = iter(load_dataset(
                "HuggingFaceFW/fineweb", name="sample-10BT",
                split="train", streaming=True,
            ))
            continue

        # LMSYS chat sample
        try:
            conv = next(chat_ds)["conversation"]
            yield tokenizer.apply_chat_template(conv, tokenize=False)
        except (StopIteration, Exception):
            chat_ds = iter(load_dataset(
                "lmsys/lmsys-chat-1m", split="train", streaming=True,
            ))
            continue


def load_fineweb_readout_task_data(
    task_name: str,
    tokenizer,
    model_name: str,
    n: int = 30000,
    max_context_tokens: int = 2000,
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    min_target_tokens: int = _READOUT_MIN_TARGET_TOKENS,
    max_target_tokens: int = _READOUT_MAX_TARGET_TOKENS,
    seed: int = 42,
) -> list[dict]:
    """Generate FutureLens/PastLens/Reconstruction variants from the HF-hosted FineWeb corpus."""
    from cot_utils import get_cot_positions, layer_percent_to_layer

    if task_name not in {"futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb"}:
        raise ValueError(f"Unsupported FineWeb readout task: {task_name}")

    rng = random.Random(seed)
    if layers is None:
        layers = [layer_percent_to_layer(model_name, p) for p in [25, 50, 75]]
    n_layers = len(layers)

    # Load invariant corpus from HF (cached on disk)
    corpus_texts = _load_fineweb_corpus(split="train")
    bos = tokenizer.bos_token or ""

    # Shuffle corpus indices deterministically
    indices = list(range(len(corpus_texts)))
    rng.shuffle(indices)

    print(f"  [{task_name}] Generating {n} examples from {len(corpus_texts)} corpus texts...")
    print(f"  [{task_name}] Target lengths use a heavy-tailed prior over {min_target_tokens}-{max_target_tokens} tokens")

    datapoints: list[dict] = []
    skipped = 0
    text_idx = 0

    while len(datapoints) < n:
        text = bos + corpus_texts[indices[text_idx % len(indices)]]
        text_idx += 1

        input_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_context_tokens + max_target_tokens)["input_ids"]
        total = len(input_ids)
        context_limit = min(max_context_tokens, total)

        if context_limit < min_target_tokens + 1:
            skipped += 1
            continue

        if task_name == "futurelens_fineweb":
            max_cutoff = min(context_limit - 1, total - min_target_tokens - 1)
            if max_cutoff < 0:
                skipped += 1
                continue
            cutoff = rng.randint(0, max_cutoff)
            k_target = _sample_heavy_tail_target_length(total - cutoff - 1, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = input_ids[:cutoff + 1]
            positions = get_cot_positions(0, len(context_ids), stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = input_ids[cutoff + 1: cutoff + 1 + k_target]
            datapoint_type = "fineweb_futurelens"
        elif task_name == "pastlens_fineweb":
            act_start_min = min_target_tokens
            act_start_max = context_limit - 1
            if act_start_max < act_start_min:
                skipped += 1
                continue
            act_start = rng.randint(act_start_min, act_start_max)
            k_target = _sample_heavy_tail_target_length(act_start, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            context_ids = input_ids[:context_limit]
            positions = get_cot_positions(act_start, len(context_ids), stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = input_ids[act_start - k_target: act_start]
            datapoint_type = "fineweb_pastlens"
        else:
            span_start_max = context_limit - min_target_tokens
            if span_start_max < 0:
                skipped += 1
                continue
            span_start = rng.randint(0, span_start_max)
            k_target = _sample_heavy_tail_target_length(context_limit - span_start, min_target_tokens=min_target_tokens, max_target_tokens=max_target_tokens, rng=rng)
            span_end = min(context_limit, span_start + k_target)
            context_ids = input_ids[:context_limit]
            positions = get_cot_positions(span_start, span_end, stride=stride, tokenizer=tokenizer, input_ids=context_ids, include_last=True)
            target_ids = context_ids[span_start:span_end]
            datapoint_type = "fineweb_reconstruction"

        if not positions:
            skipped += 1
            continue
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            skipped += 1
            continue

        all_positions = positions * n_layers
        datapoints.append({
            "datapoint_type": datapoint_type,
            "task": task_name,
            "prompt": _build_readout_prompt(task_name, len(target_ids), source="fineweb"),
            "target_response": target_text,
            "layer": layers[0],
            "layers": layers,
            "num_positions": len(all_positions),
            "context_input_ids": context_ids,
            "context_positions": all_positions,
        })

        if len(datapoints) % 10000 == 0:
            print(f"  [{task_name}] {len(datapoints)}/{n} examples...")

    if skipped > 0:
        print(f"  [{task_name}] Skipped {skipped} texts")
    print(f"  [{task_name}] Generated {len(datapoints)} examples")
    return datapoints[:n]


def load_fineweb_data(
    tokenizer,
    model_name: str,
    n: int = 50000,
    max_context_tokens: int = 2000,
    stride: int | str = "poisson",
    layers: list[int] | None = None,
    min_target_tokens: int = 5,
    max_target_tokens: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Generate PastLens-style context prediction from FineWeb + LMSYS streaming.

    Each example: tokenize web/chat text (max max_context_tokens), extract
    sampled activation positions, predict next or previous K tokens.
    Returns dicts compatible with dicts_to_training_data().

    Args:
        tokenizer: HuggingFace tokenizer.
        model_name: Model name for layer calculation.
        n: Number of examples to generate.
        max_context_tokens: Max tokens for activation context.
        stride: Position mode (same as CoT tasks; "punctuation" falls back to 5 here).
        layers: Layer indices for activation extraction.
        min_target_tokens: Minimum tokens to predict.
        max_target_tokens: Maximum tokens to predict.
        seed: Random seed for reproducibility.
    """
    from cot_utils import layer_percent_to_layer, sample_positions_in_span, _enforce_min_spacing, POISSON_MIN_SPACING

    rng = random.Random(seed)

    if layers is None:
        layers = [layer_percent_to_layer(model_name, p) for p in [25, 50, 75]]
    n_layers = len(layers)

    print(f"  [fineweb] Streaming FineWeb + LMSYS, generating {n} examples...")
    gen = _fineweb_text_generator(tokenizer)

    datapoints: list[dict] = []
    skipped = 0
    stride_step = 1 if stride == "poisson" else 5 if stride == "punctuation" else max(1, int(stride))
    min_context_tokens = max(3, stride_step * 3)

    while len(datapoints) < n:
        text = next(gen)

        # Tokenize with headroom for target tokens
        input_ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_context_tokens + max_target_tokens,
        )["input_ids"]
        L = len(input_ids)

        if L < min_context_tokens + min_target_tokens:
            skipped += 1
            continue

        k_target = rng.randint(
            min_target_tokens, min(max_target_tokens, max(min_target_tokens, L // 4)),
        )
        direction = rng.choice(["future", "past"])

        if direction == "future":
            # Context = tokens[:cutoff+1], target = tokens after cutoff
            max_cutoff = min(L - k_target - 1, max_context_tokens - 1)
            if max_cutoff < min_context_tokens:
                skipped += 1
                continue
            cutoff = rng.randint(min_context_tokens, max_cutoff)

            context_ids = input_ids[:cutoff + 1]
            if stride == "poisson":
                positions = sample_positions_in_span(0, len(context_ids) - 1, rng=rng)
                if POISSON_MIN_SPACING > 1:
                    positions = _enforce_min_spacing(positions, POISSON_MIN_SPACING)
            else:
                positions = list(range(0, len(context_ids), stride_step))
                if positions[-1] != len(context_ids) - 1:
                    positions.append(len(context_ids) - 1)

            target_ids = input_ids[cutoff + 1: cutoff + 1 + k_target]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            prompt = f"Predict the next {k_target} tokens of this text."

        else:  # past
            # Activations start at act_start; target = k_target tokens before it
            act_start_max = min(L - stride_step, max_context_tokens) - 1
            if act_start_max < k_target:
                skipped += 1
                continue
            act_start = rng.randint(k_target, act_start_max)

            # Context includes everything up to the end of the activation span
            context_end = min(act_start + max_context_tokens, L)
            context_ids = input_ids[:context_end]
            if stride == "poisson":
                positions = sample_positions_in_span(act_start, len(context_ids) - 1, rng=rng)
                if POISSON_MIN_SPACING > 1:
                    positions = _enforce_min_spacing(positions, POISSON_MIN_SPACING)
            else:
                positions = list(range(act_start, len(context_ids), stride_step))
                if positions[-1] != len(context_ids) - 1:
                    positions.append(len(context_ids) - 1)

            target_ids = input_ids[act_start - k_target: act_start]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            prompt = f"Predict the {k_target} tokens that came before the activation region."

        # Repeat positions for each layer (same format as CoT tasks)
        all_positions = positions * n_layers

        datapoints.append({
            "datapoint_type": "fineweb_context_prediction",
            "task": "fineweb",
            "prompt": prompt,
            "target_response": target_text,
            "layer": layers[0],
            "num_positions": len(all_positions),
            "context_input_ids": context_ids,
            "context_positions": all_positions,
        })

        if len(datapoints) % 10000 == 0:
            print(f"  [fineweb] {len(datapoints)}/{n} examples...")

    if skipped > 0:
        print(f"  [fineweb] Skipped {skipped} short texts")

    print(f"  [fineweb] Generated {len(datapoints)} examples")
    return datapoints[:n]


# ── Classification datasets (matching Adam's AO training) ──


_CLS_DATASETS = {
    "sst2": {
        "hf_repo": "stanfordnlp/sst2",
        "split": "train",
        "text_fn": lambda row: row["sentence"],
        "questions": [
            ("Is the sentiment of this text positive?", lambda row: "Yes" if row["label"] == 1 else "No"),
            ("Is the sentiment of this text negative?", lambda row: "Yes" if row["label"] == 0 else "No"),
        ],
    },
    "ag_news": {
        "hf_repo": "fancyzhx/ag_news",
        "split": "train",
        "text_fn": lambda row: row["text"],
        "questions": [
            ("Is this article about world news?", lambda row: "Yes" if row["label"] == 0 else "No"),
            ("Is this article about sports?", lambda row: "Yes" if row["label"] == 1 else "No"),
            ("Is this article about business?", lambda row: "Yes" if row["label"] == 2 else "No"),
            ("Is this article about science or technology?", lambda row: "Yes" if row["label"] == 3 else "No"),
        ],
    },
    "snli": {
        "hf_repo": "stanfordnlp/snli",
        "split": "train",
        "text_fn": lambda row: f"{row['premise']} {row['hypothesis']}",
        "questions": [
            ("Does the first statement entail the second?", lambda row: "Yes" if row["label"] == 0 else "No"),
        ],
        "filter_fn": lambda row: row["label"] in (0, 2),
    },
}


def load_classification_data(
    tokenizer,
    n: int = 100000,
    datasets: list[str] | None = None,
    layers: list[int] | None = None,
    stride: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Load classification training data from standard NLP datasets.

    Generates binary yes/no question-answer pairs, matching Adam's AO
    classification training. Context activations come from the raw text
    (no chat template), oracle answers questions about the content.
    """
    from datasets import load_dataset as hf_load_dataset

    if layers is None:
        layers = [9, 18, 27]
    if datasets is None:
        datasets = list(_CLS_DATASETS.keys())

    rng = random.Random(seed)
    n_layers = len(layers)
    per_ds = n // len(datasets)

    all_datapoints: list[dict] = []

    for ds_name in datasets:
        if ds_name not in _CLS_DATASETS:
            print(f"  [classification] WARNING: unknown dataset {ds_name!r}, skipping")
            continue

        cfg = _CLS_DATASETS[ds_name]
        print(f"  [classification] Loading {ds_name} from {cfg['hf_repo']}...")

        ds = hf_load_dataset(cfg["hf_repo"], split=cfg["split"])
        filter_fn = cfg.get("filter_fn")

        ds_points: list[dict] = []
        target_n = per_ds if ds_name != datasets[-1] else (n - len(all_datapoints))

        indices = list(range(len(ds)))
        rng.shuffle(indices)

        for idx in indices:
            if len(ds_points) >= target_n:
                break
            row = ds[idx]
            if filter_fn and not filter_fn(row):
                continue
            text = cfg["text_fn"](row)
            if not text or len(text.strip()) < 10:
                continue

            question_text, answer_fn = rng.choice(cfg["questions"])
            answer = answer_fn(row)

            input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
            if len(input_ids) < stride * 2:
                continue

            positions = list(range(0, len(input_ids), stride))
            if positions[-1] != len(input_ids) - 1:
                positions.append(len(input_ids) - 1)
            all_positions = positions * n_layers

            ds_points.append({
                "datapoint_type": f"classification_{ds_name}",
                "task": "classification",
                "prompt": f"Answer with 'Yes' or 'No' only. {question_text}",
                "target_response": answer,
                "layer": layers[0],
                "layers": layers,
                "num_positions": len(all_positions),
                "context_input_ids": input_ids,
                "context_positions": all_positions,
            })

        print(f"  [classification]   {ds_name}: {len(ds_points)} examples")
        all_datapoints.extend(ds_points)

    rng.shuffle(all_datapoints)
    print(f"  [classification] Total: {len(all_datapoints)} examples")
    return all_datapoints[:n]


# ── LatentQA / SPQA (matching Adam's AO training) ──


def load_latentqa_data(
    tokenizer,
    n: int = 65000,
    layers: list[int] | None = None,
    seed: int = 42,
    data_dir: str | None = None,
) -> list[dict]:
    """Load LatentQA training data from Adam's JSON files."""
    import sys

    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)
    rng = random.Random(seed)

    if data_dir is None:
        candidates = [
            Path("ao_reference/datasets/latentqa_datasets/train"),
            Path("/root/cot-oracle/ao_reference/datasets/latentqa_datasets/train"),
            Path("/root/activation_oracles/datasets/latentqa_datasets/train"),
        ]
        for p in candidates:
            if p.exists():
                data_dir = str(p)
                break
        if data_dir is None:
            raise FileNotFoundError(f"LatentQA data not found. Searched: {[str(p) for p in candidates]}")

    data_path = Path(data_dir)
    print(f"  [latentqa] Loading from {data_path}...")

    ao_ref = data_path.parent.parent.parent
    if str(ao_ref) not in sys.path:
        sys.path.insert(0, str(ao_ref))

    from nl_probes.dataset_classes.misc.latentqa_loader import DataPaths, load_latentqa_dataset

    paths = DataPaths(
        system=None,
        stimulus_completion=str(data_path / "stimulus_completion.json"),
        stimulus=str(data_path / "stimulus.json"),
        control=str(data_path / "control.json"),
        qa=str(data_path / "qa.json"),
    )
    ds = load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, add_thought_tokens=False, seed=seed)
    print(f"  [latentqa] Loaded {len(ds)} raw examples")

    masked_turn_count = {"stimulus_completion": 2, "stimulus": 2, "control": 0, "system": 0}

    datapoints: list[dict] = []
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        if len(datapoints) >= n:
            break
        item = ds[idx]
        read_prompt = item["read_prompt"]
        source = item["source"]

        add_gen_prompt = source != "stimulus_completion"
        context_str = tokenizer.apply_chat_template(read_prompt, tokenize=False, add_generation_prompt=add_gen_prompt, enable_thinking=False)
        context_ids = tokenizer.encode(context_str, add_special_tokens=False)
        if len(context_ids) < 5:
            continue

        num_masked = masked_turn_count.get(source, 0)
        if num_masked > 0:
            masked_turns = read_prompt[:num_masked]
            masked_str = tokenizer.apply_chat_template(masked_turns, tokenize=False, enable_thinking=False)
            masked_len = len(tokenizer.encode(masked_str, add_special_tokens=False))
        else:
            masked_len = 0

        positions = list(range(masked_len, len(context_ids)))
        if not positions:
            continue
        all_positions = positions * n_layers

        dialog = item["dialog"]
        datapoints.append({
            "datapoint_type": f"latentqa_{source}",
            "task": "latentqa",
            "prompt": dialog[0]["content"],
            "target_response": dialog[1]["content"],
            "layer": layers[0],
            "layers": layers,
            "num_positions": len(all_positions),
            "context_input_ids": context_ids,
            "context_positions": all_positions,
        })

    rng.shuffle(datapoints)
    print(f"  [latentqa] Generated {len(datapoints)} examples")
    return datapoints[:n]
