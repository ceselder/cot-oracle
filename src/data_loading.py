"""
Unified data loading for the CoT Oracle.

Downloads precomputed JSONL from HuggingFace, caches locally, and returns
lists of dicts in the standard format expected by dicts_to_training_data().

Also provides load_fineweb_readout_data() for FineWeb readout tasks
(futurelens/pastlens/reconstruction on streaming web text).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Generator

from tasks import TASKS, TaskDef, get_trainable_tasks


_HF_CACHE_DIR = Path(os.environ.get("COT_ORACLE_CACHE_DIR", "data/hf_cache"))

# Default prompts for tasks whose HF data doesn't include a 'prompt' field
_DEFAULT_PROMPTS: dict[str, str] = {
    "truthfulqa_hint_verbalized": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "truthfulqa_hint": "Did the model use an external hint in its reasoning? Describe the hint usage and its effect.",
    "sycophancy": "Was the model's moral judgment independently reasoned or influenced by the user's stated opinion?",
    "probe_sycophancy": "Is there sycophancy signal at this point in the reasoning?",
    "sentence_insertion": "Which step (if any) was inserted from a different problem's reasoning? Answer with the step number or 'none'.",
}


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
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line[0] != '{':
                raise ValueError(f"Corrupt JSONL at {local_path}:{i+1} — starts with {repr(line[:20])}")
            item = json.loads(line)
            # Normalize: ensure 'task' field matches the task we're loading
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
            # Map target_output → target_response (Adam's AO format, e.g. SQA)
            if "target_response" not in item and "target_output" in item:
                item["target_response"] = item["target_output"]
            # Map raw_dialog → prompt for SQA (dialog[0] is the question)
            if "prompt" not in item and "raw_dialog" in item:
                dialog = item["raw_dialog"]
                if dialog and isinstance(dialog, list) and len(dialog) > 0:
                    item["prompt"] = dialog[0].get("content", dialog[0].get("content_0", ""))
            # Map label → target_response for reasoning_termination new format
            if "target_response" not in item and "label" in item:
                label_map = {"terminates": "will_terminate", "continues": "will_continue"}
                item["target_response"] = label_map.get(item["label"], item["label"])
            # Map hinted_prompt → question for truthfulqa datasets (used by prepare_context_ids)
            if "hinted_prompt" in item and "question" not in item:
                item["question"] = item["hinted_prompt"]
            # Inject default prompt if missing (e.g. truthfulqa eval datasets)
            if "prompt" not in item or item["prompt"] is None:
                item["prompt"] = _default_prompt(task_name)
            # Map cot_field → cot_text for tasks that use a different field
            # (e.g. chunked_convqa/compqa use cot_prefix for activations)
            if task_def.cot_field != "cot_text" and task_def.cot_field in item:
                item["cot_text"] = item[task_def.cot_field]
            data.append(item)

    if n is not None and len(data) > n:
        if shuffle:
            random.shuffle(data)
        data = data[:n]

    return data


def load_all_training_data(task_config: dict[str, dict]) -> list[dict]:
    """Load train splits for all enabled tasks.

    Args:
        task_config: Maps task_name -> {"n": int, ...}.
            Set n: 0 to disable a task.

    Returns:
        Combined list of dicts from all enabled tasks, shuffled.
    """
    trainable = get_trainable_tasks()
    all_data: list[dict] = []

    for task_name, cfg in task_config.items():
        n = cfg.get("n", 0)
        if n == 0:
            continue

        if task_name not in trainable:
            raise ValueError(
                f"Task {task_name!r} is not trainable. "
                f"Trainable tasks: {sorted(trainable.keys())}"
            )

        effective_n = None if n == -1 else n
        print(f"  [data] Loading {task_name} (n={n})...")
        items = load_task_data(task_name, split="train", n=None)  # load all first

        if not items:
            raise RuntimeError(
                f"Task {task_name!r} is enabled (n={n}) but returned 0 items. "
                f"Check HF repo: {TASKS[task_name].hf_repo}"
            )

        # Epoch (repeat) if requested n > available, then truncate
        if effective_n is not None and len(items) < effective_n:
            repeats = (effective_n + len(items) - 1) // len(items)
            print(f"  [data]   {len(items)} available, repeating {repeats}x to reach {effective_n}")
            items = (items * repeats)[:effective_n]
        elif effective_n is not None:
            items = items[:effective_n]

        print(f"  [data]   -> {len(items)} examples")
        all_data.extend(items)

    random.shuffle(all_data)
    if all_data:
        print(f"  [data] Total task examples: {len(all_data)}")
    return all_data


def _tok_cache_fingerprint(items_to_tok: list[dict], tokenizer, layers: list[int]) -> str:
    """Build a hex digest that changes when the tokenization inputs change."""
    import hashlib
    h = hashlib.sha256()
    h.update(tokenizer.name_or_path.encode())
    h.update(str(layers).encode())
    h.update(str(len(items_to_tok)).encode())
    # Sample a few items for content fingerprint (hashing all is slow for 100K+)
    step = max(1, len(items_to_tok) // 200)
    for i in range(0, len(items_to_tok), step):
        item = items_to_tok[i]
        h.update((item.get("cot_text", "") + item.get("question", "") + item.get("hinted_prompt", "")).encode())
    return h.hexdigest()[:16]


def prepare_context_ids(
    items: list[dict],
    tokenizer,
    layers: list[int] | None = None,
    **_kwargs,
) -> list[dict]:
    """Compute context_input_ids and context_positions for items that need them.

    Skips items that already have context_input_ids (e.g. futurelens, fineweb
    which compute them in-memory during generation). For all others, builds
    the chat-templated input from cot_text + question/hinted_prompt, tokenizes
    it, and stores ALL token positions in the CoT region (stochastic sampler
    handles subsampling at training time).

    Results are cached to disk so subsequent runs skip tokenization.

    Args:
        items: Raw dicts from load_task_data().
        tokenizer: HuggingFace tokenizer.
        layers: Layer indices (positions are repeated per layer).

    Returns:
        Same list, mutated in place (items gain context_input_ids,
        context_positions, num_positions, layer fields).
    """
    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)

    # Count how many items need tokenization
    need_tokenize = [i for i, item in enumerate(items) if not item.get("context_input_ids")]
    if not need_tokenize:
        return items

    # Try loading from disk cache
    from filelock import FileLock
    items_to_tok = [items[i] for i in need_tokenize]
    fingerprint = _tok_cache_fingerprint(items_to_tok, tokenizer, layers)
    cache_dir = _HF_CACHE_DIR / "_tok_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fingerprint}.json"
    lock_path = cache_dir / f"{fingerprint}.json.lock"

    with FileLock(lock_path, timeout=600):
        if cache_path.exists():
            print(f"  [data] Loading tokenization cache ({len(need_tokenize)} items) from {cache_path.name}...")
            with open(cache_path) as f:
                cached = json.load(f)
            assert len(cached) == len(need_tokenize), (
                f"Cache size mismatch: {len(cached)} vs {len(need_tokenize)}"
            )
            for idx, entry in zip(need_tokenize, cached):
                items[idx]["context_input_ids"] = entry["context_input_ids"]
                items[idx]["context_positions"] = entry["context_positions"]
                items[idx]["num_positions"] = entry["num_positions"]
                items[idx]["layer"] = entry["layer"]
            print(f"  [data] Loaded tokenization cache for {len(cached)} items")
            return items

        # Cache miss — tokenize and save
        print(f"  [data] Tokenizing cot_text for {len(need_tokenize)}/{len(items)} items "
              f"(all positions, {n_layers} layers)...")

        cache_entries = []
        prepared = 0
        _log_interval = max(1, len(need_tokenize) // 20)
        for count, idx in enumerate(need_tokenize):
            item = items[idx]
            cot_text = item.get("cot_text", "")
            if not cot_text:
                raise ValueError(
                    f"Item missing both context_input_ids and cot_text (task={item.get('task', '?')}). "
                    f"Every training item needs one or the other."
                )

            user_msg = item.get("hinted_prompt") or item.get("question", "")

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

            cot_positions = list(range(prompt_len, len(full_ids)))
            if not cot_positions:
                cache_entries.append(None)
                continue

            all_positions = cot_positions * n_layers

            item["context_input_ids"] = full_ids
            item["context_positions"] = all_positions
            item["num_positions"] = len(all_positions)
            item["layer"] = layers[0]

            cache_entries.append({
                "context_input_ids": full_ids,
                "context_positions": all_positions,
                "num_positions": len(all_positions),
                "layer": layers[0],
            })
            prepared += 1

            if (count + 1) % _log_interval == 0:
                print(f"  [data]   tokenized {count + 1}/{len(need_tokenize)} "
                      f"({100 * (count + 1) / len(need_tokenize):.0f}%)")

        if prepared > 0:
            print(f"  [data] Tokenized cot_text → context_input_ids for {prepared} items "
                  f"(all positions, {n_layers} layers)")

        # Save cache (fill None entries for skipped items)
        for i, entry in enumerate(cache_entries):
            if entry is None:
                cache_entries[i] = {"context_input_ids": [], "context_positions": [], "num_positions": 0, "layer": layers[0]}
        with open(cache_path, "w") as f:
            json.dump(cache_entries, f)
        print(f"  [data] Saved tokenization cache to {cache_path.name}")

    return items


def _download_from_hf(task_def: TaskDef, split: str) -> Path:
    """Download a split from the task's HF repo, cache locally. Return local path.

    Uses a filelock to prevent multi-rank races on NFS where concurrent
    downloads corrupt the file (null bytes from overlapping writes).
    """
    from filelock import FileLock

    cache_dir = _HF_CACHE_DIR / task_def.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{split}.jsonl"
    lock_path = cache_dir / f"{split}.jsonl.lock"

    # Fast path: file exists and starts with '{' (not corrupted)
    if local_path.exists() and local_path.stat().st_size > 0:
        with open(local_path, "rb") as f:
            if f.read(1) == b"{":
                return local_path
        # Corrupted (e.g. null bytes from NFS race) — will re-download under lock
        print(f"  [data] Corrupt cache detected for {task_def.name}/{split}, re-downloading...")
        local_path.unlink()

    with FileLock(lock_path, timeout=600):
        # Re-check after acquiring lock (another rank may have downloaded)
        if local_path.exists() and local_path.stat().st_size > 0:
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

    # Try requested split, then common alternatives
    split_alternatives = {
        "train": ["train"],
        "test": ["test", "eval"],
    }
    alts = split_alternatives.get(split, [split])
    ds = None
    for alt_split in alts:
        try:
            ds = load_dataset(task_def.hf_repo, split=alt_split)
            if alt_split != split:
                print(f"  [data] Using split '{alt_split}' (requested '{split}')")
            break
        except (ValueError, KeyError):
            continue
    # For train, try combining id_train + ood_train (gives more data than either alone)
    if ds is None and split == "train":
        try:
            ds_id = load_dataset(task_def.hf_repo, split="id_train")
            ds_ood = load_dataset(task_def.hf_repo, split="ood_train")
            from datasets import concatenate_datasets
            ds = concatenate_datasets([ds_id, ds_ood])
            print(f"  [data] Combined id_train ({len(ds_id)}) + ood_train ({len(ds_ood)}) = {len(ds)}")
        except Exception:
            pass
    # Last resort: try id_train or ood_train alone
    if ds is None and split == "train":
        for alt_split in ["id_train", "ood_train"]:
            try:
                ds = load_dataset(task_def.hf_repo, split=alt_split)
                print(f"  [data] Using split '{alt_split}' (requested '{split}')")
                break
            except (ValueError, KeyError):
                continue
    if ds is None:
        raise ValueError(f"No suitable split found for {task_def.hf_repo} (tried {alts})")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")
    tmp_path.rename(local_path)


# ── FutureLens (next-token prediction from corpus) ──


def load_futurelens_data(
    tokenizer,
    n: int = 30000,
    split: str = "train",
    predict_tokens: int = 50,
    layers: list[int] | None = None,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """Generate FutureLens training/eval data from corpus-v5.

    For each corpus entry, picks random cutoff positions in the CoT and
    constructs examples where the oracle sees a single activation at that
    position and must predict the next ~50 tokens of reasoning.

    Args:
        tokenizer: HuggingFace tokenizer.
        n: Number of examples to generate.
        split: "train" (first 80% of corpus) or "test" (last 20%).
        predict_tokens: How many tokens to predict after cutoff.
        layers: Layer indices for activation extraction.
        seed: Random seed.

    Returns:
        List of dicts with context_input_ids, context_positions, etc.
    """
    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)

    rng = random.Random(seed)

    # Download corpus-v5 from HF
    corpus = _load_corpus_v5(split)
    if not corpus:
        raise RuntimeError(f"No entries found in corpus-v5 ({split} split)")

    print(f"  [futurelens] Loaded {len(corpus)} corpus entries ({split} split)")
    print(f"  [futurelens] Generating {n} examples (predict_tokens={predict_tokens})...")

    datapoints: list[dict] = []
    attempts = 0
    max_attempts = n * 5

    while len(datapoints) < n and attempts < max_attempts:
        attempts += 1
        entry = rng.choice(corpus)

        cot_text = entry.get("cot_response", "")
        if not cot_text:
            continue

        # Strip <think> tags
        think_end = cot_text.find("</think>")
        if think_end != -1:
            cot_text = cot_text[:think_end]
        cot_text = cot_text.replace("<think>", "").strip()
        if not cot_text:
            continue

        question = entry.get("question", "")

        # Tokenize question + CoT
        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        full_text = prompt_text + cot_text
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # All positions in CoT region
        cot_positions = list(range(prompt_len, len(full_ids)))
        if len(cot_positions) < 3:
            continue

        # Pick up to 3 random cutoff positions per entry
        max_k = len(cot_positions) - 1
        n_picks = min(3, max_k + 1)

        for _ in range(n_picks):
            if len(datapoints) >= n:
                break

            k = rng.randint(0, max_k)
            cutoff_pos = cot_positions[k]

            # Target: next predict_tokens tokens after cutoff
            target_start = cutoff_pos + 1
            target_end = min(target_start + predict_tokens, len(full_ids))
            if target_end - target_start < 5:
                continue

            target_ids = full_ids[target_start:target_end]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            if not target_text.strip():
                continue

            # Single activation at cutoff, repeated for each layer
            context_positions = [cutoff_pos] * n_layers

            # Context: tokens up to cutoff position
            context_slice = full_ids[:cutoff_pos + 1]

            layers_str = ", ".join(str(l) for l in layers)
            prompt = (
                f"Activations from {n_layers} positions across layers {layers_str}. "
                f"Predict the next {target_end - target_start} tokens of reasoning."
            )

            datapoints.append({
                "datapoint_type": "cot_next_step",
                "task": "futurelens",
                "prompt": prompt,
                "target_response": target_text,
                "layer": layers[0],
                "layers": layers,
                "num_positions": len(context_positions),
                "context_input_ids": context_slice,
                "context_positions": context_positions,
            })

        if len(datapoints) % 10000 == 0 and len(datapoints) > 0:
            print(f"  [futurelens] {len(datapoints)}/{n} examples...")

    print(f"  [futurelens] Generated {len(datapoints)} examples "
          f"(from {attempts} attempts, {len(corpus)} corpus entries)")
    return datapoints[:n]


def _load_corpus_v5(split: str = "train") -> list[dict]:
    """Load corpus-v5 from HF and split 80/20 for train/test.

    Corpus-v5 has only a train split on HF, so we deterministically
    split by entry index (seed=42).
    """
    from tasks import TASKS

    task_def = TASKS["futurelens"]
    cache_dir = _HF_CACHE_DIR / "futurelens_corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "corpus.jsonl"

    if not local_path.exists():
        print(f"  [futurelens] Downloading corpus from {task_def.hf_repo}...")
        try:
            from datasets import load_dataset
            ds = load_dataset(task_def.hf_repo, split="train")
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


# ── FineWeb readout tasks (futurelens/pastlens/reconstruction) ──


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


def _sample_heavy_tail_target_length(
    available: int, min_len: int = 5, max_len: int = 25,
) -> int:
    """Log-uniform sampling biased toward shorter targets.

    Produces lengths in [min_len, min(max_len, available)] with a heavy
    tail — most examples are short (5-10 tokens) but some reach 25.
    """
    import math as _math
    upper = min(max_len, available)
    if upper <= min_len:
        return min_len
    # log-uniform: exp(uniform(log(min), log(upper)))
    log_min = _math.log(min_len)
    log_max = _math.log(upper)
    return int(_math.exp(random.uniform(log_min, log_max)))


_FINEWEB_READOUT_PROMPTS: dict[str, str] = {
    "futurelens_fineweb": "Predict the next {n} tokens of this text.",
    "pastlens_fineweb": "Predict the {n} tokens that came before the activation region.",
    "reconstruction_fineweb": "Reconstruct the {n} tokens at the activation positions.",
}


def load_fineweb_readout_data(
    tokenizer,
    n: int = 15000,
    max_context_tokens: int = 2000,
    layers: list[int] | None = None,
    min_target_tokens: int = 5,
    max_target_tokens: int = 25,
    seed: int = 42,
    variant: str | None = None,
    **_kwargs,
) -> list[dict]:
    """Generate FineWeb readout data with 3 task variants.

    Single-position approach (like Adam's AO): pick one index in the text,
    feed that single activation across all layers, predict around it.

      - futurelens_fineweb: activation at pos, target = next K tokens after pos
      - pastlens_fineweb: activation at pos, target = K tokens before pos
      - reconstruction_fineweb: activation at midpoint of span, target = span tokens

    Args:
        tokenizer: HuggingFace tokenizer.
        n: Total number of examples (split ~equally across 3 variants, or all
           one variant if `variant` is specified).
        max_context_tokens: Max tokens for activation context.
        layers: Layer indices for activation extraction.
        min_target_tokens: Minimum target length.
        max_target_tokens: Maximum target length.
        seed: Random seed.
        variant: If set, only generate this variant (for eval). One of
            "futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb".
    """
    if layers is None:
        layers = [9, 18, 27]
    n_layers = len(layers)

    rng = random.Random(seed)

    all_variants = ["futurelens_fineweb", "pastlens_fineweb", "reconstruction_fineweb"]
    if variant is not None:
        # Support comma-separated string or single variant
        if isinstance(variant, str):
            requested = [v.strip() for v in variant.split(",")]
        else:
            requested = list(variant)
        for v in requested:
            assert v in all_variants, f"Unknown variant: {v}"
        variants = requested
    else:
        variants = all_variants

    print(f"  [fineweb-readout] Streaming FineWeb + LMSYS, generating {n} examples "
          f"({', '.join(variants)})...")
    gen = _fineweb_text_generator(tokenizer)

    datapoints: list[dict] = []
    skipped = 0
    min_pos = 10  # need at least 10 tokens before the activation position

    while len(datapoints) < n:
        text = next(gen)

        input_ids = tokenizer(
            text, add_special_tokens=False, truncation=True,
            max_length=max_context_tokens + max_target_tokens,
        )["input_ids"]
        L = len(input_ids)

        if L < min_pos + min_target_tokens:
            skipped += 1
            continue

        task_variant = rng.choice(variants)
        available = max(min_target_tokens, L // 4)
        k_target = _sample_heavy_tail_target_length(available, min_target_tokens, max_target_tokens)

        if task_variant == "futurelens_fineweb":
            # Single activation at pos, predict next k_target tokens
            max_pos = min(L - k_target - 1, max_context_tokens - 1)
            if max_pos < min_pos:
                skipped += 1
                continue
            pos = rng.randint(min_pos, max_pos)

            context_ids = input_ids[:pos + 1]
            target_ids = input_ids[pos + 1: pos + 1 + k_target]

        elif task_variant == "pastlens_fineweb":
            # Single activation at pos, predict k_target tokens before pos
            if L - 1 < k_target + min_pos:
                skipped += 1
                continue
            pos = rng.randint(k_target + min_pos, min(L - 1, max_context_tokens - 1))

            context_ids = input_ids[:pos + 1]
            target_ids = input_ids[pos - k_target: pos]

        else:  # reconstruction_fineweb
            # Single activation at midpoint of span, predict span tokens
            max_span_start = L - k_target - 1
            if max_span_start < 0:
                skipped += 1
                continue
            span_start = rng.randint(0, max_span_start)
            span_end = span_start + k_target
            pos = (span_start + span_end) // 2  # midpoint

            context_ids = input_ids[:max(pos + 1, span_end)]
            target_ids = input_ids[span_start:span_end]

        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if not target_text.strip():
            skipped += 1
            continue

        prompt = _FINEWEB_READOUT_PROMPTS[task_variant].format(n=len(target_ids))

        # Single position repeated for each layer
        context_positions = [pos] * n_layers

        datapoints.append({
            "datapoint_type": f"fineweb_{task_variant}",
            "task": task_variant,
            "prompt": prompt,
            "target_response": target_text,
            "layer": layers[0],
            "layers": layers,
            "num_positions": len(context_positions),
            "context_input_ids": context_ids,
            "context_positions": context_positions,
        })

        if len(datapoints) % 5000 == 0 and len(datapoints) > 0:
            print(f"  [fineweb-readout] {len(datapoints)}/{n} examples...")

    if skipped > 0:
        print(f"  [fineweb-readout] Skipped {skipped} short/empty texts")

    print(f"  [fineweb-readout] Generated {len(datapoints)} examples")
    return datapoints[:n]


# ── Classification datasets (matching Adam's AO training) ──


# Dataset configs: each entry defines how to load and question a dataset
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
        "filter_fn": lambda row: row["label"] in (0, 2),  # entailment + contradiction only
    },
}


def load_classification_data(
    tokenizer,
    n: int = 100000,
    datasets: list[str] | None = None,
    layers: list[int] | None = None,
    seed: int = 42,
    **_kwargs,
) -> list[dict]:
    """Load classification training data from standard NLP datasets.

    Generates binary yes/no question-answer pairs, matching Adam's AO
    classification training. Context activations come from the raw text
    (no chat template), oracle answers questions about the content.

    Args:
        tokenizer: HuggingFace tokenizer.
        n: Total number of examples to generate.
        datasets: Which classification datasets to use (default: all).
        layers: Layer indices for activation extraction.
        seed: Random seed.

    Returns:
        List of dicts compatible with dicts_to_training_data().
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

            # Pick a random question template
            question_text, answer_fn = rng.choice(cfg["questions"])
            answer = answer_fn(row)

            # Tokenize raw text (no chat template — activations from raw text processing)
            input_ids = tokenizer.encode(
                text, add_special_tokens=True, truncation=True, max_length=512,
            )

            if len(input_ids) < 5:
                continue

            # All token positions
            positions = list(range(len(input_ids)))

            # Repeat for each layer
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
