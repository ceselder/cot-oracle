"""Parallel corpus tokenization — shared across all dataset loaders.

Tokenizes each corpus entry exactly once with multiprocessing, producing
(context_ids, formatted_len) that every loader needs. Avoids 7 redundant
tokenization passes over the same 32K-entry corpus.

Also provides parallel ensure_boundary_positions.
"""

import json
import multiprocessing as mp
import os
from pathlib import Path

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Worker state + functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

_worker_tokenizer = None


def _init_tokenizer_worker(model_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _worker_tokenizer
    from nl_probes.utils.common import load_tokenizer
    _worker_tokenizer = load_tokenizer(model_name)


def _tokenize_entry(entry):
    """Worker: tokenize a single corpus entry → (context_ids, formatted_len)."""
    tokenizer = _worker_tokenizer
    messages = [{"role": "user", "content": entry["question"]}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    cot_text = entry["cot_response"]
    think_end = cot_text.find("</think>")
    if think_end != -1:
        cot_text = cot_text[:think_end]
    context_ids = tokenizer(formatted + cot_text, add_special_tokens=False)["input_ids"]
    formatted_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    return context_ids, len(formatted_ids)


def _compute_boundary_for_entry(entry):
    """Worker: compute sentence boundary positions for one entry."""
    from cot_utils import split_cot_into_sentences, find_sentence_boundary_positions
    tokenizer = _worker_tokenizer

    sentences = entry.get("sentences") or split_cot_into_sentences(entry["cot_response"])
    if len(sentences) < 2:
        return None

    messages = [{"role": "user", "content": entry["question"]}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    full_text = formatted + entry["cot_response"]
    boundary_positions = find_sentence_boundary_positions(tokenizer, full_text, sentences)
    return {
        "boundary_positions": boundary_positions,
        "sentences": sentences,
        "n_sentences": len(sentences),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> list[dict]:
    """Read JSONL corpus into list of dicts."""
    entries = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def pretokenize_corpus(
    entries: list[dict],
    model_name: str,
    num_workers: int | None = None,
) -> list[dict]:
    """Tokenize all corpus entries in parallel, attaching _ctx_ids and _fmt_len.

    Returns the same entries list, mutated in-place with added fields.
    Loaders can check for '_ctx_ids' to skip their own tokenization.
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    # Skip if already tokenized
    if entries and "_ctx_ids" in entries[0]:
        return entries

    if num_workers > 1 and len(entries) > 100:
        print(f"  Pre-tokenizing {len(entries)} entries with {num_workers} workers...")
        with mp.Pool(num_workers, initializer=_init_tokenizer_worker, initargs=(model_name,)) as pool:
            results = pool.map(_tokenize_entry, entries, chunksize=64)
    else:
        from nl_probes.utils.common import load_tokenizer
        global _worker_tokenizer
        _worker_tokenizer = load_tokenizer(model_name)
        results = [_tokenize_entry(e) for e in tqdm(entries, desc="  tokenizing corpus", leave=False)]

    for entry, (ctx_ids, fmt_len) in zip(entries, results):
        entry["_ctx_ids"] = ctx_ids
        entry["_fmt_len"] = fmt_len

    print(f"  Pre-tokenized {len(entries)} entries")
    return entries


def ensure_boundary_positions(
    corpus_path: str,
    model_name: str,
    num_workers: int | None = None,
) -> list[dict]:
    """Ensure all corpus entries have boundary_positions. Parallel version.

    Reads corpus, computes missing boundaries in parallel, writes back.
    Returns the loaded (and enriched) entries list.
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    entries = load_corpus(corpus_path)
    to_update = [i for i, e in enumerate(entries) if not e.get("boundary_positions")]

    if not to_update:
        print(f"All {len(entries)} corpus entries already have boundary_positions")
        return entries

    print(f"Computing boundary_positions for {len(to_update)}/{len(entries)} entries...")
    entries_needing = [entries[i] for i in to_update]

    if num_workers > 1 and len(entries_needing) > 100:
        print(f"  Using {num_workers} workers...")
        with mp.Pool(num_workers, initializer=_init_tokenizer_worker, initargs=(model_name,)) as pool:
            results = pool.map(_compute_boundary_for_entry, entries_needing, chunksize=64)
    else:
        from nl_probes.utils.common import load_tokenizer
        global _worker_tokenizer
        _worker_tokenizer = load_tokenizer(model_name)
        results = [_compute_boundary_for_entry(e) for e in tqdm(entries_needing, desc="  boundaries", leave=False)]

    updated = 0
    for idx, result in zip(to_update, results):
        if result is not None:
            entries[idx].update(result)
            updated += 1

    # Write back
    with open(corpus_path, "w") as f:
        for entry in entries:
            # Don't write _ctx_ids/_fmt_len to disk
            clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            f.write(json.dumps(clean) + "\n")

    print(f"  Updated {updated} entries with boundary_positions")
    return entries
