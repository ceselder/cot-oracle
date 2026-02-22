"""Persistent data cache for training data.

Avoids re-tokenizing + re-sampling on every run. Cache is invalidated when
any parameter that affects data generation changes (corpus path, task sizes,
model, layer config, stride config, etc.).
"""

import hashlib
import json
import os
import pickle
from pathlib import Path


def _cache_key(corpus_path, persona_corpus_path, task_sizes, model_name, **extra) -> str:
    """Deterministic hash of all params that affect generated data."""
    blob = json.dumps({
        "corpus": str(corpus_path),
        "corpus_mtime": os.path.getmtime(corpus_path) if corpus_path and Path(corpus_path).exists() else None,
        "persona_corpus": str(persona_corpus_path),
        "persona_mtime": os.path.getmtime(persona_corpus_path) if persona_corpus_path and Path(persona_corpus_path).exists() else None,
        "task_sizes": task_sizes,
        "model_name": model_name,
        **{k: v if not isinstance(v, list) else str(v) for k, v in sorted(extra.items())},
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_CACHE_DIR = Path(os.environ.get("COT_DATA_CACHE", "/ceph/scratch")) / os.environ.get("USER", "unknown") / "cot_data_cache"


def load_cached_data(corpus_path, persona_corpus_path, task_sizes, model_name, **extra):
    """Load cached (training_data, eval_datasets) if cache hit. Returns None on miss."""
    key = _cache_key(corpus_path, persona_corpus_path, task_sizes, model_name, **extra)
    cache_file = _CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"  Cache hit: {len(data[0])} training, {sum(len(v) for v in data[1].values())} eval")
        return data
    print(f"  No data cache found (key={key})")
    return None


def save_cached_data(training_data, eval_datasets, corpus_path, persona_corpus_path, task_sizes, model_name, **extra):
    """Save (training_data, eval_datasets) to persistent cache."""
    key = _cache_key(corpus_path, persona_corpus_path, task_sizes, model_name, **extra)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump((training_data, eval_datasets), f)
    print(f"Saved data cache to {cache_file} ({cache_file.stat().st_size / 1e6:.0f}MB)")
