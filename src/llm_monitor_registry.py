from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
MONITOR_METHOD_KEYS = ("weak-llm", "strong-llm")


@lru_cache(maxsize=1)
def load_train_config() -> dict[str, Any]:
    return yaml.safe_load(TRAIN_CONFIG_PATH.read_text())


def get_llm_monitor_config(cfg: dict[str, Any] | None, method_key: str) -> dict[str, Any]:
    if method_key not in MONITOR_METHOD_KEYS:
        raise ValueError(f"Unknown LLM monitor method: {method_key}")
    train_baselines = load_train_config()["baselines"]
    if cfg is not None and "baselines" in cfg:
        baselines = cfg["baselines"]
        if method_key in baselines:
            return dict(baselines[method_key])
        if method_key == "weak-llm" and "llm_monitor" in baselines:
            return dict(baselines["llm_monitor"])
    return dict(train_baselines[method_key])


def get_llm_monitor_configs(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {method_key: get_llm_monitor_config(cfg, method_key) for method_key in MONITOR_METHOD_KEYS}
