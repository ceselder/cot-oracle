"""Helpers for locating the Activation Oracles (AO) repo.

The project depends on `nl_probes` from Adam Karvonen's AO repo. This helper
adds the repo to `sys.path` in a reproducible way instead of hard-coded
machine-specific paths.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


AO_REPO_ENV = "AO_REPO_PATH"


def _looks_like_ao_repo(path: Path) -> bool:
    return (path / "nl_probes").exists() and (path / "README.md").exists()


def ensure_ao_repo_on_path() -> Path | None:
    """Ensure `nl_probes` is importable.

    Search order:
    1. Already importable in current interpreter
    2. `AO_REPO_PATH` environment variable
    3. Local clone at `<repo>/activation_oracles`
    4. Historical paths used in prior scripts
    """
    if importlib.util.find_spec("nl_probes") is not None:
        return None

    repo_root = Path(__file__).resolve().parents[2]
    candidates = []

    env_path = os.environ.get(AO_REPO_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            repo_root / "activation_oracles",
            Path("/workspace/ao_reference"),
            Path("/home/celeste/Documents/side-projects/full-stack-ao/ao_reference"),
        ]
    )

    for candidate in candidates:
        if _looks_like_ao_repo(candidate):
            sys.path.insert(0, str(candidate))
            return candidate

    checked = "\n  - ".join(str(c) for c in candidates)
    raise RuntimeError(
        "Could not import `nl_probes`. Set AO_REPO_PATH to an Activation Oracles "
        "clone or clone it to ./activation_oracles.\n"
        f"Checked:\n  - {checked}"
    )
