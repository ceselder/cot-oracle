"""Backward-compatible shim for the report module."""

from AObench.utils.report import *  # noqa: F401,F403
from AObench.utils.report import main as _main


if __name__ == "__main__":
    _main()
