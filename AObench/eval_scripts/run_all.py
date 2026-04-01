"""Compatibility wrapper for the AObench eval-suite entrypoint."""

from runpy import run_module


if __name__ == "__main__":
    run_module("AObench.open_ended_eval.run_all", run_name="__main__")

