"""Compatibility wrapper for the number-prediction eval entrypoint."""

from runpy import run_module


if __name__ == "__main__":
    run_module("AObench.open_ended_eval.number_prediction", run_name="__main__")
