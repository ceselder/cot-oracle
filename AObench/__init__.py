# AObench — Open-ended Activation Oracle benchmark
#
# Ported from Adam Karvonen's activation_oracles_dev repo:
# https://github.com/adamkarvonen/activation_oracles_dev/tree/main/nl_probes/open_ended_eval
#
# Original code by Adam Karvonen. Copied with permission for standalone use
# and modification in the cot-oracle project.

from pathlib import Path

AOBENCH_ROOT = Path(__file__).resolve().parent


def dataset_path(relative: str) -> str:
    """Resolve a dataset path relative to AObench/."""
    return str(AOBENCH_ROOT / relative)
