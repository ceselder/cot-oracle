"""Core AO utilities used by training and eval scripts."""

from .ao import (
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    EarlyStopException,
    add_hook,
    batch_generate_cot,
    choose_attn_implementation,
    collect_activations,
    collect_activations_at_positions,
    find_sentence_boundary_positions,
    generate_cot,
    generate_direct_answer,
    get_hf_submodule,
    get_steering_hook,
    layer_percent_to_layer,
    load_extra_adapter,
    load_model_with_ao,
    run_oracle_on_activations,
    split_cot_into_sentences,
)

from .ao_repo import ensure_ao_repo_on_path

__all__ = [
    "AO_CHECKPOINTS",
    "SPECIAL_TOKEN",
    "EarlyStopException",
    "add_hook",
    "batch_generate_cot",
    "choose_attn_implementation",
    "collect_activations",
    "collect_activations_at_positions",
    "ensure_ao_repo_on_path",
    "find_sentence_boundary_positions",
    "generate_cot",
    "generate_direct_answer",
    "get_hf_submodule",
    "get_steering_hook",
    "layer_percent_to_layer",
    "load_extra_adapter",
    "load_model_with_ao",
    "run_oracle_on_activations",
    "split_cot_into_sentences",
]
