"""
Layer sampling and prefix utilities for random-layer-subset training.

No ML dependencies — pure Python + stdlib.
"""

import random

SPECIAL_TOKEN = " ?"


def sample_num_layers(max_layers: int = 36, mean: int = 5) -> int:
    """Poisson(mean) truncated to [1, max_layers]."""
    # Knuth algorithm for Poisson sampling (no numpy needed)
    import math
    L = math.exp(-mean)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            break
    return max(1, min(max_layers, k - 1))


def sample_layers(max_layers: int = 36, mean: int = 5) -> list[int]:
    """Sample a random subset of layers, sorted ascending."""
    k = sample_num_layers(max_layers, mean)
    return sorted(random.sample(range(max_layers), k))


def build_random_layer_prefix(layers: list[int], num_positions_per_layer: int) -> str:
    """Build prefix like 'L5: ? ? ? ? L11: ? ? ? ?\\n'.

    Each layer block has num_positions_per_layer placeholder tokens.
    All use the same SPECIAL_TOKEN (' ?') since layer identity is
    conveyed by the 'L{n}:' label.
    """
    parts = []
    for layer in layers:
        block = f"L{layer}:" + SPECIAL_TOKEN * num_positions_per_layer
        parts.append(block)
    return " ".join(parts) + " \n"


def find_all_special_positions(
    token_ids: list[int],
    special_token_id: int,
    expected_count: int,
) -> list[int]:
    """Find all positions of special_token_id in token_ids.

    Unlike find_pattern_in_tokens, does NOT require consecutiveness —
    the 'L{n}:' labels between blocks break the consecutive run.
    """
    positions = [i for i, tid in enumerate(token_ids) if tid == special_token_id]
    # Target response may contain the special token; take only the first expected_count
    assert len(positions) >= expected_count, (
        f"Expected {expected_count} special tokens, found {len(positions)}"
    )
    return positions[:expected_count]


def layers_to_quartile_bin(layers: list[int], max_layers: int = 36) -> str:
    """Map a layer set to a 4-char quartile string like '1010'.

    Q1: layers 0..max_layers//4-1
    Q2: layers max_layers//4..max_layers//2-1
    Q3: layers max_layers//2..3*max_layers//4-1
    Q4: layers 3*max_layers//4..max_layers-1

    Each char is '1' if any layer falls in that quartile, '0' otherwise.
    """
    q_size = max_layers / 4
    bits = ['0', '0', '0', '0']
    for layer in layers:
        q = min(int(layer / q_size), 3)
        bits[q] = '1'
    return "".join(bits)
