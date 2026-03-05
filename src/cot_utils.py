"""
Lightweight CoT utilities (no peft/torch dependency).

Extracted from ao_lib.py for use in CPU-only scripts like corpus generation.
"""

import bisect
import math
import random
import re
from functools import lru_cache


def split_cot_into_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences. Removes <think> tags first."""
    text = re.sub(r'<think>|</think>', '', cot_text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def find_sentence_boundary_positions(
    tokenizer,
    formatted_text: str,
    sentences: list[str],
) -> list[int]:
    """
    Find the token positions of the last token of each sentence in the formatted text.
    Returns a list of token positions (indices into the tokenized sequence).

    Uses a single pass to build char→token mapping instead of per-token decode.
    """
    tokens = tokenizer(formatted_text, add_special_tokens=False)
    token_ids = tokens["input_ids"]
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids[0].tolist() if len(token_ids.shape) > 1 else token_ids.tolist()
    elif isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    full_text_decoded = tokenizer.decode(token_ids)

    # Build cumulative char offsets in ONE pass (the old code re-scanned per sentence)
    cum_chars = []
    char_count = 0
    for t_id in token_ids:
        char_count += len(tokenizer.decode([t_id]))
        cum_chars.append(char_count)

    positions = []
    search_start = 0

    for sentence in sentences:
        anchor = sentence[-20:] if len(sentence) > 20 else sentence
        idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            anchor = sentence[-10:] if len(sentence) > 10 else sentence
            idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            continue

        char_end = idx + len(anchor)
        search_start = char_end

        # Binary search for the token whose cumulative chars >= char_end
        token_pos = bisect.bisect_left(cum_chars, char_end)
        if token_pos < len(token_ids):
            positions.append(token_pos)

    return positions


PUNCTUATION_CHARS = frozenset(".,;:?!")


def get_cot_punctuation_positions(
    prompt_token_count: int,
    total_token_count: int,
    tokenizer,
    input_ids: list[int],
    fallback_stride: int = 5,
    include_last: bool = True,
) -> list[int]:
    """Get positions of punctuation tokens within the CoT region.

    Decodes each token in the CoT region and checks if the decoded text ends
    with a punctuation character (. , ; : ? !).  Since Qwen3-8B SentencePiece
    often merges punctuation with surrounding text (e.g. " sentence." is one
    token), checking the *last character* of the decoded string is the right
    heuristic.

    Falls back to stride-based positions if fewer than 2 punctuation positions
    are found.
    """
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1

    if cot_end - cot_start + 1 < 2:
        return []

    positions: list[int] = []
    for pos in range(cot_start, cot_end + 1):
        token_id = input_ids[pos]
        decoded = tokenizer.decode([token_id])
        if decoded and decoded.rstrip().endswith(tuple(PUNCTUATION_CHARS)):
            positions.append(pos)

    # Optionally include the very last CoT token
    if include_last and positions and positions[-1] != cot_end:
        positions.append(cot_end)

    # Deduplicate while preserving order
    deduped: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    positions = deduped

    # Fall back to stride-based if too few punctuation positions
    if len(positions) < 2:
        return get_cot_stride_positions(
            prompt_token_count, total_token_count, stride=fallback_stride,
            include_last=include_last,
        )

    return positions


def get_cot_stride_positions(
    prompt_token_count: int,
    total_token_count: int,
    stride: int,
    include_last: bool = True,
) -> list[int]:
    """Get fixed-stride positions over the CoT token region.

    Positions start right after the prompt tokens and proceed every `stride` tokens.
    Optionally includes the last token to keep late-CoT signal.
    """
    cot_start = max(0, prompt_token_count)
    cot_end = total_token_count - 1

    if cot_end - cot_start + 1 < 2:
        return []

    positions = list(range(cot_start, cot_end + 1, max(1, stride)))
    if include_last and positions and positions[-1] != cot_end:
        positions.append(cot_end)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    positions = deduped

    if len(positions) < 2:
        return [cot_start, cot_end]

    return positions


def get_cot_positions(
    prompt_token_count: int,
    total_token_count: int,
    stride: int | str,
    tokenizer=None,
    input_ids: list[int] | None = None,
    include_last: bool = True,
) -> list[int]:
    """Unified position dispatcher: stride-based or punctuation-based.

    Args:
        stride: int for fixed-stride, or "punctuation" for punctuation boundaries.
                No default — must be explicitly provided.
        tokenizer: Required when stride="punctuation".
        input_ids: Required when stride="punctuation".
    """
    if stride is None:
        raise ValueError(
            "stride must be explicitly set (int or 'punctuation'). "
            "Check config activations.stride or --stride CLI flag."
        )
    if stride == "punctuation":
        assert tokenizer is not None and input_ids is not None, (
            "tokenizer and input_ids required for punctuation mode"
        )
        return get_cot_punctuation_positions(
            prompt_token_count, total_token_count,
            tokenizer, input_ids,
            include_last=include_last,
        )
    return get_cot_stride_positions(
        prompt_token_count, total_token_count,
        stride=int(stride), include_last=include_last,
    )


def sample_poisson_positions(
    base_positions: list[int],
    rng: random.Random | None = None,
    max_k: int = 100,
    include_boundaries: bool = True,
) -> list[int]:
    """Graduated position subsampling with Poisson-process tail.

    Schedule:
        20% → last position only
        15% → last 2 positions
        15% → last 3 positions
        50% → Poisson-process (log-uniform k, iid draws, always include first AND last)

    When include_boundaries=True (default), the sparse branches (last-1/2/3)
    also include the first position.
    """
    sampler = rng or random
    K = len(base_positions)
    if K <= 1:
        return base_positions[-1:]
    r = sampler.random()
    if r < 0.20:
        # Last position only
        if include_boundaries and K >= 2:
            return sorted({base_positions[0], base_positions[-1]})
        return [base_positions[-1]]
    elif r < 0.35:
        # Last 2 positions
        picked = set(base_positions[-2:])
        if include_boundaries:
            picked.add(base_positions[0])
        return sorted(picked)
    elif r < 0.50:
        # Last 3 positions
        picked = set(base_positions[-3:])
        if include_boundaries:
            picked.add(base_positions[0])
        return sorted(picked)
    # 50%: Poisson process — log-uniform k, always include first and last
    lo, hi = 2, min(max_k, K)
    k = int(round(math.exp(sampler.uniform(math.log(lo), math.log(hi)))))
    k = max(2, min(k, K))
    picked = set(sampler.sample(base_positions, k))
    if include_boundaries:
        picked.add(base_positions[0])
    picked.add(base_positions[-1])
    return sorted(picked)


@lru_cache(maxsize=512)
def _zipf_cdf(K: int, target_mean_10x: int) -> tuple[float, tuple[float, ...]]:
    """Solve for Zipf exponent s and build CDF for sampling k ~ Zipf(s) on {1,...,K}.

    P(k) ∝ k^(-s).  Monotonically falling for s > 0, heavy-tailed (power law).
    target_mean_10x = 10 * target_mean (int for lru_cache hashability).

    Returns (s, cumulative_cdf_tuple).
    """
    target_mean = target_mean_10x / 10.0

    def _mean(s: float) -> float:
        num = sum(k ** (1 - s) for k in range(1, K + 1))
        denom = sum(k ** (-s) for k in range(1, K + 1))
        return num / denom

    # Bisection: s=0 → uniform (mean=(K+1)/2); s→∞ → mean→1
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _mean(mid) > target_mean:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2

    # Build CDF for inverse-CDF sampling
    weights = [k ** (-s) for k in range(1, K + 1)]
    total = sum(weights)
    cdf: list[float] = []
    cumsum = 0.0
    for w in weights:
        cumsum += w / total
        cdf.append(cumsum)

    return s, tuple(cdf)


def _sample_from_cdf(cdf: tuple[float, ...], sampler: random.Random) -> int:
    """Sample from discrete distribution via inverse-CDF (binary search). Returns 1-indexed."""
    u = sampler.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo + 1


# β such that P(Beta(β,1) > 0.9) = 0.7  ⇒  0.9^β = 0.3  → ~70% of ticks in trailing 10%
END_CONCENTRATION = -math.log(10 / 3) / math.log(0.9)  # ≈ 11.43

MAX_ENDWEIGHTED_K = 500


def sample_endweighted_positions(
    base_positions: list[int],
    rng: random.Random | None = None,
    target_mean_k: int = 50,
    end_concentration: float = END_CONCENTRATION,
) -> list[int]:
    """End-weighted stochastic position sampling.

    1. Draw k ~ Zipf(s) on {1,...,min(K, MAX_ENDWEIGHTED_K)} with mean ≈ target_mean_k.
       Discrete power law: P(k) ∝ k^(-s). Monotonically falling, heavy-tailed.
    2. Select k positions via Gumbel-top-k weighted by Beta(end_concentration, 1),
       concentrating ~70% of selected positions in the trailing 10% of the CoT.
    3. Always includes the last position.
    """
    sampler = rng or random
    K = len(base_positions)
    if K <= 1:
        return list(base_positions)

    # --- Draw k from Zipf (capped at MAX_ENDWEIGHTED_K) ---
    K_eff = min(K, MAX_ENDWEIGHTED_K)
    max_falling_mean = (K_eff + 1) / 2
    effective_target = min(float(target_mean_k), max_falling_mean - 0.5)
    if effective_target < 1.5:
        return list(base_positions)

    _, cdf = _zipf_cdf(K_eff, int(effective_target * 10))
    k = _sample_from_cdf(cdf, sampler)

    if k >= K:
        return list(base_positions)

    # --- Select positions via Gumbel-top-k ---
    # Weight profile: Beta(β, 1) → w_i ∝ ((i+0.5)/K)^(β-1), concentrated at end
    beta_m1 = end_concentration - 1
    log_weights = [beta_m1 * math.log((i + 0.5) / K) for i in range(K)]

    # Gumbel-max trick: key_i = log(w_i) + Gumbel(0,1), take top-k
    keys = [lw - math.log(-math.log(max(sampler.random(), 1e-15))) for lw in log_weights]

    # Always include last position; pick top-(k-1) from rest
    selected = {K - 1}
    rest_with_keys = sorted(((keys[i], i) for i in range(K - 1)), reverse=True)
    for _, idx in rest_with_keys[:k - 1]:
        selected.add(idx)

    return sorted(base_positions[i] for i in selected)


def sparse_sample_positions(
    positions: list[int],
    n_layers: int = 3,
) -> list[int]:
    """Randomly subsample CoT positions.

    Positions are stored as [base_positions] * n_layers.

    For each example we:
      1. Extract base positions (first 1/n_layers of the list)
      2. Sample a random count n_feed ~ Uniform(1, K) with 1 double-weighted
      3. Always include the last CoT position
      4. Re-expand to n_layers

    This forces the oracle to work with sparse, incomplete activation evidence.
    """
    if not positions:
        return positions

    total = len(positions)

    # Extract base positions (un-expand the layer repetition)
    if n_layers > 1 and total % n_layers == 0:
        k = total // n_layers
        base = positions[:k]
    else:
        base = positions
        n_layers = 1

    n_cot = len(base)
    if n_cot <= 2:
        return positions

    # Sample how many positions to keep: uniform over [1, n_cot]
    # with 1 double-weighted (oracle sometimes sees minimal evidence)
    pool = [1] + list(range(1, n_cot + 1))
    n_feed = random.choice(pool)

    if n_feed >= n_cot:
        sampled = base
    else:
        # Always include the last position, sample rest randomly
        last = base[-1]
        rest = base[:-1]
        picked = sorted(random.sample(rest, min(n_feed - 1, len(rest))))
        sampled = picked + [last]

    return sampled * n_layers


# Layer count lookup (no torch dependency)
LAYER_COUNTS = {
    "Qwen/Qwen3-0.6B": 28,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-4B": 36,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-14B": 40,
    "Qwen/Qwen3-32B": 64,
}


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


# Set by train.py main() — dataset loaders read this instead of hardcoding [25,50,75]
CONFIGURED_LAYERS: list[int] | None = None


def get_injection_layers(model_name: str) -> list[int]:
    """Return configured injection layers, or default [25%, 50%, 75%]."""
    if CONFIGURED_LAYERS is not None:
        return list(CONFIGURED_LAYERS)
    return [layer_percent_to_layer(model_name, p) for p in [25, 50, 75]]
