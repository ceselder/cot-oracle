"""Tiered rubric (0/1/2) + binary correctness → scalar reward → group advantages."""

from __future__ import annotations

from dataclasses import dataclass, field

CRITERIA_NAMES = [
    "passes_swap_test",
    "specific_and_falsifiable",
    "adds_insight",
    "not_provably_wrong",
    "follows_instructions",
]


@dataclass
class RubricResult:
    rollout_idx: int
    criteria: dict[str, int] = field(default_factory=dict)  # 0, 1, or 2

    def reward(self, weights: dict[str, float]) -> float:
        # If provably wrong (score 0), entire reward is 0
        if self.criteria.get("not_provably_wrong", 0) == 0:
            return 0.0
        total = sum(weights.get(k, 1.0) * 2 for k in CRITERIA_NAMES)
        earned = sum(
            weights.get(k, 1.0) * self.criteria.get(k, 0)
            for k in CRITERIA_NAMES
        )
        return earned / total if total > 0 else 0.0


def compute_rewards(
    rubrics: list[RubricResult],
    weights: dict[str, float],
) -> list[float]:
    return [r.reward(weights) for r in rubrics]


def compute_group_advantages(
    rewards: list[float],
    normalize: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """Compute GRPO advantages: (r_i - mean) / std within a group."""
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = var_r ** 0.5
    if normalize and std_r > eps:
        return [(r - mean_r) / std_r for r in rewards]
    return [r - mean_r for r in rewards]
