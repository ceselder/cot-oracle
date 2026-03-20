"""Binary rubric → scalar reward → group advantages."""

from __future__ import annotations

from dataclasses import dataclass, field

CRITERIA_NAMES = [
    "not_provably_wrong",
    "specific",
    "follows_instructions",
    "passes_swap_test",
    "concise",
    "not_just_restating_text",
    "numbers_if_applicable",
    "confident_when_verifiable",
    "hedged_when_uncertain",
    "useful_to_a_human",
    "falsifiable",
]


@dataclass
class RubricResult:
    rollout_idx: int
    criteria: dict[str, bool] = field(default_factory=dict)

    def reward(self, weights: dict[str, float]) -> float:
        total = sum(weights.get(k, 1.0) for k in CRITERIA_NAMES)
        earned = sum(
            weights.get(k, 1.0) * float(self.criteria.get(k, False))
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
