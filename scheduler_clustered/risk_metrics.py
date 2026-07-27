"""Weighted expectation, VaR, and CVaR for finite scenario samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskSummary:
    expected_loss: float
    var: float
    cvar: float
    alpha: float
    tail_ids: tuple[str, ...]
    normalized_probabilities: dict[str, float]


def normalize_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    values = {str(key): float(value) for key, value in probabilities.items()}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("Scenario probabilities must be non-negative.")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Scenario probabilities sum to zero.")
    return {key: value / total for key, value in values.items()}


def weighted_var_cvar(
    losses: Mapping[str, float],
    probabilities: Mapping[str, float],
    *,
    alpha: float = 0.95,
) -> RiskSummary:
    """Compute weighted empirical expectation, VaR, and CVaR.

    The CVaR calculation allocates exactly ``1-alpha`` probability mass to the
    upper tail, including a fractional contribution from the VaR atom when
    required.  This is important for small and non-uniform scenario sets.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    if set(losses) != set(probabilities):
        raise ValueError("Loss and probability mappings must have identical keys.")
    if not losses:
        return RiskSummary(0.0, 0.0, 0.0, alpha, (), {})

    normalized = normalize_probabilities(probabilities)
    ordered = sorted(
        (
            (float(losses[scenario]), normalized[scenario], str(scenario))
            for scenario in losses
        ),
        key=lambda item: item[0],
    )
    expected = sum(loss * probability for loss, probability, _ in ordered)

    cumulative = 0.0
    var = ordered[-1][0]
    for loss, probability, _ in ordered:
        cumulative += probability
        if cumulative + 1e-15 >= alpha:
            var = loss
            break

    remaining = 1.0 - alpha
    tail_loss = 0.0
    tail_ids: list[str] = []
    for loss, probability, scenario_id in reversed(ordered):
        if remaining <= 1e-15:
            break
        used = min(probability, remaining)
        if used > 0.0:
            tail_loss += used * loss
            tail_ids.append(scenario_id)
            remaining -= used

    cvar = tail_loss / (1.0 - alpha)
    return RiskSummary(
        expected_loss=float(expected),
        var=float(var),
        cvar=float(cvar),
        alpha=float(alpha),
        tail_ids=tuple(tail_ids),
        normalized_probabilities=normalized,
    )
