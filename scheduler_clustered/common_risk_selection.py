"""Pure selection utilities for a common-pool risk comparison.

The functions in this module deliberately contain no solver dependencies.  They
select a risk-neutral schedule and a CVaR schedule from the *same* evaluated
candidate pool, subject to the same utility floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, TypeVar

from .risk_metrics import RiskSummary, normalize_probabilities


class RiskCandidate(Protocol):
    """Structural interface required by the common-pool selectors."""

    pool_index: int
    selection_utility: float
    risk: RiskSummary
    scenario_losses: Mapping[str, float]


CandidateT = TypeVar("CandidateT", bound=RiskCandidate)


@dataclass(frozen=True)
class PairedLossSummary:
    """Probability-weighted paired comparison of two loss vectors."""

    expected_difference: float
    minimum_difference: float
    maximum_difference: float
    probability_cvar_better: float
    probability_risk_neutral_better: float
    probability_tie: float
    scenario_differences: dict[str, float]


def compute_utility_floor(
    best_utility: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float = 0.0,
) -> float:
    """Return the minimum utility admitted in the common comparison set.

    The admissible loss of utility is the larger of the configured relative and
    absolute tolerances.  This avoids a vanishing relative tolerance when the
    utility scale is close to zero.
    """
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative.")
    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative.")
    allowed_drop = max(
        float(absolute_tolerance),
        float(relative_tolerance) * max(abs(float(best_utility)), 1.0),
    )
    return float(best_utility) - allowed_drop


def select_common_pool_candidates(
    candidates: Sequence[CandidateT],
    *,
    relative_utility_tolerance: float = 0.01,
    absolute_utility_tolerance: float = 0.0,
    numerical_tolerance: float = 1e-10,
) -> tuple[CandidateT, CandidateT, float, tuple[CandidateT, ...]]:
    """Select risk-neutral and CVaR schedules from one candidate pool.

    The risk-neutral schedule minimizes expected loss.  The CVaR schedule
    minimizes CVaR.  Both are selected from the same utility-admissible subset,
    and deterministic tie-breaking is used for reproducibility.
    """
    if not candidates:
        raise ValueError("At least one evaluated candidate is required.")
    if numerical_tolerance < 0.0:
        raise ValueError("numerical_tolerance must be non-negative.")

    best_utility = max(float(candidate.selection_utility) for candidate in candidates)
    floor = compute_utility_floor(
        best_utility,
        relative_tolerance=relative_utility_tolerance,
        absolute_tolerance=absolute_utility_tolerance,
    )
    eligible = tuple(
        candidate
        for candidate in candidates
        if float(candidate.selection_utility) + numerical_tolerance >= floor
    )
    if not eligible:
        raise RuntimeError("The utility filter removed every candidate.")

    risk_neutral = min(
        eligible,
        key=lambda candidate: (
            float(candidate.risk.expected_loss),
            float(candidate.risk.cvar),
            -float(candidate.selection_utility),
            int(candidate.pool_index),
        ),
    )
    cvar = min(
        eligible,
        key=lambda candidate: (
            float(candidate.risk.cvar),
            float(candidate.risk.expected_loss),
            -float(candidate.selection_utility),
            int(candidate.pool_index),
        ),
    )
    return risk_neutral, cvar, float(floor), eligible


def paired_loss_summary(
    risk_neutral_losses: Mapping[str, float],
    cvar_losses: Mapping[str, float],
    probabilities: Mapping[str, float],
    *,
    tie_tolerance: float = 1e-9,
) -> PairedLossSummary:
    """Compare CVaR-minus-risk-neutral loss on identical scenario IDs."""
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be non-negative.")
    keys = set(risk_neutral_losses)
    if keys != set(cvar_losses) or keys != set(probabilities):
        raise ValueError(
            "Both loss mappings and the probability mapping must have identical keys."
        )
    normalized = normalize_probabilities(probabilities)
    differences = {
        scenario_id: float(cvar_losses[scenario_id])
        - float(risk_neutral_losses[scenario_id])
        for scenario_id in sorted(keys)
    }
    expected = sum(
        normalized[scenario_id] * difference
        for scenario_id, difference in differences.items()
    )
    p_cvar = sum(
        normalized[scenario_id]
        for scenario_id, difference in differences.items()
        if difference < -tie_tolerance
    )
    p_risk_neutral = sum(
        normalized[scenario_id]
        for scenario_id, difference in differences.items()
        if difference > tie_tolerance
    )
    p_tie = max(0.0, 1.0 - p_cvar - p_risk_neutral)
    values = tuple(differences.values())
    return PairedLossSummary(
        expected_difference=float(expected),
        minimum_difference=float(min(values, default=0.0)),
        maximum_difference=float(max(values, default=0.0)),
        probability_cvar_better=float(p_cvar),
        probability_risk_neutral_better=float(p_risk_neutral),
        probability_tie=float(p_tie),
        scenario_differences=differences,
    )
