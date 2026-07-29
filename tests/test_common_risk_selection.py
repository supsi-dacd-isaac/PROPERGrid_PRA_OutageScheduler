from __future__ import annotations

from dataclasses import dataclass

import pytest

from scheduler_clustered.common_risk_selection import (
    compute_utility_floor,
    paired_loss_summary,
    select_common_pool_candidates,
)
from scheduler_clustered.risk_metrics import RiskSummary


@dataclass
class Candidate:
    pool_index: int
    selection_utility: float
    risk: RiskSummary
    scenario_losses: dict[str, float]


def risk(expected: float, cvar: float, *, var: float | None = None) -> RiskSummary:
    return RiskSummary(
        expected_loss=expected,
        var=cvar if var is None else var,
        cvar=cvar,
        alpha=0.95,
        tail_ids=("s2",),
        normalized_probabilities={"s1": 0.5, "s2": 0.5},
    )


@pytest.mark.unit
def test_utility_floor_uses_larger_absolute_or_relative_tolerance() -> None:
    assert compute_utility_floor(100.0, relative_tolerance=0.01) == pytest.approx(99.0)
    assert compute_utility_floor(
        100.0,
        relative_tolerance=0.01,
        absolute_tolerance=3.0,
    ) == pytest.approx(97.0)
    assert compute_utility_floor(-2.0, relative_tolerance=0.10) == pytest.approx(-2.2)


@pytest.mark.unit
def test_selects_expected_and_cvar_from_same_eligible_pool() -> None:
    candidates = [
        Candidate(1, 100.0, risk(3.0, 9.0), {"s1": 1.0, "s2": 5.0}),
        Candidate(2, 99.5, risk(4.0, 6.0), {"s1": 3.0, "s2": 5.0}),
        Candidate(3, 95.0, risk(1.0, 2.0), {"s1": 1.0, "s2": 1.0}),
    ]
    risk_neutral, cvar, floor, eligible = select_common_pool_candidates(
        candidates,
        relative_utility_tolerance=0.01,
    )
    assert floor == pytest.approx(99.0)
    assert {candidate.pool_index for candidate in eligible} == {1, 2}
    assert risk_neutral.pool_index == 1
    assert cvar.pool_index == 2


@pytest.mark.unit
def test_selection_tie_breaking_is_deterministic() -> None:
    candidates = [
        Candidate(4, 10.0, risk(2.0, 5.0), {"s1": 1.0, "s2": 3.0}),
        Candidate(2, 10.0, risk(2.0, 5.0), {"s1": 1.0, "s2": 3.0}),
    ]
    risk_neutral, cvar, _, _ = select_common_pool_candidates(
        candidates,
        relative_utility_tolerance=0.0,
    )
    assert risk_neutral.pool_index == 2
    assert cvar.pool_index == 2


@pytest.mark.unit
def test_paired_summary_uses_common_probabilities() -> None:
    summary = paired_loss_summary(
        {"s1": 2.0, "s2": 8.0, "s3": 4.0},
        {"s1": 1.0, "s2": 9.0, "s3": 4.0},
        {"s1": 0.2, "s2": 0.3, "s3": 0.5},
    )
    assert summary.expected_difference == pytest.approx(0.1)
    assert summary.minimum_difference == pytest.approx(-1.0)
    assert summary.maximum_difference == pytest.approx(1.0)
    assert summary.probability_cvar_better == pytest.approx(0.2)
    assert summary.probability_risk_neutral_better == pytest.approx(0.3)
    assert summary.probability_tie == pytest.approx(0.5)


@pytest.mark.unit
def test_paired_summary_honours_tie_tolerance_and_normalises_weights() -> None:
    summary = paired_loss_summary(
        {"a": 1.0, "b": 1.0},
        {"a": 1.0 + 1e-10, "b": 0.0},
        {"a": 9.0, "b": 1.0},
        tie_tolerance=1e-9,
    )
    assert summary.probability_tie == pytest.approx(0.9)
    assert summary.probability_cvar_better == pytest.approx(0.1)
    assert summary.probability_risk_neutral_better == 0.0


@pytest.mark.unit
def test_selector_input_validation() -> None:
    with pytest.raises(ValueError, match="At least one"):
        select_common_pool_candidates([])
    with pytest.raises(ValueError, match="non-negative"):
        compute_utility_floor(1.0, relative_tolerance=-0.1)
    with pytest.raises(ValueError, match="identical keys"):
        paired_loss_summary({"s1": 1.0}, {"s2": 1.0}, {"s1": 1.0})
