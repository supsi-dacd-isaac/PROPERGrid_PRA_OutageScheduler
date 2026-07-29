from __future__ import annotations

import pytest

from scheduler_clustered.risk_metrics import (
    normalize_probabilities,
    weighted_var_cvar,
)


@pytest.mark.unit
def test_probability_normalisation() -> None:
    result = normalize_probabilities({"a": 2.0, "b": 3.0})
    assert result == pytest.approx({"a": 0.4, "b": 0.6})


@pytest.mark.unit
def test_exact_weighted_cvar_includes_fractional_var_atom() -> None:
    result = weighted_var_cvar(
        {"normal": 0.0, "moderate": 10.0, "extreme": 100.0},
        {"normal": 0.80, "moderate": 0.15, "extreme": 0.05},
        alpha=0.90,
    )
    assert result.expected_loss == pytest.approx(6.5)
    assert result.var == pytest.approx(10.0)
    assert result.cvar == pytest.approx(55.0)
    assert set(result.tail_ids) == {"moderate", "extreme"}


@pytest.mark.unit
def test_weighted_cvar_is_invariant_to_probability_scale() -> None:
    losses = {"s1": 1.0, "s2": 3.0, "s3": 9.0}
    first = weighted_var_cvar(losses, {"s1": 1, "s2": 2, "s3": 1}, alpha=0.75)
    second = weighted_var_cvar(losses, {"s1": 10, "s2": 20, "s3": 10}, alpha=0.75)
    assert first == second


@pytest.mark.unit
def test_empty_loss_distribution_returns_zero_summary() -> None:
    result = weighted_var_cvar({}, {}, alpha=0.95)
    assert result.expected_loss == 0.0
    assert result.var == 0.0
    assert result.cvar == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        weighted_var_cvar({"s": 1.0}, {"s": 1.0}, alpha=alpha)


@pytest.mark.unit
def test_invalid_probability_and_key_mismatch_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        normalize_probabilities({"a": -1.0, "b": 2.0})
    with pytest.raises(ValueError, match="sum to zero"):
        normalize_probabilities({"a": 0.0})
    with pytest.raises(ValueError, match="identical keys"):
        weighted_var_cvar({"a": 1.0}, {"b": 1.0})
