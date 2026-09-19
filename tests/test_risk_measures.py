import numpy as np

from optimizers.risk_measures import (
    paired_cvar_difference_interval,
    summarise_losses,
    weighted_cvar,
    weighted_quantile,
)


def test_weighted_cvar_uses_fractional_var_atom() -> None:
    losses = np.array([0.0, 10.0, 20.0])
    probabilities = np.array([0.80, 0.15, 0.05])
    assert weighted_quantile(losses, 0.90, probabilities) == 10.0
    assert np.isclose(weighted_cvar(losses, 0.90, probabilities), 15.0)


def test_risk_summary() -> None:
    losses = np.array([0.0, 2.0, 4.0, 8.0])
    summary = summarise_losses(losses, beta=0.75)
    assert summary.expected == 3.5
    assert summary.var == 4.0
    assert summary.cvar == 8.0
    assert summary.maximum == 8.0
    assert summary.probability_positive == 0.75


def test_paired_cvar_difference_sign() -> None:
    reference = np.array([0.0, 0.0, 10.0, 20.0])
    cvar = np.array([0.0, 0.0, 5.0, 10.0])
    estimate, lower, upper = paired_cvar_difference_interval(
        cvar,
        reference,
        beta=0.75,
        n_bootstrap=100,
        seed=1,
    )
    assert estimate < 0
    assert lower <= upper
