"""Pure numerical risk metrics used in optimisation reporting and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskSummary:
    expected: float
    var: float
    cvar: float
    q99: float
    maximum: float
    probability_positive: float

    def to_dict(self) -> dict[str, float]:
        return {
            "expected": self.expected,
            "var": self.var,
            "cvar": self.cvar,
            "q99": self.q99,
            "maximum": self.maximum,
            "probability_positive": self.probability_positive,
        }


def _normalise_inputs(values: np.ndarray, probabilities: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("At least one loss value is required.")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Loss values must be finite.")
    if probabilities is None:
        p = np.full(x.size, 1.0 / x.size)
    else:
        p = np.asarray(probabilities, dtype=float).reshape(-1)
        if p.size != x.size:
            raise ValueError("Loss and probability vectors must have the same length.")
        if np.isnan(p).any() or np.isinf(p).any() or (p < 0).any():
            raise ValueError("Probabilities must be finite and non-negative.")
        total = p.sum()
        if total <= 0:
            raise ValueError("Probability mass must be positive.")
        p = p / total
    return x, p


def weighted_mean(values: np.ndarray, probabilities: np.ndarray | None = None) -> float:
    x, p = _normalise_inputs(values, probabilities)
    return float(np.dot(x, p))


def weighted_quantile(
    values: np.ndarray,
    quantile: float,
    probabilities: np.ndarray | None = None,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    x, p = _normalise_inputs(values, probabilities)
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    p_sorted = p[order]
    cumulative = np.cumsum(p_sorted)
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(x_sorted[min(index, x_sorted.size - 1)])


def weighted_cvar(
    values: np.ndarray,
    beta: float = 0.95,
    probabilities: np.ndarray | None = None,
) -> float:
    """Exact finite-distribution upper-tail CVaR, including a fractional VaR atom."""
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1).")
    x, p = _normalise_inputs(values, probabilities)
    order = np.argsort(x)[::-1]
    x_sorted = x[order]
    p_sorted = p[order]
    tail_mass = 1.0 - beta
    remaining = tail_mass
    tail_integral = 0.0
    for value, mass in zip(x_sorted, p_sorted):
        take = min(float(mass), remaining)
        tail_integral += take * float(value)
        remaining -= take
        if remaining <= 1e-15:
            break
    if remaining > 1e-12:
        raise RuntimeError("Insufficient probability mass while computing CVaR.")
    return float(tail_integral / tail_mass)


def summarise_losses(
    values: np.ndarray,
    beta: float = 0.95,
    probabilities: np.ndarray | None = None,
    positive_tolerance: float = 1.0e-9,
) -> RiskSummary:
    x, p = _normalise_inputs(values, probabilities)
    return RiskSummary(
        expected=float(np.dot(x, p)),
        var=weighted_quantile(x, beta, p),
        cvar=weighted_cvar(x, beta, p),
        q99=weighted_quantile(x, 0.99, p),
        maximum=float(np.max(x)),
        probability_positive=float(np.sum(p[x > positive_tolerance])),
    )


def paired_cvar_difference_interval(
    losses_cvar: np.ndarray,
    losses_reference: np.ndarray,
    beta: float = 0.95,
    probabilities: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260729,
) -> tuple[float, float, float]:
    """Bootstrap interval for CVaR(CVaR schedule) minus CVaR(reference schedule)."""
    cvar_losses, p = _normalise_inputs(losses_cvar, probabilities)
    reference_losses, _ = _normalise_inputs(losses_reference, p)
    if cvar_losses.size != reference_losses.size:
        raise ValueError("Paired loss vectors must have the same length.")
    estimate = weighted_cvar(cvar_losses, beta, p) - weighted_cvar(reference_losses, beta, p)
    if n_bootstrap <= 0:
        return estimate, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = cvar_losses.size
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.choice(n, size=n, replace=True, p=p)
        samples[index] = weighted_cvar(cvar_losses[draw], beta) - weighted_cvar(reference_losses[draw], beta)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(estimate), float(lower), float(upper)
