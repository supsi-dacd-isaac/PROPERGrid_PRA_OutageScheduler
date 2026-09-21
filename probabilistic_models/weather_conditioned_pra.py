"""Weather-conditioned component rates and first-order N-1 PRA utilities.

The functions in this module deliberately separate three quantities:

* a baseline component failure rate;
* a forecast wind covariate mapped to an exposed branch; and
* a user-specified exposure--rate relationship.

The default exposure--rate curve is an *illustrative sensitivity model*.  It
is not a fitted fragility curve and must not be interpreted as one.  Its
maximum multiplier is anchored to the ratio between the supplied H1 and H2
single-line rates, while the wind thresholds follow the public MeteoSwiss
wind-speed classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindRateCurve:
    """Smooth illustrative multiplier for a baseline line-failure rate.

    Parameters are expressed in km/h because the MeteoSwiss public legend and
    the forecast CSV files use km/h.  Below ``onset_kmh`` the multiplier is
    one.  Between ``onset_kmh`` and ``severe_kmh`` it increases smoothly to
    ``maximum_multiplier`` and remains capped above ``severe_kmh``.
    """

    onset_kmh: float = 12.0
    severe_kmh: float = 108.0
    maximum_multiplier: float = 6.81e-7 / 2.26e-7
    exponent: float = 1.0
    calibration_status: str = (
        "ILLUSTRATIVE SENSITIVITY CURVE - not calibrated from joint outage/weather data"
    )

    def __post_init__(self) -> None:
        if self.onset_kmh < 0:
            raise ValueError("onset_kmh must be non-negative")
        if self.severe_kmh <= self.onset_kmh:
            raise ValueError("severe_kmh must exceed onset_kmh")
        if self.maximum_multiplier < 1:
            raise ValueError("maximum_multiplier must be at least one")
        if self.exponent <= 0:
            raise ValueError("exponent must be positive")

    def multiplier(self, wind_kmh) -> np.ndarray:
        """Return bounded failure-rate multipliers for wind values."""

        wind = np.asarray(wind_kmh, dtype=float)
        scaled = np.clip(
            (wind - self.onset_kmh) / (self.severe_kmh - self.onset_kmh),
            0.0,
            1.0,
        )
        result = 1.0 + (self.maximum_multiplier - 1.0) * scaled**self.exponent
        return np.where(np.isfinite(wind), result, np.nan)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def condition_rates_on_wind(
    baseline_rates_per_hour: Mapping[str, float],
    wind_by_asset_kmh: Mapping[str, float] | pd.Series,
    *,
    curve: WindRateCurve,
) -> dict[str, float]:
    """Apply the wind multiplier only to assets with a finite exposure.

    Assets absent from ``wind_by_asset_kmh`` retain their baseline rate.  This
    is used, for example, to leave transformers unaffected when only overhead
    line wind exposure is represented.
    """

    conditioned = {str(key): float(value) for key, value in baseline_rates_per_hour.items()}
    if any(value < 0 for value in conditioned.values()):
        raise ValueError("baseline failure rates must be non-negative")

    wind = pd.Series(wind_by_asset_kmh, dtype=float)
    common = wind.index.intersection(pd.Index(conditioned))
    if common.empty:
        return conditioned
    values = wind.loc[common].to_numpy(float)
    multipliers = curve.multiplier(values)
    for asset_id, multiplier in zip(common.astype(str), multipliers):
        if np.isfinite(multiplier):
            conditioned[asset_id] *= float(multiplier)
    return conditioned


@dataclass(frozen=True)
class N1ProbabilityMass:
    """Exact probability mass of the zero- and one-failure Poisson states."""

    normal_probability: float
    n1_probabilities: dict[str, float]
    omitted_multiple_failure_probability: float


def exact_n1_probability_mass(
    rates_per_hour: Mapping[str, float],
    horizon_hours: float,
) -> N1ProbabilityMass:
    """Return exact zero- and single-failure probabilities.

    Independent Poisson processes are assumed.  The omitted mass contains all
    states with two or more component failures during the interval and is
    reported explicitly instead of being silently assigned to the N-1 set.
    """

    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    identifiers = [str(key) for key in rates_per_hour]
    rates = np.asarray([rates_per_hour[key] for key in rates_per_hour], dtype=float)
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rates_per_hour must contain finite non-negative values")

    cumulative = float(rates.sum() * horizon_hours)
    normal = float(np.exp(-cumulative))
    # P(exactly asset i fails at least once and every other asset has no event)
    # = (1-exp(-lambda_i*h)) * exp(-sum_{j!=i} lambda_j*h).
    n1 = {}
    for asset_id, rate in zip(identifiers, rates):
        other_survival = np.exp(-(rates.sum() - rate) * horizon_hours)
        n1[asset_id] = float((1.0 - np.exp(-rate * horizon_hours)) * other_survival)
    omitted = max(1.0 - normal - sum(n1.values()), 0.0)
    return N1ProbabilityMass(normal, n1, float(omitted))


def weighted_tail_metrics(
    losses,
    probabilities,
    *,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute expectation, weighted VaR and weighted CVaR for discrete loss."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    loss = np.asarray(losses, dtype=float)
    weight = np.asarray(probabilities, dtype=float)
    if loss.ndim != 1 or weight.shape != loss.shape or loss.size == 0:
        raise ValueError("losses and probabilities must be equally sized non-empty vectors")
    if np.any(~np.isfinite(loss)) or np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("losses/weights must be finite and weights non-negative")
    mass = float(weight.sum())
    if mass <= 0:
        raise ValueError("probability mass must be positive")
    weight = weight / mass
    order = np.argsort(loss, kind="stable")
    sorted_loss = loss[order]
    sorted_weight = weight[order]
    cumulative = np.cumsum(sorted_weight)
    position = min(int(np.searchsorted(cumulative, alpha, side="left")), loss.size - 1)
    var = float(sorted_loss[position])
    # Rockafellar--Uryasev discrete representation handles probability mass at VaR.
    cvar = float(var + np.sum(weight * np.maximum(loss - var, 0.0)) / (1.0 - alpha))
    return {
        "expected_loss": float(np.sum(weight * loss)),
        "var": var,
        "cvar": cvar,
        "normalization_mass": mass,
    }
