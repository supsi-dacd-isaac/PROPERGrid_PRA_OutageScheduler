"""Calibrated and explicit-placeholder weather regime models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from probabilistic_models.base import as_rng


class WeatherRegime(str, Enum):
    """Regime labels used by the supplied component-rate table."""

    H1_ADVERSE = "H1_adverse"
    H2_NORMAL = "H2_normal"


@dataclass(frozen=True)
class WeatherModelMetadata:
    calibrated: bool
    status: str
    assumptions: tuple[str, ...]


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, index])
    if stationary.sum() < 0:
        stationary = -stationary
    stationary = np.maximum(stationary, 0.0)
    if stationary.sum() <= 0:
        stationary = np.ones(transition.shape[0])
    return stationary / stationary.sum()


class WeatherRegimeModel:
    """Two-state discrete-time Markov model for normal/adverse weather."""

    states = (WeatherRegime.H1_ADVERSE, WeatherRegime.H2_NORMAL)

    def __init__(self, transition_matrix: np.ndarray, metadata: WeatherModelMetadata):
        transition = np.asarray(transition_matrix, dtype=float)
        if transition.shape != (2, 2) or np.any(transition < 0):
            raise ValueError("transition_matrix must be a non-negative 2x2 array.")
        row_sums = transition.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Every transition-matrix row must have positive mass.")
        self.transition_matrix = transition / row_sums[:, None]
        self.initial_probabilities = _stationary_distribution(self.transition_matrix)
        self.metadata = metadata

    def sample_regimes(
        self,
        n_trajectories: int,
        n_steps: int = 1,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw categorical regime trajectories with shape ``(paths, steps)``."""

        if n_trajectories < 1 or n_steps < 1:
            raise ValueError("n_trajectories and n_steps must be positive.")
        rng = as_rng(random_state)
        encoded = np.empty((int(n_trajectories), int(n_steps)), dtype=np.int8)
        encoded[:, 0] = rng.choice(2, size=int(n_trajectories), p=self.initial_probabilities)
        for step in range(1, int(n_steps)):
            uniforms = rng.random(int(n_trajectories))
            probability_h1 = self.transition_matrix[encoded[:, step - 1], 0]
            encoded[:, step] = (uniforms >= probability_h1).astype(np.int8)
        labels = np.asarray([state.value for state in self.states], dtype=object)
        return labels[encoded]


class PlaceholderWeatherModel(WeatherRegimeModel):
    """Transparent fallback used when no meteorological observations are supplied.

    Defaults imply a 2% hourly probability of entering adverse weather from the
    normal state and 70% persistence once adverse. They are demonstrative,
    deliberately labelled placeholders, and must not be used for decisions.
    """

    def __init__(self, *, normal_to_adverse: float = 0.02, adverse_persistence: float = 0.70):
        if not 0.0 <= normal_to_adverse <= 1.0 or not 0.0 <= adverse_persistence <= 1.0:
            raise ValueError("Weather transition probabilities must lie in [0, 1].")
        transition = np.array(
            [
                [adverse_persistence, 1.0 - adverse_persistence],
                [normal_to_adverse, 1.0 - normal_to_adverse],
            ]
        )
        metadata = WeatherModelMetadata(
            calibrated=False,
            status="PLACEHOLDER - replace before operational interpretation",
            assumptions=(
                f"P(H1_adverse[t+1] | H2_normal[t]) = {normal_to_adverse:.4g} per model step",
                f"P(H1_adverse[t+1] | H1_adverse[t]) = {adverse_persistence:.4g}",
                "H1 is interpreted as adverse and H2 as normal because the supplied H1 line rates are higher.",
            ),
        )
        super().__init__(transition, metadata)


class EmpiricalWeatherModel(WeatherRegimeModel):
    """Estimate regime transitions from observed labels or a wind threshold."""

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame | Iterable[str],
        *,
        regime_column: str | None = None,
        wind_speed_column: str = "wind_speed_m_s",
        adverse_wind_threshold: float | None = None,
        threshold_quantile: float = 0.95,
        laplace_smoothing: float = 0.5,
    ) -> "EmpiricalWeatherModel":
        if isinstance(data, pd.DataFrame):
            if regime_column is not None:
                raw = data[regime_column].astype(str).to_numpy()
                adverse = np.array(
                    [value in {WeatherRegime.H1_ADVERSE.value, "H1", "adverse", "1"} for value in raw],
                    dtype=bool,
                )
                assumption = f"Observed regime labels from column {regime_column!r}."
            else:
                wind = pd.to_numeric(data[wind_speed_column], errors="coerce").to_numpy()
                if not np.all(np.isfinite(wind)):
                    raise ValueError("Weather wind-speed column contains missing/non-numeric values.")
                threshold = (
                    float(adverse_wind_threshold)
                    if adverse_wind_threshold is not None
                    else float(np.quantile(wind, threshold_quantile))
                )
                adverse = wind >= threshold
                assumption = f"H1_adverse defined by {wind_speed_column} >= {threshold:.6g}."
        else:
            raw = np.asarray(list(data), dtype=str)
            adverse = np.array(
                [value in {WeatherRegime.H1_ADVERSE.value, "H1", "adverse", "1"} for value in raw],
                dtype=bool,
            )
            assumption = "Observed regime labels supplied as a sequence."
        if adverse.size < 3:
            raise ValueError("At least three ordered weather observations are required.")
        encoded = np.where(adverse, 0, 1)
        counts = np.full((2, 2), float(laplace_smoothing))
        for previous, current in zip(encoded[:-1], encoded[1:]):
            counts[previous, current] += 1.0
        transition = counts / counts.sum(axis=1, keepdims=True)
        metadata = WeatherModelMetadata(
            calibrated=True,
            status="calibrated from supplied ordered weather observations",
            assumptions=(assumption, f"Laplace smoothing = {laplace_smoothing:.4g}."),
        )
        return cls(transition, metadata)


def weather_forecaster(
    data: pd.DataFrame | Iterable[str] | None = None,
    **kwargs,
) -> WeatherRegimeModel:
    """Factory retaining the old function name while making fallback status explicit."""

    if data is None:
        return PlaceholderWeatherModel(**kwargs)
    return EmpiricalWeatherModel.fit(data, **kwargs)
