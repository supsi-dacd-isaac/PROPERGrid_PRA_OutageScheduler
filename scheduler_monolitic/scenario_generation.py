"""Common demand-scenario generation for risk-neutral and CVaR formulations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ScenarioConfig


@dataclass(frozen=True)
class ScenarioBank:
    """Demand trajectories and probabilities used by an optimisation or validation run."""

    demands: dict[str, pd.DataFrame]
    probabilities: dict[str, float]
    seed: int
    metadata: dict[str, object]

    @property
    def labels(self) -> list[str]:
        return list(self.demands)

    def validate(self, reference: pd.DataFrame | None = None) -> None:
        if not self.demands:
            raise ValueError("ScenarioBank must contain at least one scenario.")
        if set(self.demands) != set(self.probabilities):
            raise ValueError("Demand and probability labels do not match.")
        probabilities = np.asarray(list(self.probabilities.values()), dtype=float)
        if np.any(probabilities < 0):
            raise ValueError("Scenario probabilities must be non-negative.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Scenario probabilities must sum to one.")
        shapes = {frame.shape for frame in self.demands.values()}
        if len(shapes) != 1:
            raise ValueError("All scenario demand tables must have the same shape.")
        for label, frame in self.demands.items():
            if frame.isna().any().any():
                raise ValueError(f"Scenario {label} contains NaN values.")
            if (frame.to_numpy() < 0).any():
                raise ValueError(f"Scenario {label} contains negative demand.")
        if reference is not None:
            first = next(iter(self.demands.values()))
            if first.shape != reference.shape:
                raise ValueError("Scenario and reference demand shapes differ.")
            if list(first.columns) != list(reference.columns):
                raise ValueError("Scenario and reference demand columns differ.")

    @classmethod
    def nominal(cls, demand: pd.DataFrame, label: str = "scenario_000") -> "ScenarioBank":
        frame = demand.astype(float).copy()
        return cls(
            demands={label: frame},
            probabilities={label: 1.0},
            seed=0,
            metadata={"sampling_scheme": "nominal", "n_scenarios": 1},
        )


def _ar1_standard_normal(rng: np.random.Generator, n_steps: int, rho: float) -> np.ndarray:
    innovations = rng.standard_normal(n_steps)
    values = np.empty(n_steps, dtype=float)
    values[0] = innovations[0]
    scale = np.sqrt(max(1.0 - rho**2, 0.0))
    for index in range(1, n_steps):
        values[index] = rho * values[index - 1] + scale * innovations[index]
    return values


def _stratified_global_scores(n_scenarios: int, tail_fraction: float) -> np.ndarray:
    """Return deterministic standard-normal quantiles with extra upper-tail coverage."""
    n_tail = max(1, int(round(n_scenarios * tail_fraction)))
    n_body = n_scenarios - n_tail
    quantiles: list[float] = []
    if n_body:
        quantiles.extend((np.arange(n_body) + 0.5) / n_body * 0.90)
    quantiles.extend(0.90 + (np.arange(n_tail) + 0.5) / n_tail * 0.10)
    normal = NormalDist()
    return np.asarray([normal.inv_cdf(float(np.clip(q, 1e-8, 1 - 1e-8))) for q in quantiles])


def generate_demand_scenarios(
    nominal_demand: pd.DataFrame,
    config: ScenarioConfig | None = None,
) -> ScenarioBank:
    """
    Generate positive, temporally correlated multiplicative demand scenarios.

    A scenario-wide score controls its broad severity, while AR(1) temporal and
    independent local components preserve variation across time and buses. The
    same generated bank must be passed to both optimisation formulations.
    """
    config = config or ScenarioConfig()
    config.validate()

    nominal = nominal_demand.astype(float).copy()
    if nominal.empty:
        raise ValueError("nominal_demand is empty.")
    if nominal.isna().any().any() or (nominal.to_numpy() < 0).any():
        raise ValueError("nominal_demand must be finite and non-negative.")

    rng = np.random.default_rng(config.seed)
    n_steps, n_buses = nominal.shape

    if config.sampling_scheme == "stratified_tail":
        global_scores = _stratified_global_scores(config.n_scenarios, config.tail_fraction)
        rng.shuffle(global_scores)
    else:
        global_scores = rng.standard_normal(config.n_scenarios)

    demands: dict[str, pd.DataFrame] = {}
    probabilities: dict[str, float] = {}
    probability = 1.0 / config.n_scenarios

    for scenario_index, global_score in enumerate(global_scores):
        temporal = _ar1_standard_normal(rng, n_steps, config.temporal_correlation)
        local = rng.standard_normal((n_steps, n_buses))

        log_multiplier = (
            config.sigma_global * (0.65 * global_score + 0.35 * temporal[:, None])
            + config.sigma_local * local
        )
        variance_correction = 0.5 * (config.sigma_global**2 + config.sigma_local**2)
        multiplier = np.exp(log_multiplier - variance_correction)
        sampled = np.maximum(nominal.to_numpy() * multiplier, 0.0)

        label = f"scenario_{scenario_index:03d}"
        demands[label] = pd.DataFrame(sampled, index=nominal.index, columns=nominal.columns)
        probabilities[label] = probability

    bank = ScenarioBank(
        demands=demands,
        probabilities=probabilities,
        seed=config.seed,
        metadata={
            "sampling_scheme": config.sampling_scheme,
            "n_scenarios": config.n_scenarios,
            "sigma_global": config.sigma_global,
            "sigma_local": config.sigma_local,
            "temporal_correlation": config.temporal_correlation,
            "tail_fraction": config.tail_fraction,
        },
    )
    bank.validate(reference=nominal)
    return bank


def align_demand_columns(demand: pd.DataFrame, bus_names: Iterable[str]) -> pd.DataFrame:
    """Return a copy whose columns exactly follow the model bus order."""
    buses = list(bus_names)
    frame = demand.copy()
    if set(buses).issubset(frame.columns):
        return frame.loc[:, buses].astype(float)
    if len(frame.columns) != len(buses):
        raise ValueError(
            f"Demand has {len(frame.columns)} columns but the network has {len(buses)} buses."
        )
    frame.columns = buses
    return frame.astype(float)
