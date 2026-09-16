"""Reproducible demand-scenario generation for outage clusters.

Two samplers are provided.

``GaussianCriticalStateScenarioSampler``
    The default for the deterministic-versus-CVaR comparison.  It first takes
    the network-critical operating states selected for a constant-topology
    cluster and generates correlated Gaussian perturbations around those
    states.  Complete nodal demand vectors are retained, negative demand is
    truncated to zero, and common random numbers are used across candidate
    schedules.

``EmpiricalClusterScenarioSampler``
    A backwards-compatible empirical sampler that resamples complete nodal
    demand vectors from the periods contained in a cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import hashlib
import math
from statistics import NormalDist

import numpy as np

from .outage_clusters import OutageCluster


@dataclass(frozen=True)
class DemandScenario:
    scenario_id: str
    probability: float
    source_time: str
    demand: np.ndarray
    total_demand: float
    global_multiplier: float
    anchor_rank: int = 0
    sampling_model: str = "empirical"


def _nearest_psd_correlation(matrix: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    """Return a symmetric positive-semidefinite correlation matrix."""
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    psd = (vectors * values) @ vectors.T
    diagonal = np.sqrt(np.maximum(np.diag(psd), floor))
    correlation = psd / np.outer(diagonal, diagonal)
    np.fill_diagonal(correlation, 1.0)
    return correlation


class GaussianCriticalStateScenarioSampler:
    """Generate correlated Gaussian loads around selected critical states.

    The scenario bank is schedule-comparable because the same standard-normal
    vectors and scenario identifiers are reused for every candidate.  A
    scenario is anchored to one of the selected critical states in round-robin
    order, so the configured ``scenario_count`` is distributed as evenly as
    possible over the critical-state indices.

    Spatial correlation is estimated from historical relative nodal-demand
    deviations and shrunk toward the identity matrix for numerical stability.
    The perturbation model is

    ``d_omega = max(0, d_anchor * (1 + sigma_rel * z_spatial)
                         * exp(-0.5*sigma_global^2 + sigma_global*z_global))``.
    """

    def __init__(
        self,
        data: Mapping,
        *,
        scenario_count: int = 48,
        seed: int = 42,
        relative_sigma: float = 0.05,
        global_sigma: float = 0.03,
        correlation_shrinkage: float = 0.20,
        probabilities: Sequence[float] | None = None,
        antithetic: bool = True,
    ):
        if scenario_count < 2:
            raise ValueError("scenario_count must be at least two for CVaR.")
        if relative_sigma < 0.0 or global_sigma < 0.0:
            raise ValueError("Gaussian standard deviations cannot be negative.")
        if not 0.0 <= correlation_shrinkage <= 1.0:
            raise ValueError("correlation_shrinkage must lie in [0, 1].")

        self.times = list(data["T"])
        self.time_index = {time: index for index, time in enumerate(self.times)}
        self.demand = data["nodal_demand"].to_numpy(dtype=float)
        self.scenario_count = int(scenario_count)
        self.seed = int(seed)
        self.relative_sigma = float(relative_sigma)
        self.global_sigma = float(global_sigma)
        self.correlation_shrinkage = float(correlation_shrinkage)
        self.antithetic = bool(antithetic)

        if self.demand.ndim != 2 or self.demand.shape[0] != len(self.times):
            raise ValueError("nodal_demand must be a time-by-bus table.")
        if np.any(~np.isfinite(self.demand)) or np.any(self.demand < -1e-9):
            raise ValueError("nodal_demand must be finite and non-negative.")

        self._correlation = self._estimate_spatial_correlation()
        self._factor = self._factorize(self._correlation)
        self._spatial_draws, self._global_draws = self._build_common_draws()

        if probabilities is None:
            self.probabilities = np.full(
                self.scenario_count, 1.0 / self.scenario_count, dtype=float
            )
        else:
            values = np.asarray(probabilities, dtype=float)
            if values.shape != (self.scenario_count,):
                raise ValueError(
                    "The probability vector must have scenario_count entries."
                )
            if np.any(values < 0.0) or values.sum() <= 0.0:
                raise ValueError(
                    "Scenario probabilities must be non-negative and nonzero."
                )
            self.probabilities = values / values.sum()

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(
            f"scenario_{index:03d}" for index in range(self.scenario_count)
        )

    def probability_mapping(self) -> dict[str, float]:
        return {
            scenario_id: float(self.probabilities[index])
            for index, scenario_id in enumerate(self.scenario_ids)
        }

    def _estimate_spatial_correlation(self) -> np.ndarray:
        n_buses = self.demand.shape[1]
        means = np.mean(self.demand, axis=0)
        scale = np.where(means > 1e-9, means, 1.0)
        relative = (self.demand - means) / scale
        if relative.shape[0] < 2:
            empirical = np.eye(n_buses)
        else:
            empirical = np.eye(n_buses)
            variable = np.flatnonzero(np.std(relative, axis=0) > 1e-12)
            if variable.size >= 2:
                sub = np.corrcoef(relative[:, variable], rowvar=False)
                sub = np.nan_to_num(
                    np.asarray(sub, dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                empirical[np.ix_(variable, variable)] = sub
        shrunk = (
            (1.0 - self.correlation_shrinkage) * empirical
            + self.correlation_shrinkage * np.eye(n_buses)
        )
        return _nearest_psd_correlation(shrunk)

    @staticmethod
    def _factorize(correlation: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            values, vectors = np.linalg.eigh(correlation)
            return vectors @ np.diag(np.sqrt(np.maximum(values, 1e-10)))

    def _build_common_draws(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        n_buses = self.demand.shape[1]
        if self.antithetic:
            base_count = (self.scenario_count + 1) // 2
            base_spatial = rng.standard_normal((base_count, n_buses))
            base_global = rng.standard_normal(base_count)
            spatial = np.vstack([base_spatial, -base_spatial])[: self.scenario_count]
            global_draws = np.concatenate([base_global, -base_global])[
                : self.scenario_count
            ]
        else:
            spatial = rng.standard_normal((self.scenario_count, n_buses))
            global_draws = rng.standard_normal(self.scenario_count)
        correlated = spatial @ self._factor.T
        return correlated, global_draws

    def scenarios_for_cluster(
        self,
        cluster: OutageCluster,
        representatives: Sequence[object],
    ) -> list[DemandScenario]:
        if not representatives:
            raise ValueError(
                f"At least one critical state is required for {cluster.cluster_id}."
            )
        anchor_times = [str(getattr(state, "time")) for state in representatives]
        for time in anchor_times:
            if time not in self.time_index:
                raise KeyError(f"Unknown critical-state time {time!r}.")

        scenarios: list[DemandScenario] = []
        for index, scenario_id in enumerate(self.scenario_ids):
            anchor_rank = index % len(anchor_times)
            source_time = anchor_times[anchor_rank]
            anchor = np.asarray(
                self.demand[self.time_index[source_time]], dtype=float
            ).copy()
            local_factor = 1.0 + self.relative_sigma * self._spatial_draws[index]
            global_multiplier = float(
                np.exp(
                    -0.5 * self.global_sigma**2
                    + self.global_sigma * self._global_draws[index]
                )
            )
            sampled = np.maximum(0.0, anchor * local_factor * global_multiplier)
            scenarios.append(
                DemandScenario(
                    scenario_id=scenario_id,
                    probability=float(self.probabilities[index]),
                    source_time=source_time,
                    demand=sampled,
                    total_demand=float(np.sum(sampled)),
                    global_multiplier=global_multiplier,
                    anchor_rank=anchor_rank,
                    sampling_model="gaussian_critical_states",
                )
            )
        return scenarios


class EmpiricalClusterScenarioSampler:
    """Generate common-random-number empirical scenarios for each cluster."""

    def __init__(
        self,
        data: Mapping,
        *,
        scenario_count: int = 20,
        seed: int = 42,
        global_lognormal_sigma: float = 0.0,
        probabilities: Sequence[float] | None = None,
        sampling_scheme: str = "stratified_tail",
    ):
        if scenario_count < 2:
            raise ValueError("scenario_count must be at least two for CVaR.")
        if global_lognormal_sigma < 0.0:
            raise ValueError("global_lognormal_sigma cannot be negative.")
        self.times = list(data["T"])
        self.time_index = {time: index for index, time in enumerate(self.times)}
        self.demand = data["nodal_demand"].to_numpy(dtype=float)
        self.scenario_count = int(scenario_count)
        self.seed = int(seed)
        self.sigma = float(global_lognormal_sigma)
        if sampling_scheme not in {"random", "stratified_tail"}:
            raise ValueError(
                "sampling_scheme must be 'random' or 'stratified_tail'."
            )
        self.sampling_scheme = sampling_scheme

        rng = np.random.default_rng(self.seed)
        if self.sampling_scheme == "stratified_tail":
            if self.scenario_count == 2:
                n_body, n_shoulder, n_tail = 1, 0, 1
            else:
                n_tail = max(1, self.scenario_count // 4)
                n_shoulder = max(1, self.scenario_count // 4)
                n_body = self.scenario_count - n_shoulder - n_tail
            pieces = [np.linspace(0.0, 0.80, n_body + 1)]
            if n_shoulder > 0:
                pieces.append(np.linspace(0.80, 0.95, n_shoulder + 1)[1:])
            pieces.append(np.linspace(0.95, 1.0, n_tail + 1)[1:])
            edges = np.concatenate(pieces)
            if len(edges) != self.scenario_count + 1:
                raise RuntimeError(
                    "Internal stratified-scenario construction failed."
                )
            self.quantiles = 0.5 * (edges[:-1] + edges[1:])
            default_probabilities = np.diff(edges)
            normal = NormalDist()
            standard_normal = np.asarray(
                [
                    normal.inv_cdf(min(max(q, 1e-9), 1.0 - 1e-9))
                    for q in self.quantiles
                ],
                dtype=float,
            )
        else:
            self.quantiles = rng.uniform(0.0, 1.0, size=self.scenario_count)
            default_probabilities = np.full(
                self.scenario_count,
                1.0 / self.scenario_count,
                dtype=float,
            )
            standard_normal = rng.standard_normal(self.scenario_count)

        self.multipliers = np.exp(
            -0.5 * self.sigma**2 + self.sigma * standard_normal
        )
        if probabilities is None:
            self.probabilities = (
                default_probabilities / default_probabilities.sum()
            )
        else:
            values = np.asarray(probabilities, dtype=float)
            if values.shape != (self.scenario_count,):
                raise ValueError(
                    "The probability vector must have scenario_count entries."
                )
            if np.any(values < 0.0) or values.sum() <= 0.0:
                raise ValueError(
                    "Scenario probabilities must be non-negative and nonzero."
                )
            self.probabilities = values / values.sum()

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(
            f"scenario_{index:03d}" for index in range(self.scenario_count)
        )

    def probability_mapping(self) -> dict[str, float]:
        return {
            scenario_id: float(self.probabilities[index])
            for index, scenario_id in enumerate(self.scenario_ids)
        }

    def _ordered_cluster_times(self, cluster: OutageCluster) -> list[str]:
        return sorted(
            cluster.times,
            key=lambda time: (
                float(np.sum(self.demand[self.time_index[time]])),
                self.time_index[time],
            ),
        )

    def scenarios_for_cluster(
        self,
        cluster: OutageCluster,
        representatives: Sequence[object] | None = None,
    ) -> list[DemandScenario]:
        del representatives
        if not cluster.times:
            raise ValueError(f"Cluster {cluster.cluster_id} contains no periods.")
        ordered_times = self._ordered_cluster_times(cluster)
        scenarios: list[DemandScenario] = []
        for index, scenario_id in enumerate(self.scenario_ids):
            quantile = min(
                max(float(self.quantiles[index]), 0.0),
                np.nextafter(1.0, 0.0),
            )
            source_index = min(
                len(ordered_times) - 1,
                int(math.floor(quantile * len(ordered_times))),
            )
            source_time = ordered_times[source_index]
            multiplier = float(self.multipliers[index])
            demand = (
                np.asarray(
                    self.demand[self.time_index[source_time]], dtype=float
                ).copy()
                * multiplier
            )
            scenarios.append(
                DemandScenario(
                    scenario_id=scenario_id,
                    probability=float(self.probabilities[index]),
                    source_time=source_time,
                    demand=demand,
                    total_demand=float(np.sum(demand)),
                    global_multiplier=multiplier,
                    anchor_rank=0,
                    sampling_model="empirical_cluster",
                )
            )
        return scenarios


def demand_fingerprint(demand: Sequence[float] | np.ndarray) -> str:
    """Stable compact key for caching scenario-specific LP solutions."""
    values = np.asarray(demand, dtype=np.float64).reshape(-1)
    return hashlib.blake2b(values.tobytes(), digest_size=12).hexdigest()
