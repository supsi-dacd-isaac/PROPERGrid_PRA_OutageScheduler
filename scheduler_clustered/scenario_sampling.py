"""Reproducible demand-scenario sampling for outage clusters.

The built-in empirical sampler preserves the observed spatial dependence among
nodal loads by resampling complete nodal-demand vectors from the periods within
an outage cluster.  Optional common lognormal multipliers can represent a
calibrated aggregate forecast-error distribution without destroying the nodal
pattern.  For research use, replace or override this sampler with externally
calibrated scenarios (for example, copula-based nodal trajectories).
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
        # Common random numbers make candidate schedules more comparable even
        # when their cluster boundaries differ. Tail-stratified quadrature
        # allocates explicit probability mass above the CVaR threshold, so a
        # moderate scenario set remains informative.
        if self.sampling_scheme == "stratified_tail":
            if self.scenario_count == 2:
                n_body, n_shoulder, n_tail = 1, 0, 1
            else:
                n_tail = max(1, self.scenario_count // 4)
                n_shoulder = max(1, self.scenario_count // 4)
                n_body = self.scenario_count - n_shoulder - n_tail
            pieces = [np.linspace(0.0, 0.80, n_body + 1)]
            if n_shoulder > 0:
                pieces.append(
                    np.linspace(0.80, 0.95, n_shoulder + 1)[1:]
                )
            pieces.append(np.linspace(0.95, 1.0, n_tail + 1)[1:])
            edges = np.concatenate(pieces)
            if len(edges) != self.scenario_count + 1:
                raise RuntimeError("Internal stratified-scenario construction failed.")
            self.quantiles = 0.5 * (edges[:-1] + edges[1:])
            default_probabilities = np.diff(edges)
            normal = NormalDist()
            standard_normal = np.asarray(
                [normal.inv_cdf(min(max(q, 1e-9), 1.0 - 1e-9)) for q in self.quantiles],
                dtype=float,
            )
        else:
            self.quantiles = rng.uniform(0.0, 1.0, size=self.scenario_count)
            default_probabilities = np.full(
                self.scenario_count, 1.0 / self.scenario_count, dtype=float
            )
            standard_normal = rng.standard_normal(self.scenario_count)

        self.multipliers = np.exp(
            -0.5 * self.sigma**2 + self.sigma * standard_normal
        )
        if probabilities is None:
            self.probabilities = default_probabilities / default_probabilities.sum()
        else:
            values = np.asarray(probabilities, dtype=float)
            if values.shape != (self.scenario_count,):
                raise ValueError(
                    "The probability vector must have scenario_count entries."
                )
            if np.any(values < 0.0) or values.sum() <= 0.0:
                raise ValueError("Scenario probabilities must be non-negative and nonzero.")
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
        # Sorting by total demand maps a common quantile to a comparable demand
        # percentile in every candidate-specific cluster while preserving the
        # full nodal pattern of the selected historical period.
        return sorted(
            cluster.times,
            key=lambda time: (
                float(np.sum(self.demand[self.time_index[time]])),
                self.time_index[time],
            ),
        )

    def scenarios_for_cluster(self, cluster: OutageCluster) -> list[DemandScenario]:
        if not cluster.times:
            raise ValueError(f"Cluster {cluster.cluster_id} contains no periods.")
        ordered_times = self._ordered_cluster_times(cluster)
        scenarios: list[DemandScenario] = []
        for index, scenario_id in enumerate(self.scenario_ids):
            quantile = min(max(float(self.quantiles[index]), 0.0), np.nextafter(1.0, 0.0))
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
                )
            )
        return scenarios


def demand_fingerprint(demand: Sequence[float] | np.ndarray) -> str:
    """Stable compact key for caching scenario-specific LP solutions."""
    values = np.asarray(demand, dtype=np.float64).reshape(-1)
    return hashlib.blake2b(values.tobytes(), digest_size=12).hexdigest()
