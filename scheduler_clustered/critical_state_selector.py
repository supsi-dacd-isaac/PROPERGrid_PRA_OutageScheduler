"""Selection of one deterministic representative operating state per cluster."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from .cluster_scopf_oracle import ClusterSCOPFOracle, ClusterStateResult
from .outage_clusters import OutageCluster


@dataclass(frozen=True)
class RepresentativeState:
    time: str
    selection_mode: str
    screening_score: float
    total_demand: float
    base_load_shedding: float
    worst_screened_contingency: str | None


class CriticalStateSelector:
    """Choose one state using peak demand or PTDF/LODF criticality.

    For scalability, network-aware selection first shortlists the highest-load
    periods in a cluster and then evaluates their topology-aware contingency
    screening scores. This avoids solving a base OPF for every period in long
    clusters while retaining spatial information in the demand realization.
    """

    def __init__(
        self,
        data: Mapping,
        oracle: ClusterSCOPFOracle,
        *,
        mode: str = "network_screening",
        candidate_times: int = 12,
    ):
        if mode not in {"network_screening", "peak_total_demand"}:
            raise ValueError(
                "Representative-state mode must be 'network_screening' or "
                "'peak_total_demand'."
            )
        self.data = data
        self.oracle = oracle
        self.mode = mode
        self.candidate_times = int(candidate_times)
        self.times = list(data["T"])
        self.time_index = {time: index for index, time in enumerate(self.times)}
        self.demand = data["nodal_demand"].to_numpy(dtype=float)
        self._cache: dict[
            tuple[tuple[str, ...], str],
            tuple[ClusterStateResult, float, str | None],
        ] = {}

    def _total_demand(self, time: str) -> float:
        return float(np.sum(self.demand[self.time_index[time]]))

    def _shortlist(self, cluster: OutageCluster) -> list[str]:
        ordered = sorted(
            cluster.times,
            key=lambda time: (-self._total_demand(time), self.time_index[time]),
        )
        if self.candidate_times <= 0:
            return ordered
        return ordered[: min(self.candidate_times, len(ordered))]

    def _screen_time(
        self,
        time: str,
        active_outages: Sequence[str],
    ) -> tuple[ClusterStateResult, float, str | None]:
        key = (tuple(sorted(active_outages)), time)
        if key in self._cache:
            return self._cache[key]

        normal_result = self.oracle.solve(
            time,
            active_outages,
            contingencies=(),
        )
        normal = normal_result.normal_state
        outaged_lines = {
            outage
            for outage in active_outages
            if outage in self.oracle.network.line_index
        }
        active_indices = self.oracle.network.active_line_indices(outaged_lines)
        scores = self.oracle.network.branch_screening_scores(
            active_indices,
            normal.injections,
        )

        worst_score = -math.inf
        worst_contingency: str | None = None
        for contingency in self.oracle.effective_contingencies(active_outages):
            metadata = self.oracle.metadata[contingency]
            if metadata["type"] == "line":
                score = float(scores.get(str(metadata["element"]), -math.inf))
            elif metadata["type"] == "generator":
                lost_generation = abs(
                    normal.generation.get(str(metadata["element"]), 0.0)
                )
                score = lost_generation / max(
                    1.0, float(np.sum(self.oracle.network.pmax))
                )
            else:
                score = -math.inf
            if score > worst_score:
                worst_score = score
                worst_contingency = contingency

        result = (normal, float(worst_score), worst_contingency)
        self._cache[key] = result
        return result


    def screened_normal(
        self,
        time: str,
        active_outages: Sequence[str],
    ) -> tuple[ClusterStateResult, float, str | None]:
        """Return the cached normal state and screening result."""
        return self._screen_time(time, active_outages)

    def select(self, cluster: OutageCluster) -> RepresentativeState:
        if not cluster.times:
            raise ValueError(f"Cluster {cluster.cluster_id} contains no periods.")

        if self.mode == "peak_total_demand":
            time = max(
                cluster.times,
                key=lambda candidate: (
                    self._total_demand(candidate),
                    -self.time_index[candidate],
                ),
            )
            return RepresentativeState(
                time=time,
                selection_mode=self.mode,
                screening_score=float("nan"),
                total_demand=self._total_demand(time),
                base_load_shedding=float("nan"),
                worst_screened_contingency=None,
            )

        best_time: str | None = None
        best_normal: ClusterStateResult | None = None
        best_score = -math.inf
        best_contingency: str | None = None
        best_key: tuple[float, float, float, float] | None = None

        for time in self._shortlist(cluster):
            normal, screening_score, contingency = self._screen_time(
                time, cluster.active_outages
            )
            finite_score = (
                1e6 if math.isinf(screening_score) else screening_score
            )
            ranking = (
                float(normal.load_shedding > 1e-7),
                float(normal.load_shedding),
                finite_score,
                self._total_demand(time),
            )
            if best_key is None or ranking > best_key:
                best_key = ranking
                best_time = time
                best_normal = normal
                best_score = screening_score
                best_contingency = contingency

        if best_time is None or best_normal is None:
            raise RuntimeError(
                f"Could not select a representative state for {cluster.cluster_id}."
            )
        return RepresentativeState(
            time=best_time,
            selection_mode=self.mode,
            screening_score=float(best_score),
            total_demand=self._total_demand(best_time),
            base_load_shedding=float(best_normal.load_shedding),
            worst_screened_contingency=best_contingency,
        )
