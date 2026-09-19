"""Selection of the most critical operating states in an outage cluster."""
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
    rank: int = 1


class CriticalStateSelector:
    """Rank operating states by load or topology-aware N-1 criticality.

    ``select_many`` returns the requested number of distinct critical periods.
    For scalability, network-aware ranking first shortlists the highest-load
    periods and then evaluates a normal-state SCOPF plus PTDF/LODF screening.
    ``select`` is retained as a compatibility wrapper returning rank one.
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

    def _shortlist(self, cluster: OutageCluster, count: int) -> list[str]:
        ordered = sorted(
            cluster.times,
            key=lambda time: (-self._total_demand(time), self.time_index[time]),
        )
        minimum = max(1, int(count))
        if self.candidate_times <= 0:
            return ordered
        limit = max(minimum, self.candidate_times)
        return ordered[: min(limit, len(ordered))]

    def _screen_time(
        self,
        time: str,
        active_outages: Sequence[str],
    ) -> tuple[ClusterStateResult, float, str | None]:
        key = (tuple(sorted(active_outages)), time)
        if key in self._cache:
            return self._cache[key]

        normal_result = self.oracle.solve(time, active_outages, contingencies=())
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

    def select_many(
        self,
        cluster: OutageCluster,
        count: int = 3,
    ) -> tuple[RepresentativeState, ...]:
        """Return up to ``count`` distinct critical periods, worst first."""
        if not cluster.times:
            raise ValueError(f"Cluster {cluster.cluster_id} contains no periods.")
        requested = max(1, int(count))
        selected_count = min(requested, len(cluster.times))

        if self.mode == "peak_total_demand":
            ordered = sorted(
                cluster.times,
                key=lambda time: (
                    -self._total_demand(time),
                    self.time_index[time],
                ),
            )[:selected_count]
            return tuple(
                RepresentativeState(
                    time=time,
                    selection_mode=self.mode,
                    screening_score=float("nan"),
                    total_demand=self._total_demand(time),
                    base_load_shedding=float("nan"),
                    worst_screened_contingency=None,
                    rank=rank,
                )
                for rank, time in enumerate(ordered, start=1)
            )

        ranked: list[
            tuple[
                tuple[float, float, float, float, float],
                str,
                ClusterStateResult,
                float,
                str | None,
            ]
        ] = []
        for time in self._shortlist(cluster, selected_count):
            normal, screening_score, contingency = self._screen_time(
                time, cluster.active_outages
            )
            finite_score = 1e6 if math.isinf(screening_score) else screening_score
            ranking = (
                float(normal.load_shedding > 1e-7),
                float(normal.load_shedding),
                float(finite_score),
                self._total_demand(time),
                -float(self.time_index[time]),
            )
            ranked.append(
                (ranking, time, normal, screening_score, contingency)
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        result: list[RepresentativeState] = []
        for rank, (_, time, normal, score, contingency) in enumerate(
            ranked[:selected_count], start=1
        ):
            result.append(
                RepresentativeState(
                    time=time,
                    selection_mode=self.mode,
                    screening_score=float(score),
                    total_demand=self._total_demand(time),
                    base_load_shedding=float(normal.load_shedding),
                    worst_screened_contingency=contingency,
                    rank=rank,
                )
            )
        if not result:
            raise RuntimeError(
                f"Could not select critical states for {cluster.cluster_id}."
            )
        return tuple(result)

    def select(self, cluster: OutageCluster) -> RepresentativeState:
        """Compatibility wrapper returning the highest-ranked state."""
        return self.select_many(cluster, count=1)[0]
