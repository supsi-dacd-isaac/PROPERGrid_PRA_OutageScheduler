"""Deterministic outage-cluster master/slave decomposition.

Two operating modes are provided:

``penalty``
    Every slave is feasible through load shedding. The master is guided by
    learned linear security penalties and no-good exploration. This mode
    always retains the best evaluated complete schedule and is the recommended
    deterministic soft-security formulation.

``budget``
    Outage clusters whose incremental curtailment exceeds a configured budget
    generate overlap cuts. This supplies a deterministic security certificate
    for the selected representative states, but the complete schedule may be
    infeasible unless outage deferral is enabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import json
import logging
import math

import numpy as np

from .cluster_scopf_oracle import (
    ClusterSCOPFOracle,
    ClusterSCOPFResult,
)
from .critical_state_selector import (
    CriticalStateSelector,
    RepresentativeState,
)
from .master_problem import (
    ConflictCut,
    MasterSolution,
    MasterState,
    NoGoodCut,
    SchedulingMaster,
    no_good_from_solution,
)
from .outage_clusters import OutageCluster, build_outage_clusters
from .serialization import json_safe

logger = logging.getLogger(__name__)


@dataclass
class ClusteredDeterministicConfig:
    max_iterations: int = 50
    master_time_limit: float = 180.0
    master_mip_gap: float = 0.01
    master_threads: int = 0
    representative_state_mode: str = "network_screening"
    representative_candidate_times: int = 12
    contingency_top_k: int | None = None
    mandatory_contingencies: tuple[str, ...] = ()
    security_mode: str = "penalty"  # penalty or budget
    security_penalty_weight: float = 10.0
    total_dns_component_weight: float = 0.10
    risk_proxy_learning_rate: float = 0.40
    maximum_incremental_dns_budget: float = 0.25
    total_incremental_dns_budget: float = math.inf
    max_cluster_cuts_per_iteration: int = 5
    max_candidates: int = 20
    patience: int = 8
    stop_on_first_budget_feasible: bool = True
    full_contingency_validation: bool = True
    results_path: str = "clustered_deterministic_results.json"
    iis_path: str = "cluster_master_infeasibility.ilp"


@dataclass
class ClusterEvaluation:
    cluster: OutageCluster
    representative: RepresentativeState
    candidate: ClusterSCOPFResult
    baseline: ClusterSCOPFResult
    maximum_incremental_dns: float
    total_incremental_dns: float
    normalized_severity: float
    budget_feasible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster": self.cluster.to_dict(),
            "representative": asdict(self.representative),
            "candidate": {
                "time": self.candidate.time,
                "active_planned_outages": self.candidate.active_planned_outages,
                "contingencies": self.candidate.contingencies,
                "status": self.candidate.status,
                "objective": self.candidate.objective,
                "maximum_dns": self.candidate.maximum_dns,
                "total_dns": self.candidate.total_dns,
                "worst_contingency": self.candidate.worst_contingency,
            },
            "baseline": {
                "maximum_dns": self.baseline.maximum_dns,
                "total_dns": self.baseline.total_dns,
                "worst_contingency": self.baseline.worst_contingency,
            },
            "maximum_incremental_dns": self.maximum_incremental_dns,
            "total_incremental_dns": self.total_incremental_dns,
            "normalized_severity": self.normalized_severity,
            "budget_feasible": self.budget_feasible,
        }


@dataclass
class ClusterCandidateRecord:
    iteration: int
    start_times: dict[str, str]
    deferred_outages: tuple[str, ...]
    maintenance_utility: float
    deterministic_security_cost: float
    score: float
    budget_feasible: bool
    clusters: list[ClusterEvaluation]

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "start_times": self.start_times,
            "deferred_outages": self.deferred_outages,
            "maintenance_utility": self.maintenance_utility,
            "deterministic_security_cost": self.deterministic_security_cost,
            "score": self.score,
            "budget_feasible": self.budget_feasible,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


@dataclass
class ClusteredDeterministicResult:
    best_schedule: dict[str, str] | None
    deferred_outages: tuple[str, ...]
    best_active_outages: dict[str, tuple[str, ...]] | None
    best_score: float | None
    deterministic_security_cost: float | None
    iterations: int
    candidates: list[ClusterCandidateRecord]
    conflict_cuts_added: int
    no_good_cuts_added: int
    termination_reason: str
    full_validation: list[ClusterEvaluation] = field(default_factory=list)


class ClusteredDeterministicScheduler:
    """Solve the deterministic cluster-based scheduling formulation."""

    def __init__(
        self,
        data: Mapping,
        config: ClusteredDeterministicConfig | None = None,
    ):
        self.data = data
        self.config = config or ClusteredDeterministicConfig()
        if self.config.security_mode not in {"penalty", "budget"}:
            raise ValueError("security_mode must be 'penalty' or 'budget'.")

        self.state = MasterState()
        self.oracle = ClusterSCOPFOracle(data)
        self.selector = CriticalStateSelector(
            data,
            self.oracle,
            mode=self.config.representative_state_mode,
            candidate_times=self.config.representative_candidate_times,
        )
        total_demand = data["nodal_demand"].sum(axis=1).to_numpy(dtype=float)
        self.peak_demand = max(1.0, float(np.max(total_demand)))
        self.horizon = max(1, len(data["T"]))
        self.baseline_cache: dict[
            tuple[str, tuple[str, ...]], ClusterSCOPFResult
        ] = {}
        self.candidates: list[ClusterCandidateRecord] = []
        self.best_solution: MasterSolution | None = None
        self.best_score = -math.inf
        self.best_security_cost: float | None = None
        self.best_cluster_evaluations: list[ClusterEvaluation] = []
        self._seen_conflicts: set[tuple[str, tuple[str, ...]]] = set()
        self._seen_schedules: set[
            tuple[tuple[tuple[str, str], ...], tuple[str, ...]]
        ] = set()

    def _selected_contingencies(
        self,
        representative: RepresentativeState,
        active_outages: Sequence[str],
    ) -> tuple[str, ...]:
        if self.config.contingency_top_k is None:
            return self.oracle.effective_contingencies(active_outages)
        normal, _, _ = self.selector.screened_normal(
            representative.time, active_outages
        )
        return self.oracle.rank_contingencies(
            normal,
            active_outages,
            top_k=self.config.contingency_top_k,
            mandatory=self.config.mandatory_contingencies,
        )

    def _baseline(
        self,
        time: str,
        contingencies: Sequence[str],
    ) -> ClusterSCOPFResult:
        key = (time, tuple(contingencies))
        if key not in self.baseline_cache:
            self.baseline_cache[key] = self.oracle.solve(
                time,
                (),
                contingencies=contingencies,
            )
        return self.baseline_cache[key]

    def _evaluate_cluster(self, cluster: OutageCluster) -> ClusterEvaluation:
        representative = self.selector.select(cluster)
        contingencies = self._selected_contingencies(
            representative, cluster.active_outages
        )
        candidate = self.oracle.solve(
            representative.time,
            cluster.active_outages,
            contingencies=contingencies,
        )
        baseline = self._baseline(representative.time, contingencies)

        maximum_increment = max(
            0.0, candidate.maximum_dns - baseline.maximum_dns
        )
        total_increment = max(0.0, candidate.total_dns - baseline.total_dns)
        average_increment = total_increment / max(
            1, len(candidate.contingencies) + 1
        )
        duration_weight = cluster.duration_steps / self.horizon
        normalized = duration_weight * (
            maximum_increment
            + self.config.total_dns_component_weight * average_increment
        ) / self.peak_demand
        budget_feasible = (
            maximum_increment
            <= self.config.maximum_incremental_dns_budget + 1e-9
            and total_increment
            <= self.config.total_incremental_dns_budget + 1e-9
        )
        return ClusterEvaluation(
            cluster=cluster,
            representative=representative,
            candidate=candidate,
            baseline=baseline,
            maximum_incremental_dns=float(maximum_increment),
            total_incremental_dns=float(total_increment),
            normalized_severity=float(normalized),
            budget_feasible=budget_feasible,
        )

    def _evaluate_solution(
        self,
        solution: MasterSolution,
        *,
        force_full_contingencies: bool = False,
    ) -> list[ClusterEvaluation]:
        clusters = build_outage_clusters(
            self.data["T"],
            solution.active_outages,
            include_empty=False,
        )
        if not force_full_contingencies:
            return [self._evaluate_cluster(cluster) for cluster in clusters]

        original_top_k = self.config.contingency_top_k
        try:
            self.config.contingency_top_k = None
            return [self._evaluate_cluster(cluster) for cluster in clusters]
        finally:
            self.config.contingency_top_k = original_top_k

    def _security_cost(
        self, cluster_evaluations: Sequence[ClusterEvaluation]
    ) -> float:
        return float(
            sum(cluster.normalized_severity for cluster in cluster_evaluations)
        )

    def _candidate_score(
        self,
        solution: MasterSolution,
        security_cost: float,
    ) -> float:
        return float(
            solution.maintenance_utility
            - self.config.security_penalty_weight * security_cost
        )

    def _update_proxy(
        self,
        cluster_evaluations: Sequence[ClusterEvaluation],
    ) -> None:
        learning_rate = float(self.config.risk_proxy_learning_rate)
        for evaluation in cluster_evaluations:
            outages = evaluation.cluster.active_outages
            times = evaluation.cluster.times
            if not outages or not times or evaluation.normalized_severity <= 0:
                continue
            observed = (
                self.config.security_penalty_weight
                * evaluation.normalized_severity
                / (len(outages) * len(times))
            )
            for outage in outages:
                for time in times:
                    key = (outage, time)
                    previous = float(
                        self.state.risk_coefficients.get(key, 0.0)
                    )
                    self.state.risk_coefficients[key] = (
                        (1.0 - learning_rate) * previous
                        + learning_rate * observed
                    )

    def _add_no_good(self, solution: MasterSolution, label: str) -> bool:
        signature = solution.decision_signature
        if signature in self._seen_schedules:
            return False
        self._seen_schedules.add(signature)
        self.state.no_goods.append(
            no_good_from_solution(solution, label=label)
        )
        return True

    def _add_cluster_cuts(
        self,
        evaluations: Sequence[ClusterEvaluation],
    ) -> int:
        violating = sorted(
            (
                evaluation
                for evaluation in evaluations
                if not evaluation.budget_feasible
            ),
            key=lambda evaluation: (
                evaluation.maximum_incremental_dns,
                evaluation.total_incremental_dns,
            ),
            reverse=True,
        )
        added = 0
        for evaluation in violating[
            : self.config.max_cluster_cuts_per_iteration
        ]:
            outages = tuple(sorted(evaluation.cluster.active_outages))
            key = (evaluation.representative.time, outages)
            if not outages or key in self._seen_conflicts:
                continue
            self._seen_conflicts.add(key)
            self.state.conflicts.append(
                ConflictCut(
                    time=evaluation.representative.time,
                    outages=outages,
                    label="cluster_budget",
                )
            )
            added += 1
        return added

    def _save(self, result: ClusteredDeterministicResult) -> None:
        serial = {
            "best_schedule": result.best_schedule,
            "deferred_outages": result.deferred_outages,
            "best_active_outages": result.best_active_outages,
            "best_score": result.best_score,
            "deterministic_security_cost": result.deterministic_security_cost,
            "iterations": result.iterations,
            "candidates": [candidate.to_dict() for candidate in result.candidates],
            "conflict_cuts_added": result.conflict_cuts_added,
            "no_good_cuts_added": result.no_good_cuts_added,
            "termination_reason": result.termination_reason,
            "full_validation": [
                evaluation.to_dict() for evaluation in result.full_validation
            ],
            "config": asdict(self.config),
        }
        Path(self.config.results_path).write_text(
            json.dumps(json_safe(serial), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def solve(self) -> ClusteredDeterministicResult:
        termination = "iteration_limit"
        no_improvement = 0
        last_iteration = 0

        for iteration in range(1, self.config.max_iterations + 1):
            last_iteration = iteration
            logger.info(
                "Iteration %d: %d cluster cuts, %d no-good cuts, %d learned coefficients",
                iteration,
                len(self.state.conflicts),
                len(self.state.no_goods),
                len(self.state.risk_coefficients),
            )
            master = SchedulingMaster(self.data, self.state)
            solution = master.solve(
                mip_gap=self.config.master_mip_gap,
                time_limit=self.config.master_time_limit,
                threads=self.config.master_threads,
                iis_path=self.config.iis_path,
            )
            if not solution.start_times and not solution.deferred_outages:
                termination = "master_infeasible_or_no_solution"
                break

            cluster_evaluations = self._evaluate_solution(solution)
            security_cost = self._security_cost(cluster_evaluations)
            score = self._candidate_score(solution, security_cost)
            budget_feasible = all(
                evaluation.budget_feasible
                for evaluation in cluster_evaluations
            )
            candidate = ClusterCandidateRecord(
                iteration=iteration,
                start_times=dict(solution.start_times),
                deferred_outages=tuple(solution.deferred_outages),
                maintenance_utility=float(solution.maintenance_utility),
                deterministic_security_cost=security_cost,
                score=score,
                budget_feasible=budget_feasible,
                clusters=list(cluster_evaluations),
            )
            self.candidates.append(candidate)

            logger.info(
                "Candidate %d: %d clusters, security cost=%.6g, score=%.6g, budget feasible=%s",
                iteration,
                len(cluster_evaluations),
                security_cost,
                score,
                budget_feasible,
            )
            for evaluation in cluster_evaluations:
                logger.info(
                    "%s [%s..%s], outages=%s, representative=%s, "
                    "max incremental DNS=%.6g MW, total incremental DNS=%.6g MW",
                    evaluation.cluster.cluster_id,
                    evaluation.cluster.start_time,
                    evaluation.cluster.end_time,
                    evaluation.cluster.active_outages,
                    evaluation.representative.time,
                    evaluation.maximum_incremental_dns,
                    evaluation.total_incremental_dns,
                )

            acceptable_candidate = (
                self.config.security_mode == "penalty" or budget_feasible
            )
            if acceptable_candidate and score > self.best_score + 1e-12:
                self.best_score = score
                self.best_solution = solution
                self.best_security_cost = security_cost
                self.best_cluster_evaluations = list(cluster_evaluations)
                no_improvement = 0
            else:
                no_improvement += 1

            self._update_proxy(cluster_evaluations)

            if self.config.security_mode == "budget" and not budget_feasible:
                cuts = self._add_cluster_cuts(cluster_evaluations)
                if cuts == 0:
                    self._add_no_good(solution, "unresolved_cluster_budget")
                continue

            if (
                self.config.security_mode == "budget"
                and budget_feasible
                and self.config.stop_on_first_budget_feasible
            ):
                termination = "first_budget_feasible_schedule"
                break

            if len(self.candidates) >= self.config.max_candidates:
                termination = "candidate_limit"
                break
            if no_improvement >= self.config.patience:
                termination = "no_improvement"
                break
            self._add_no_good(solution, "deterministic_exploration")

        full_validation: list[ClusterEvaluation] = []
        if (
            self.best_solution is not None
            and self.config.full_contingency_validation
            and self.config.contingency_top_k is not None
        ):
            logger.info(
                "Validating the best schedule using the complete contingency set."
            )
            full_validation = self._evaluate_solution(
                self.best_solution,
                force_full_contingencies=True,
            )

        result = ClusteredDeterministicResult(
            best_schedule=(
                None
                if self.best_solution is None
                else dict(self.best_solution.start_times)
            ),
            deferred_outages=(
                ()
                if self.best_solution is None
                else tuple(self.best_solution.deferred_outages)
            ),
            best_active_outages=(
                None
                if self.best_solution is None
                else dict(self.best_solution.active_outages)
            ),
            best_score=(
                None if self.best_solution is None else float(self.best_score)
            ),
            deterministic_security_cost=self.best_security_cost,
            iterations=last_iteration,
            candidates=list(self.candidates),
            conflict_cuts_added=len(self.state.conflicts),
            no_good_cuts_added=len(self.state.no_goods),
            termination_reason=termination,
            full_validation=full_validation,
        )
        self._save(result)
        return result
