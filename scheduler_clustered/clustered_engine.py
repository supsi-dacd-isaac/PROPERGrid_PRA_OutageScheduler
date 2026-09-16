"""Deterministic outage-cluster master/slave decomposition.

The master schedules or defers outages using compact start variables.  For each
complete schedule, maximal constant-topology clusters are constructed.  The
slave evaluates the ``q`` most critical operating periods in every cluster with
one soft preventive/corrective N-1 SCOPF per period.  The exact evaluated
schedule score is

``maintenance utility - beta * deterministic DNS severity``.

The master contains a learned linear proxy of the slave severity.  Consequently
the outer method is a simulation-optimisation matheuristic rather than a
classical exact Benders implementation; every retained candidate is nevertheless
scored using the full slave output defined above.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import json
import logging
import math
import threading

import numpy as np

from .cluster_scopf_oracle import ClusterSCOPFOracle, ClusterSCOPFResult
from .critical_state_selector import CriticalStateSelector, RepresentativeState
from .master_problem import (
    ConflictCut,
    MasterSolution,
    MasterState,
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
    oracle_workers: int = 1
    representative_state_mode: str = "network_screening"
    representative_candidate_times: int = 12
    critical_state_count: int = 3
    include_empty_clusters: bool = False
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
class DeterministicStateEvaluation:
    representative: RepresentativeState
    contingencies: tuple[str, ...]
    candidate: ClusterSCOPFResult
    baseline: ClusterSCOPFResult
    maximum_incremental_dns: float
    total_incremental_dns: float
    mean_incremental_dns: float
    nodal_incremental_dns: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "representative": asdict(self.representative),
            "contingencies": self.contingencies,
            "candidate": _scopf_summary(self.candidate),
            "baseline": _scopf_summary(self.baseline),
            "maximum_incremental_dns": self.maximum_incremental_dns,
            "total_incremental_dns": self.total_incremental_dns,
            "mean_incremental_dns": self.mean_incremental_dns,
            "nodal_incremental_dns": self.nodal_incremental_dns,
        }


def _scopf_summary(result: ClusterSCOPFResult) -> dict[str, object]:
    return {
        "time": result.time,
        "active_planned_outages": result.active_planned_outages,
        "contingencies": result.contingencies,
        "status": result.status,
        "objective": result.objective,
        "maximum_dns": result.maximum_dns,
        "total_dns": result.total_dns,
        "worst_contingency": result.worst_contingency,
        "nodal_dns_by_state": {
            state_label: {
                bus: float(value)
                for bus, value in state.load_shedding_by_bus.items()
                if abs(float(value)) > 1e-9
            }
            for state_label, state in result.states.items()
        },
    }


def _summed_nodal_dns(result: ClusterSCOPFResult) -> dict[str, float]:
    totals: dict[str, float] = {}
    for state in result.states.values():
        for bus, value in state.load_shedding_by_bus.items():
            totals[bus] = totals.get(bus, 0.0) + float(value)
    return totals


@dataclass
class ClusterEvaluation:
    cluster: OutageCluster
    state_evaluations: list[DeterministicStateEvaluation]
    maximum_incremental_dns: float
    total_incremental_dns: float
    mean_incremental_dns: float
    nodal_incremental_dns: dict[str, float]
    normalized_severity: float
    budget_feasible: bool

    @property
    def worst_state(self) -> DeterministicStateEvaluation:
        return max(
            self.state_evaluations,
            key=lambda item: (
                item.maximum_incremental_dns,
                item.total_incremental_dns,
                item.representative.total_demand,
            ),
        )

    @property
    def representative(self) -> RepresentativeState:
        return self.worst_state.representative

    @property
    def representatives(self) -> tuple[RepresentativeState, ...]:
        return tuple(item.representative for item in self.state_evaluations)

    @property
    def candidate(self) -> ClusterSCOPFResult:
        return self.worst_state.candidate

    @property
    def baseline(self) -> ClusterSCOPFResult:
        return self.worst_state.baseline

    def to_dict(self) -> dict[str, object]:
        worst = self.worst_state
        return {
            "cluster": self.cluster.to_dict(),
            # Compatibility fields used by the existing plotting layer.
            "representative": asdict(worst.representative),
            "representatives": [
                asdict(item.representative) for item in self.state_evaluations
            ],
            "candidate": _scopf_summary(worst.candidate),
            "baseline": _scopf_summary(worst.baseline),
            "state_evaluations": [
                item.to_dict() for item in self.state_evaluations
            ],
            "maximum_incremental_dns": self.maximum_incremental_dns,
            "total_incremental_dns": self.total_incremental_dns,
            "mean_incremental_dns": self.mean_incremental_dns,
            "nodal_incremental_dns": self.nodal_incremental_dns,
            "normalized_severity": self.normalized_severity,
            "budget_feasible": self.budget_feasible,
        }


@dataclass
class ClusterCandidateRecord:
    iteration: int
    start_times: dict[str, str]
    deferred_outages: tuple[str, ...]
    scheduled_outage_count: int
    coverage_utility: float
    timing_utility: float
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
            "scheduled_outage_count": self.scheduled_outage_count,
            "coverage_utility": self.coverage_utility,
            "timing_utility": self.timing_utility,
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
    """Solve the deterministic three-critical-state cluster formulation."""

    def __init__(
        self,
        data: Mapping,
        config: ClusteredDeterministicConfig | None = None,
    ):
        self.data = data
        self.config = config or ClusteredDeterministicConfig()
        if self.config.security_mode not in {"penalty", "budget"}:
            raise ValueError("security_mode must be 'penalty' or 'budget'.")
        if self.config.critical_state_count <= 0:
            raise ValueError("critical_state_count must be positive.")

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
        self._baseline_lock = threading.Lock()
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
        with self._baseline_lock:
            cached = self.baseline_cache.get(key)
        if cached is not None:
            return cached
        solved = self.oracle.solve(time, (), contingencies=contingencies)
        with self._baseline_lock:
            return self.baseline_cache.setdefault(key, solved)

    def _evaluate_state(
        self,
        cluster: OutageCluster,
        representative: RepresentativeState,
    ) -> DeterministicStateEvaluation:
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
        mean_increment = total_increment / max(
            1, len(candidate.contingencies) + 1
        )
        candidate_nodal = _summed_nodal_dns(candidate)
        baseline_nodal = _summed_nodal_dns(baseline)
        buses = set(candidate_nodal) | set(baseline_nodal)
        nodal_increment = {
            bus: max(
                0.0,
                float(candidate_nodal.get(bus, 0.0))
                - float(baseline_nodal.get(bus, 0.0)),
            )
            for bus in sorted(buses)
        }
        return DeterministicStateEvaluation(
            representative=representative,
            contingencies=tuple(contingencies),
            candidate=candidate,
            baseline=baseline,
            maximum_incremental_dns=float(maximum_increment),
            total_incremental_dns=float(total_increment),
            mean_incremental_dns=float(mean_increment),
            nodal_incremental_dns=nodal_increment,
        )

    def _evaluate_cluster(self, cluster: OutageCluster) -> ClusterEvaluation:
        representatives = self.selector.select_many(
            cluster, count=self.config.critical_state_count
        )
        if self.config.oracle_workers <= 1 or len(representatives) <= 1:
            state_evaluations = [
                self._evaluate_state(cluster, representative)
                for representative in representatives
            ]
        else:
            by_time: dict[str, DeterministicStateEvaluation] = {}
            with ThreadPoolExecutor(
                max_workers=self.config.oracle_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_state, cluster, representative
                    ): representative.time
                    for representative in representatives
                }
                for future in as_completed(futures):
                    by_time[futures[future]] = future.result()
            state_evaluations = [
                by_time[representative.time]
                for representative in representatives
            ]
        maximum_increment = max(
            state.maximum_incremental_dns for state in state_evaluations
        )
        total_increment = sum(
            state.total_incremental_dns for state in state_evaluations
        )
        mean_increment = float(
            np.mean([state.mean_incremental_dns for state in state_evaluations])
        )
        nodal_increment: dict[str, float] = {}
        for state in state_evaluations:
            for bus, value in state.nodal_incremental_dns.items():
                nodal_increment[bus] = nodal_increment.get(bus, 0.0) + float(value)

        duration_weight = cluster.duration_steps / self.horizon
        normalized = duration_weight * (
            maximum_increment
            + self.config.total_dns_component_weight * mean_increment
        ) / self.peak_demand
        budget_feasible = (
            maximum_increment
            <= self.config.maximum_incremental_dns_budget + 1e-9
            and total_increment
            <= self.config.total_incremental_dns_budget + 1e-9
        )
        return ClusterEvaluation(
            cluster=cluster,
            state_evaluations=state_evaluations,
            maximum_incremental_dns=float(maximum_increment),
            total_incremental_dns=float(total_increment),
            mean_incremental_dns=float(mean_increment),
            nodal_incremental_dns=nodal_increment,
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
            include_empty=self.config.include_empty_clusters,
        )
        if not force_full_contingencies:
            return [self._evaluate_cluster(cluster) for cluster in clusters]

        original_top_k = self.config.contingency_top_k
        try:
            self.config.contingency_top_k = None
            return [self._evaluate_cluster(cluster) for cluster in clusters]
        finally:
            self.config.contingency_top_k = original_top_k

    @staticmethod
    def _security_cost(
        cluster_evaluations: Sequence[ClusterEvaluation],
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
                    previous = float(self.state.risk_coefficients.get(key, 0.0))
                    self.state.risk_coefficients[key] = (
                        (1.0 - learning_rate) * previous
                        + learning_rate * observed
                    )

    def _add_no_good(self, solution: MasterSolution, label: str) -> bool:
        signature = solution.decision_signature
        if signature in self._seen_schedules:
            return False
        self._seen_schedules.add(signature)
        self.state.no_goods.append(no_good_from_solution(solution, label=label))
        return True

    def _add_cluster_cuts(
        self,
        evaluations: Sequence[ClusterEvaluation],
    ) -> int:
        violating = sorted(
            (evaluation for evaluation in evaluations if not evaluation.budget_feasible),
            key=lambda evaluation: (
                evaluation.maximum_incremental_dns,
                evaluation.total_incremental_dns,
            ),
            reverse=True,
        )
        added = 0
        for evaluation in violating[: self.config.max_cluster_cuts_per_iteration]:
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
            "deterministic_risk_definition": (
                "For every constant-topology cluster, evaluate the q most "
                "critical periods. Severity is duration weighted and combines "
                "maximum incremental DNS with the mean summed nodal DNS over "
                "normal and N-1 states."
            ),
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
            if solution.objective is None:
                termination = "master_infeasible_or_no_solution"
                break

            cluster_evaluations = self._evaluate_solution(solution)
            security_cost = self._security_cost(cluster_evaluations)
            score = self._candidate_score(solution, security_cost)
            budget_feasible = all(
                evaluation.budget_feasible for evaluation in cluster_evaluations
            )
            candidate = ClusterCandidateRecord(
                iteration=iteration,
                start_times=dict(solution.start_times),
                deferred_outages=tuple(solution.deferred_outages),
                scheduled_outage_count=solution.scheduled_outage_count,
                coverage_utility=float(solution.coverage_utility),
                timing_utility=float(solution.timing_utility),
                maintenance_utility=float(solution.maintenance_utility),
                deterministic_security_cost=security_cost,
                score=score,
                budget_feasible=budget_feasible,
                clusters=list(cluster_evaluations),
            )
            self.candidates.append(candidate)

            logger.info(
                "Candidate %d: scheduled=%d, clusters=%d, security cost=%.6g, "
                "score=%.6g, budget feasible=%s",
                iteration,
                solution.scheduled_outage_count,
                len(cluster_evaluations),
                security_cost,
                score,
                budget_feasible,
            )
            for evaluation in cluster_evaluations:
                logger.info(
                    "%s [%s..%s], outages=%s, critical states=%s, "
                    "max incremental DNS=%.6g MW, summed incremental DNS=%.6g MW",
                    evaluation.cluster.cluster_id,
                    evaluation.cluster.start_time,
                    evaluation.cluster.end_time,
                    evaluation.cluster.active_outages,
                    tuple(state.time for state in evaluation.representatives),
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
                self.best_solution, force_full_contingencies=True
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
