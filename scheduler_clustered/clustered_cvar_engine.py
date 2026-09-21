"""Clustered stochastic outage scheduling with finite-sample CVaR of DNS.

For each candidate schedule, constant-topology outage clusters are built.  The
``q`` most critical operating states in each cluster anchor a reproducible bank
of correlated Gaussian nodal-demand scenarios.  Every scenario is evaluated by
the same soft preventive/corrective N-1 SCOPF used by the deterministic
scheduler.  The schedule loss combines worst-state DNS and summed nodal DNS,
weighted by cluster duration, and finite-sample CVaR is computed exactly.

The outer method is a simulation-optimisation matheuristic.  A learned linear
risk proxy guides the compact scheduling master, while every complete candidate
is evaluated with the exact finite scenario bank.  When a deterministic
benchmark schedule is supplied and ``selection_rule`` is
``utility_constrained_cvar``, that benchmark is included in the candidate set;
therefore the selected CVaR schedule cannot have higher *in-sample* CVaR among
utility-admissible candidates.  Independent paired validation remains necessary
for an out-of-sample claim.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import logging
import math
import threading

import numpy as np

from .cluster_scopf_oracle import ClusterSCOPFOracle, ClusterSCOPFResult
from .critical_state_selector import CriticalStateSelector, RepresentativeState
from .master_problem import (
    MasterSolution,
    MasterState,
    SchedulingMaster,
    no_good_from_solution,
    solution_from_start_times,
)
from .outage_clusters import OutageCluster, build_outage_clusters
from .risk_metrics import RiskSummary, weighted_var_cvar
from .serialization import json_safe
from .scenario_sampling import (
    DemandScenario,
    EmpiricalClusterScenarioSampler,
    GaussianCriticalStateScenarioSampler,
    demand_fingerprint,
)

logger = logging.getLogger(__name__)


def _summed_nodal_dns(result: ClusterSCOPFResult) -> dict[str, float]:
    totals: dict[str, float] = {}
    for state in result.states.values():
        for bus, value in state.load_shedding_by_bus.items():
            totals[bus] = totals.get(bus, 0.0) + float(value)
    return totals


@dataclass
class ClusteredCVaRConfig:
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

    scenario_model: str = "gaussian_critical_states"
    scenario_count: int = 48
    gaussian_samples_per_critical_state: int | None = 16
    scenario_seed: int = 42
    gaussian_relative_sigma: float = 0.05
    gaussian_global_sigma: float = 0.03
    gaussian_correlation_shrinkage: float = 0.20
    gaussian_antithetic: bool = True
    # Legacy empirical-sampler settings.
    scenario_sampling_scheme: str = "stratified_tail"
    global_lognormal_sigma: float = 0.0

    maximum_dns_loss_weight: float = 1.0
    total_dns_component_weight: float = 0.10
    risk_mode: str = "cvar_objective"  # cvar_objective or cvar_constraint
    cvar_alpha: float = 0.95
    cvar_penalty_weight: float = 10.0
    expected_loss_penalty_weight: float = 0.0
    cvar_limit_mw: float | None = None
    risk_proxy_learning_rate: float = 0.40

    selection_rule: str = "utility_constrained_cvar"
    relative_utility_tolerance: float = 0.01
    absolute_utility_tolerance: float = 0.0
    max_candidates: int = 20
    patience: int = 8
    min_candidate_changes: int = 1
    full_contingency_validation: bool = True
    results_path: str = "clustered_cvar_results.json"
    iis_path: str = "cluster_cvar_master_infeasibility.ilp"


@dataclass
class ScenarioClusterResult:
    scenario_id: str
    probability: float
    source_time: str
    anchor_rank: int
    sampling_model: str
    total_demand: float
    global_multiplier: float
    candidate_maximum_dns: float
    candidate_total_dns: float
    baseline_maximum_dns: float
    baseline_total_dns: float
    incremental_maximum_dns: float
    incremental_total_dns: float
    mean_incremental_dns: float
    loss: float
    nodal_incremental_dns: dict[str, float]
    worst_contingency: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterCVaREvaluation:
    cluster: OutageCluster
    representatives: tuple[RepresentativeState, ...]
    contingencies: tuple[str, ...]
    scenario_results: list[ScenarioClusterResult]
    risk: RiskSummary
    normalized_severity: float

    @property
    def representative(self) -> RepresentativeState:
        return self.representatives[0]

    @property
    def worst_scenario(self) -> ScenarioClusterResult:
        return max(self.scenario_results, key=lambda result: result.loss)

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst_scenario
        return {
            "cluster": self.cluster.to_dict(),
            "representative": asdict(self.representative),
            "representatives": [asdict(item) for item in self.representatives],
            "candidate": {
                "time": worst.source_time,
                "active_planned_outages": self.cluster.active_outages,
                "contingencies": self.contingencies,
                "maximum_dns": worst.candidate_maximum_dns,
                "total_dns": worst.candidate_total_dns,
                "worst_contingency": worst.worst_contingency,
            },
            "baseline": {
                "maximum_dns": worst.baseline_maximum_dns,
                "total_dns": worst.baseline_total_dns,
                "worst_contingency": None,
            },
            "maximum_incremental_dns": max(
                result.incremental_maximum_dns
                for result in self.scenario_results
            ),
            "total_incremental_dns": sum(
                result.incremental_total_dns for result in self.scenario_results
            ),
            "mean_incremental_dns": float(
                np.mean(
                    [result.mean_incremental_dns for result in self.scenario_results]
                )
            ),
            "worst_scenario_nodal_incremental_dns": dict(
                worst.nodal_incremental_dns
            ),
            "normalized_severity": self.normalized_severity,
            "budget_feasible": True,
            "risk": asdict(self.risk),
            "scenario_losses": {
                result.scenario_id: result.loss
                for result in self.scenario_results
            },
            "scenario_results": [
                result.to_dict() for result in self.scenario_results
            ],
        }


@dataclass
class ClusteredCVaRCandidateRecord:
    iteration: int
    source: str
    start_times: dict[str, str]
    deferred_outages: tuple[str, ...]
    scheduled_outage_count: int
    coverage_utility: float
    timing_utility: float
    maintenance_utility: float
    score: float
    risk: RiskSummary
    scenario_losses: dict[str, float]
    clusters: list[ClusterCVaREvaluation]
    risk_feasible: bool
    utility_admissible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "source": self.source,
            "start_times": self.start_times,
            "deferred_outages": self.deferred_outages,
            "scheduled_outage_count": self.scheduled_outage_count,
            "coverage_utility": self.coverage_utility,
            "timing_utility": self.timing_utility,
            "maintenance_utility": self.maintenance_utility,
            "score": self.score,
            "risk": asdict(self.risk),
            "scenario_losses": self.scenario_losses,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "risk_feasible": self.risk_feasible,
            "utility_admissible": self.utility_admissible,
            "budget_feasible": self.risk_feasible and self.utility_admissible,
        }


@dataclass
class ClusteredCVaRResult:
    best_schedule: dict[str, str] | None
    deferred_outages: tuple[str, ...]
    best_active_outages: dict[str, tuple[str, ...]] | None
    best_score: float | None
    best_risk: RiskSummary | None
    best_scenario_losses: dict[str, float]
    best_scenario_probabilities: dict[str, float]
    utility_floor: float | None
    benchmark_included: bool
    iterations: int
    candidates: list[ClusteredCVaRCandidateRecord]
    no_good_cuts_added: int
    termination_reason: str
    full_validation_risk: RiskSummary | None = None
    full_validation_clusters: list[ClusterCVaREvaluation] = field(default_factory=list)


class ClusteredCVaRScheduler:
    """CVaR-guided outage-cluster scheduler."""

    def __init__(
        self,
        data: Mapping,
        config: ClusteredCVaRConfig | None = None,
        *,
        benchmark_schedules: Sequence[MasterSolution | Mapping[str, str]] = (),
    ):
        self.data = data
        self.config = config or ClusteredCVaRConfig()
        if self.config.risk_mode not in {"cvar_objective", "cvar_constraint"}:
            raise ValueError(
                "risk_mode must be 'cvar_objective' or 'cvar_constraint'."
            )
        if (
            self.config.risk_mode == "cvar_constraint"
            and self.config.cvar_limit_mw is None
        ):
            raise ValueError("cvar_constraint mode requires cvar_limit_mw.")
        if self.config.selection_rule not in {
            "weighted_score",
            "utility_constrained_cvar",
        }:
            raise ValueError(
                "selection_rule must be 'weighted_score' or "
                "'utility_constrained_cvar'."
            )
        if self.config.critical_state_count <= 0:
            raise ValueError("critical_state_count must be positive.")
        outage_count = len(data["names"]["outages"])
        if not 1 <= self.config.min_candidate_changes <= outage_count:
            raise ValueError(
                "min_candidate_changes must be between 1 and the number "
                f"of planned outages ({outage_count})."
            )

        self.state = MasterState()
        self.oracle = ClusterSCOPFOracle(data)
        self.selector = CriticalStateSelector(
            data,
            self.oracle,
            mode=self.config.representative_state_mode,
            candidate_times=self.config.representative_candidate_times,
        )
        effective_scenario_count = self.config.scenario_count
        if self.config.gaussian_samples_per_critical_state is not None:
            if self.config.gaussian_samples_per_critical_state <= 0:
                raise ValueError(
                    "gaussian_samples_per_critical_state must be positive."
                )
            effective_scenario_count = (
                self.config.gaussian_samples_per_critical_state
                * self.config.critical_state_count
            )
        if self.config.scenario_model == "gaussian_critical_states":
            self.sampler = GaussianCriticalStateScenarioSampler(
                data,
                scenario_count=effective_scenario_count,
                seed=self.config.scenario_seed,
                relative_sigma=self.config.gaussian_relative_sigma,
                global_sigma=self.config.gaussian_global_sigma,
                correlation_shrinkage=(
                    self.config.gaussian_correlation_shrinkage
                ),
                antithetic=self.config.gaussian_antithetic,
            )
        elif self.config.scenario_model == "empirical_cluster":
            self.sampler = EmpiricalClusterScenarioSampler(
                data,
                scenario_count=self.config.scenario_count,
                seed=self.config.scenario_seed,
                global_lognormal_sigma=self.config.global_lognormal_sigma,
                sampling_scheme=self.config.scenario_sampling_scheme,
            )
        else:
            raise ValueError(
                "scenario_model must be 'gaussian_critical_states' or "
                "'empirical_cluster'."
            )
        self.probabilities = self.sampler.probability_mapping()
        total_demand = data["nodal_demand"].sum(axis=1).to_numpy(dtype=float)
        self.peak_demand = max(1.0, float(np.max(total_demand)))
        self.horizon = max(1, len(data["T"]))

        self.candidates: list[ClusteredCVaRCandidateRecord] = []
        self.best_solution: MasterSolution | None = None
        self.best_score = -math.inf
        self.best_risk: RiskSummary | None = None
        self.best_losses: dict[str, float] = {}
        self.best_clusters: list[ClusterCVaREvaluation] = []
        self.utility_floor: float | None = None
        self.benchmark_solutions = self._normalise_benchmarks(benchmark_schedules)
        self._seen_schedules: set[
            tuple[tuple[tuple[str, str], ...], tuple[str, ...]]
        ] = set()
        self._solve_cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], str],
            ClusterSCOPFResult,
        ] = {}
        self._cache_lock = threading.Lock()
        self._solve_key_locks: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], str], threading.Lock
        ] = {}

    def _normalise_benchmarks(
        self,
        schedules: Sequence[MasterSolution | Mapping[str, str]],
    ) -> list[MasterSolution]:
        result: list[MasterSolution] = []
        all_outages = set(self.data["names"]["outages"])
        for schedule in schedules:
            if isinstance(schedule, MasterSolution):
                result.append(schedule)
                continue
            starts = {str(key): str(value) for key, value in schedule.items()}
            deferred = tuple(sorted(all_outages - set(starts)))
            result.append(
                solution_from_start_times(
                    self.data,
                    starts,
                    deferred_outages=deferred,
                )
            )
        return result

    def _selected_contingencies(
        self,
        representatives: Sequence[RepresentativeState],
        active_outages: Sequence[str],
        *,
        force_full: bool,
    ) -> tuple[str, ...]:
        if force_full or self.config.contingency_top_k is None:
            return self.oracle.effective_contingencies(active_outages)
        selected: set[str] = set(self.config.mandatory_contingencies)
        for representative in representatives:
            normal, _, _ = self.selector.screened_normal(
                representative.time, active_outages
            )
            selected.update(
                self.oracle.rank_contingencies(
                    normal,
                    active_outages,
                    top_k=self.config.contingency_top_k,
                    mandatory=self.config.mandatory_contingencies,
                )
            )
        configured = self.oracle.effective_contingencies(active_outages)
        return tuple(item for item in configured if item in selected)

    def _cached_solve(
        self,
        *,
        source_time: str,
        active_outages: Sequence[str],
        contingencies: Sequence[str],
        demand: np.ndarray,
    ) -> ClusterSCOPFResult:
        key = (
            source_time,
            tuple(sorted(active_outages)),
            tuple(contingencies),
            demand_fingerprint(demand),
        )
        with self._cache_lock:
            cached = self._solve_cache.get(key)
            key_lock = self._solve_key_locks.setdefault(key, threading.Lock())
        if cached is not None:
            return cached
        with key_lock:
            with self._cache_lock:
                cached = self._solve_cache.get(key)
            if cached is None:
                cached = self.oracle.solve(
                    source_time,
                    active_outages,
                    contingencies=contingencies,
                    demand_override=demand,
                )
                with self._cache_lock:
                    self._solve_cache[key] = cached
            return cached

    def _evaluate_scenario(
        self,
        cluster: OutageCluster,
        scenario: DemandScenario,
        contingencies: Sequence[str],
    ) -> ScenarioClusterResult:
        candidate = self._cached_solve(
            source_time=scenario.source_time,
            active_outages=cluster.active_outages,
            contingencies=contingencies,
            demand=scenario.demand,
        )
        baseline = self._cached_solve(
            source_time=scenario.source_time,
            active_outages=(),
            contingencies=contingencies,
            demand=scenario.demand,
        )
        maximum_increment = max(
            0.0, candidate.maximum_dns - baseline.maximum_dns
        )
        total_increment = max(0.0, candidate.total_dns - baseline.total_dns)
        mean_increment = total_increment / max(
            1, len(candidate.contingencies) + 1
        )
        loss = (
            self.config.maximum_dns_loss_weight * maximum_increment
            + self.config.total_dns_component_weight * mean_increment
        )
        candidate_nodal = _summed_nodal_dns(candidate)
        baseline_nodal = _summed_nodal_dns(baseline)
        nodal_increment = {
            bus: max(
                0.0,
                float(candidate_nodal.get(bus, 0.0))
                - float(baseline_nodal.get(bus, 0.0)),
            )
            for bus in sorted(set(candidate_nodal) | set(baseline_nodal))
            if max(
                0.0,
                float(candidate_nodal.get(bus, 0.0))
                - float(baseline_nodal.get(bus, 0.0)),
            ) > 1e-9
        }
        return ScenarioClusterResult(
            scenario_id=scenario.scenario_id,
            probability=scenario.probability,
            source_time=scenario.source_time,
            anchor_rank=scenario.anchor_rank,
            sampling_model=scenario.sampling_model,
            total_demand=scenario.total_demand,
            global_multiplier=scenario.global_multiplier,
            candidate_maximum_dns=float(candidate.maximum_dns),
            candidate_total_dns=float(candidate.total_dns),
            baseline_maximum_dns=float(baseline.maximum_dns),
            baseline_total_dns=float(baseline.total_dns),
            incremental_maximum_dns=float(maximum_increment),
            incremental_total_dns=float(total_increment),
            mean_incremental_dns=float(mean_increment),
            loss=float(loss),
            nodal_incremental_dns=nodal_increment,
            worst_contingency=candidate.worst_contingency,
        )

    def _evaluate_cluster(
        self,
        cluster: OutageCluster,
        *,
        force_full_contingencies: bool,
    ) -> ClusterCVaREvaluation:
        representatives = self.selector.select_many(
            cluster, count=self.config.critical_state_count
        )
        contingencies = self._selected_contingencies(
            representatives,
            cluster.active_outages,
            force_full=force_full_contingencies,
        )
        scenarios = self.sampler.scenarios_for_cluster(
            cluster, representatives
        )

        if self.config.oracle_workers <= 1:
            scenario_results = [
                self._evaluate_scenario(cluster, scenario, contingencies)
                for scenario in scenarios
            ]
        else:
            by_id: dict[str, ScenarioClusterResult] = {}
            with ThreadPoolExecutor(
                max_workers=self.config.oracle_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_scenario,
                        cluster,
                        scenario,
                        contingencies,
                    ): scenario.scenario_id
                    for scenario in scenarios
                }
                for future in as_completed(futures):
                    by_id[futures[future]] = future.result()
            scenario_results = [
                by_id[scenario.scenario_id] for scenario in scenarios
            ]

        losses = {
            result.scenario_id: result.loss for result in scenario_results
        }
        risk = weighted_var_cvar(
            losses, self.probabilities, alpha=self.config.cvar_alpha
        )
        duration_weight = cluster.duration_steps / self.horizon
        normalized = duration_weight * risk.cvar / self.peak_demand
        return ClusterCVaREvaluation(
            cluster=cluster,
            representatives=tuple(representatives),
            contingencies=tuple(contingencies),
            scenario_results=scenario_results,
            risk=risk,
            normalized_severity=float(normalized),
        )

    def _evaluate_solution(
        self,
        solution: MasterSolution,
        *,
        force_full_contingencies: bool = False,
    ) -> tuple[list[ClusterCVaREvaluation], dict[str, float], RiskSummary]:
        clusters = build_outage_clusters(
            self.data["T"],
            solution.active_outages,
            include_empty=self.config.include_empty_clusters,
        )
        evaluations = [
            self._evaluate_cluster(
                cluster,
                force_full_contingencies=force_full_contingencies,
            )
            for cluster in clusters
        ]
        schedule_losses = {
            scenario_id: 0.0 for scenario_id in self.sampler.scenario_ids
        }
        for evaluation in evaluations:
            duration_weight = evaluation.cluster.duration_steps / self.horizon
            for result in evaluation.scenario_results:
                schedule_losses[result.scenario_id] += (
                    duration_weight * result.loss
                )
        risk = weighted_var_cvar(
            schedule_losses,
            self.probabilities,
            alpha=self.config.cvar_alpha,
        )
        return evaluations, schedule_losses, risk

    def _score(self, solution: MasterSolution, risk: RiskSummary) -> float:
        normalized_expected = risk.expected_loss / self.peak_demand
        normalized_cvar = risk.cvar / self.peak_demand
        return float(
            solution.maintenance_utility
            - self.config.expected_loss_penalty_weight * normalized_expected
            - self.config.cvar_penalty_weight * normalized_cvar
        )

    def _risk_feasible(self, risk: RiskSummary) -> bool:
        if self.config.risk_mode != "cvar_constraint":
            return True
        assert self.config.cvar_limit_mw is not None
        return risk.cvar <= self.config.cvar_limit_mw + 1e-9

    def _initialise_utility_floor(self, utility: float) -> None:
        if self.utility_floor is not None:
            return
        allowed_drop = max(
            self.config.absolute_utility_tolerance,
            self.config.relative_utility_tolerance * max(abs(utility), 1.0),
        )
        self.utility_floor = float(utility - allowed_drop)

    def _is_better(
        self,
        solution: MasterSolution,
        risk: RiskSummary,
        score: float,
    ) -> bool:
        if not self._risk_feasible(risk):
            return False
        if self.config.selection_rule == "weighted_score":
            return score > self.best_score + 1e-10

        self._initialise_utility_floor(solution.maintenance_utility)
        assert self.utility_floor is not None
        if solution.maintenance_utility + 1e-10 < self.utility_floor:
            return False
        if self.best_solution is None or self.best_risk is None:
            return True
        if risk.cvar < self.best_risk.cvar - 1e-10:
            return True
        if abs(risk.cvar - self.best_risk.cvar) <= 1e-10:
            if risk.expected_loss < self.best_risk.expected_loss - 1e-10:
                return True
            if (
                abs(risk.expected_loss - self.best_risk.expected_loss) <= 1e-10
                and solution.maintenance_utility
                > self.best_solution.maintenance_utility + 1e-10
            ):
                return True
        return False

    def _record_candidate(
        self,
        solution: MasterSolution,
        *,
        iteration: int,
        source: str,
    ) -> tuple[ClusteredCVaRCandidateRecord, bool]:
        evaluations, losses, risk = self._evaluate_solution(solution)
        score = self._score(solution, risk)
        feasible = self._risk_feasible(risk)
        self._initialise_utility_floor(solution.maintenance_utility)
        utility_admissible = (
            self.utility_floor is None
            or solution.maintenance_utility + 1e-10 >= self.utility_floor
        )
        record = ClusteredCVaRCandidateRecord(
            iteration=iteration,
            source=source,
            start_times=dict(solution.start_times),
            deferred_outages=solution.deferred_outages,
            scheduled_outage_count=solution.scheduled_outage_count,
            coverage_utility=float(solution.coverage_utility),
            timing_utility=float(solution.timing_utility),
            maintenance_utility=float(solution.maintenance_utility),
            score=float(score),
            risk=risk,
            scenario_losses=losses,
            clusters=evaluations,
            risk_feasible=feasible,
            utility_admissible=utility_admissible,
        )
        self.candidates.append(record)
        improved = self._is_better(solution, risk, score)
        if improved:
            self.best_score = score
            self.best_solution = solution
            self.best_risk = risk
            self.best_losses = losses
            self.best_clusters = evaluations
        logger.info(
            "CVaR candidate %s/%d: scheduled=%d, clusters=%d, E[L]=%.6g MW, "
            "VaR=%.6g MW, CVaR=%.6g MW, utility=%.6g, admissible=%s, improved=%s",
            source,
            iteration,
            solution.scheduled_outage_count,
            len(evaluations),
            risk.expected_loss,
            risk.var,
            risk.cvar,
            solution.maintenance_utility,
            utility_admissible,
            improved,
        )
        return record, improved

    def _update_proxy(
        self,
        evaluations: Sequence[ClusterCVaREvaluation],
        risk: RiskSummary,
    ) -> None:
        if not risk.tail_ids:
            return
        learning_rate = float(self.config.risk_proxy_learning_rate)
        tail = set(risk.tail_ids)
        for evaluation in evaluations:
            outages = evaluation.cluster.active_outages
            times = evaluation.cluster.times
            if not outages or not times:
                continue
            tail_contributions = [
                result.loss
                for result in evaluation.scenario_results
                if result.scenario_id in tail
            ]
            if not tail_contributions:
                continue
            duration_weight = evaluation.cluster.duration_steps / self.horizon
            observed_cluster = (
                self.config.cvar_penalty_weight
                * duration_weight
                * float(np.mean(tail_contributions))
                / self.peak_demand
            )
            observed = observed_cluster / (len(outages) * len(times))
            for outage in outages:
                for time in times:
                    key = (outage, time)
                    previous = float(self.state.risk_coefficients.get(key, 0.0))
                    self.state.risk_coefficients[key] = (
                        (1.0 - learning_rate) * previous
                        + learning_rate * observed
                    )

    def _add_no_good(self, solution: MasterSolution, *, label: str) -> bool:
        signature = solution.decision_signature
        if signature in self._seen_schedules:
            return False
        self._seen_schedules.add(signature)
        self.state.no_goods.append(
            no_good_from_solution(
                solution,
                label=label,
                min_changes=self.config.min_candidate_changes,
            )
        )
        return True

    def _save(self, result: ClusteredCVaRResult) -> None:
        serial = {
            "best_schedule": result.best_schedule,
            "deferred_outages": result.deferred_outages,
            "best_active_outages": result.best_active_outages,
            "best_score": result.best_score,
            "best_risk": (
                None if result.best_risk is None else asdict(result.best_risk)
            ),
            "best_scenario_losses": result.best_scenario_losses,
            "best_scenario_probabilities": result.best_scenario_probabilities,
            "utility_floor": result.utility_floor,
            "benchmark_included": result.benchmark_included,
            "iterations": result.iterations,
            "candidates": [candidate.to_dict() for candidate in result.candidates],
            "no_good_cuts_added": result.no_good_cuts_added,
            "termination_reason": result.termination_reason,
            "full_validation_risk": (
                None
                if result.full_validation_risk is None
                else asdict(result.full_validation_risk)
            ),
            "full_validation_clusters": [
                cluster.to_dict() for cluster in result.full_validation_clusters
            ],
            "config": asdict(self.config),
            "oracle_statistics": self.oracle.statistics(),
            "loss_definition": (
                "Schedule scenario loss = sum over constant-topology clusters "
                "of (cluster duration / horizon) times [maximum incremental DNS "
                "+ total_dns_component_weight times mean incremental summed "
                "nodal DNS over normal and N-1 states]."
            ),
            "guarantee_note": (
                "With utility_constrained_cvar and a deterministic benchmark, "
                "the selected schedule has no higher CVaR on the optimisation "
                "scenario bank than that admissible benchmark. This does not "
                "guarantee lower independent validation CVaR."
            ),
        }
        Path(self.config.results_path).write_text(
            json.dumps(json_safe(serial), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def solve(self) -> ClusteredCVaRResult:
        termination = "iteration_limit"
        no_improvement = 0
        last_iteration = 0

        # Include deterministic or externally supplied schedules in the CVaR
        # candidate set before the risk-guided search.  This provides the
        # in-sample tail-risk guardrail described in the module docstring.
        for benchmark_index, solution in enumerate(self.benchmark_solutions, start=1):
            if solution.decision_signature in self._seen_schedules:
                continue
            self._seen_schedules.add(solution.decision_signature)
            benchmark_record, improved = self._record_candidate(
                solution,
                iteration=0,
                source=f"benchmark_{benchmark_index}",
            )
            self._update_proxy(
                benchmark_record.clusters, benchmark_record.risk
            )
            if not improved:
                no_improvement += 1
            self.state.no_goods.append(
                no_good_from_solution(
                    solution,
                    label="benchmark_exploration",
                    min_changes=self.config.min_candidate_changes,
                )
            )

        for iteration in range(1, self.config.max_iterations + 1):
            last_iteration = iteration
            logger.info(
                "CVaR iteration %d: %d no-good cuts, %d learned coefficients",
                iteration,
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

            record, improved = self._record_candidate(
                solution, iteration=iteration, source="risk_search"
            )
            if improved:
                no_improvement = 0
            else:
                no_improvement += 1

            self._update_proxy(record.clusters, record.risk)
            self._add_no_good(
                solution,
                label=(
                    "cvar_exploration"
                    if record.risk_feasible
                    else "cvar_constraint"
                ),
            )

            searched_candidates = sum(
                candidate.source == "risk_search" for candidate in self.candidates
            )
            if searched_candidates >= self.config.max_candidates:
                termination = "candidate_limit"
                break
            if no_improvement >= self.config.patience:
                termination = "no_improvement"
                break

        full_risk: RiskSummary | None = None
        full_clusters: list[ClusterCVaREvaluation] = []
        if (
            self.best_solution is not None
            and self.config.full_contingency_validation
            and self.config.contingency_top_k is not None
        ):
            logger.info(
                "Validating the best CVaR schedule with the full contingency set."
            )
            full_clusters, _, full_risk = self._evaluate_solution(
                self.best_solution, force_full_contingencies=True
            )
            logger.info(
                "Full validation: E[L]=%.6g MW, VaR=%.6g MW, CVaR=%.6g MW",
                full_risk.expected_loss,
                full_risk.var,
                full_risk.cvar,
            )

        result = ClusteredCVaRResult(
            best_schedule=(
                None
                if self.best_solution is None
                else dict(self.best_solution.start_times)
            ),
            deferred_outages=(
                ()
                if self.best_solution is None
                else self.best_solution.deferred_outages
            ),
            best_active_outages=(
                None
                if self.best_solution is None
                else self.best_solution.active_outages
            ),
            best_score=(None if self.best_solution is None else self.best_score),
            best_risk=self.best_risk,
            best_scenario_losses=self.best_losses,
            best_scenario_probabilities=self.probabilities,
            utility_floor=self.utility_floor,
            benchmark_included=bool(self.benchmark_solutions),
            iterations=last_iteration,
            candidates=self.candidates,
            no_good_cuts_added=len(self.state.no_goods),
            termination_reason=termination,
            full_validation_risk=full_risk,
            full_validation_clusters=full_clusters,
        )
        self._save(result)
        return result
