"""Clustered stochastic outage scheduling with finite-sample CVaR of DNS.

For each candidate outage schedule, constant-topology outage clusters are built.
A common, reproducible set of nodal-demand scenarios is sampled within every
cluster.  Each scenario is evaluated through a soft joint N-1 SCOPF and the
schedule-level loss is the duration-weighted incremental worst-contingency DNS.
The empirical CVaR is computed exactly for the finite weighted scenario set.

The outer scheduling algorithm is a simulation-optimization matheuristic: the
finite-sample CVaR of every evaluated complete schedule is exact, while learned
linear penalties guide the compact master toward lower-tail-risk schedules.
It is not an exact extensive-form or classical Benders solution of the full
mixed-integer stochastic program.
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
from .master_problem import MasterSolution, MasterState, SchedulingMaster, no_good_from_solution
from .outage_clusters import OutageCluster, build_outage_clusters
from .risk_metrics import RiskSummary, weighted_var_cvar
from .serialization import json_safe
from .scenario_sampling import (
    DemandScenario,
    EmpiricalClusterScenarioSampler,
    demand_fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteredCVaRConfig:
    max_iterations: int = 50
    master_time_limit: float = 180.0
    master_mip_gap: float = 0.01
    master_threads: int = 0
    oracle_workers: int = 1
    representative_state_mode: str = "network_screening"
    representative_candidate_times: int = 12
    contingency_top_k: int | None = 10
    mandatory_contingencies: tuple[str, ...] = ()

    scenario_count: int = 16
    scenario_seed: int = 42
    scenario_sampling_scheme: str = "stratified_tail"
    global_lognormal_sigma: float = 0.0

    risk_mode: str = "cvar_objective"  # cvar_objective or cvar_constraint
    cvar_alpha: float = 0.95
    cvar_penalty_weight: float = 10.0
    expected_loss_penalty_weight: float = 0.0
    cvar_limit_mw: float | None = None
    risk_proxy_learning_rate: float = 0.40

    max_candidates: int = 20
    patience: int = 8
    full_contingency_validation: bool = True
    results_path: str = "clustered_cvar_results.json"
    iis_path: str = "cluster_cvar_master_infeasibility.ilp"


@dataclass
class ScenarioClusterResult:
    scenario_id: str
    probability: float
    source_time: str
    total_demand: float
    global_multiplier: float
    candidate_maximum_dns: float
    candidate_total_dns: float
    baseline_maximum_dns: float
    baseline_total_dns: float
    incremental_maximum_dns: float
    incremental_total_dns: float
    worst_contingency: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterCVaREvaluation:
    cluster: OutageCluster
    representative: RepresentativeState
    contingencies: tuple[str, ...]
    scenario_results: list[ScenarioClusterResult]
    risk: RiskSummary
    normalized_severity: float

    @property
    def worst_scenario(self) -> ScenarioClusterResult:
        return max(
            self.scenario_results,
            key=lambda result: result.incremental_maximum_dns,
        )

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst_scenario
        # Deterministic-compatible fields allow the common visualization layer
        # to plot a CVaR result without a separate schema branch.
        return {
            "cluster": self.cluster.to_dict(),
            "representative": asdict(self.representative),
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
                result.incremental_maximum_dns
                for result in self.scenario_results
            ),
            "normalized_severity": self.normalized_severity,
            "budget_feasible": True,
            "risk": asdict(self.risk),
            "scenario_losses": {
                result.scenario_id: result.incremental_maximum_dns
                for result in self.scenario_results
            },
            "scenario_results": [
                result.to_dict() for result in self.scenario_results
            ],
        }


@dataclass
class ClusteredCVaRCandidateRecord:
    iteration: int
    start_times: dict[str, str]
    deferred_outages: tuple[str, ...]
    maintenance_utility: float
    score: float
    risk: RiskSummary
    scenario_losses: dict[str, float]
    clusters: list[ClusterCVaREvaluation]
    risk_feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "start_times": self.start_times,
            "deferred_outages": self.deferred_outages,
            "maintenance_utility": self.maintenance_utility,
            "score": self.score,
            "risk": asdict(self.risk),
            "scenario_losses": self.scenario_losses,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "risk_feasible": self.risk_feasible,
            "budget_feasible": self.risk_feasible,
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
    ):
        self.data = data
        self.config = config or ClusteredCVaRConfig()
        if self.config.risk_mode not in {"cvar_objective", "cvar_constraint"}:
            raise ValueError(
                "risk_mode must be 'cvar_objective' or 'cvar_constraint'."
            )
        if self.config.risk_mode == "cvar_constraint" and self.config.cvar_limit_mw is None:
            raise ValueError("cvar_constraint mode requires cvar_limit_mw.")

        self.state = MasterState()
        self.oracle = ClusterSCOPFOracle(data)
        self.selector = CriticalStateSelector(
            data,
            self.oracle,
            mode=self.config.representative_state_mode,
            candidate_times=self.config.representative_candidate_times,
        )
        self.sampler = EmpiricalClusterScenarioSampler(
            data,
            scenario_count=self.config.scenario_count,
            seed=self.config.scenario_seed,
            global_lognormal_sigma=self.config.global_lognormal_sigma,
            sampling_scheme=self.config.scenario_sampling_scheme,
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
        self._seen_schedules: set[
            tuple[tuple[tuple[str, str], ...], tuple[str, ...]]
        ] = set()
        self._solve_cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], str],
            ClusterSCOPFResult,
        ] = {}
        self._cache_lock = threading.Lock()

    def _selected_contingencies(
        self,
        representative: RepresentativeState,
        active_outages: Sequence[str],
        *,
        force_full: bool,
    ) -> tuple[str, ...]:
        if force_full or self.config.contingency_top_k is None:
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
        if cached is not None:
            return cached
        solved = self.oracle.solve(
            source_time,
            active_outages,
            contingencies=contingencies,
            demand_override=demand,
        )
        with self._cache_lock:
            self._solve_cache[key] = solved
        return solved

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
        return ScenarioClusterResult(
            scenario_id=scenario.scenario_id,
            probability=scenario.probability,
            source_time=scenario.source_time,
            total_demand=scenario.total_demand,
            global_multiplier=scenario.global_multiplier,
            candidate_maximum_dns=float(candidate.maximum_dns),
            candidate_total_dns=float(candidate.total_dns),
            baseline_maximum_dns=float(baseline.maximum_dns),
            baseline_total_dns=float(baseline.total_dns),
            incremental_maximum_dns=float(maximum_increment),
            incremental_total_dns=float(total_increment),
            worst_contingency=candidate.worst_contingency,
        )

    def _evaluate_cluster(
        self,
        cluster: OutageCluster,
        *,
        force_full_contingencies: bool,
    ) -> ClusterCVaREvaluation:
        representative = self.selector.select(cluster)
        contingencies = self._selected_contingencies(
            representative,
            cluster.active_outages,
            force_full=force_full_contingencies,
        )
        scenarios = self.sampler.scenarios_for_cluster(cluster)

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
            result.scenario_id: result.incremental_maximum_dns
            for result in scenario_results
        }
        risk = weighted_var_cvar(
            losses,
            self.probabilities,
            alpha=self.config.cvar_alpha,
        )
        duration_weight = cluster.duration_steps / self.horizon
        normalized = duration_weight * risk.cvar / self.peak_demand
        return ClusterCVaREvaluation(
            cluster=cluster,
            representative=representative,
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
            include_empty=False,
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
                    duration_weight * result.incremental_maximum_dns
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
                result.incremental_maximum_dns
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
                    previous = float(
                        self.state.risk_coefficients.get(key, 0.0)
                    )
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
            no_good_from_solution(solution, label=label)
        )
        return True

    def _save(self, result: ClusteredCVaRResult) -> None:
        serial = {
            "best_schedule": result.best_schedule,
            "deferred_outages": result.deferred_outages,
            "best_active_outages": result.best_active_outages,
            "best_score": result.best_score,
            "best_risk": None if result.best_risk is None else asdict(result.best_risk),
            "best_scenario_losses": result.best_scenario_losses,
            "best_scenario_probabilities": result.best_scenario_probabilities,
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
            "loss_definition": (
                "Schedule scenario loss = sum over outage clusters of "
                "(cluster duration / horizon) times positive incremental "
                "worst-contingency DNS relative to the no-maintenance baseline."
            ),
        }
        Path(self.config.results_path).write_text(
            json.dumps(json_safe(serial), indent=2, allow_nan=False), encoding="utf-8"
        )

    def solve(self) -> ClusteredCVaRResult:
        termination = "iteration_limit"
        no_improvement = 0
        last_iteration = 0

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

            evaluations, losses, risk = self._evaluate_solution(solution)
            score = self._score(solution, risk)
            feasible = self._risk_feasible(risk)
            record = ClusteredCVaRCandidateRecord(
                iteration=iteration,
                start_times=dict(solution.start_times),
                deferred_outages=solution.deferred_outages,
                maintenance_utility=float(solution.maintenance_utility),
                score=float(score),
                risk=risk,
                scenario_losses=losses,
                clusters=evaluations,
                risk_feasible=feasible,
            )
            self.candidates.append(record)
            logger.info(
                "CVaR candidate %d: clusters=%d, E[DNS]=%.6g MW, "
                "VaR=%.6g MW, CVaR=%.6g MW, score=%.6g, feasible=%s",
                iteration,
                len(evaluations),
                risk.expected_loss,
                risk.var,
                risk.cvar,
                score,
                feasible,
            )

            if feasible and score > self.best_score + 1e-10:
                self.best_score = score
                self.best_solution = solution
                self.best_risk = risk
                self.best_losses = losses
                self.best_clusters = evaluations
                no_improvement = 0
            else:
                no_improvement += 1

            self._update_proxy(evaluations, risk)
            self._add_no_good(
                solution,
                label="cvar_exploration" if feasible else "cvar_constraint",
            )

            if len(self.candidates) >= self.config.max_candidates:
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
            full_clusters, full_losses, full_risk = self._evaluate_solution(
                self.best_solution,
                force_full_contingencies=True,
            )
            logger.info(
                "Full validation: E[DNS]=%.6g MW, VaR=%.6g MW, CVaR=%.6g MW",
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
            iterations=last_iteration,
            candidates=self.candidates,
            no_good_cuts_added=len(self.state.no_goods),
            termination_reason=termination,
            full_validation_risk=full_risk,
            full_validation_clusters=full_clusters,
        )
        self._save(result)
        return result
