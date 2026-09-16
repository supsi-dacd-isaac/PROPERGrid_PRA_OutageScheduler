"""Compact start-index outage-scheduling master problem.

The formulation uses one binary start variable for each admissible outage start.
This representation enforces a single uninterrupted outage block of exactly the
requested duration without introducing period-by-period transition binaries.
Optional deferral variables let the master select a feasible subset when all
requested outages cannot be accommodated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from gurobipy import GRB, LinExpr, Model, quicksum


@dataclass(frozen=True)
class ConflictCut:
    """Forbid one simultaneous outage set at a witness period."""

    time: str
    outages: tuple[str, ...]
    label: str = "cluster_security"


@dataclass(frozen=True)
class NoGoodCut:
    """Exclude one complete schedule/deferral decision."""

    starts: tuple[tuple[str, str], ...]
    deferred: tuple[str, ...] = ()
    label: str = "exploration"


@dataclass
class MasterState:
    conflicts: list[ConflictCut] = field(default_factory=list)
    no_goods: list[NoGoodCut] = field(default_factory=list)
    forbidden_starts: set[tuple[str, str]] = field(default_factory=set)
    risk_coefficients: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass
class MasterSolution:
    status: int
    objective: float | None
    start_times: dict[str, str]
    deferred_outages: tuple[str, ...]
    active_outages: dict[str, tuple[str, ...]]
    maintenance_utility: float
    proxy_security_penalty: float
    coverage_utility: float = 0.0
    timing_utility: float = 0.0
    scheduled_outage_count: int = 0

    @property
    def decision_signature(
        self,
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        return (
            tuple(sorted(self.start_times.items())),
            tuple(sorted(self.deferred_outages)),
        )


def duration_steps(data: Mapping, outage: str) -> int:
    duration = int(round(float(data["durations"][outage])))
    if duration <= 0:
        raise ValueError(
            f"Outage {outage!r} has non-positive duration {duration}."
        )
    return duration


def _resolve_window_indices(
    data: Mapping,
    outage: str,
    horizon: int,
) -> tuple[int, int]:
    times = list(data["T"])
    duration = duration_steps(data, outage)
    latest_default = horizon - duration
    raw = data.get("outage_start_windows", {}).get(outage)
    if raw is None:
        return 0, latest_default

    if isinstance(raw, Mapping):
        first = raw.get("start", 0)
        last = raw.get("end", latest_default)
        end_is_completion = bool(raw.get("end_is_completion", False))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        first, last = raw
        end_is_completion = bool(data.get("window_end_is_completion", False))
    else:
        raise ValueError(f"Unsupported outage window for {outage}: {raw!r}")

    def to_index(value: object) -> int:
        if isinstance(value, str):
            try:
                return times.index(value)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown time label {value!r} in window for {outage}."
                ) from exc
        return int(value)

    first_index = max(0, to_index(first))
    last_index = to_index(last)
    if end_is_completion:
        last_index -= duration - 1
    last_index = min(latest_default, last_index)
    if first_index > last_index:
        raise ValueError(
            f"No feasible starts for {outage}: range=({first_index}, "
            f"{last_index}), duration={duration}, horizon={horizon}."
        )
    return first_index, last_index


def feasible_starts(data: Mapping) -> dict[str, tuple[str, ...]]:
    times = list(data["T"])
    horizon = len(times)
    starts: dict[str, tuple[str, ...]] = {}
    for outage in data["names"]["outages"]:
        first, last = _resolve_window_indices(data, outage, horizon)
        starts[outage] = tuple(times[index] for index in range(first, last + 1))
    return starts


def covering_starts(
    data: Mapping,
    starts: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str], tuple[str, ...]]:
    times = list(data["T"])
    index = {time: position for position, time in enumerate(times)}
    cover: dict[tuple[str, str], tuple[str, ...]] = {}
    for outage, outage_starts in starts.items():
        duration = duration_steps(data, outage)
        start_indices = {index[start]: start for start in outage_starts}
        for time, time_index in index.items():
            cover[(time, outage)] = tuple(
                start
                for start_index, start in start_indices.items()
                if start_index <= time_index < start_index + duration
            )
    return cover


def build_active_outages(
    data: Mapping,
    start_times: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Expand start decisions into period-wise active outage sets."""
    times = list(data["T"])
    time_index = {time: index for index, time in enumerate(times)}
    starts = feasible_starts(data)
    for outage, start in start_times.items():
        if outage not in starts:
            raise KeyError(f"Unknown outage {outage!r}.")
        if start not in starts[outage]:
            raise ValueError(
                f"Start {start!r} is not admissible for outage {outage!r}."
            )

    active: dict[str, tuple[str, ...]] = {}
    for time in times:
        current_index = time_index[time]
        current: list[str] = []
        for outage, start in start_times.items():
            start_index = time_index[start]
            if start_index <= current_index < start_index + duration_steps(data, outage):
                current.append(outage)
        active[time] = tuple(sorted(current))
    return active


def validate_outage_consistency(
    data: Mapping,
    start_times: Mapping[str, str],
    active_outages: Mapping[str, Sequence[str]],
) -> None:
    """Verify exact duration, contiguity, and simultaneous-outage limits."""
    times = list(data["T"])
    expected = build_active_outages(data, start_times)
    for time in times:
        actual = tuple(sorted(set(active_outages.get(time, ()))))
        if actual != expected[time]:
            raise ValueError(
                f"Inconsistent active-outage set at {time}: expected "
                f"{expected[time]}, received {actual}."
            )

    max_tasks = data["max_tasks"]
    for time in times:
        capacity = (
            int(max_tasks.get(time, 0))
            if isinstance(max_tasks, Mapping)
            else int(max_tasks)
        )
        if len(expected[time]) > capacity:
            raise ValueError(
                f"Simultaneous-outage limit exceeded at {time}: "
                f"{len(expected[time])} > {capacity}."
            )


def _priority(data: Mapping, outage: str) -> float:
    return float(data.get("priority", {}).get(outage, 1.0))


def _utility_components(
    data: Mapping,
    start_times: Mapping[str, str],
) -> tuple[float, float, float]:
    times = list(data["T"])
    index = {time: position for position, time in enumerate(times)}
    horizon = max(1, len(times) - 1)
    coverage = sum(_priority(data, outage) for outage in start_times)
    timing = sum(
        _priority(data, outage) * (1.0 - index[start] / horizon)
        for outage, start in start_times.items()
    )
    maintenance = (
        float(data.get("outage_coverage_reward", 1.0)) * coverage
        + float(data.get("outage_priority_timing_reward", 0.1)) * timing
    )
    return float(coverage), float(timing), float(maintenance)


def solution_from_start_times(
    data: Mapping,
    start_times: Mapping[str, str],
    *,
    deferred_outages: Sequence[str] = (),
    status: int = GRB.OPTIMAL,
) -> MasterSolution:
    """Create a validated solution object from externally supplied starts."""
    all_outages = set(data["names"]["outages"])
    selected = {str(key): str(value) for key, value in start_times.items()}
    deferred = tuple(sorted(str(value) for value in deferred_outages))
    if set(selected) & set(deferred):
        raise ValueError("An outage cannot be both scheduled and deferred.")
    if set(selected) | set(deferred) != all_outages:
        missing = all_outages - (set(selected) | set(deferred))
        extra = (set(selected) | set(deferred)) - all_outages
        raise ValueError(
            f"External schedule must classify every outage; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )
    active = build_active_outages(data, selected)
    validate_outage_consistency(data, selected, active)
    coverage, timing, maintenance = _utility_components(data, selected)
    return MasterSolution(
        status=status,
        objective=maintenance,
        start_times=selected,
        deferred_outages=deferred,
        active_outages=active,
        maintenance_utility=maintenance,
        proxy_security_penalty=0.0,
        coverage_utility=coverage,
        timing_utility=timing,
        scheduled_outage_count=len(selected),
    )


def _expression_value(expression: LinExpr) -> float:
    return float(expression.getValue())


class SchedulingMaster:
    """Start-only outage scheduler with optional deferral and learned risk."""

    def __init__(self, data: Mapping, state: MasterState | None = None):
        self.data = data
        self.state = state or MasterState()
        self.times = list(data["T"])
        self.outages = list(data["names"]["outages"])
        self.starts = feasible_starts(data)
        self.cover = covering_starts(data, self.starts)
        self.allow_deferral = bool(data.get("allow_outage_deferral", False))
        self.model: Model | None = None
        self.y = None
        self.defer = None
        self.x_expr: dict[tuple[str, str], LinExpr] = {}
        self.selected_expr: dict[str, LinExpr] = {}
        self._maintenance_expr: LinExpr | None = None
        self._coverage_expr: LinExpr | None = None
        self._timing_expr: LinExpr | None = None
        self._risk_expr: LinExpr | None = None

    def _add_optional_constraints(self, model: Model) -> None:
        # Pairwise or group incompatibilities.  A list [a,b,c] means that at
        # most one of the listed outages may be active in a period.
        groups = list(self.data.get("mutually_exclusive_outages", []))
        groups.extend(self.data.get("incompatible_outage_groups", []))
        for group_index, group in enumerate(groups):
            valid = [outage for outage in group if outage in self.outages]
            if len(valid) < 2:
                continue
            for time in self.times:
                model.addConstr(
                    quicksum(self.x_expr[(time, outage)] for outage in valid) <= 1,
                    name=f"Incompatibility_{group_index}_{time}",
                )

        capacities = self.data.get("resource_capacities", {})
        usage = self.data.get("resource_usage", {})
        for resource, capacity_by_time in capacities.items():
            for time in self.times:
                capacity = (
                    float(capacity_by_time.get(time, 0.0))
                    if isinstance(capacity_by_time, Mapping)
                    else float(capacity_by_time)
                )
                model.addConstr(
                    quicksum(
                        float(usage.get(resource, {}).get(outage, 0.0))
                        * self.x_expr[(time, outage)]
                        for outage in self.outages
                    )
                    <= capacity,
                    name=f"Resource_{resource}_{time}",
                )

        time_index = {time: index for index, time in enumerate(self.times)}
        big_m = 2 * len(self.times) + max(
            duration_steps(self.data, outage) for outage in self.outages
        )
        for relation_index, relation in enumerate(self.data.get("precedence", [])):
            before, after = relation[0], relation[1]
            lag = int(relation[2]) if len(relation) >= 3 else 0
            finish_before = quicksum(
                (time_index[start] + duration_steps(self.data, before) + lag)
                * self.y[before, start]
                for start in self.starts[before]
            )
            start_after = quicksum(
                time_index[start] * self.y[after, start]
                for start in self.starts[after]
            )
            relaxation = 0.0
            if self.allow_deferral:
                relaxation = big_m * (self.defer[before] + self.defer[after])
            model.addConstr(
                finish_before <= start_after + relaxation,
                name=f"Precedence_{relation_index}",
            )

    def build(self) -> Model:
        model = Model("ClusteredOutageSchedulingMaster")
        model.Params.OutputFlag = int(self.data.get("master_output_flag", 0))

        feasible_keys = [
            (outage, start)
            for outage in self.outages
            for start in self.starts[outage]
        ]
        self.y = model.addVars(feasible_keys, vtype=GRB.BINARY, name="outage_start")
        if self.allow_deferral:
            self.defer = model.addVars(
                self.outages, vtype=GRB.BINARY, name="outage_deferred"
            )

        for outage in self.outages:
            schedule_sum = quicksum(
                self.y[outage, start] for start in self.starts[outage]
            )
            self.selected_expr[outage] = schedule_sum
            if self.allow_deferral:
                model.addConstr(
                    schedule_sum + self.defer[outage] == 1,
                    name=f"ScheduleOrDefer_{outage}",
                )
            else:
                model.addConstr(schedule_sum == 1, name=f"OneStart_{outage}")

        for outage, start in self.state.forbidden_starts:
            if (outage, start) in self.y:
                self.y[outage, start].UB = 0.0

        max_tasks = self.data["max_tasks"]
        for time in self.times:
            for outage in self.outages:
                self.x_expr[(time, outage)] = quicksum(
                    self.y[outage, start]
                    for start in self.cover[(time, outage)]
                )
            capacity = (
                int(max_tasks.get(time, 0))
                if isinstance(max_tasks, Mapping)
                else int(max_tasks)
            )
            model.addConstr(
                quicksum(
                    self.x_expr[(time, outage)] for outage in self.outages
                )
                <= capacity,
                name=f"MaxTasks_{time}",
            )

        self._add_optional_constraints(model)

        for cut_index, cut in enumerate(self.state.conflicts):
            valid = tuple(outage for outage in cut.outages if outage in self.outages)
            if not valid:
                continue
            model.addConstr(
                quicksum(self.x_expr[(cut.time, outage)] for outage in valid)
                <= len(valid) - 1,
                name=f"Conflict_{cut.label}_{cut_index}",
            )

        for cut_index, cut in enumerate(self.state.no_goods):
            selected_terms = [
                self.y[outage, start]
                for outage, start in cut.starts
                if (outage, start) in self.y
            ]
            if self.allow_deferral:
                selected_terms.extend(
                    self.defer[outage]
                    for outage in cut.deferred
                    if outage in self.outages
                )
            if selected_terms:
                model.addConstr(
                    quicksum(selected_terms) <= len(selected_terms) - 1,
                    name=f"NoGood_{cut.label}_{cut_index}",
                )

        time_index = {time: index for index, time in enumerate(self.times)}
        timing_horizon = max(1, len(self.times) - 1)
        self._coverage_expr = quicksum(
            _priority(self.data, outage) * self.selected_expr[outage]
            for outage in self.outages
        )
        self._timing_expr = quicksum(
            _priority(self.data, outage)
            * (1.0 - time_index[start] / timing_horizon)
            * self.y[outage, start]
            for outage in self.outages
            for start in self.starts[outage]
        )
        self._maintenance_expr = (
            float(self.data.get("outage_coverage_reward", 1.0))
            * self._coverage_expr
            + float(self.data.get("outage_priority_timing_reward", 0.1))
            * self._timing_expr
        )
        self._risk_expr = quicksum(
            float(self.state.risk_coefficients.get((outage, time), 0.0))
            * self.x_expr[(time, outage)]
            for outage in self.outages
            for time in self.times
        )

        deferral_penalty = 0.0
        if self.allow_deferral:
            penalties = self.data.get("defer_penalty", {})
            deferral_penalty = quicksum(
                float(
                    penalties.get(
                        outage,
                        self.data.get("default_defer_penalty", 0.0)
                        * _priority(self.data, outage),
                    )
                )
                * self.defer[outage]
                for outage in self.outages
            )

        model.setObjective(
            float(self.data.get("maintenance_weight", 1.0))
            * self._maintenance_expr
            - float(self.data.get("security_proxy_weight", 1.0))
            * self._risk_expr
            - deferral_penalty,
            GRB.MAXIMIZE,
        )
        self.model = model
        return model

    def solve(
        self,
        *,
        mip_gap: float = 0.01,
        time_limit: float = 300.0,
        threads: int = 0,
        iis_path: str = "cluster_master_infeasibility.ilp",
    ) -> MasterSolution:
        model = self.build()
        model.Params.MIPGap = float(mip_gap)
        model.Params.TimeLimit = float(time_limit)
        model.Params.Presolve = 2
        model.Params.Heuristics = float(self.data.get("master_heuristics", 0.1))
        model.Params.Cuts = int(self.data.get("master_cuts", 1))
        if threads > 0:
            model.Params.Threads = int(threads)
        model.optimize()

        acceptable = {
            GRB.OPTIMAL,
            GRB.TIME_LIMIT,
            GRB.SUBOPTIMAL,
            GRB.NODE_LIMIT,
            GRB.ITERATION_LIMIT,
        }
        if model.Status not in acceptable or model.SolCount <= 0:
            if model.Status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write(str(Path(iis_path)))
            return MasterSolution(
                status=model.Status,
                objective=None,
                start_times={},
                deferred_outages=(),
                active_outages={},
                maintenance_utility=float("nan"),
                proxy_security_penalty=float("nan"),
            )

        deferred = tuple(
            sorted(
                outage
                for outage in self.outages
                if self.allow_deferral and self.defer[outage].X > 0.5
            )
        )
        selected: dict[str, str] = {}
        for outage in self.outages:
            if outage in deferred:
                continue
            selected[outage] = max(
                self.starts[outage], key=lambda start: self.y[outage, start].X
            )

        active = build_active_outages(self.data, selected)
        validate_outage_consistency(self.data, selected, active)

        return MasterSolution(
            status=model.Status,
            objective=float(model.ObjVal),
            start_times=selected,
            deferred_outages=deferred,
            active_outages=active,
            maintenance_utility=_expression_value(self._maintenance_expr),
            proxy_security_penalty=_expression_value(self._risk_expr),
            coverage_utility=_expression_value(self._coverage_expr),
            timing_utility=_expression_value(self._timing_expr),
            scheduled_outage_count=len(selected),
        )


def no_good_from_solution(
    solution: MasterSolution,
    *,
    label: str = "exploration",
) -> NoGoodCut:
    return NoGoodCut(
        starts=tuple(sorted(solution.start_times.items())),
        deferred=tuple(sorted(solution.deferred_outages)),
        label=label,
    )
