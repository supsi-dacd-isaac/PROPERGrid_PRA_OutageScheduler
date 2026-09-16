"""Joint soft preventive/corrective N-1 SCOPF for one outage cluster.

The model uses one common pre-contingency dispatch and contingency-specific
corrective recourse. Load shedding and generation spillage guarantee complete
recourse, including islanded topologies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
import threading

import numpy as np
from gurobipy import GRB, Env, Model, quicksum

from .network_model import NetworkModel


@dataclass
class ClusterStateResult:
    contingency: str | None
    load_shedding: float
    spillage: float
    load_shedding_by_bus: dict[str, float]
    spillage_by_bus: dict[str, float]
    generation: dict[str, float]
    flows: dict[str, float]
    injections: np.ndarray


@dataclass
class ClusterSCOPFResult:
    time: str
    active_planned_outages: tuple[str, ...]
    contingencies: tuple[str, ...]
    status: int
    objective: float
    maximum_dns: float
    total_dns: float
    worst_contingency: str | None
    states: dict[str, ClusterStateResult] = field(default_factory=dict)

    @property
    def normal_state(self) -> ClusterStateResult:
        return self.states["normal"]


class ClusterSCOPFOracle:
    """Build and solve one continuous joint soft SCOPF per cluster state."""

    def __init__(self, data: Mapping):
        self.data = data
        self.network = NetworkModel(data)
        self.times = list(data["T"])
        self.time_index = {
            time: index for index, time in enumerate(self.times)
        }
        self.demand = data["nodal_demand"].to_numpy(dtype=float)
        self.metadata = data["contingency_metadata"]
        self.all_contingencies = tuple(
            data["names"].get("contingencies", [])
        )
        self.corrective_fraction = float(
            data.get("corrective_redispatch_fraction", 1.0)
        )
        self.corrective_up = data.get("corrective_up", {})
        self.corrective_down = data.get("corrective_down", {})
        self.generation_cost = data.get("generation_cost", {})
        self._thread_local = threading.local()

    def _environment(self) -> Env | None:
        if not bool(self.data.get("oracle_isolated_env", True)):
            return None
        environment = getattr(self._thread_local, "environment", None)
        if environment is None:
            environment = Env(empty=True)
            environment.setParam("OutputFlag", 0)
            environment.start()
            self._thread_local.environment = environment
        return environment

    def _contingency_metadata(self, contingency: str | None) -> Mapping:
        if contingency is None:
            return {"type": "normal", "element": None}
        if contingency not in self.metadata:
            raise KeyError(
                f"Missing metadata for contingency {contingency!r}."
            )
        return self.metadata[contingency]

    def effective_contingencies(
        self,
        active_planned_outages: Sequence[str],
        contingencies: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Remove failures of assets already unavailable for maintenance."""
        active = set(active_planned_outages)
        requested = (
            self.all_contingencies
            if contingencies is None
            else tuple(contingencies)
        )
        effective: list[str] = []
        for contingency in requested:
            metadata = self._contingency_metadata(contingency)
            if metadata.get("element") in active:
                continue
            effective.append(contingency)
        return tuple(effective)

    def _demand_at(self, time: str) -> np.ndarray:
        return np.asarray(
            self.demand[self.time_index[time]], dtype=float
        ).copy()

    def _generator_static_bounds(
        self,
        active_planned_outages: set[str],
        contingency: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = self.network.pmin.copy()
        upper = self.network.pmax.copy()
        metadata = self._contingency_metadata(contingency)
        for generator_index, generator in enumerate(self.network.generators):
            unavailable = generator in active_planned_outages
            unavailable = unavailable or (
                metadata["type"] == "generator"
                and metadata["element"] == generator
            )
            if unavailable:
                lower[generator_index] = 0.0
                upper[generator_index] = 0.0
        return lower, upper

    def _active_lines(
        self,
        active_planned_outages: set[str],
        contingency: str | None,
    ) -> tuple[int, ...]:
        outaged_lines = {
            outage
            for outage in active_planned_outages
            if outage in self.network.line_index
        }
        metadata = self._contingency_metadata(contingency)
        if metadata["type"] == "line":
            outaged_lines.add(str(metadata["element"]))
        return self.network.active_line_indices(outaged_lines)

    def _add_state_network(
        self,
        model: Model,
        state_label: str,
        contingency: str | None,
        active_outages: set[str],
        demand: np.ndarray,
        pg,
        theta,
        flow,
        shed,
        spill,
    ) -> tuple[int, ...]:
        active_lines = self._active_lines(active_outages, contingency)
        topology = self.network.topology(active_lines)

        for line_index in active_lines:
            row = self.network.bf[line_index]
            nonzero = np.flatnonzero(np.abs(row) > 1e-12)
            model.addConstr(
                flow[state_label, line_index]
                == self.network.base_mva
                * quicksum(
                    float(row[bus_index])
                    * theta[state_label, bus_index]
                    for bus_index in nonzero
                )
                + float(self.network.pfinj_mw[line_index]),
                name=f"DC_{state_label}_{line_index}",
            )

        for component_index, component in enumerate(
            topology.connected_components
        ):
            reference = (
                self.network.preferred_reference
                if self.network.preferred_reference in component
                else component[0]
            )
            model.addConstr(
                theta[state_label, reference] == 0.0,
                name=f"Reference_{state_label}_{component_index}_{reference}",
            )

        incident: list[list[tuple[int, float]]] = [
            [] for _ in self.network.buses
        ]
        for line_index in active_lines:
            for bus_index in np.flatnonzero(
                np.abs(self.network.cft[line_index]) > 1e-12
            ):
                incident[int(bus_index)].append(
                    (
                        line_index,
                        float(self.network.cft[line_index, bus_index]),
                    )
                )

        for bus_index in range(len(self.network.buses)):
            generation = quicksum(
                pg[state_label, generator_index]
                for generator_index in self.network.generators_by_bus[bus_index]
            )
            net_outflow = quicksum(
                coefficient * flow[state_label, line_index]
                for line_index, coefficient in incident[bus_index]
            )
            model.addConstr(
                generation
                - net_outflow
                + shed[state_label, bus_index]
                - spill[state_label, bus_index]
                == float(demand[bus_index]),
                name=f"Balance_{state_label}_{bus_index}",
            )
        return active_lines

    def rank_contingencies(
        self,
        normal_state: ClusterStateResult,
        active_planned_outages: Sequence[str],
        *,
        top_k: int | None,
        mandatory: Iterable[str] = (),
    ) -> tuple[str, ...]:
        """Rank N-1 states using PTDF/LODF and lost-generation proxies."""
        active = set(active_planned_outages)
        outaged_lines = {
            outage
            for outage in active
            if outage in self.network.line_index
        }
        active_indices = self.network.active_line_indices(outaged_lines)
        line_scores = self.network.branch_screening_scores(
            active_indices, normal_state.injections
        )
        mandatory_set = set(mandatory)
        candidates: list[tuple[float, str]] = []

        for contingency in self.effective_contingencies(active):
            metadata = self._contingency_metadata(contingency)
            if metadata["type"] == "line":
                score = line_scores.get(
                    str(metadata["element"]), -np.inf
                )
            elif metadata["type"] == "generator":
                lost = abs(
                    normal_state.generation.get(
                        str(metadata["element"]), 0.0
                    )
                )
                score = lost / max(1.0, float(np.sum(self.network.pmax)))
            else:
                score = -np.inf
            if contingency in mandatory_set:
                score = float("inf")
            candidates.append((float(score), contingency))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        if top_k is None or top_k >= len(candidates):
            return tuple(contingency for _, contingency in candidates)
        selected = {
            contingency for _, contingency in candidates[:top_k]
        } | mandatory_set
        return tuple(
            contingency
            for _, contingency in candidates
            if contingency in selected
        )

    def solve(
        self,
        time: str,
        active_planned_outages: Sequence[str],
        *,
        contingencies: Sequence[str] | None = None,
        demand_override: Sequence[float] | np.ndarray | None = None,
    ) -> ClusterSCOPFResult:
        active_outages = set(active_planned_outages)
        effective = self.effective_contingencies(
            active_planned_outages, contingencies
        )
        state_contingencies: tuple[str | None, ...] = (None, *effective)
        state_labels = {
            contingency: (
                "normal" if contingency is None else str(contingency)
            )
            for contingency in state_contingencies
        }
        if demand_override is None:
            demand = self._demand_at(time)
        else:
            demand = np.asarray(demand_override, dtype=float).reshape(-1).copy()
            if demand.shape != (len(self.network.buses),):
                raise ValueError(
                    "demand_override must contain exactly one value per bus; "
                    f"expected {len(self.network.buses)}, received {demand.shape}."
                )
            if np.any(~np.isfinite(demand)) or np.any(demand < -1e-9):
                raise ValueError(
                    "demand_override must be finite and non-negative."
                )
            demand = np.maximum(demand, 0.0)

        environment = self._environment()
        if environment is None:
            model = Model(f"cluster_scopf::{time}")
        else:
            model = Model(f"cluster_scopf::{time}", env=environment)
        model.Params.OutputFlag = 0
        model.Params.Threads = int(self.data.get("cluster_oracle_threads", 1))
        model.Params.Method = int(self.data.get("cluster_oracle_method", 1))
        model.Params.Presolve = 2
        model.Params.FeasibilityTol = float(
            self.data.get("oracle_feasibility_tol", 1e-7)
        )

        pg_keys: list[tuple[str, int]] = []
        theta_keys: list[tuple[str, int]] = []
        flow_keys: list[tuple[str, int]] = []
        bus_keys: list[tuple[str, int]] = []
        active_lines_by_state: dict[str, tuple[int, ...]] = {}

        for contingency in state_contingencies:
            label = state_labels[contingency]
            active_lines = self._active_lines(active_outages, contingency)
            active_lines_by_state[label] = active_lines
            pg_keys.extend(
                (label, index)
                for index in range(len(self.network.generators))
            )
            theta_keys.extend(
                (label, index)
                for index in range(len(self.network.buses))
            )
            flow_keys.extend((label, index) for index in active_lines)
            bus_keys.extend(
                (label, index)
                for index in range(len(self.network.buses))
            )

        pg = model.addVars(pg_keys, lb=-GRB.INFINITY, name="pg")
        theta = model.addVars(
            theta_keys, lb=-GRB.INFINITY, name="theta"
        )
        flow = model.addVars(flow_keys, lb=-GRB.INFINITY, name="flow")
        shed = model.addVars(bus_keys, lb=0.0, name="shed")
        spill = model.addVars(bus_keys, lb=0.0, name="spill")

        for contingency in state_contingencies:
            label = state_labels[contingency]
            lower, upper = self._generator_static_bounds(
                active_outages, contingency
            )
            for generator_index in range(len(self.network.generators)):
                pg[label, generator_index].LB = float(lower[generator_index])
                pg[label, generator_index].UB = float(upper[generator_index])
            for line_index in active_lines_by_state[label]:
                flow[label, line_index].LB = -float(
                    self.network.limits[line_index]
                )
                flow[label, line_index].UB = float(
                    self.network.limits[line_index]
                )
            for bus_index in range(len(self.network.buses)):
                shed[label, bus_index].UB = max(
                    0.0, float(demand[bus_index])
                )

            self._add_state_network(
                model,
                label,
                contingency,
                active_outages,
                demand,
                pg,
                theta,
                flow,
                shed,
                spill,
            )

        normal_label = "normal"
        redispatch_deviation = 0.0
        redispatch_cost = float(self.data.get("redispatch_cost", 0.0))
        for contingency in effective:
            label = state_labels[contingency]
            metadata = self._contingency_metadata(contingency)
            for generator_index, generator in enumerate(
                self.network.generators
            ):
                failed = (
                    metadata["type"] == "generator"
                    and metadata["element"] == generator
                )
                planned = generator in active_outages
                if failed or planned:
                    continue
                up = float(
                    self.corrective_up.get(
                        generator,
                        self.corrective_fraction
                        * self.network.pmax[generator_index],
                    )
                )
                down = float(
                    self.corrective_down.get(
                        generator,
                        self.corrective_fraction
                        * self.network.pmax[generator_index],
                    )
                )
                model.addConstr(
                    pg[label, generator_index]
                    - pg[normal_label, generator_index]
                    <= up,
                    name=f"CorrectiveUp_{label}_{generator_index}",
                )
                model.addConstr(
                    pg[normal_label, generator_index]
                    - pg[label, generator_index]
                    <= down,
                    name=f"CorrectiveDown_{label}_{generator_index}",
                )
                if redispatch_cost > 0.0:
                    positive = model.addVar(
                        lb=0.0,
                        name=f"RedispatchPositive_{label}_{generator_index}",
                    )
                    negative = model.addVar(
                        lb=0.0,
                        name=f"RedispatchNegative_{label}_{generator_index}",
                    )
                    model.addConstr(
                        pg[label, generator_index]
                        - pg[normal_label, generator_index]
                        == positive - negative,
                        name=f"RedispatchBalance_{label}_{generator_index}",
                    )
                    redispatch_deviation += positive + negative

        dns_by_state = {
            label: quicksum(
                shed[label, bus_index]
                for bus_index in range(len(self.network.buses))
            )
            for label in state_labels.values()
        }
        worst_dns = model.addVar(lb=0.0, name="maximum_dns")
        for label, dns_expression in dns_by_state.items():
            model.addConstr(
                worst_dns >= dns_expression,
                name=f"MaximumDNS_{label}",
            )

        total_dns = quicksum(dns_by_state.values())
        total_spillage = quicksum(
            spill[label, bus_index]
            for label in state_labels.values()
            for bus_index in range(len(self.network.buses))
        )
        normal_generation_cost = quicksum(
            float(self.generation_cost.get(generator, 1.0))
            * pg[normal_label, generator_index]
            for generator_index, generator in enumerate(
                self.network.generators
            )
        )

        objective = (
            float(self.data.get("cluster_max_dns_weight", 1e6))
            * worst_dns
            + float(self.data.get("cluster_total_dns_weight", 1e3))
            * total_dns
            + float(self.data.get("spillage_cost", 1.0))
            * total_spillage
            + float(self.data.get("cluster_generation_cost_weight", 1e-3))
            * normal_generation_cost
            + redispatch_cost * redispatch_deviation
        )
        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        acceptable = {GRB.OPTIMAL, GRB.SUBOPTIMAL}
        if model.Status not in acceptable or model.SolCount <= 0:
            result = ClusterSCOPFResult(
                time=time,
                active_planned_outages=tuple(sorted(active_outages)),
                contingencies=effective,
                status=model.Status,
                objective=float("inf"),
                maximum_dns=float("inf"),
                total_dns=float("inf"),
                worst_contingency=None,
                states={},
            )
            model.dispose()
            return result

        states: dict[str, ClusterStateResult] = {}
        maximum_dns = -1.0
        worst_contingency: str | None = None
        total_dns_value = 0.0

        for contingency in state_contingencies:
            label = state_labels[contingency]
            generation = {
                generator: float(pg[label, generator_index].X)
                for generator_index, generator in enumerate(
                    self.network.generators
                )
            }
            flows = {
                self.network.lines[line_index]: float(
                    flow[label, line_index].X
                )
                for line_index in active_lines_by_state[label]
            }
            shed_values = np.asarray(
                [
                    float(shed[label, bus_index].X)
                    for bus_index in range(len(self.network.buses))
                ],
                dtype=float,
            )
            spill_values = np.asarray(
                [
                    float(spill[label, bus_index].X)
                    for bus_index in range(len(self.network.buses))
                ],
                dtype=float,
            )
            injections = -demand + shed_values - spill_values
            for generator_index, bus_index in enumerate(
                self.network.generator_bus
            ):
                injections[int(bus_index)] += float(
                    pg[label, generator_index].X
                )
            dns_value = float(np.sum(shed_values))
            total_dns_value += dns_value
            if dns_value > maximum_dns:
                maximum_dns = dns_value
                worst_contingency = contingency
            states[label] = ClusterStateResult(
                contingency=contingency,
                load_shedding=dns_value,
                spillage=float(np.sum(spill_values)),
                load_shedding_by_bus={
                    bus: float(shed_values[bus_index])
                    for bus_index, bus in enumerate(self.network.buses)
                },
                spillage_by_bus={
                    bus: float(spill_values[bus_index])
                    for bus_index, bus in enumerate(self.network.buses)
                },
                generation=generation,
                flows=flows,
                injections=injections,
            )

        result = ClusterSCOPFResult(
            time=time,
            active_planned_outages=tuple(sorted(active_outages)),
            contingencies=effective,
            status=model.Status,
            objective=float(model.ObjVal),
            maximum_dns=float(maximum_dns),
            total_dns=float(total_dns_value),
            worst_contingency=worst_contingency,
            states=states,
        )
        model.dispose()
        return result
