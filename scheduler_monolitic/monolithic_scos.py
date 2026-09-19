"""Unified monolithic risk-neutral and CVaR security-constrained outage scheduler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from gurobipy import GRB, LinExpr, Model, quicksum
except ImportError as exc:  # pragma: no cover - depends on commercial solver installation
    raise ImportError(
        "gurobipy is required for scheduler_monolitic.monolithic_scos. "
        "Install Gurobi and activate a valid licence."
    ) from exc

from .config import ModelConfig, RiskConfig, SolverConfig
from .data_adapter import PreparedSCOSData, prepare_scos_data
from .postprocessing import SCOSResult, extract_result
from .scenario_generation import ScenarioBank
from .model_size import estimate_extensive_form_size

LOGGER = logging.getLogger(__name__)


def contingency_asset(contingency: str) -> str:
    """Map ``n1_line_3`` to ``line_3`` and ``n1_gen_2`` to ``gen_2``."""
    if contingency.startswith("n1_"):
        return contingency[3:]
    return contingency


def _schedule_value(
    schedule: pd.DataFrame | Mapping[str, Any],
    outage: str,
    time: str,
    duration: int = 1,
) -> int:
    if isinstance(schedule, pd.DataFrame):
        if outage not in schedule.index:
            raise KeyError(f"Fixed schedule is missing outage {outage}.")
        if time in schedule.columns:
            value = schedule.loc[outage, time]
        else:
            index = int(time.replace("step_", ""))
            value = schedule.loc[outage].iloc[index]
        return int(round(float(value)))

    item = schedule[outage]
    if isinstance(item, Mapping):
        return int(round(float(item[time])))
    if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
        index = int(time.replace("step_", ""))
        return int(round(float(item[index])))
    start = int(item)
    index = int(time.replace("step_", ""))
    return int(start <= index < start + duration)


@dataclass
class ModelArtifacts:
    model: Model
    data: PreparedSCOSData
    variables: dict[str, Any]
    expressions: dict[str, Any]
    risk_config: RiskConfig
    model_config: ModelConfig


class MonolithicSCOS:
    """
    Common stochastic SCOS model.

    The risk-neutral and CVaR formulations share the same outage, demand,
    contingency, network and recourse representations. They differ only in the
    risk functional used to rank feasible schedules.
    """

    def __init__(
        self,
        data: dict[str, Any],
        scenario_bank: ScenarioBank,
        risk_config: RiskConfig | None = None,
        model_config: ModelConfig | None = None,
        solver_config: SolverConfig | None = None,
        *,
        fixed_schedule: pd.DataFrame | Mapping[str, Any] | None = None,
        reference_schedule: pd.DataFrame | None = None,
        reference_utility: float | None = None,
    ) -> None:
        self.risk_config = risk_config or RiskConfig()
        self.model_config = model_config or ModelConfig()
        self.solver_config = solver_config or SolverConfig()
        self.risk_config.validate()
        self.model_config.validate()
        self.solver_config.validate()

        self.data = prepare_scos_data(
            data,
            scenario_bank,
            contingency_top_k=self.model_config.contingency_top_k,
        )
        self.fixed_schedule = fixed_schedule
        self.reference_schedule = reference_schedule
        self.reference_utility = reference_utility

        self.size_estimate = estimate_extensive_form_size(
            n_times=len(self.data.times),
            n_buses=len(self.data.buses),
            n_lines=len(self.data.lines),
            n_generators=len(self.data.generators),
            n_outages=len(self.data.outages),
            n_scenarios=len(self.data.scenarios),
            n_contingencies=len(self.data.contingencies),
            use_dc_power_flow=self.model_config.use_dc_power_flow,
            include_base_state=self.model_config.include_base_state,
            include_cvar_variables=self.risk_config.formulation in {"cvar", "weighted"},
        )
        LOGGER.info(
            "Monolithic preflight: T=%d, S=%d, C=%d; %s.",
            len(self.data.times),
            len(self.data.scenarios),
            len(self.data.contingencies),
            self.size_estimate.summary(),
        )
        variable_limit = self.model_config.max_estimated_variables
        if (
            variable_limit is not None
            and self.size_estimate.variables > variable_limit
            and not self.model_config.allow_large_model
        ):
            raise RuntimeError(
                "The requested monolithic extensive form is too large for the "
                "configured safety limit. It would create "
                f"{self.size_estimate.summary()}. The variable limit is "
                f"{variable_limit:,}. Reduce the temporal resolution (recommended: "
                "--aggregation-time W), reduce training scenarios/contingencies, "
                "or pass --allow-large-model only when sufficient RAM is available."
            )

        self.model = Model(f"Monolithic_SCOS_{self.risk_config.formulation}")
        self.variables: dict[str, Any] = {}
        self.expressions: dict[str, Any] = {}
        self._built = False

    def build(self) -> ModelArtifacts:
        if self._built:
            return self.artifacts
        LOGGER.info("Building schedule variables and constraints.")
        self._add_schedule_variables_and_constraints()
        LOGGER.info("Adding operational variables.")
        self._add_operational_variables()
        LOGGER.info("Adding generation constraints.")
        self._add_generation_constraints()
        LOGGER.info("Adding sparse network constraints.")
        self._add_network_constraints()
        LOGGER.info("Adding sparse nodal-balance constraints.")
        self._add_power_balance_constraints()
        LOGGER.info("Adding risk functional.")
        self._add_risk_functional()
        self._set_objective()
        self.solver_config.apply(self.model)
        self.model.update()
        self._built = True
        return self.artifacts

    @property
    def artifacts(self) -> ModelArtifacts:
        return ModelArtifacts(
            model=self.model,
            data=self.data,
            variables=self.variables,
            expressions=self.expressions,
            risk_config=self.risk_config,
            model_config=self.model_config,
        )

    def _operational_constraint_name(self, value: str) -> str:
        return value if self.model_config.name_operational_constraints else ""

    @staticmethod
    def _linexpr(coefficients: list[float], variables: list[Any]) -> Any:
        if not variables:
            return 0.0
        return LinExpr(coefficients, variables)

    def _add_schedule_variables_and_constraints(self) -> None:
        d = self.data
        x = self.model.addVars(d.times, d.outages, vtype=GRB.BINARY, name="planned_outage_indicator")

        start_keys: list[tuple[str, str]] = []
        feasible_start_times: dict[str, list[str]] = {}
        for outage in d.outages:
            latest = len(d.times) - d.duration[outage]
            feasible = d.times[: latest + 1]
            feasible_start_times[outage] = feasible
            start_keys.extend((time, outage) for time in feasible)
        start = self.model.addVars(start_keys, vtype=GRB.BINARY, name="outage_start")

        for outage in d.outages:
            feasible = feasible_start_times[outage]
            self.model.addConstr(
                quicksum(start[time, outage] for time in feasible) == 1,
                name=f"one_start[{outage}]",
            )
            duration = d.duration[outage]
            for time_index, time in enumerate(d.times):
                covering = [
                    start[start_time, outage]
                    for start_index, start_time in enumerate(feasible)
                    if start_index <= time_index < start_index + duration
                ]
                self.model.addConstr(
                    x[time, outage] == quicksum(covering),
                    name=f"schedule_convolution[{time},{outage}]",
                )

        for time in d.times:
            self.model.addConstr(
                quicksum(x[time, outage] for outage in d.outages) <= d.max_tasks,
                name=f"maximum_simultaneous_outages[{time}]",
            )

        if self.fixed_schedule is not None:
            for time in d.times:
                for outage in d.outages:
                    value = _schedule_value(self.fixed_schedule, outage, time, d.duration[outage])
                    x[time, outage].LB = value
                    x[time, outage].UB = value

        if self.reference_schedule is not None:
            for time in d.times:
                for outage in d.outages:
                    x[time, outage].Start = _schedule_value(self.reference_schedule, outage, time, d.duration[outage])

        utility = quicksum(
            d.priority[outage]
            * (len(d.times) - start_index)
            / len(d.times)
            * start[time, outage]
            for outage in d.outages
            for start_index, time in enumerate(feasible_start_times[outage])
        )
        utility_scale = max(sum(d.priority.values()), 1.0)

        self.variables.update({"x": x, "start": start})
        self.expressions.update(
            {
                "maintenance_utility": utility,
                "maintenance_utility_normalised": utility / utility_scale,
                "utility_scale": utility_scale,
            }
        )

    def _add_operational_variables(self) -> None:
        d = self.data
        T, B, L, G, S, C = d.times, d.buses, d.lines, d.generators, d.scenarios, d.contingencies
        angle_bound = self.model_config.angle_bound

        variables = {
            "p": self.model.addVars(T, G, S, lb=0.0, name="power_generation"),
            "flow": self.model.addVars(T, L, S, lb=-GRB.INFINITY, name="line_flow"),
            "theta": self.model.addVars(T, B, S, lb=-angle_bound, ub=angle_bound, name="voltage_angle"),
            "shed": self.model.addVars(T, B, S, lb=0.0, name="load_shedding"),
            "spill": self.model.addVars(T, B, S, lb=0.0, name="generation_spillage"),
            "p_c": self.model.addVars(T, G, S, C, lb=0.0, name="power_generation_contingency"),
            "flow_c": self.model.addVars(T, L, S, C, lb=-GRB.INFINITY, name="line_flow_contingency"),
            "theta_c": self.model.addVars(
                T, B, S, C, lb=-angle_bound, ub=angle_bound, name="voltage_angle_contingency"
            ),
            "shed_c": self.model.addVars(T, B, S, C, lb=0.0, name="load_shedding_contingency"),
            "spill_c": self.model.addVars(T, B, S, C, lb=0.0, name="generation_spillage_contingency"),
        }
        self.variables.update(variables)

    def _generator_availability(self, time: str, generator: str) -> Any:
        if generator in self.data.outages:
            return 1 - self.variables["x"][time, generator]
        return 1.0

    def _line_availability(self, time: str, line: str) -> Any:
        if line in self.data.outages:
            return 1 - self.variables["x"][time, line]
        return 1.0

    def _add_generation_constraints(self) -> None:
        d = self.data
        p, p_c = self.variables["p"], self.variables["p_c"]
        redispatch_fraction = self.model_config.corrective_redispatch_fraction
        failed_assets = {contingency: contingency_asset(contingency) for contingency in d.contingencies}
        report_interval = max(1, len(d.times) // 10)

        for time_index, time in enumerate(d.times):
            if time_index % report_interval == 0 or time_index == len(d.times) - 1:
                LOGGER.info(
                    "Generation constraints: period %d/%d.",
                    time_index + 1,
                    len(d.times),
                )
            for generator in d.generators:
                availability = self._generator_availability(time, generator)
                redispatch = redispatch_fraction * d.p_max[generator]
                for scenario in d.scenarios:
                    self.model.addConstr(
                        p[time, generator, scenario] <= d.p_max[generator] * availability,
                        name=self._operational_constraint_name(
                            f"generator_upper[{time},{generator},{scenario}]"
                        ),
                    )
                    self.model.addConstr(
                        p[time, generator, scenario] >= d.p_min[generator] * availability,
                        name=self._operational_constraint_name(
                            f"generator_lower[{time},{generator},{scenario}]"
                        ),
                    )

                    for contingency in d.contingencies:
                        if failed_assets[contingency] == generator:
                            self.model.addConstr(
                                p_c[time, generator, scenario, contingency] == 0.0,
                                name=self._operational_constraint_name(
                                    f"failed_generator[{time},{generator},{scenario},{contingency}]"
                                ),
                            )
                            continue

                        self.model.addConstr(
                            p_c[time, generator, scenario, contingency]
                            <= d.p_max[generator] * availability,
                            name=self._operational_constraint_name(
                                f"generator_cont_upper[{time},{generator},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            p_c[time, generator, scenario, contingency]
                            >= d.p_min[generator] * availability,
                            name=self._operational_constraint_name(
                                f"generator_cont_lower[{time},{generator},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            p_c[time, generator, scenario, contingency]
                            - p[time, generator, scenario]
                            <= redispatch,
                            name=self._operational_constraint_name(
                                f"redispatch_up[{time},{generator},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            p[time, generator, scenario]
                            - p_c[time, generator, scenario, contingency]
                            <= redispatch,
                            name=self._operational_constraint_name(
                                f"redispatch_down[{time},{generator},{scenario},{contingency}]"
                            ),
                        )

    def _add_network_constraints(self) -> None:
        d = self.data
        x = self.variables["x"]
        flow, theta = self.variables["flow"], self.variables["theta"]
        flow_c, theta_c = self.variables["flow_c"], self.variables["theta_c"]
        failed_assets = {contingency: contingency_asset(contingency) for contingency in d.contingencies}
        report_interval = max(1, len(d.times) // 10)

        for time_index, time in enumerate(d.times):
            if time_index % report_interval == 0 or time_index == len(d.times) - 1:
                LOGGER.info(
                    "Network constraints: period %d/%d.",
                    time_index + 1,
                    len(d.times),
                )
            for scenario in d.scenarios:
                for reference_bus in d.reference_buses:
                    self.model.addConstr(
                        theta[time, reference_bus, scenario] == 0.0,
                        name=self._operational_constraint_name(
                            f"reference_angle[{time},{reference_bus},{scenario}]"
                        ),
                    )
                    for contingency in d.contingencies:
                        self.model.addConstr(
                            theta_c[time, reference_bus, scenario, contingency] == 0.0,
                            name=self._operational_constraint_name(
                                f"reference_angle_cont[{time},{reference_bus},{scenario},{contingency}]"
                            ),
                        )

                for line in d.lines:
                    availability = self._line_availability(time, line)
                    limit = d.flow_limit[line]
                    self.model.addConstr(
                        flow[time, line, scenario] <= limit * availability,
                        name=self._operational_constraint_name(
                            f"flow_upper[{time},{line},{scenario}]"
                        ),
                    )
                    self.model.addConstr(
                        flow[time, line, scenario] >= -limit * availability,
                        name=self._operational_constraint_name(
                            f"flow_lower[{time},{line},{scenario}]"
                        ),
                    )

                    if self.model_config.use_dc_power_flow:
                        dc_terms = d.b_flow_terms[line]
                        expression = self._linexpr(
                            [coefficient for _, coefficient in dc_terms],
                            [theta[time, bus, scenario] for bus, _ in dc_terms],
                        )
                        if line in d.outages:
                            self.model.addGenConstrIndicator(
                                x[time, line],
                                0,
                                flow[time, line, scenario] == expression,
                                name=self._operational_constraint_name(
                                    f"dc_flow_active[{time},{line},{scenario}]"
                                ),
                            )
                        else:
                            self.model.addConstr(
                                flow[time, line, scenario] == expression,
                                name=self._operational_constraint_name(
                                    f"dc_flow[{time},{line},{scenario}]"
                                ),
                            )

                    for contingency in d.contingencies:
                        if failed_assets[contingency] == line:
                            self.model.addConstr(
                                flow_c[time, line, scenario, contingency] == 0.0,
                                name=self._operational_constraint_name(
                                    f"failed_line[{time},{line},{scenario},{contingency}]"
                                ),
                            )
                            continue

                        self.model.addConstr(
                            flow_c[time, line, scenario, contingency] <= limit * availability,
                            name=self._operational_constraint_name(
                                f"flow_cont_upper[{time},{line},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            flow_c[time, line, scenario, contingency] >= -limit * availability,
                            name=self._operational_constraint_name(
                                f"flow_cont_lower[{time},{line},{scenario},{contingency}]"
                            ),
                        )
                        if self.model_config.use_dc_power_flow:
                            dc_terms = d.b_flow_terms[line]
                            expression_c = self._linexpr(
                                [coefficient for _, coefficient in dc_terms],
                                [
                                    theta_c[time, bus, scenario, contingency]
                                    for bus, _ in dc_terms
                                ],
                            )
                            if line in d.outages:
                                self.model.addGenConstrIndicator(
                                    x[time, line],
                                    0,
                                    flow_c[time, line, scenario, contingency] == expression_c,
                                    name=self._operational_constraint_name(
                                        f"dc_flow_cont_active[{time},{line},{scenario},{contingency}]"
                                    ),
                                )
                            else:
                                self.model.addConstr(
                                    flow_c[time, line, scenario, contingency] == expression_c,
                                    name=self._operational_constraint_name(
                                        f"dc_flow_cont[{time},{line},{scenario},{contingency}]"
                                    ),
                                )

    def _add_power_balance_constraints(self) -> None:
        d = self.data
        p, flow, shed, spill = (
            self.variables["p"],
            self.variables["flow"],
            self.variables["shed"],
            self.variables["spill"],
        )
        p_c, flow_c, shed_c, spill_c = (
            self.variables["p_c"],
            self.variables["flow_c"],
            self.variables["shed_c"],
            self.variables["spill_c"],
        )
        total_pmax = sum(d.p_max.values())
        report_interval = max(1, len(d.scenarios) // 10)

        for scenario_index, scenario in enumerate(d.scenarios):
            if scenario_index % report_interval == 0 or scenario_index == len(d.scenarios) - 1:
                LOGGER.info(
                    "Nodal balances: scenario %d/%d.",
                    scenario_index + 1,
                    len(d.scenarios),
                )
            demand = d.scenario_demands[scenario]
            for time_index, time in enumerate(d.times):
                for bus_index, bus in enumerate(d.buses):
                    generators_at_bus = d.bus_to_generators[bus]
                    generation = quicksum(
                        p[time, generator, scenario]
                        for generator in generators_at_bus
                    )
                    incidence_terms = d.incidence_terms[bus]
                    network_export = self._linexpr(
                        [coefficient for _, coefficient in incidence_terms],
                        [flow[time, line, scenario] for line, _ in incidence_terms],
                    )
                    demand_value = float(demand.iloc[time_index, bus_index])
                    self.model.addConstr(
                        generation
                        - network_export
                        + shed[time, bus, scenario]
                        - spill[time, bus, scenario]
                        == demand_value,
                        name=self._operational_constraint_name(
                            f"power_balance[{time},{bus},{scenario}]"
                        ),
                    )
                    self.model.addConstr(
                        shed[time, bus, scenario] <= demand_value,
                        name=self._operational_constraint_name(
                            f"shed_upper[{time},{bus},{scenario}]"
                        ),
                    )
                    self.model.addConstr(
                        spill[time, bus, scenario] <= total_pmax,
                        name=self._operational_constraint_name(
                            f"spill_upper[{time},{bus},{scenario}]"
                        ),
                    )

                    for contingency in d.contingencies:
                        generation_c = quicksum(
                            p_c[time, generator, scenario, contingency]
                            for generator in generators_at_bus
                        )
                        network_export_c = self._linexpr(
                            [coefficient for _, coefficient in incidence_terms],
                            [
                                flow_c[time, line, scenario, contingency]
                                for line, _ in incidence_terms
                            ],
                        )
                        self.model.addConstr(
                            generation_c
                            - network_export_c
                            + shed_c[time, bus, scenario, contingency]
                            - spill_c[time, bus, scenario, contingency]
                            == demand_value,
                            name=self._operational_constraint_name(
                                f"power_balance_cont[{time},{bus},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            shed_c[time, bus, scenario, contingency] <= demand_value,
                            name=self._operational_constraint_name(
                                f"shed_cont_upper[{time},{bus},{scenario},{contingency}]"
                            ),
                        )
                        self.model.addConstr(
                            spill_c[time, bus, scenario, contingency] <= total_pmax,
                            name=self._operational_constraint_name(
                                f"spill_cont_upper[{time},{bus},{scenario},{contingency}]"
                            ),
                        )

    def _add_risk_functional(self) -> None:
        d = self.data
        shed, shed_c = self.variables["shed"], self.variables["shed_c"]
        spill, spill_c = self.variables["spill"], self.variables["spill_c"]

        worst_dns = self.model.addVars(d.times, d.scenarios, lb=0.0, name="worst_state_dns")
        scenario_loss = self.model.addVars(d.scenarios, lb=0.0, name="scenario_loss")

        for scenario in d.scenarios:
            for time in d.times:
                if self.model_config.include_base_state:
                    self.model.addConstr(
                        worst_dns[time, scenario]
                        >= quicksum(shed[time, bus, scenario] for bus in d.buses),
                        name=f"worst_dns_base[{time},{scenario}]",
                    )
                for contingency in d.contingencies:
                    self.model.addConstr(
                        worst_dns[time, scenario]
                        >= quicksum(shed_c[time, bus, scenario, contingency] for bus in d.buses),
                        name=f"worst_dns_cont[{time},{scenario},{contingency}]",
                    )
            self.model.addConstr(
                scenario_loss[scenario]
                == quicksum(d.time_weights[time] * worst_dns[time, scenario] for time in d.times),
                name=f"scenario_loss_definition[{scenario}]",
            )

        expected_loss = quicksum(
            d.scenario_probabilities[scenario] * scenario_loss[scenario]
            for scenario in d.scenarios
        )
        expected_generation = quicksum(
            d.scenario_probabilities[scenario]
            * d.time_weights[time]
            * self.variables["p"][time, generator, scenario]
            for scenario in d.scenarios
            for time in d.times
            for generator in d.generators
        )
        expected_spill = quicksum(
            d.scenario_probabilities[scenario]
            * d.time_weights[time]
            * (
                quicksum(spill[time, bus, scenario] for bus in d.buses)
                + quicksum(
                    spill_c[time, bus, scenario, contingency]
                    for bus in d.buses
                    for contingency in d.contingencies
                )
                / max(len(d.contingencies), 1)
            )
            for scenario in d.scenarios
            for time in d.times
        )

        loss_scale = max(d.peak_demand * sum(d.time_weights.values()), 1.0)
        generation_scale = loss_scale
        self.variables.update({"worst_dns": worst_dns, "scenario_loss": scenario_loss})
        self.expressions.update(
            {
                "expected_loss": expected_loss,
                "expected_loss_normalised": expected_loss / loss_scale,
                "expected_generation": expected_generation,
                "expected_generation_normalised": expected_generation / generation_scale,
                "expected_spill": expected_spill,
                "expected_spill_normalised": expected_spill / generation_scale,
                "loss_scale": loss_scale,
                "generation_scale": generation_scale,
            }
        )

        if self.risk_config.formulation in {"cvar", "weighted"}:
            eta = self.model.addVar(lb=0.0, name="var_threshold")
            excess = self.model.addVars(d.scenarios, lb=0.0, name="cvar_excess")
            cvar_value = self.model.addVar(lb=0.0, name="cvar_value")
            for scenario in d.scenarios:
                self.model.addConstr(
                    excess[scenario] >= scenario_loss[scenario] - eta,
                    name=f"cvar_excess_definition[{scenario}]",
                )
            self.model.addConstr(
                cvar_value
                == eta
                + (1.0 / (1.0 - self.risk_config.beta))
                * quicksum(
                    d.scenario_probabilities[scenario] * excess[scenario]
                    for scenario in d.scenarios
                ),
                name="cvar_definition",
            )
            self.variables.update({"eta": eta, "excess": excess, "cvar_value": cvar_value})
            self.expressions.update(
                {
                    "cvar": cvar_value,
                    "cvar_normalised": cvar_value / loss_scale,
                }
            )

    def _set_objective(self) -> None:
        rc = self.risk_config
        utility = self.expressions["maintenance_utility"]
        utility_normalised = self.expressions["maintenance_utility_normalised"]
        expected_loss = self.expressions["expected_loss_normalised"]
        generation = self.expressions["expected_generation_normalised"]
        spill = self.expressions["expected_spill_normalised"]

        if self.reference_utility is not None:
            self.model.addConstr(
                utility >= self.reference_utility - rc.utility_tolerance,
                name="reference_utility_floor",
            )

        self.model.ModelSense = GRB.MINIMIZE
        if rc.formulation == "expected":
            self.model.setObjectiveN(expected_loss, index=0, priority=3, weight=1.0, name="expected_loss")
            self.model.setObjectiveN(
                rc.generation_weight * generation + self.model_config.spill_penalty * spill,
                index=1,
                priority=2,
                weight=1.0,
                name="operating_effort",
            )
            self.model.setObjectiveN(-utility_normalised, index=2, priority=1, weight=1.0, name="utility")
        elif rc.formulation == "cvar":
            cvar = self.expressions["cvar_normalised"]
            self.model.setObjectiveN(cvar, index=0, priority=4, weight=1.0, name="cvar")
            self.model.setObjectiveN(expected_loss, index=1, priority=3, weight=1.0, name="expected_loss")
            self.model.setObjectiveN(
                rc.generation_weight * generation + self.model_config.spill_penalty * spill,
                index=2,
                priority=2,
                weight=1.0,
                name="operating_effort",
            )
            self.model.setObjectiveN(-utility_normalised, index=3, priority=1, weight=1.0, name="utility")
        else:
            cvar = self.expressions["cvar_normalised"]
            objective = (
                rc.expected_weight * expected_loss
                + rc.cvar_weight * cvar
                + rc.generation_weight * generation
                + self.model_config.spill_penalty * spill
                - rc.utility_weight * utility_normalised
            )
            self.model.setObjective(objective, GRB.MINIMIZE)

    def solve(self, *, output_dir: str | None = None, run_name: str | None = None) -> SCOSResult:
        self.build()
        LOGGER.info(
            "Solving %s formulation with %d scenarios and %d contingencies.",
            self.risk_config.formulation,
            len(self.data.scenarios),
            len(self.data.contingencies),
        )
        self.model.optimize()
        result = extract_result(self.artifacts)
        if output_dir is not None:
            from .postprocessing import save_result_bundle

            save_result_bundle(result, output_dir, run_name or self.risk_config.formulation)
        return result


def solve_scos(
    data: dict[str, Any],
    scenario_bank: ScenarioBank,
    risk_config: RiskConfig | None = None,
    model_config: ModelConfig | None = None,
    solver_config: SolverConfig | None = None,
    **kwargs: Any,
) -> SCOSResult:
    """Functional interface to :class:`MonolithicSCOS`."""
    scheduler = MonolithicSCOS(
        data=data,
        scenario_bank=scenario_bank,
        risk_config=risk_config,
        model_config=model_config,
        solver_config=solver_config,
        fixed_schedule=kwargs.pop("fixed_schedule", None),
        reference_schedule=kwargs.pop("reference_schedule", None),
        reference_utility=kwargs.pop("reference_utility", None),
    )
    if kwargs:
        raise TypeError(f"Unknown solve_scos keyword arguments: {sorted(kwargs)}")
    return scheduler.solve()
