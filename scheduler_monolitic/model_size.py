"""Preflight estimates for the monolithic extensive-form SCOS model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSizeEstimate:
    n_times: int
    n_buses: int
    n_lines: int
    n_generators: int
    n_outages: int
    n_scenarios: int
    n_contingencies: int
    variables: int
    constraints: int

    @property
    def variables_million(self) -> float:
        return self.variables / 1_000_000.0

    @property
    def constraints_million(self) -> float:
        return self.constraints / 1_000_000.0

    def summary(self) -> str:
        return (
            f"approximately {self.variables:,} variables "
            f"({self.variables_million:.2f} million) and "
            f"{self.constraints:,} constraints "
            f"({self.constraints_million:.2f} million)"
        )


def estimate_extensive_form_size(
    *,
    n_times: int,
    n_buses: int,
    n_lines: int,
    n_generators: int,
    n_outages: int,
    n_scenarios: int,
    n_contingencies: int,
    use_dc_power_flow: bool = True,
    include_base_state: bool = True,
    include_cvar_variables: bool = True,
) -> ModelSizeEstimate:
    """Return a conservative count before any Gurobi objects are created."""

    T, B, L, G, O, S, C = (
        int(n_times),
        int(n_buses),
        int(n_lines),
        int(n_generators),
        int(n_outages),
        int(n_scenarios),
        int(n_contingencies),
    )

    # Scheduling binaries: activity and feasible-start indicators.
    schedule_variables = 2 * T * O

    # Base-state recourse: generation, flows, angles, shedding and spillage.
    base_variables_per_ts = G + L + 2 * B + (B if use_dc_power_flow else 0)
    base_variables = T * S * base_variables_per_ts

    # Contingency recourse uses the same state vector for every contingency.
    contingency_variables_per_tsc = G + L + 2 * B + (B if use_dc_power_flow else 0)
    contingency_variables = T * S * C * contingency_variables_per_tsc

    # Worst-state DNS, one scenario loss, and optional CVaR auxiliaries.
    risk_variables = T * S + S
    if include_cvar_variables:
        risk_variables += S + 2  # excess_s, eta, CVaR

    variables = schedule_variables + base_variables + contingency_variables + risk_variables

    # Scheduling constraints: one start, convolution and max simultaneous tasks.
    schedule_constraints = O + T * O + T

    # Generator bounds and corrective redispatch bounds.
    generation_constraints = 2 * T * G * S
    generation_constraints += 4 * T * G * S * C

    # Flow bounds and optional DC equations; failed-line equalities are covered
    # conservatively by the same count.
    flow_factor = 3 if use_dc_power_flow else 2
    network_constraints = flow_factor * T * L * S
    network_constraints += flow_factor * T * L * S * C
    if use_dc_power_flow:
        network_constraints += T * S * (1 + C)  # reference angles

    # Nodal balance plus upper bounds on shedding and spillage.
    balance_constraints = 3 * T * B * S
    balance_constraints += 3 * T * B * S * C

    # Worst-state inequalities, scenario-loss definitions, and CVaR equations.
    risk_constraints = T * S * (C + (1 if include_base_state else 0)) + S
    if include_cvar_variables:
        risk_constraints += S + 1

    constraints = (
        schedule_constraints
        + generation_constraints
        + network_constraints
        + balance_constraints
        + risk_constraints
    )

    return ModelSizeEstimate(
        n_times=T,
        n_buses=B,
        n_lines=L,
        n_generators=G,
        n_outages=O,
        n_scenarios=S,
        n_contingencies=C,
        variables=variables,
        constraints=constraints,
    )
