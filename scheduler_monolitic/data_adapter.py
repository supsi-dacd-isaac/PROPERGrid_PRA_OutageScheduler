"""Data loading and model-data preparation for the monolithic scheduler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scenario_generation import ScenarioBank, align_demand_columns

LOGGER = logging.getLogger(__name__)


def _dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    if hasattr(matrix, "A"):
        return np.asarray(matrix.A)
    return np.asarray(matrix)


@dataclass(frozen=True)
class PreparedSCOSData:
    raw: dict[str, Any]
    times: list[str]
    buses: list[str]
    lines: list[str]
    generators: list[str]
    outages: list[str]
    contingencies: list[str]
    scenarios: list[str]
    scenario_demands: dict[str, pd.DataFrame]
    scenario_probabilities: dict[str, float]
    incidence: np.ndarray
    b_flow: np.ndarray
    incidence_terms: dict[str, tuple[tuple[str, float], ...]]
    b_flow_terms: dict[str, tuple[tuple[str, float], ...]]
    bus_to_generators: dict[str, list[str]]
    p_min: dict[str, float]
    p_max: dict[str, float]
    flow_limit: dict[str, float]
    duration: dict[str, int]
    priority: dict[str, float]
    max_tasks: int
    reference_buses: list[str]
    peak_demand: float
    time_weights: dict[str, float]

    @property
    def names(self) -> dict[str, list[str]]:
        return {
            "times": self.times,
            "buses": self.buses,
            "lines": self.lines,
            "generators": self.generators,
            "outages": self.outages,
            "contingencies": self.contingencies,
            "scenarios": self.scenarios,
        }


def build_contingency_names(data: dict[str, Any], lines: list[str], generators: list[str]) -> list[str]:
    names = data.get("n_minus1_names")
    if names:
        return list(names)
    return [f"n1_{line}" for line in lines] + [f"n1_{generator}" for generator in generators]


def prepare_scos_data(
    data: dict[str, Any],
    scenario_bank: ScenarioBank,
    contingency_top_k: int | None = None,
) -> PreparedSCOSData:
    """Validate legacy DATA and convert it to a consistent internal representation."""
    required = {
        "network",
        "nodal_demand",
        "outages",
        "num_buses",
        "num_branches",
        "branch_capacity",
        "max_number_of_maintenance_tasks",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"DATA is missing required fields: {sorted(missing)}")

    net = data["network"]
    times = [f"step_{index}" for index in range(len(data["nodal_demand"]))]
    buses = [f"bus_{index}" for index in range(int(data["num_buses"]))]
    lines = [f"line_{index}" for index in range(int(data["num_branches"]))]
    generators = [f"gen_{index}" for index in net.gen.index.tolist()]
    outages = list(data["outages"]["names"])
    contingencies = build_contingency_names(data, lines, generators)
    if contingency_top_k is not None:
        contingencies = contingencies[:contingency_top_k]
    valid_assets = set(lines) | set(generators)
    invalid_contingencies = [
        contingency for contingency in contingencies
        if (contingency[3:] if contingency.startswith("n1_") else contingency) not in valid_assets
    ]
    if invalid_contingencies:
        raise ValueError(
            "Contingencies reference unknown assets: " + ", ".join(invalid_contingencies)
        )

    scenario_bank.validate()
    aligned_demands = {
        label: align_demand_columns(frame, buses).set_axis(times, axis=0)
        for label, frame in scenario_bank.demands.items()
    }
    nominal = align_demand_columns(data["nodal_demand"], buses).set_axis(times, axis=0)
    scenario_bank_aligned = ScenarioBank(
        demands=aligned_demands,
        probabilities=dict(scenario_bank.probabilities),
        seed=scenario_bank.seed,
        metadata=dict(scenario_bank.metadata),
    )
    scenario_bank_aligned.validate(reference=nominal)

    internal = net._ppc["internal"]
    incidence = _dense(internal["Cft"]).T.astype(float)
    b_flow = np.real(_dense(internal["Bf"])).astype(float)
    if incidence.shape != (len(buses), len(lines)):
        raise ValueError(
            f"Incidence matrix shape {incidence.shape} does not match "
            f"({len(buses)}, {len(lines)})."
        )
    if b_flow.shape != (len(lines), len(buses)):
        raise ValueError(
            f"Bf shape {b_flow.shape} does not match ({len(lines)}, {len(buses)})."
        )

    tolerance = 1.0e-12
    incidence_terms = {
        bus: tuple(
            (lines[line_index], float(incidence[bus_index, line_index]))
            for line_index in np.flatnonzero(np.abs(incidence[bus_index, :]) > tolerance)
        )
        for bus_index, bus in enumerate(buses)
    }
    b_flow_terms = {
        line: tuple(
            (buses[bus_index], float(b_flow[line_index, bus_index]))
            for bus_index in np.flatnonzero(np.abs(b_flow[line_index, :]) > tolerance)
        )
        for line_index, line in enumerate(lines)
    }

    bus_to_generators = {bus: [] for bus in buses}
    for generator, bus_index in zip(generators, net.gen["bus"].tolist()):
        bus_name = f"bus_{int(bus_index)}"
        if bus_name not in bus_to_generators:
            raise ValueError(f"Generator {generator} references unknown bus {bus_name}.")
        bus_to_generators[bus_name].append(generator)

    p_max = {
        generator: float(value if value > 0 else 200.0)
        for generator, value in zip(generators, net.gen["max_p_mw"].tolist())
    }
    p_min = {
        generator: max(0.0, float(value))
        for generator, value in zip(generators, net.gen["min_p_mw"].tolist())
    }
    flow_limit = {
        line: float(value)
        for line, value in zip(lines, data["branch_capacity"])
    }
    if len(flow_limit) != len(lines):
        raise ValueError("branch_capacity length does not match the number of branches.")

    duration = {
        outage: int(value)
        for outage, value in zip(outages, data["outages"]["expected_duration_steps"])
    }
    priority = {
        outage: float(value)
        for outage, value in zip(outages, data["outages"]["priorities"])
    }
    for outage in outages:
        if outage not in lines and outage not in generators:
            raise ValueError(f"Planned outage {outage} is neither a line nor a generator.")
        if duration[outage] < 1 or duration[outage] > len(times):
            raise ValueError(f"Invalid duration for {outage}: {duration[outage]}.")

    reference_buses = list(data.get("ref_buses") or [buses[0]])
    for bus in reference_buses:
        if bus not in buses:
            raise ValueError(f"Unknown reference bus: {bus}")

    peak_demand = max(float(frame.sum(axis=1).max()) for frame in aligned_demands.values())
    raw_weights = data.get("time_weights")
    if raw_weights is None:
        time_weights = {time: 1.0 for time in times}
    elif isinstance(raw_weights, dict):
        time_weights = {time: float(raw_weights[time]) for time in times}
    else:
        sequence = list(raw_weights)
        if len(sequence) != len(times):
            raise ValueError("time_weights length does not match the scheduling horizon.")
        time_weights = {time: float(value) for time, value in zip(times, sequence)}

    return PreparedSCOSData(
        raw=data,
        times=times,
        buses=buses,
        lines=lines,
        generators=generators,
        outages=outages,
        contingencies=contingencies,
        scenarios=list(aligned_demands),
        scenario_demands=aligned_demands,
        scenario_probabilities=dict(scenario_bank.probabilities),
        incidence=incidence,
        b_flow=b_flow,
        incidence_terms=incidence_terms,
        b_flow_terms=b_flow_terms,
        bus_to_generators=bus_to_generators,
        p_min=p_min,
        p_max=p_max,
        flow_limit=flow_limit,
        duration=duration,
        priority=priority,
        max_tasks=int(data["max_number_of_maintenance_tasks"]),
        reference_buses=reference_buses,
        peak_demand=max(peak_demand, 1.0),
        time_weights=time_weights,
    )


def load_data_from_conf_grid_case(
    conf_path: str | Path | None = None,
    *,
    aggregation_time_override: str | None = None,
) -> dict[str, Any]:
    """
    Load the repository's legacy configuration and build its DATA dictionary.

    This function intentionally imports ``utils.data_preporcess`` lazily because
    that module belongs to the surrounding PROPER repository rather than this
    package.
    """
    try:
        from utils.data_preporcess import (  # type: ignore
            aggregate_hourly_demand,
            aggregate_step_costs_and_durations,
            data_loader,
        )
    except ImportError as exc:
        raise ImportError(
            "The surrounding PROPER repository must provide utils.data_preporcess."
        ) from exc

    repository_root = Path(__file__).resolve().parents[1]
    if conf_path is None:
        conf_path = repository_root / "config/wp3_optim" / "IEEE24_scheduler.json"
    conf_path = Path(conf_path).expanduser().resolve()
    if not conf_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")

    network, hourly_demand, loaded_config = data_loader(str(conf_path))
    config = dict(loaded_config)
    aggregation_step = aggregation_time_override or config["aggregation_time"]
    config["aggregation_time"] = aggregation_step
    scaling_factor = float(config.get("load_scaling_factor", 1.0))
    nodal_demand = aggregate_hourly_demand(
        hourly_demand * scaling_factor,
        aggregation_step=aggregation_step,
    )

    comp_types = config["comp_types"]
    comp_ids = config["comp_ids"]
    outage_names = [f"{asset_type}_{asset_id}" for asset_type, asset_id in zip(comp_types, comp_ids)]
    costs, durations = aggregate_step_costs_and_durations(
        config["cost_per_days"],
        config["expected_duration_days"],
        aggregation_step=aggregation_step,
    )
    outages = {
        "indices": comp_ids,
        "names": outage_names,
        "type": comp_types,
        "expected_duration_steps": durations,
        "cost_per_step": costs,
        "priorities": config["priorities"],
    }

    num_branches = len(network.line) + len(network.trafo)
    num_buses = len(network.bus)
    branch_capacity = [
        175.0 if value <= 1 else 500.0 for value in network.line["max_i_ka"]
    ] + [400.0 for _ in network.trafo.index]

    n_minus1_names = config.get("n_minus1_names")
    if n_minus1_names is None:
        n_contingencies = int(config.get("n_minus1_top_k", min(10, num_branches)))
        n_minus1_names = [f"n1_line_{index}" for index in range(n_contingencies)]

    data = {
        "max_number_of_maintenance_tasks": int(config["max_number_of_maintenance_tasks"]),
        "nodal_demand": nodal_demand,
        "outages": outages,
        "config": config,
        "network": network,
        "num_buses": num_buses,
        "num_branches": num_branches,
        "ref_buses": config.get("ref_buses", ["bus_12"]),
        "branch_capacity": branch_capacity,
        "n_minus1_names": n_minus1_names,
        "VoLL": float(config.get("VoLL", 1.0e6)),
    }
    LOGGER.info("Loaded %s with %d time steps.", config.get("case_name", conf_path.stem), len(nodal_demand))
    return data
