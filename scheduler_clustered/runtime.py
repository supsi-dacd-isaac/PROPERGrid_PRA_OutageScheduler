"""Runtime helpers shared by the command-line entry points."""
from __future__ import annotations

from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar
import json

from .data_adapter import augment_for_decomposition, validate_decomposition_data

T = TypeVar("T")


def prepare_data_compat(config_path: str | Path) -> dict[str, Any]:
    """Load PROPER scheduler data across the historical package layouts."""
    errors: list[Exception] = []
    for module_name in ("scheduler.dataprocess", "optimizers.dataprocess"):
        try:
            module = import_module(module_name)
            return module.prepare_data(conf_path=str(config_path))
        except (ImportError, AttributeError) as exc:
            errors.append(exc)
    raise ImportError(
        "Could not import prepare_data from scheduler.dataprocess or "
        "optimizers.dataprocess."
    ) from errors[-1]


def configured_decomposition_data(
    config_path: str | Path,
    *,
    allow_outage_deferral: bool = True,
) -> dict[str, Any]:
    raw_data = prepare_data_compat(config_path)
    data = augment_for_decomposition(
        raw_data,
        include_all_line_contingencies=True,
        include_generator_contingencies=False,
        prefer_ppc_branch_limits=True,
    )
    data.setdefault("master_output_flag", 0)
    data["allow_outage_deferral"] = bool(allow_outage_deferral)
    data.setdefault("security_proxy_weight", 1.0)
    data.setdefault("outage_coverage_reward", 1.0)
    data.setdefault("outage_priority_timing_reward", 0.1)
    data.setdefault("default_defer_penalty", 0.0)
    data.setdefault("corrective_redispatch_fraction", 0.20)
    data.setdefault("redispatch_cost", 10.0)
    data.setdefault("cluster_max_dns_weight", 1e6)
    data.setdefault("cluster_total_dns_weight", 1e3)
    data.setdefault("cluster_generation_cost_weight", 1e-3)
    data.setdefault("cluster_oracle_threads", 1)
    data.setdefault("oracle_isolated_env", True)
    return data


def dataclass_config_from_json(
    config_type: type[T],
    path: str | Path | None,
    section: str,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> T:
    """Instantiate a config dataclass from one optional JSON section."""
    values: dict[str, Any] = dict(defaults or {})
    if path is not None:
        source = Path(path)
        if source.exists():
            document = json.loads(source.read_text(encoding="utf-8"))
            raw = document.get(section, document)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Configuration section {section!r} must be a mapping."
                )
            values.update(raw)
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {config_type.__name__} settings: {unknown}."
        )
    return config_type(**values)


def apply_data_overrides_from_json(
    data: dict[str, Any],
    path: str | Path | None,
) -> dict[str, Any]:
    """Apply optional master/oracle data settings from a clustered JSON file.

    The accepted section names are ``data_overrides`` and the legacy
    ``optional_data_fields``. Nested mappings are merged one level deep so a
    partial priority or window mapping does not erase values prepared by the
    repository data loader.
    """
    if path is None:
        return data
    source = Path(path)
    if not source.exists():
        return data
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("The clustered configuration root must be a mapping.")
    overrides: dict[str, Any] = {}
    for section in ("optional_data_fields", "data_overrides"):
        raw = document.get(section)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"Configuration section {section!r} must be a mapping.")
        overrides.update(raw)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(data.get(key), Mapping):
            merged = dict(data[key])
            merged.update(value)
            data[key] = merged
        else:
            data[key] = value
    benchmark_slice = document.get("benchmark_slice")
    if benchmark_slice is not None:
        if not isinstance(benchmark_slice, Mapping):
            raise ValueError("Configuration section 'benchmark_slice' must be a mapping.")
        apply_benchmark_slice(data, benchmark_slice)
    validate_decomposition_data(data)
    return data


def apply_benchmark_slice(data: dict[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Create a controlled horizon/outage view for scalability experiments only.

    The grid topology is unchanged. Horizon rows are taken in order and repeated
    cyclically only when the requested horizon is longer than the source data.
    Outage windows are normalised by default so horizon scaling measures problem
    size rather than calendar-window infeasibility. The applied transformation is
    retained in ``data['benchmark_slice']`` for reporting.
    """
    source_times = list(data["T"])
    if not source_times:
        raise ValueError("Cannot benchmark an empty source horizon.")
    horizon = int(settings.get("horizon_steps", len(source_times)))
    if horizon <= 0:
        raise ValueError("benchmark_slice.horizon_steps must be positive.")

    source_positions = [index % len(source_times) for index in range(horizon)]
    if horizon <= len(source_times):
        times = source_times[:horizon]
    else:
        times = [f"benchmark_t{index + 1:04d}" for index in range(horizon)]
    demand = data["nodal_demand"].iloc[source_positions].copy()
    demand.index = times
    data["T"], data["nodal_demand"] = times, demand

    max_tasks = data.get("max_tasks")
    if isinstance(max_tasks, Mapping):
        data["max_tasks"] = {
            target: max_tasks[source_times[position]]
            for target, position in zip(times, source_positions)
        }
    capacities = data.get("resource_capacities", {})
    data["resource_capacities"] = {
        resource: (
            {
                target: values[source_times[position]]
                for target, position in zip(times, source_positions)
            }
            if isinstance(values, Mapping)
            else values
        )
        for resource, values in capacities.items()
    }

    available = list(data["names"]["outages"])
    explicit = settings.get("outages")
    if explicit is not None:
        if isinstance(explicit, (str, bytes)) or not isinstance(explicit, Sequence):
            raise ValueError("benchmark_slice.outages must be a sequence of outage names.")
        selected = [str(value) for value in explicit]
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown benchmark outages: {unknown}")
    else:
        count = int(settings.get("outage_count", len(available)))
        if count <= 0 or count > len(available):
            raise ValueError(f"benchmark_slice.outage_count must be in [1, {len(available)}].")
        selected = available[:count]
    selected_set = set(selected)
    data["names"] = dict(data["names"])
    data["names"]["outages"] = selected

    def filter_mapping(name: str) -> None:
        if isinstance(data.get(name), Mapping):
            data[name] = {key: value for key, value in data[name].items() if key in selected_set}

    for name in ("durations", "priority", "defer_penalty", "outage_start_windows"):
        filter_mapping(name)
    too_long = {name: value for name, value in data["durations"].items() if int(round(float(value))) > horizon}
    if too_long:
        raise ValueError(f"Outage durations exceed the benchmark horizon {horizon}: {too_long}")

    usage = data.get("resource_usage", {})
    data["resource_usage"] = {
        resource: {outage: value for outage, value in values.items() if outage in selected_set}
        for resource, values in usage.items()
    }
    for name in ("mutually_exclusive_outages", "incompatible_outage_groups"):
        groups = [[outage for outage in group if outage in selected_set] for group in data.get(name, [])]
        data[name] = [group for group in groups if len(group) >= 2]
    data["precedence"] = [
        relation for relation in data.get("precedence", [])
        if len(relation) >= 2 and relation[0] in selected_set and relation[1] in selected_set
    ]
    if bool(settings.get("normalise_windows", True)):
        data["outage_start_windows"] = {
            outage: [0, horizon - int(round(float(data["durations"][outage])))]
            for outage in selected
        }

    data["benchmark_slice"] = {
        "source_horizon_steps": len(source_times),
        "horizon_steps": horizon,
        "source_outage_count": len(available),
        "outage_count": len(selected),
        "outages": selected,
        "normalised_windows": bool(settings.get("normalise_windows", True)),
    }
    return data
