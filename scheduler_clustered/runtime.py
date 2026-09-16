"""Runtime helpers shared by the command-line entry points."""
from __future__ import annotations

from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, TypeVar
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
    validate_decomposition_data(data)
    return data
