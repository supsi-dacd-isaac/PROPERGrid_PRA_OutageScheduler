"""Validation, diagnostics, and augmentation for the decomposed scheduler."""
from __future__ import annotations

from typing import Iterable, Mapping
import copy
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# PYPOWER/MATPOWER branch column index for RATE_A.
_RATE_A = 5


def _line_contingency_name(line: str) -> str:
    return f"n1_{line}"


def _generator_contingency_name(generator: str) -> str:
    return f"n1_{generator}"


def _extract_ppc_branch_limits(data: Mapping, line_names: list[str]) -> dict[str, float]:
    """Extract positive RATE_A values from pandapower's initialized PPC."""
    network = data.get("network")
    if network is None or not hasattr(network, "_ppc") or network._ppc is None:
        return {}
    branch = np.asarray(network._ppc.get("branch", []), dtype=float)
    if branch.ndim != 2 or branch.shape[0] != len(line_names) or branch.shape[1] <= _RATE_A:
        return {}
    rate_a = branch[:, _RATE_A]
    return {
        line: float(limit)
        for line, limit in zip(line_names, rate_a)
        if np.isfinite(limit) and limit > 0.0
    }


def _extract_dc_metadata(result: dict) -> None:
    """Attach base-MVA and phase-shift injections when available."""
    network = result.get("network")
    if network is None or not hasattr(network, "_ppc") or network._ppc is None:
        return
    ppc = network._ppc
    internal = ppc.get("internal", {})
    result["base_mva"] = float(ppc.get("baseMVA", result.get("base_mva", 1.0)))
    n_lines = len(result["names"]["lines"])
    pfinj = np.asarray(internal.get("Pfinj", np.zeros(n_lines)), dtype=float).reshape(-1)
    if pfinj.size == n_lines:
        result["Pfinj"] = pfinj


def augment_for_decomposition(
    data: dict,
    include_all_line_contingencies: bool = True,
    include_generator_contingencies: bool = False,
    excluded_contingencies: Iterable[str] = (),
    prefer_ppc_branch_limits: bool = False,
) -> dict:
    """Return data with complete structured contingency and matrix metadata.

    Parameters
    ----------
    prefer_ppc_branch_limits:
        Replace existing branch limits with positive MATPOWER/PYPOWER ``RATE_A``
        values when the pandapower PPC is available. Existing limits are kept
        for branches whose RATE_A is zero or unavailable.
    """
    result = copy.copy(data)
    result["names"] = copy.deepcopy(data["names"])
    result["f_lim"] = dict(data["f_lim"])
    names = result["names"]

    contingencies: list[str] = []
    if include_all_line_contingencies:
        contingencies.extend(_line_contingency_name(line) for line in names["lines"])
    else:
        contingencies.extend(names.get("contingencies", []))
    if include_generator_contingencies:
        contingencies.extend(_generator_contingency_name(gen) for gen in names["generators"])

    excluded = set(excluded_contingencies)
    contingencies = [c for c in dict.fromkeys(contingencies) if c not in excluded]
    names["contingencies"] = contingencies

    metadata: dict[str, dict[str, str]] = {}
    for contingency in contingencies:
        if contingency.startswith("n1_line_"):
            metadata[contingency] = {
                "type": "line",
                "element": contingency[len("n1_"):],
            }
        elif contingency.startswith("n1_gen_"):
            metadata[contingency] = {
                "type": "generator",
                "element": contingency[len("n1_"):],
            }
        else:
            existing = data.get("contingency_metadata", {}).get(contingency)
            if existing is None:
                raise ValueError(f"Cannot infer metadata for contingency {contingency!r}.")
            metadata[contingency] = dict(existing)
    result["contingency_metadata"] = metadata

    s = np.asarray(result["S"], dtype=float)
    bf = np.asarray(result["B_mat"], dtype=float)
    expected_s = (len(names["buses"]), len(names["lines"]))
    expected_bf = (len(names["lines"]), len(names["buses"]))
    if s.shape != expected_s:
        raise ValueError(f"S shape {s.shape}; expected {expected_s}.")
    if bf.shape != expected_bf:
        raise ValueError(f"B_mat shape {bf.shape}; expected {expected_bf}.")
    result["S"] = s
    result["B_mat"] = bf
    result["C_branch_bus"] = np.asarray(result.get("C_branch_bus", s.T), dtype=float)

    _extract_dc_metadata(result)

    if prefer_ppc_branch_limits:
        ppc_limits = _extract_ppc_branch_limits(result, list(names["lines"]))
        replaced = 0
        for line, limit in ppc_limits.items():
            current = float(result["f_lim"].get(line, 0.0))
            if not math.isclose(current, limit, rel_tol=1e-10, abs_tol=1e-10):
                replaced += 1
            result["f_lim"][line] = limit
        result["line_limit_source"] = "PPC_RATE_A_with_fallback"
        logger.info("Applied PPC RATE_A limits to %d of %d branches.", len(ppc_limits), len(names["lines"]))
        if replaced:
            logger.warning("Replaced %d existing branch limits using PPC RATE_A.", replaced)

    validate_decomposition_data(result)
    return result


def power_data_diagnostics(data: Mapping) -> dict[str, object]:
    """Return diagnostics for unit, capacity, and model-coverage problems."""
    names = data["names"]
    demand = data["nodal_demand"].to_numpy(dtype=float)
    pmax = np.asarray([float(data["p_max"][g]) for g in names["generators"]], dtype=float)
    limits = np.asarray([float(data["f_lim"][line]) for line in names["lines"]], dtype=float)
    peak = float(np.max(np.sum(demand, axis=1))) if demand.size else 0.0
    installed = float(np.sum(pmax))
    unique_limits = sorted({round(float(value), 8) for value in limits})

    warnings: list[str] = []
    if installed + 1e-9 < peak:
        warnings.append(
            f"Peak demand ({peak:.3f}) exceeds represented generator capacity ({installed:.3f})."
        )
    if len(unique_limits) <= 4 and len(limits) >= 10:
        warnings.append(
            "Branch limits contain only a few repeated values; verify that they are physical MW/MVA ratings, "
            "not heuristic placeholders."
        )
    if np.any(limits <= 0):
        warnings.append("One or more branch limits are non-positive.")

    network = data.get("network")
    ext_grid_count = 0
    network_gen_count = None
    if network is not None:
        try:
            ext_grid_count = len(network.ext_grid)
            network_gen_count = len(network.gen)
        except Exception:
            pass
    if ext_grid_count > 0 and all(not str(g).startswith("ext_grid_") for g in names["generators"]):
        warnings.append(
            f"The pandapower network contains {ext_grid_count} ext_grid element(s), but none are represented "
            "in the scheduler generator set. Verify slack/import capacity explicitly."
        )

    return {
        "buses": len(names["buses"]),
        "branches": len(names["lines"]),
        "generators_represented": len(names["generators"]),
        "network_gen_count": network_gen_count,
        "network_ext_grid_count": ext_grid_count,
        "peak_total_demand": peak,
        "represented_pmax": installed,
        "capacity_margin": installed - peak,
        "line_limit_min": float(np.min(limits)) if limits.size else None,
        "line_limit_max": float(np.max(limits)) if limits.size else None,
        "line_limit_unique_count": len(unique_limits),
        "line_limit_unique_values": unique_limits[:20],
        "base_mva": float(data.get("base_mva", 1.0)),
        "warnings": warnings,
    }


def validate_decomposition_data(data: Mapping) -> None:
    required = [
        "T", "names", "durations", "max_tasks", "nodal_demand",
        "p_min", "p_max", "f_lim", "g2bus", "S", "B_mat",
        "contingency_metadata",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing decomposition data fields: {missing}")

    names = data["names"]
    for group in ("outages", "lines", "buses", "generators", "contingencies"):
        if group not in names:
            raise KeyError(f"data['names'] lacks {group!r}.")
        if len(names[group]) != len(set(names[group])):
            raise ValueError(f"Duplicate names in group {group!r}.")

    unknown_outages = set(names["outages"]) - (set(names["lines"]) | set(names["generators"]))
    if unknown_outages:
        raise ValueError(
            "Every planned outage must currently identify a line or generator. "
            f"Unknown assets: {sorted(unknown_outages)}"
        )
    if set(data["durations"]) != set(names["outages"]):
        missing_duration = set(names["outages"]) - set(data["durations"])
        if missing_duration:
            raise ValueError(f"Missing outage durations: {sorted(missing_duration)}")

    for contingency in names["contingencies"]:
        if contingency not in data["contingency_metadata"]:
            raise ValueError(f"Missing metadata for contingency {contingency}.")
        metadata = data["contingency_metadata"][contingency]
        if metadata["type"] == "line" and metadata["element"] not in names["lines"]:
            raise ValueError(
                f"Contingency {contingency} references unknown line {metadata['element']}."
            )
        if metadata["type"] == "generator" and metadata["element"] not in names["generators"]:
            raise ValueError(
                f"Contingency {contingency} references unknown generator {metadata['element']}."
            )
