"""Severity metrics for contingency analysis."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def branch_overload_excess(branch_results: pd.DataFrame, *, threshold: float = 100.0) -> float:
    if branch_results.empty or "loading_percent" not in branch_results:
        return 0.0
    loading = branch_results["loading_percent"].astype(float).to_numpy()
    return float(np.nansum(np.maximum(loading - threshold, 0.0)))


def max_branch_loading(branch_results: pd.DataFrame) -> float:
    if branch_results.empty or "loading_percent" not in branch_results:
        return float("nan")
    return float(np.nanmax(branch_results["loading_percent"].astype(float).to_numpy()))


def number_of_overloads(branch_results: pd.DataFrame, *, threshold: float = 100.0) -> int:
    if branch_results.empty or "loading_percent" not in branch_results:
        return 0
    return int(np.nansum(branch_results["loading_percent"].astype(float).to_numpy() > threshold))


def voltage_violation_excess(
    bus_results: pd.DataFrame,
    *,
    low_vm_pu: float = 0.92,
    high_vm_pu: float = 1.08,
) -> float:
    if bus_results.empty or "vm_pu" not in bus_results:
        return 0.0
    v = bus_results["vm_pu"].astype(float).to_numpy()
    low = np.maximum(low_vm_pu - v, 0.0)
    high = np.maximum(v - high_vm_pu, 0.0)
    return float(np.nansum(low + high))


def compute_severity(
    branch_results: pd.DataFrame,
    bus_results: pd.DataFrame | None = None,
    *,
    metric: str = "overload_excess",
    loading_threshold: float = 100.0,
    low_vm_pu: float = 0.92,
    high_vm_pu: float = 1.08,
    dns_mw: float | None = None,
    duration_h: float = 1.0,
) -> float:
    """Compute a scalar severity metric.

    Supported metrics: ``overload_excess``, ``max_loading``, ``n_overloads``,
    ``voltage_excess``, ``ens_mwh`` and ``combined_overload_voltage``.
    """
    metric = metric.lower()
    bus_results = bus_results if bus_results is not None else pd.DataFrame()
    if metric == "overload_excess":
        return branch_overload_excess(branch_results, threshold=loading_threshold)
    if metric == "max_loading":
        return max_branch_loading(branch_results)
    if metric == "n_overloads":
        return float(number_of_overloads(branch_results, threshold=loading_threshold))
    if metric == "voltage_excess":
        return voltage_violation_excess(bus_results, low_vm_pu=low_vm_pu, high_vm_pu=high_vm_pu)
    if metric == "ens_mwh":
        return float(0.0 if dns_mw is None else dns_mw * duration_h)
    if metric == "combined_overload_voltage":
        return branch_overload_excess(branch_results, threshold=loading_threshold) + 100.0 * voltage_violation_excess(
            bus_results, low_vm_pu=low_vm_pu, high_vm_pu=high_vm_pu
        )
    raise ValueError(f"Unsupported severity metric {metric!r}")


def summarize_branch_state(
    branch_results: pd.DataFrame,
    *,
    loading_threshold: float = 100.0,
) -> dict[str, Any]:
    return {
        "max_loading_pct": max_branch_loading(branch_results),
        "n_overloads": number_of_overloads(branch_results, threshold=loading_threshold),
        "overload_excess": branch_overload_excess(branch_results, threshold=loading_threshold),
    }
