"""Deterministic post-contingency analysis utilities for PRA/PSA.

This module replaces the old experimental contingency_analysis.py with a
self-contained implementation that works with the newer contingency objects
from ``pra_psa.contingencies`` and with legacy dictionary contingencies.

Main public entry point
-----------------------
    run_contingency_analysis(...)

The function returns a ContingencyAnalysisResult with three dataframes:
    - summary: one row per operating point and contingency
    - components: line/trafo post-contingency loading schedule_results
    - bus: bus-voltage schedule_results

It also keeps backward-compatible helpers:
    - get_OPF_gen
    - get_PF_loading
    - apply_load
    - apply_nk_contingency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import copy
import itertools

import numpy as np
import pandas as pd
import pandapower as pp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ContingencyAnalysisResult:
    """Container for post-contingency analysis schedule_results."""

    summary: pd.DataFrame
    components: pd.DataFrame
    bus: pd.DataFrame

    def to_csv(self, output_dir: str | Path) -> None:
        """Write all result tables to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(output_dir / "contingency_summary.csv", index=False)
        self.components.to_csv(output_dir / "contingency_component_results.csv", index=False)
        self.bus.to_csv(output_dir / "contingency_bus_results.csv", index=False)


# ---------------------------------------------------------------------------
# Backward-compatible basic helpers
# ---------------------------------------------------------------------------


def get_OPF_gen(net, opf_solver=pp.rundcopp):
    """Run OPF and return generator dispatch and the solved network."""
    opf_solver(net)
    p = net.res_gen.p_mw.copy() if hasattr(net, "res_gen") and "p_mw" in net.res_gen else pd.Series(dtype=float)
    q = net.res_gen.q_mvar.copy() if hasattr(net, "res_gen") and "q_mvar" in net.res_gen else pd.Series(dtype=float)
    return p, q, net



def get_PF_loading(net, pf_solver=pp.rundcpp):
    """Run power flow and return line loading percentages and the solved network."""
    pf_solver(net)
    return net.res_line.loading_percent.copy(), net



def apply_load(network, p_load):
    """Apply load vector to pandapower loads in their current order.

    This legacy helper assumes ``len(p_load) == len(network.load)``.
    For bus-level time-series use ``run_contingency_analysis`` directly;
    it can map bus-level loads to pandapower load elements.
    """
    if len(p_load) != len(network.load):
        raise ValueError(
            f"apply_load expects one value per pandapower load: "
            f"got {len(p_load)} values for {len(network.load)} loads."
        )
    for load_idx, p_mw in zip(network.load.index, p_load):
        network.load.at[load_idx, "p_mw"] = float(p_mw)
    return network



def apply_reference_dispatch(network, reference_p_mw=None, reference_q_mvar=None):
    """Fix generator active/reactive dispatch to reference values."""
    if reference_p_mw is not None:
        for gen_idx, p_mw in zip(network.gen.index, reference_p_mw):
            p_mw = float(p_mw)
            network.gen.at[gen_idx, "p_mw"] = p_mw
            network.gen.at[gen_idx, "max_p_mw"] = p_mw
            network.gen.at[gen_idx, "min_p_mw"] = p_mw

    if reference_q_mvar is not None:
        for gen_idx, q_mvar in zip(network.gen.index, reference_q_mvar):
            q_mvar = float(q_mvar)
            network.gen.at[gen_idx, "max_q_mvar"] = q_mvar
            network.gen.at[gen_idx, "min_q_mvar"] = q_mvar

    if hasattr(network, "res_ext_grid") and not network.res_ext_grid.empty:
        if "p_mw" in network.res_ext_grid.columns:
            network.ext_grid["min_p_mw"] = network.res_ext_grid["p_mw"].values
            network.ext_grid["max_p_mw"] = network.res_ext_grid["p_mw"].values

    return network


# ---------------------------------------------------------------------------
# Contingency normalization and application
# ---------------------------------------------------------------------------


def _normalize_element_type(element_type: str) -> str:
    e = str(element_type).strip().lower()
    aliases = {
        "branch": "line",
        "lines": "line",
        "transformer": "trafo",
        "transformers": "trafo",
        "generator": "gen",
        "generators": "gen",
    }
    return aliases.get(e, e)



def _outage_to_tuple(outage: Any) -> tuple[str, int]:
    """Convert an outage object/dict/tuple to (element_type, element_index)."""
    if isinstance(outage, dict):
        element_type = outage.get("element_type", outage.get("type"))
        element_index = outage.get("element_index", outage.get("index"))
    elif isinstance(outage, (tuple, list)) and len(outage) >= 2:
        element_type, element_index = outage[0], outage[1]
    else:
        element_type = getattr(outage, "element_type", getattr(outage, "type", None))
        element_index = getattr(outage, "element_index", getattr(outage, "index", None))

    if element_type is None or element_index is None:
        raise ValueError(f"Cannot parse outage specification: {outage!r}")

    return _normalize_element_type(element_type), int(element_index)



def _contingency_outages(contingency: Any) -> list[tuple[str, int]]:
    """Return normalized outages for new Contingency objects or legacy inputs."""
    if contingency is None:
        return []

    if isinstance(contingency, dict) and "outages" not in contingency:
        return [_outage_to_tuple(contingency)]

    if isinstance(contingency, (tuple, list)):
        # A single outage can be represented as ("line", 3).
        if len(contingency) >= 2 and isinstance(contingency[0], str):
            return [_outage_to_tuple(contingency)]
        return [_outage_to_tuple(o) for o in contingency]

    outages = getattr(contingency, "outages", None)
    if outages is None and isinstance(contingency, dict):
        outages = contingency.get("outages")

    if outages is None:
        return [_outage_to_tuple(contingency)]

    return [_outage_to_tuple(o) for o in outages]



def _contingency_id(contingency: Any, fallback_index: int | None = None) -> str:
    if contingency is None:
        return "base_case"

    for attr in ("contingency_id", "id", "name"):
        value = getattr(contingency, attr, None)
        if value is not None:
            return str(value)

    if isinstance(contingency, dict):
        for key in ("contingency_id", "id", "name"):
            if key in contingency:
                return str(contingency[key])

    outages = _contingency_outages(contingency)
    if outages:
        return "+".join(f"{e}_{i}" for e, i in outages)

    return f"contingency_{fallback_index}" if fallback_index is not None else "contingency"



def apply_nk_contingency(network, failure_event):
    """Return a deep-copied network with the specified outage(s) applied.

    Supports both old dictionary syntax and the new Contingency/Outage classes.
    """
    contingency_network = copy.deepcopy(network)

    for element_type, element_index in _contingency_outages(failure_event):
        if element_type == "line":
            if element_index not in contingency_network.line.index:
                raise KeyError(f"Line index {element_index} not found in net.line")
            contingency_network.line.at[element_index, "in_service"] = False
        elif element_type == "trafo":
            if element_index not in contingency_network.trafo.index:
                raise KeyError(f"Trafo index {element_index} not found in net.trafo")
            contingency_network.trafo.at[element_index, "in_service"] = False
        elif element_type == "gen":
            if element_index not in contingency_network.gen.index:
                raise KeyError(f"Generator index {element_index} not found in net.gen")
            contingency_network.gen.at[element_index, "in_service"] = False
        elif element_type == "ext_grid":
            if element_index not in contingency_network.ext_grid.index:
                raise KeyError(f"External-grid index {element_index} not found in net.ext_grid")
            contingency_network.ext_grid.at[element_index, "in_service"] = False
        else:
            raise ValueError(
                f"Invalid element type {element_type!r}. "
                "Use one of {'line', 'trafo', 'gen', 'ext_grid'}."
            )

    return contingency_network


# ---------------------------------------------------------------------------
# Operating point handling
# ---------------------------------------------------------------------------


def _series_value_by_bus(row: pd.Series, bus_idx: Any, bus_name: Any | None = None) -> float | None:
    """Retrieve a bus-level value from a Series using several common conventions."""
    candidates = [bus_idx, str(bus_idx)]
    try:
        candidates.append(int(bus_idx))
    except Exception:
        pass

    if bus_name is not None and not pd.isna(bus_name):
        candidates.extend([bus_name, str(bus_name)])

    for col in candidates:
        if col in row.index:
            value = row.loc[col]
            if pd.notna(value):
                return float(value)

    return None



def _apply_operating_point_to_loads(net, operating_point: pd.Series | np.ndarray | Sequence[float] | None) -> None:
    """Map an operating point to pandapower loads.

    Supported formats
    -----------------
    1. length == len(net.load): one value per pandapower load.
    2. length == len(net.bus): one bus-level demand value per bus.
       Values are distributed to load elements connected to the bus.
    3. pandas Series indexed by bus index, bus name, or load index.
    """
    if operating_point is None or len(net.load) == 0:
        return

    if isinstance(operating_point, pd.Series):
        row = operating_point
        values = row.to_numpy(dtype=float)
    else:
        row = None
        values = np.asarray(operating_point, dtype=float)

    # Case 1: one value per load.
    if len(values) == len(net.load):
        net.load.loc[:, "p_mw"] = values
        return

    # Case 2: bus-level vector in net.bus order.
    if len(values) == len(net.bus):
        bus_value = dict(zip(net.bus.index, values))
    else:
        bus_value = {}
        if row is not None:
            for bus_idx in net.bus.index:
                bus_name = net.bus.at[bus_idx, "name"] if "name" in net.bus.columns else None
                value = _series_value_by_bus(row, bus_idx, bus_name)
                if value is not None:
                    bus_value[bus_idx] = value

    if not bus_value:
        raise ValueError(
            "Could not map operating point to loads. Provide either one value per load, "
            "one value per bus, or a Series indexed by bus index/name."
        )

    # Distribute bus-level demand to load rows connected to each bus.
    for bus_idx, total_p_mw in bus_value.items():
        load_idx = net.load.index[net.load["bus"] == bus_idx].tolist()
        if not load_idx:
            continue

        base = pd.to_numeric(net.load.loc[load_idx, "p_mw"], errors="coerce").fillna(0.0).to_numpy(float)
        base_abs_sum = float(np.sum(np.abs(base)))

        if base_abs_sum > 1e-9:
            weights = np.abs(base) / base_abs_sum
        else:
            weights = np.full(len(load_idx), 1.0 / len(load_idx))

        net.load.loc[load_idx, "p_mw"] = weights * float(total_p_mw)



def _iter_operating_points(operating_points) -> Iterable[tuple[Any, Any]]:
    """Yield (operating_point_id, operating_point) pairs."""
    if operating_points is None:
        yield 0, None
        return

    if isinstance(operating_points, pd.DataFrame):
        for idx, row in operating_points.iterrows():
            yield idx, row
        return

    if isinstance(operating_points, pd.Series):
        yield operating_points.name if operating_points.name is not None else 0, operating_points
        return

    arr = np.asarray(operating_points)
    if arr.ndim == 1:
        yield 0, arr
    elif arr.ndim == 2:
        for i, row in enumerate(arr):
            yield i, row
    else:
        raise ValueError("operating_points must be None, Series, DataFrame, or 1D/2D array-like.")


# ---------------------------------------------------------------------------
# Power-flow and severity utilities
# ---------------------------------------------------------------------------


def _run_power_flow(net, pf_mode: str) -> tuple[bool, str | None]:
    mode = str(pf_mode).lower()
    try:
        if mode in {"dc", "dcpf", "rundcpp"}:
            pp.rundcpp(net)
        elif mode in {"ac", "acpf", "runpp"}:
            pp.runpp(net, calculate_voltage_angles=True, init="auto")
        else:
            raise ValueError(f"Unsupported pf_mode={pf_mode!r}; use 'dc' or 'ac'.")
        return bool(getattr(net, "converged", True)), None
    except Exception as exc:
        return False, str(exc)



def _safe_loading_table(net, element: str) -> pd.DataFrame:
    res = getattr(net, f"res_{element}", pd.DataFrame())
    elm = getattr(net, element, pd.DataFrame())

    if res is None or res.empty or elm is None or elm.empty:
        return pd.DataFrame(columns=["element_index", "loading_percent", "p_from_mw"])

    out = pd.DataFrame(index=elm.index)
    out["element_index"] = elm.index
    if "loading_percent" in res.columns:
        out["loading_percent"] = pd.to_numeric(res["loading_percent"], errors="coerce")
    else:
        out["loading_percent"] = np.nan

    if "p_from_mw" in res.columns:
        out["p_from_mw"] = pd.to_numeric(res["p_from_mw"], errors="coerce")
    elif "p_hv_mw" in res.columns:
        out["p_from_mw"] = pd.to_numeric(res["p_hv_mw"], errors="coerce")
    else:
        out["p_from_mw"] = np.nan

    if "in_service" in elm.columns:
        out["in_service"] = elm["in_service"].astype(bool)
    else:
        out["in_service"] = True

    return out.reset_index(drop=True)



def _voltage_results(net, v_min_pu: float, v_max_pu: float) -> pd.DataFrame:
    if not hasattr(net, "res_bus") or net.res_bus.empty:
        return pd.DataFrame(columns=["bus", "vm_pu", "va_degree", "voltage_violation_pu"])

    out = pd.DataFrame({"bus": net.bus.index})
    if "vm_pu" in net.res_bus.columns:
        out["vm_pu"] = pd.to_numeric(net.res_bus["vm_pu"], errors="coerce").to_numpy(float)
    else:
        out["vm_pu"] = np.nan

    if "va_degree" in net.res_bus.columns:
        out["va_degree"] = pd.to_numeric(net.res_bus["va_degree"], errors="coerce").to_numpy(float)
    else:
        out["va_degree"] = np.nan

    vm = out["vm_pu"].to_numpy(float)
    low = np.maximum(v_min_pu - vm, 0.0)
    high = np.maximum(vm - v_max_pu, 0.0)
    out["voltage_violation_pu"] = np.nan_to_num(low + high, nan=0.0)
    return out



def _build_component_rows(
    net,
    *,
    operating_point_id: Any,
    contingency_id: str,
    loading_threshold: float,
) -> pd.DataFrame:
    rows = []
    for element in ("line", "trafo"):
        table = _safe_loading_table(net, element)
        if table.empty:
            continue
        for _, row in table.iterrows():
            loading = row.get("loading_percent", np.nan)
            overload = max(float(loading) - loading_threshold, 0.0) if np.isfinite(loading) else np.nan
            rows.append(
                {
                    "operating_point_id": operating_point_id,
                    "contingency_id": contingency_id,
                    "element_type": element,
                    "element_index": int(row["element_index"]),
                    "in_service": bool(row.get("in_service", True)),
                    "loading_percent": loading,
                    "p_from_mw": row.get("p_from_mw", np.nan),
                    "overload_percent": overload,
                    "violated": bool(np.isfinite(overload) and overload > 0.0),
                }
            )
    return pd.DataFrame(rows)



def _compute_summary(
    component_rows: pd.DataFrame,
    bus_rows: pd.DataFrame,
    *,
    operating_point_id: Any,
    contingency_id: str,
    n_outages: int,
    converged: bool,
    error: str | None,
    loading_threshold: float,
    severity_metric: str,
) -> dict[str, Any]:
    if component_rows.empty:
        max_loading = np.nan
        n_overloads = 0
        overload_sum = np.nan
    else:
        loadings = pd.to_numeric(component_rows["loading_percent"], errors="coerce")
        overloads = pd.to_numeric(component_rows["overload_percent"], errors="coerce")
        max_loading = float(loadings.max()) if loadings.notna().any() else np.nan
        n_overloads = int((overloads > 0).sum())
        overload_sum = float(overloads.clip(lower=0.0).sum()) if overloads.notna().any() else np.nan

    if bus_rows.empty or "voltage_violation_pu" not in bus_rows.columns:
        max_voltage_violation = np.nan
        voltage_violation_sum = np.nan
    else:
        vviol = pd.to_numeric(bus_rows["voltage_violation_pu"], errors="coerce")
        max_voltage_violation = float(vviol.max()) if vviol.notna().any() else np.nan
        voltage_violation_sum = float(vviol.sum()) if vviol.notna().any() else np.nan

    metric = severity_metric.lower()
    if metric in {"overload_excess", "overload_sum", "loading_excess"}:
        severity = overload_sum
    elif metric in {"n_overloads", "violation_count"}:
        severity = float(n_overloads)
    elif metric in {"voltage_violation", "voltage_excess"}:
        severity = voltage_violation_sum
    elif metric in {"combined", "overload_plus_voltage"}:
        severity = (0.0 if not np.isfinite(overload_sum) else overload_sum) + 100.0 * (
            0.0 if not np.isfinite(voltage_violation_sum) else voltage_violation_sum
        )
    else:
        raise ValueError(
            f"Unsupported severity_metric={severity_metric!r}. Use one of "
            "'overload_excess', 'n_overloads', 'voltage_violation', 'combined'."
        )

    return {
        "operating_point_id": operating_point_id,
        "contingency_id": contingency_id,
        "n_outages": n_outages,
        "converged": bool(converged),
        "error": error,
        "loading_threshold": float(loading_threshold),
        "max_loading_percent": max_loading,
        "n_overloads": n_overloads,
        "overload_excess_sum": overload_sum,
        "max_voltage_violation_pu": max_voltage_violation,
        "voltage_violation_sum_pu": voltage_violation_sum,
        "severity_metric": severity_metric,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def run_contingency_analysis(
    net,
    *,
    operating_points=None,
    contingencies: Sequence[Any] | None = None,
    pf_mode: str = "dc",
    include_base_case: bool = True,
    loading_threshold: float = 100.0,
    severity_metric: str = "overload_excess",
    v_min_pu: float = 0.95,
    v_max_pu: float = 1.05,
    progress: bool = False,
) -> ContingencyAnalysisResult:
    """Run deterministic post-contingency analysis.

    Parameters
    ----------
    net:
        pandapower network.
    operating_points:
        None, one operating point, or a collection of operating points.
        A DataFrame is interpreted as one operating point per row.
    contingencies:
        Iterable of Contingency objects, Outage objects, legacy dictionaries,
        or tuples. If None, only the base case is run when include_base_case=True.
    pf_mode:
        "dc" for pp.rundcpp or "ac" for pp.runpp.
    include_base_case:
        If True, run the undamaged network before the contingencies for each
        operating point.
    loading_threshold:
        Percentage loading above which a line/trafo is counted as overloaded.
    severity_metric:
        "overload_excess", "n_overloads", "voltage_violation", or "combined".
    progress:
        Print progress information.

    Returns
    -------
    ContingencyAnalysisResult
        Structured result container with summary, components, and bus tables.
    """
    if contingencies is None:
        contingencies = []

    cases: list[Any] = []
    if include_base_case:
        cases.append(None)
    cases.extend(list(contingencies))

    if not cases:
        raise ValueError("No cases to run. Provide contingencies or set include_base_case=True.")

    summary_rows: list[dict[str, Any]] = []
    component_tables: list[pd.DataFrame] = []
    bus_tables: list[pd.DataFrame] = []

    operating_point_list = list(_iter_operating_points(operating_points))
    total_runs = len(operating_point_list) * len(cases)
    run_counter = 0

    for op_id, op in operating_point_list:
        # Start from a clean copy of the original system for every operating point.
        op_net = copy.deepcopy(net)
        _apply_operating_point_to_loads(op_net, op)

        for case_idx, contingency in enumerate(cases):
            run_counter += 1
            contingency_id = _contingency_id(contingency, case_idx)
            outages = _contingency_outages(contingency)

            if progress:
                pct = run_counter / total_runs
                bar = "#" * int(pct * 40)
                print(f"\r[{bar:<40}] {pct:.0%} OP={op_id} case={contingency_id}", end="")
            try:
                case_net = copy.deepcopy(op_net) if contingency is None else apply_nk_contingency(op_net, contingency)
                converged, error = _run_power_flow(case_net, pf_mode)

                comp = _build_component_rows(
                    case_net,
                    operating_point_id=op_id,
                    contingency_id=contingency_id,
                    loading_threshold=loading_threshold,
                )

                bus = _voltage_results(case_net, v_min_pu=v_min_pu, v_max_pu=v_max_pu)
                if not bus.empty:
                    bus.insert(0, "contingency_id", contingency_id)
                    bus.insert(0, "operating_point_id", op_id)

                summary = _compute_summary(
                    comp,
                    bus,
                    operating_point_id=op_id,
                    contingency_id=contingency_id,
                    n_outages=len(outages),
                    converged=converged,
                    error=error,
                    loading_threshold=loading_threshold,
                    severity_metric=severity_metric,
                )

            except Exception as exc:
                comp = pd.DataFrame()
                bus = pd.DataFrame()
                summary = _compute_summary(
                    comp,
                    bus,
                    operating_point_id=op_id,
                    contingency_id=contingency_id,
                    n_outages=len(outages),
                    converged=False,
                    error=str(exc),
                    loading_threshold=loading_threshold,
                    severity_metric=severity_metric,
                )

            summary_rows.append(summary)
            if not comp.empty:
                component_tables.append(comp)
            if not bus.empty:
                bus_tables.append(bus)

    summary_df = pd.DataFrame(summary_rows)
    components_df = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    bus_df = pd.concat(bus_tables, ignore_index=True) if bus_tables else pd.DataFrame()

    return ContingencyAnalysisResult(summary=summary_df, components=components_df, bus=bus_df)


# ---------------------------------------------------------------------------
# Optional brute-force LODF helper retained from the old module, corrected
# ---------------------------------------------------------------------------


def calculate_lodf_and_shift(network, pf_solver=pp.rundcpp):
    """Brute-force estimate of line-flow shifts and empirical LODF matrix.

    This is slower than analytical LODF computation but useful as a diagnostic.
    It runs a base power flow, then outages each line one at a time.
    """
    net0 = copy.deepcopy(network)
    pf_solver(net0)
    original_flow = pd.to_numeric(net0.res_line["p_from_mw"], errors="coerce").copy()

    shift_rows = []
    lodf_rows = []

    for outaged_line_idx in net0.line.index:
        case = copy.deepcopy(network)
        if outaged_line_idx not in case.line.index:
            continue
        pre_flow = float(original_flow.loc[outaged_line_idx])
        case.line.at[outaged_line_idx, "in_service"] = False

        try:
            pf_solver(case)
            shifted_flow = pd.to_numeric(case.res_line["p_from_mw"], errors="coerce")
            flow_shift = shifted_flow - original_flow
            if abs(pre_flow) > 1e-9:
                lodf = flow_shift / pre_flow
            else:
                lodf = pd.Series(np.nan, index=original_flow.index)
        except Exception:
            flow_shift = pd.Series(np.nan, index=original_flow.index)
            lodf = pd.Series(np.nan, index=original_flow.index)

        shift_rows.append(flow_shift.rename(outaged_line_idx))
        lodf_rows.append(lodf.rename(outaged_line_idx))

    p_f_shift_df = pd.DataFrame(shift_rows)
    lodf_df = pd.DataFrame(lodf_rows)
    return p_f_shift_df, lodf_df