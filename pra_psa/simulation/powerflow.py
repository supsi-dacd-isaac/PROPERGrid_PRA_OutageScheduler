"""Power-flow utilities used by contingency and cascading analyses."""

from __future__ import annotations

from typing import Any, Mapping
import copy
import numpy as np
import pandas as pd


def require_pandapower():
    try:
        import pandapower as pp
    except ImportError as exc:
        raise ImportError("pandapower is required for this function. Install it with `pip install pandapower`.") from exc
    return pp


def deepcopy_net(net: Any) -> Any:
    if hasattr(net, "deepcopy"):
        return net.deepcopy()
    return copy.deepcopy(net)


def run_power_flow(
    net: Any,
    *,
    mode: str = "dc",
    distributed_slack: bool = True,
    enforce_q_lims: bool = False,
    **kwargs: Any,
) -> Any:
    """Run AC or DC power flow on a pandapower network."""
    pp = require_pandapower()
    mode = mode.lower()
    try:
        if mode in {"dc", "dcpf", "rundcpp"}:
            pp.rundcpp(net, **kwargs)
        elif mode in {"ac", "acpf", "runpp"}:
            try:
                pp.runpp(net, distributed_slack=distributed_slack, enforce_q_lims=enforce_q_lims, **kwargs)
            except TypeError:
                pp.runpp(net, enforce_q_lims=enforce_q_lims, **kwargs)
        else:
            raise ValueError("mode must be 'dc' or 'ac'")
    except Exception:
        # Do not hide pandapower convergence errors. The analysis layer will
        # catch them and record non-converged cases.
        raise
    return net


def apply_load_profile(
    net: Any,
    load_profile: Mapping | pd.Series | np.ndarray | list | None,
    *,
    q_over_p: float | None = None,
) -> Any:
    """Apply a bus-level or load-level active-power profile to a network.

    Accepted shapes:
    - length == number of loads: assigned directly to ``net.load.p_mw``;
    - length == number of buses: mapped to loads by bus;
    - pandas Series indexed by external MATPOWER bus IDs, pandapower bus
      indices, or strings such as ``bus_0``.
    """
    if load_profile is None:
        return net
    if not hasattr(net, "load") or len(net.load) == 0:
        return net

    if isinstance(load_profile, pd.Series):
        profile = load_profile.copy()
    elif isinstance(load_profile, Mapping):
        profile = pd.Series(load_profile)
    else:
        arr = np.asarray(load_profile, dtype=float).reshape(-1)
        if len(arr) == len(net.load):
            net.load.loc[:, "p_mw"] = arr
            if q_over_p is not None and "q_mvar" in net.load:
                net.load.loc[:, "q_mvar"] = arr * q_over_p
            return net
        if len(arr) == len(net.bus):
            bus_ids = _external_bus_ids(net)
            profile = pd.Series(arr, index=bus_ids)
        else:
            raise ValueError(
                f"Load profile length {len(arr)} is incompatible with "
                f"n_load={len(net.load)} and n_bus={len(net.bus)}."
            )

    values = []
    for _, load in net.load.iterrows():
        bus_idx = load["bus"]
        candidates = _bus_key_candidates(net, bus_idx)
        value = None
        for key in candidates:
            if key in profile.index:
                value = profile.loc[key]
                break
        if value is None:
            # If the bus has no entry, assign zero. This is safe for bus-level profiles containing zero-demand buses.
            value = 0.0
        values.append(float(value))
    net.load.loc[:, "p_mw"] = values
    if q_over_p is not None and "q_mvar" in net.load:
        net.load.loc[:, "q_mvar"] = np.asarray(values) * q_over_p
    return net


def _external_bus_ids(net: Any) -> list:
    if "matpower_bus_id" in net.bus.columns:
        return net.bus["matpower_bus_id"].astype(int).tolist()
    return list(net.bus.index)


def _bus_key_candidates(net: Any, bus_idx: int) -> list:
    candidates = [bus_idx, int(bus_idx), str(bus_idx)]
    if "matpower_bus_id" in net.bus.columns:
        ext = int(net.bus.at[bus_idx, "matpower_bus_id"])
        candidates.extend([ext, str(ext), f"bus_{ext}", f"bus_{ext - 1}"])
    candidates.extend([f"bus_{bus_idx}", f"bus_{int(bus_idx)}"])
    return candidates


def extract_branch_results(net: Any) -> pd.DataFrame:
    """Return line and transformer active flows and loadings as a long table."""
    rows: list[dict[str, Any]] = []
    if hasattr(net, "line") and hasattr(net, "res_line") and len(net.line):
        for idx in net.line.index:
            res = net.res_line.loc[idx]
            rows.append(
                {
                    "element_type": "line",
                    "element_index": int(idx),
                    "from_bus": int(net.line.at[idx, "from_bus"]),
                    "to_bus": int(net.line.at[idx, "to_bus"]),
                    "p_mw": _get_first_available(res, ["p_from_mw", "p_to_mw"], default=np.nan),
                    "q_mvar": _get_first_available(res, ["q_from_mvar", "q_to_mvar"], default=np.nan),
                    "loading_percent": float(res.get("loading_percent", np.nan)),
                    "in_service": bool(net.line.at[idx, "in_service"]),
                }
            )
    if hasattr(net, "trafo") and hasattr(net, "res_trafo") and len(net.trafo):
        for idx in net.trafo.index:
            res = net.res_trafo.loc[idx]
            rows.append(
                {
                    "element_type": "trafo",
                    "element_index": int(idx),
                    "from_bus": int(net.trafo.at[idx, "hv_bus"]),
                    "to_bus": int(net.trafo.at[idx, "lv_bus"]),
                    "p_mw": _get_first_available(res, ["p_hv_mw", "p_lv_mw"], default=np.nan),
                    "q_mvar": _get_first_available(res, ["q_hv_mvar", "q_lv_mvar"], default=np.nan),
                    "loading_percent": float(res.get("loading_percent", np.nan)),
                    "in_service": bool(net.trafo.at[idx, "in_service"]),
                }
            )
    return pd.DataFrame(rows)


def extract_bus_results(net: Any) -> pd.DataFrame:
    if not hasattr(net, "res_bus"):
        return pd.DataFrame()
    out = net.res_bus.copy()
    out.insert(0, "bus", out.index.astype(int))
    if hasattr(net, "bus") and "matpower_bus_id" in net.bus.columns:
        out["matpower_bus_id"] = net.bus.loc[out.index, "matpower_bus_id"].astype(int).values
    return out.reset_index(drop=True)


def network_island_count(net: Any) -> int | None:
    """Return number of connected electrical islands, if pandapower topology is available."""
    try:
        import pandapower.topology as top
        import networkx as nx
        graph = top.create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True)
        return nx.number_connected_components(graph)
    except Exception:
        return None


def extract_branch_flows_and_loading(net: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning arrays of branch flows and loadings."""
    br = extract_branch_results(net)
    return br["p_mw"].to_numpy(dtype=float), br["loading_percent"].to_numpy(dtype=float)


def _get_first_available(row: pd.Series, keys: list[str], default=np.nan):
    for key in keys:
        if key in row:
            return float(row[key])
    return default
