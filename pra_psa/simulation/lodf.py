"""LODF/PTDF-based contingency screening.

The LODF layer is intended as a fast screening/ranking method. Final severity
estimates should still be validated with AC or DC power-flow simulations.
"""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import pandas as pd

from pra_psa.core.models import Contingency, normalize_contingencies
from pra_psa.simulation.powerflow import require_pandapower, run_power_flow


def compute_lodf_matrix(net: Any, *, ensure_pf: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute LODF matrix and return base branch flows and ratings.

    Returns
    -------
    lodf:
        Branch-by-branch LODF matrix in pandapower internal PPC branch order.
    base_flows_mw:
        Base-case active branch flows in MW.
    ratings_mva:
        Branch ratings used to estimate loading percentages.
    branch_meta:
        Mapping between PPC branch rows and pandapower element IDs.
    """
    pp = require_pandapower()
    if ensure_pf or not hasattr(net, "_ppc"):
        run_power_flow(net, mode="dc")
    try:
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pypower.makeLODF import makeLODF
        from pandapower.pypower.idx_brch import PF, RATE_A
    except Exception as exc:
        raise ImportError("Could not import pandapower/pypower LODF utilities") from exc

    ppc = net._ppc.get("internal", net._ppc)
    ptdf = makePTDF(baseMVA=ppc["baseMVA"], bus=ppc["bus"], branch=ppc["branch"])
    lodf = makeLODF(ppc["branch"], PTDF=ptdf)
    lodf = np.nan_to_num(lodf, nan=0.0, posinf=0.0, neginf=0.0)
    base_flows = ppc["branch"][:, PF].astype(float).copy()
    ratings = ppc["branch"][:, RATE_A].astype(float).copy()
    branch_meta = _ppc_branch_metadata(net, n_branch=len(base_flows))
    return lodf, base_flows, ratings, branch_meta


def screen_contingencies_lodf(
    net: Any,
    contingencies: Iterable[Any],
    *,
    loading_threshold: float = 100.0,
) -> pd.DataFrame:
    """Rank single-line contingencies using a DC LODF approximation."""
    cont_list = normalize_contingencies(contingencies)
    lodf, base_flows, ratings, branch_meta = compute_lodf_matrix(net)
    ppc_line_by_pp_idx = _ppc_index_lookup(branch_meta, "line")

    rows = []
    for contingency in cont_list:
        if contingency.order != 1 or contingency.outages[0].element_type != "line":
            rows.append(_unsupported_lodf_row(contingency, "Only single-line contingencies are supported"))
            continue
        outage = contingency.outages[0]
        if outage.element_index not in ppc_line_by_pp_idx:
            rows.append(_unsupported_lodf_row(contingency, "Line not found in PPC branch order"))
            continue
        j = ppc_line_by_pp_idx[outage.element_index]
        post_flows = base_flows + lodf[:, j] * base_flows[j]
        post_flows[j] = 0.0
        loading = np.full_like(post_flows, np.nan, dtype=float)
        valid = ratings > 0
        loading[valid] = np.abs(post_flows[valid]) / ratings[valid] * 100.0
        overload_excess = float(np.nansum(np.maximum(loading - loading_threshold, 0.0)))
        rows.append(
            {
                "contingency_id": contingency.contingency_id,
                "outaged_element": outage.short_id,
                "supported": True,
                "max_loading_pct_lodf": float(np.nanmax(loading)),
                "n_overloads_lodf": int(np.nansum(loading > loading_threshold)),
                "overload_excess_lodf": overload_excess,
                "error": "",
            }
        )
    return pd.DataFrame(rows).sort_values("overload_excess_lodf", ascending=False, na_position="last")


def _ppc_branch_metadata(net: Any, *, n_branch: int) -> pd.DataFrame:
    rows = []
    lookup = getattr(net, "_pd2ppc_lookups", {}).get("branch", {})
    for et in ["line", "trafo"]:
        if et in lookup:
            start, end = lookup[et]
            table = getattr(net, et)
            for pp_idx, ppc_idx in zip(table.index, range(start, end)):
                rows.append({"ppc_branch_index": ppc_idx, "element_type": et, "element_index": int(pp_idx)})
    if not rows:
        rows = [{"ppc_branch_index": i, "element_type": "branch", "element_index": i} for i in range(n_branch)]
    return pd.DataFrame(rows)


def _ppc_index_lookup(branch_meta: pd.DataFrame, element_type: str) -> dict[int, int]:
    subset = branch_meta[branch_meta["element_type"] == element_type]
    return {int(row.element_index): int(row.ppc_branch_index) for row in subset.itertuples()}


def _unsupported_lodf_row(contingency: Contingency, error: str) -> dict:
    return {
        "contingency_id": contingency.contingency_id,
        "outaged_element": ";".join(o.short_id for o in contingency.outages),
        "supported": False,
        "max_loading_pct_lodf": np.nan,
        "n_overloads_lodf": np.nan,
        "overload_excess_lodf": np.nan,
        "error": error,
    }
