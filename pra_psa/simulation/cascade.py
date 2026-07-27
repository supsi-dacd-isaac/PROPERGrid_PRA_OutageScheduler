"""Simplified cascading-failure simulator.

This module is deliberately conservative: it provides a first deterministic
cascade layer for PRA experiments, not a replacement for a full high-fidelity
CASCADE implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from pra_psa.core.contingencies import apply_contingency
from pra_psa.core.models import Contingency
from pra_psa.simulation.powerflow import deepcopy_net, extract_branch_results, network_island_count, run_power_flow
from pra_psa.severity import branch_overload_excess


@dataclass
class CascadeResult:
    summary: pd.DataFrame
    final_net: Any


def run_cascade(
    net: Any,
    initiating_contingency: Any,
    *,
    pf_mode: str = "dc",
    overload_threshold: float = 100.0,
    max_steps: int = 10,
    trip_rule: str = "most_overloaded",
) -> CascadeResult:
    """Run an iterative overload-tripping cascade approximation."""
    cont = Contingency.from_any(initiating_contingency)
    case_net = apply_contingency(deepcopy_net(net), cont, copy_net=False)
    failed = [o.short_id for o in cont.outages]
    rows = []

    for step in range(max_steps + 1):
        try:
            run_power_flow(case_net, mode=pf_mode)
            branches = extract_branch_results(case_net)
            islands = network_island_count(case_net)
            excess = branch_overload_excess(branches, threshold=overload_threshold)
            max_loading = float(np.nanmax(branches["loading_percent"])) if not branches.empty else np.nan
            n_overloads = int(np.nansum(branches["loading_percent"].astype(float).to_numpy() > overload_threshold)) if not branches.empty else 0
            rows.append(
                {
                    "step": step,
                    "converged": True,
                    "n_islands": islands,
                    "n_overloads": n_overloads,
                    "max_loading_pct": max_loading,
                    "overload_excess": excess,
                    "failed_elements": ";".join(failed),
                    "tripped_this_step": "",
                    "error": "",
                }
            )
            if n_overloads == 0:
                break
            trip = _select_trip(branches, threshold=overload_threshold, rule=trip_rule)
            if trip is None:
                break
            _trip_element(case_net, trip["element_type"], int(trip["element_index"]))
            failed.append(f"{trip['element_type']}:{int(trip['element_index'])}")
            rows[-1]["tripped_this_step"] = failed[-1]
        except Exception as exc:
            rows.append(
                {
                    "step": step,
                    "converged": False,
                    "n_islands": network_island_count(case_net),
                    "n_overloads": np.nan,
                    "max_loading_pct": np.nan,
                    "overload_excess": np.nan,
                    "failed_elements": ";".join(failed),
                    "tripped_this_step": "",
                    "error": repr(exc),
                }
            )
            break
    return CascadeResult(summary=pd.DataFrame(rows), final_net=case_net)


def _select_trip(branches: pd.DataFrame, *, threshold: float, rule: str) -> dict | None:
    candidates = branches[(branches["in_service"]) & (branches["loading_percent"].astype(float) > threshold)]
    if candidates.empty:
        return None
    if rule != "most_overloaded":
        raise ValueError("Currently only trip_rule='most_overloaded' is supported")
    row = candidates.sort_values("loading_percent", ascending=False).iloc[0]
    return row.to_dict()


def _trip_element(net: Any, element_type: str, element_index: int) -> None:
    table = getattr(net, element_type)
    table.at[element_index, "in_service"] = False
