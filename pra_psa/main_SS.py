"""
Subset Simulation example for pandapower case14.

The limit-state function is:

    g(u) = max_{c in N-1 line contingencies} max_l loading_l(u, c) - threshold

Failure is g(u) >= 0.

The workflow is:
    1. create base network;
    2. apply the base load;
    3. solve DC-OPF on the intact network;
    4. freeze generator dispatch from the OPF;
    5. keep the external grid in service as the slack/reference element;
    6. map standard-normal uncertainty u to load multipliers;
    7. run DC power flow for every N-1 line contingency;
    8. return the worst loading margin.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandapower as pp
from pandapower.networks import case14

from pra_psa.core.contingency import generate_n1_contingencies
from pra_psa.sampler.subset_simulation import (
    SubsetSym,
    compare_subset_simulation_with_mc,
)


@dataclass(frozen=True)
class ReferenceDispatch:
    """Preventive dispatch obtained from the intact base-load OPF."""

    gen_p_mw: np.ndarray
    gen_q_mvar: np.ndarray | None
    ext_grid_p_mw: float | None
    ext_grid_q_mvar: float | None
    load_p_mw: np.ndarray
    load_q_mvar: np.ndarray | None


def _as_line_contingency(event) -> dict:
    """Normalise legacy contingency formats to {'element_type', 'element_index'}."""

    if isinstance(event, dict):
        if "element_type" in event and "element_index" in event:
            return {
                "element_type": event["element_type"],
                "element_index": int(event["element_index"]),
            }
        if "type" in event and "index" in event:
            return {"element_type": event["type"], "element_index": int(event["index"])}

    if isinstance(event, tuple) and len(event) == 2:
        element_type, element_index = event
        return {"element_type": element_type, "element_index": int(element_index)}

    if isinstance(event, (int, np.integer)):
        return {"element_type": "line", "element_index": int(event)}

    raise ValueError(f"Unsupported contingency format: {event!r}")


def add_linear_generation_costs_if_missing(net) -> None:
    """Ensure that pandapower OPF has generation cost functions."""

    if "poly_cost" not in net or net.poly_cost.empty:
        # Small positive costs. The exact values are not important for the PRA
        # demonstration; they only make the OPF well-defined.
        for ext_grid_index in net.ext_grid.index:
            pp.create_poly_cost(
                net,
                element=ext_grid_index,
                et="ext_grid",
                cp1_eur_per_mw=20.0,
            )
        for gen_index in net.gen.index:
            pp.create_poly_cost(
                net,
                element=gen_index,
                et="gen",
                cp1_eur_per_mw=10.0 + float(gen_index),
            )


def _numeric_series(values, index, default: float = 0.0):
    """Return a numeric pandas Series aligned with the target index."""
    import pandas as pd

    if values is None:
        return pd.Series(default, index=index, dtype=float)

    series = pd.Series(values, index=index)
    series = pd.to_numeric(series, errors="coerce")
    return series.fillna(default).astype(float)


def prepare_opf_bounds(net) -> None:
    """Add broad controllability bounds required by pandapower OPF.

    This version avoids pandas ``fillna(ndarray)`` errors by always using
    scalar values or index-aligned Series.
    """

    net.ext_grid["in_service"] = True
    if "slack" in net.gen.columns:
        net.gen["slack"] = False

    # External grid: keep it controllable during OPF but retain it as the
    # PF slack/reference afterwards.
    net.ext_grid["controllable"] = True
    net.ext_grid["min_p_mw"] = _numeric_series(
        net.ext_grid.get("min_p_mw"), net.ext_grid.index, -1e4
    )
    net.ext_grid["max_p_mw"] = _numeric_series(
        net.ext_grid.get("max_p_mw"), net.ext_grid.index, 1e4
    )
    net.ext_grid["min_q_mvar"] = _numeric_series(
        net.ext_grid.get("min_q_mvar"), net.ext_grid.index, -1e4
    )
    net.ext_grid["max_q_mvar"] = _numeric_series(
        net.ext_grid.get("max_q_mvar"), net.ext_grid.index, 1e4
    )

    # Generators: ensure controllability and complete P/Q bounds.
    net.gen["controllable"] = True

    current_p = _numeric_series(net.gen.get("p_mw"), net.gen.index, 0.0)
    fallback_max_p = np.maximum(current_p.to_numpy(dtype=float), 1.0)

    if "min_p_mw" not in net.gen.columns:
        net.gen["min_p_mw"] = 0.0
    else:
        net.gen["min_p_mw"] = _numeric_series(
            net.gen["min_p_mw"], net.gen.index, 0.0
        )

    if "max_p_mw" not in net.gen.columns:
        net.gen["max_p_mw"] = fallback_max_p
    else:
        max_p = _numeric_series(net.gen["max_p_mw"], net.gen.index, np.nan)
        fallback = _numeric_series(fallback_max_p, net.gen.index, 1.0)
        net.gen["max_p_mw"] = max_p.where(max_p.notna(), fallback)

    # Avoid impossible bounds if existing case data are inconsistent.
    net.gen["max_p_mw"] = np.maximum(
        net.gen["max_p_mw"].to_numpy(dtype=float),
        net.gen["min_p_mw"].to_numpy(dtype=float) + 1e-6,
    )

    if "min_q_mvar" not in net.gen.columns:
        net.gen["min_q_mvar"] = -1e4
    else:
        net.gen["min_q_mvar"] = _numeric_series(
            net.gen["min_q_mvar"], net.gen.index, -1e4
        )

    if "max_q_mvar" not in net.gen.columns:
        net.gen["max_q_mvar"] = 1e4
    else:
        net.gen["max_q_mvar"] = _numeric_series(
            net.gen["max_q_mvar"], net.gen.index, 1e4
        )

    net.gen["max_q_mvar"] = np.maximum(
        net.gen["max_q_mvar"].to_numpy(dtype=float),
        net.gen["min_q_mvar"].to_numpy(dtype=float) + 1e-6,
    )

    if len(net.line):
        net.line["max_loading_percent"] = 100.0

def solve_reference_dispatch(
    net,
    *,
    base_load_scale: float = 1.0,
    opf_solver=pp.rundcopp,
) -> tuple[object, ReferenceDispatch]:
    """
    Solve intact-network OPF and return a network with fixed preventive dispatch.

    The external grid is retained as the PF slack/reference. Its OPF exchange is
    stored for diagnostics, but in a subsequent PF its active power remains an
    output balancing variable.
    """

    opf_net = copy.deepcopy(net)

    nominal_p = opf_net.load["p_mw"].to_numpy(dtype=float).copy()
    nominal_q = (
        opf_net.load["q_mvar"].to_numpy(dtype=float).copy()
        if "q_mvar" in opf_net.load.columns
        else None
    )

    opf_net.load.loc[:, "p_mw"] = nominal_p * base_load_scale
    if nominal_q is not None:
        opf_net.load.loc[:, "q_mvar"] = nominal_q * base_load_scale

    prepare_opf_bounds(opf_net)
    add_linear_generation_costs_if_missing(opf_net)

    try:
        opf_solver(opf_net)
    except Exception:
        # DC-OPF may fail for some pandapower versions/cases without complete
        # OPF metadata. Fall back to a DC PF and freeze the existing dispatch.
        pp.rundcpp(opf_net)

    if hasattr(opf_net, "res_gen") and not opf_net.res_gen.empty:
        gen_p = opf_net.res_gen["p_mw"].to_numpy(dtype=float).copy()
        gen_q = (
            opf_net.res_gen["q_mvar"].to_numpy(dtype=float).copy()
            if "q_mvar" in opf_net.res_gen.columns
            else None
        )
    else:
        gen_p = opf_net.gen["p_mw"].to_numpy(dtype=float).copy()
        gen_q = (
            opf_net.gen["q_mvar"].to_numpy(dtype=float).copy()
            if "q_mvar" in opf_net.gen.columns
            else None
        )

    ext_p = (
        float(opf_net.res_ext_grid["p_mw"].sum())
        if hasattr(opf_net, "res_ext_grid") and not opf_net.res_ext_grid.empty
        else None
    )
    ext_q = (
        float(opf_net.res_ext_grid["q_mvar"].sum())
        if hasattr(opf_net, "res_ext_grid")
        and not opf_net.res_ext_grid.empty
        and "q_mvar" in opf_net.res_ext_grid.columns
        else None
    )

    fixed_net = copy.deepcopy(opf_net)
    fixed_net.gen.loc[:, "p_mw"] = gen_p
    if gen_q is not None and "q_mvar" in fixed_net.gen.columns:
        fixed_net.gen.loc[:, "q_mvar"] = gen_q

    fixed_net.ext_grid["in_service"] = True
    if "slack" in fixed_net.gen.columns:
        fixed_net.gen["slack"] = False

    dispatch = ReferenceDispatch(
        gen_p_mw=gen_p,
        gen_q_mvar=gen_q,
        ext_grid_p_mw=ext_p,
        ext_grid_q_mvar=ext_q,
        load_p_mw=nominal_p * base_load_scale,
        load_q_mvar=None if nominal_q is None else nominal_q * base_load_scale,
    )

    return fixed_net, dispatch


def standard_normal_to_lognormal_multiplier(
    u: np.ndarray,
    *,
    sigma: float | np.ndarray = 0.05,
) -> np.ndarray:
    """Mean-one lognormal load multipliers from standard-normal variables."""

    u = np.asarray(u, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    return np.exp(-0.5 * sigma_array**2 + sigma_array * u)


def apply_line_contingency(net, contingency: dict) -> None:
    """Apply one line contingency in-place."""

    contingency = _as_line_contingency(contingency)
    if contingency["element_type"] != "line":
        raise ValueError(f"Expected line contingency, received {contingency}.")
    net.line.at[contingency["element_index"], "in_service"] = False


def make_case14_worst_n1_loading_limit_state(
    fixed_dispatch_net,
    dispatch: ReferenceDispatch,
    contingencies,
    *,
    sigma: float = 0.05,
    loading_threshold_percent: float = 90.0,
    pf_solver=pp.rundcpp,
    include_base_case: bool = True,
    external_grid_tolerance_mw: float | None = None,
):
    """
    Build g(u) for case14 worst N-1 loading.

    ``dimension`` equals the number of loads, because each uncertain variable
    controls one load multiplier.

    Failure convention:
        g(u) >= 0
    """

    contingencies = [_as_line_contingency(c) for c in contingencies]
    nominal_p = dispatch.load_p_mw.copy()
    nominal_q = None if dispatch.load_q_mvar is None else dispatch.load_q_mvar.copy()
    dimension = len(nominal_p)

    def run_pf_and_margin(net) -> float:
        try:
            pf_solver(net)
        except Exception:
            return float("inf")

        if not getattr(net, "converged", True):
            return float("inf")

        loading = net.res_line["loading_percent"].to_numpy(dtype=float)
        margin_loading = float(np.nanmax(np.abs(loading)) - loading_threshold_percent)

        if (
            external_grid_tolerance_mw is not None
            and dispatch.ext_grid_p_mw is not None
            and hasattr(net, "res_ext_grid")
            and not net.res_ext_grid.empty
        ):
            p_ext = float(net.res_ext_grid["p_mw"].sum())
            margin_ext = abs(p_ext - dispatch.ext_grid_p_mw) - external_grid_tolerance_mw
            return max(margin_loading, margin_ext)

        return margin_loading

    def limit_state(u: np.ndarray) -> float:
        multipliers = standard_normal_to_lognormal_multiplier(u, sigma=sigma)
        worst_margin = -float("inf")

        states = [None] if include_base_case else []
        states += contingencies

        for contingency in states:
            trial = copy.deepcopy(fixed_dispatch_net)
            trial.load.loc[:, "p_mw"] = nominal_p * multipliers
            if nominal_q is not None:
                trial.load.loc[:, "q_mvar"] = nominal_q * multipliers

            if contingency is not None:
                apply_line_contingency(trial, contingency)

            margin = run_pf_and_margin(trial)
            worst_margin = max(worst_margin, margin)

        return float(worst_margin)

    return limit_state, dimension


if __name__ == "__main__":

    net = case14()

    # Legacy generator returns line contingencies. The helper below also accepts
    # simple integer line indices.
    contingencies = generate_n1_contingencies(
        n_elements=len(net.line),
        element_type="line",
    )

    fixed_dispatch_net, reference = solve_reference_dispatch(
        net,
        base_load_scale=1.0,
        opf_solver=pp.rundcopp,
    )

    limit_state, dimension = make_case14_worst_n1_loading_limit_state(
        fixed_dispatch_net,
        reference,
        contingencies,
        sigma=0.05,
        loading_threshold_percent=90.0,
        pf_solver=pp.rundcpp,
        include_base_case=True,
        # Set this to a number if external-grid balancing must also count as
        # failure, e.g. external_grid_tolerance_mw=50.0.
        external_grid_tolerance_mw=None,
    )

    result = SubsetSym(
        limit_state,
        dimension=dimension,
        n_samples=1000,
        p0=0.1,
        proposal_std=0.8,
        seed=7,
        failure_value_on_exception=float("inf"),
    )

    print("Estimated failure probability:", result.probability)
    print("Summary:", result.summary())
    print("Thresholds:", [level.threshold for level in result.levels])
    print("Level diagnostics:")
    for level in result.levels:
        print(level)

    # Optional comparison with crude Monte Carlo. This is useful for debugging
    # but expensive when each limit-state call loops over all contingencies.
    compare_subset_simulation_with_mc(
        limit_state,
        dimension=dimension,
        subset_n_samples=1000,
        subset_p0=0.1,
        subset_proposal_std=0.8,
        mc_n_samples=5000,
        failure_threshold=0.0,
        seed=11,
        exact_probability=None,
        figure_path=Path("outputs/subset_simulation_case14/mc_vs_subset.png"),
    )