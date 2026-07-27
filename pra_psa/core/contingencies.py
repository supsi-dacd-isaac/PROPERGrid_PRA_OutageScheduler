"""Contingency generation and application utilities."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Sequence
import random

from pra_psa.core.models import Contingency, Outage, normalize_contingencies, normalize_element_type
from pra_psa.simulation.powerflow import deepcopy_net


def build_n1_contingencies(
    net: Any,
    *,
    include: Sequence[str] = ("line", "trafo"),
    only_in_service: bool = True,
) -> list[Contingency]:
    """Build all N-1 contingencies for selected pandapower element types."""
    contingencies: list[Contingency] = []
    for element_type in include:
        et = normalize_element_type(element_type)
        if not hasattr(net, et):
            continue
        table = getattr(net, et)
        for idx in table.index:
            if only_in_service and "in_service" in table.columns and not bool(table.at[idx, "in_service"]):
                continue
            outage = Outage(et, int(idx))
            contingencies.append(Contingency((outage,), contingency_id=f"N-1::{outage.short_id}"))
    return contingencies


def build_nk_contingencies(
    net: Any,
    *,
    k: int = 2,
    candidate_pool: Iterable[Any] | None = None,
    include: Sequence[str] = ("line", "trafo"),
    max_contingencies: int | None = None,
    random_state: int | None = None,
) -> list[Contingency]:
    """Build N-k contingencies from a candidate outage pool."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if candidate_pool is None:
        pool = [c.outages[0] for c in build_n1_contingencies(net, include=include)]
    else:
        pool = [Outage.from_any(o) if not isinstance(o, Contingency) else o.outages[0] for o in candidate_pool]
    combos = list(combinations(pool, k))
    if max_contingencies is not None and len(combos) > max_contingencies:
        rng = random.Random(random_state)
        combos = rng.sample(combos, max_contingencies)
    return [Contingency(tuple(c), contingency_id=f"N-{k}::" + "+".join(o.short_id for o in c)) for c in combos]


def apply_contingency(net: Any, contingency: Any, *, copy_net: bool = True) -> Any:
    """Apply a contingency by setting affected elements out of service."""
    cont = Contingency.from_any(contingency)
    out = deepcopy_net(net) if copy_net else net
    for outage in cont.outages:
        if not hasattr(out, outage.element_type):
            raise ValueError(f"Network has no table {outage.element_type!r}")
        table = getattr(out, outage.element_type)
        if outage.element_index not in table.index:
            raise KeyError(f"{outage.element_type} index {outage.element_index} not found")
        if "in_service" not in table.columns:
            raise ValueError(f"Element table {outage.element_type!r} has no in_service column")
        table.at[outage.element_index, "in_service"] = False
    return out


def apply_nk_contingency(network: Any, failure_event: Any):
    """Backward-compatible alias for older code."""
    return apply_contingency(network, failure_event, copy_net=True)


def contingency_ids(contingencies: Iterable[Any]) -> list[str]:
    return [c.contingency_id for c in normalize_contingencies(contingencies)]
