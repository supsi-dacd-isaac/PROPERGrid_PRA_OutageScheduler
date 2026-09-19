"""Sequential DC cascading-failure simulator inspired by the Cascades workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from .network import DCNetwork, PowerFlowState


TripPolicy = Literal["most_overloaded", "all_overloaded"]


@dataclass(frozen=True)
class CascadeStage:
    stage: int
    active_outages: tuple[str, ...]
    tripped_after_stage: tuple[str, ...]
    demand_not_served_mw: float
    number_of_islands: int
    maximum_loading_ratio: float


@dataclass(frozen=True)
class CascadeResult:
    initiating_outages: tuple[str, ...]
    planned_outages: tuple[str, ...]
    propagated_outages: tuple[str, ...]
    final_outages: tuple[str, ...]
    stages: tuple[CascadeStage, ...]
    final_state: PowerFlowState
    stable: bool
    termination_reason: str

    @property
    def demand_not_served_mw(self) -> float:
        return self.final_state.demand_not_served_mw

    def stage_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(stage) for stage in self.stages])

    def save(self, directory: str | Path, prefix: str = "cascade") -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.stage_frame().to_csv(output / f"{prefix}_stages.csv", index=False)


class CascadingFailureSimulator:
    """Iteratively trip overloads until the DC snapshot is stable.

    The simulator implements the steady-state skeleton used in the cited
    Cascades studies: apply initiating/planned outages, identify islands,
    approximate frequency control by redispatch and proportional load shedding,
    recalculate flows, and trip the most overloaded branch. It is a draft, not
    an AC/transient-stability reproduction: voltage protection, reactive power,
    relay delays, dynamic frequency, and hidden failures are outside scope.
    """

    def __init__(
        self,
        network: DCNetwork,
        *,
        overload_threshold: float = 1.0,
        max_stages: int = 50,
        trip_policy: TripPolicy = "most_overloaded",
        tolerance: float = 1e-9,
    ):
        if overload_threshold <= 0 or max_stages < 0:
            raise ValueError("overload_threshold must be positive and max_stages non-negative.")
        if trip_policy not in {"most_overloaded", "all_overloaded"}:
            raise ValueError(f"Unsupported trip_policy {trip_policy!r}.")
        self.network = network
        self.overload_threshold = float(overload_threshold)
        self.max_stages = int(max_stages)
        self.trip_policy = trip_policy
        self.tolerance = float(tolerance)

    def _select_trips(self, state: PowerFlowState, outages: set[str]) -> tuple[str, ...]:
        overloaded = {
            asset_id: loading
            for asset_id, loading in state.branch_loading_ratio.items()
            if asset_id not in outages and loading > self.overload_threshold + self.tolerance
        }
        if not overloaded:
            return ()
        if self.trip_policy == "all_overloaded":
            return tuple(sorted(overloaded, key=lambda item: (-overloaded[item], item)))
        maximum = max(overloaded.values())
        candidates = [item for item, value in overloaded.items() if abs(value - maximum) <= self.tolerance]
        return (min(candidates),)

    def simulate(
        self,
        *,
        load_mw=None,
        initiating_outages: Iterable[str] = (),
        planned_outages: Iterable[str] = (),
    ) -> CascadeResult:
        network = self.network if load_mw is None else self.network.with_load(load_mw)
        initiating = tuple(dict.fromkeys(map(str, initiating_outages)))
        planned = tuple(dict.fromkeys(map(str, planned_outages)))
        known = set(network.branch_ids)
        unknown = (set(initiating) | set(planned)) - known
        if unknown:
            raise KeyError(f"Unknown initial branch outages: {sorted(unknown)}")
        outages = set(initiating) | set(planned)
        propagated: list[str] = []
        records: list[CascadeStage] = []
        final_state = network.solve_dc(outages)
        stable = False
        termination = "maximum cascade stages reached"
        for stage in range(self.max_stages + 1):
            final_state = network.solve_dc(outages)
            selected = self._select_trips(final_state, outages)
            maximum_loading = max(final_state.branch_loading_ratio.values(), default=0.0)
            records.append(
                CascadeStage(
                    stage=stage,
                    active_outages=tuple(sorted(outages)),
                    tripped_after_stage=selected,
                    demand_not_served_mw=final_state.demand_not_served_mw,
                    number_of_islands=len(final_state.islands),
                    maximum_loading_ratio=float(maximum_loading),
                )
            )
            if not selected:
                stable = True
                termination = "stable: no branch exceeds the overload threshold"
                break
            if stage == self.max_stages:
                break
            outages.update(selected)
            propagated.extend(selected)
        return CascadeResult(
            initiating_outages=initiating,
            planned_outages=planned,
            propagated_outages=tuple(propagated),
            final_outages=tuple(sorted(outages)),
            stages=tuple(records),
            final_state=final_state,
            stable=stable,
            termination_reason=termination,
        )

