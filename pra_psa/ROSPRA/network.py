"""Small, dependency-light DC network model used by the cascade prototype."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Branch:
    asset_id: str
    from_bus: int
    to_bus: int
    reactance_pu: float
    rating_mw: float
    component_kind: str = "single_line"
    length_km: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PowerFlowState:
    bus_angles_rad: FloatArray
    branch_flow_mw: dict[str, float]
    branch_loading_ratio: dict[str, float]
    generation_mw: FloatArray
    served_load_mw: FloatArray
    load_shed_mw: FloatArray
    islands: tuple[tuple[int, ...], ...]

    @property
    def demand_not_served_mw(self) -> float:
        return float(np.sum(self.load_shed_mw))


@dataclass(frozen=True)
class DCNetwork:
    """Static DC network snapshot with bus-level load and generation."""

    n_buses: int
    branches: tuple[Branch, ...]
    load_mw: FloatArray
    generation_mw: FloatArray
    max_generation_mw: FloatArray
    bus_names: tuple[str, ...] | None = None
    base_mva: float = 100.0

    def __post_init__(self) -> None:
        if self.n_buses < 1:
            raise ValueError("n_buses must be positive.")
        for name in ("load_mw", "generation_mw", "max_generation_mw"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (self.n_buses,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be a finite non-negative vector of length {self.n_buses}.")
            object.__setattr__(self, name, value.copy())
        if np.any(self.generation_mw > self.max_generation_mw + 1e-9):
            raise ValueError("generation_mw cannot exceed max_generation_mw.")
        identifiers = set()
        for branch in self.branches:
            if branch.asset_id in identifiers:
                raise ValueError(f"Duplicate branch id {branch.asset_id!r}.")
            identifiers.add(branch.asset_id)
            if not 0 <= branch.from_bus < self.n_buses or not 0 <= branch.to_bus < self.n_buses:
                raise ValueError(f"Branch {branch.asset_id!r} references an invalid bus.")
            if branch.from_bus == branch.to_bus or branch.reactance_pu <= 0 or branch.rating_mw <= 0:
                raise ValueError(f"Branch {branch.asset_id!r} has invalid electrical parameters.")
        if self.bus_names is None:
            object.__setattr__(self, "bus_names", tuple(f"bus_{index + 1}" for index in range(self.n_buses)))
        elif len(self.bus_names) != self.n_buses:
            raise ValueError("bus_names must contain one label per bus.")

    @property
    def branch_ids(self) -> tuple[str, ...]:
        return tuple(branch.asset_id for branch in self.branches)

    def with_load(self, load_mw: ArrayLike) -> "DCNetwork":
        load = np.asarray(load_mw, dtype=float)
        if load.shape != (self.n_buses,):
            raise ValueError(f"Expected a load vector of shape {(self.n_buses,)}, received {load.shape}.")
        return replace(self, load_mw=load)

    def connected_components(self, outaged_branch_ids: Iterable[str] = ()) -> tuple[tuple[int, ...], ...]:
        outaged = set(outaged_branch_ids)
        adjacency = [set() for _ in range(self.n_buses)]
        for branch in self.branches:
            if branch.asset_id not in outaged:
                adjacency[branch.from_bus].add(branch.to_bus)
                adjacency[branch.to_bus].add(branch.from_bus)
        unseen = set(range(self.n_buses))
        islands: list[tuple[int, ...]] = []
        while unseen:
            root = min(unseen)
            stack = [root]
            component = []
            unseen.remove(root)
            while stack:
                node = stack.pop()
                component.append(node)
                neighbours = adjacency[node] & unseen
                unseen.difference_update(neighbours)
                stack.extend(neighbours)
            islands.append(tuple(sorted(component)))
        return tuple(islands)

    @staticmethod
    def _dispatch_island(
        buses: np.ndarray,
        load: FloatArray,
        generation: FloatArray,
        capacity: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        island_load = load[buses]
        island_generation = np.minimum(generation[buses], capacity[buses])
        island_capacity = capacity[buses]
        demand = float(np.sum(island_load))
        available_capacity = float(np.sum(island_capacity))
        served_total = min(demand, available_capacity)
        if demand > 0:
            served = island_load * (served_total / demand)
        else:
            served = np.zeros_like(island_load)
        shed = island_load - served

        base_total = float(np.sum(island_generation))
        if served_total <= 0 or available_capacity <= 0:
            dispatch = np.zeros_like(island_generation)
        elif base_total > served_total and base_total > 0:
            dispatch = island_generation * (served_total / base_total)
        else:
            headroom = np.maximum(island_capacity - island_generation, 0.0)
            needed = served_total - base_total
            if needed > 0 and headroom.sum() > 0:
                dispatch = island_generation + headroom * (needed / headroom.sum())
            elif base_total > 0:
                dispatch = island_generation * (served_total / base_total)
            else:
                dispatch = island_capacity * (served_total / available_capacity)
        correction = served_total - float(dispatch.sum())
        if abs(correction) > 1e-9 and dispatch.size:
            dispatch[int(np.argmax(island_capacity))] += correction
        return dispatch, served, shed

    def solve_dc(self, outaged_branch_ids: Iterable[str] = ()) -> PowerFlowState:
        """Balance islands, shed infeasible load, and solve linear DC flows."""

        outaged = set(outaged_branch_ids)
        unknown = outaged - set(self.branch_ids)
        if unknown:
            raise KeyError(f"Unknown branch ids: {sorted(unknown)}")
        islands = self.connected_components(outaged)
        dispatch = np.zeros(self.n_buses)
        served = np.zeros(self.n_buses)
        shed = np.zeros(self.n_buses)
        angles = np.zeros(self.n_buses)
        branch_flow: dict[str, float] = {branch.asset_id: 0.0 for branch in self.branches}
        branch_loading: dict[str, float] = {branch.asset_id: 0.0 for branch in self.branches}

        active_branches = [branch for branch in self.branches if branch.asset_id not in outaged]
        for island in islands:
            buses = np.asarray(island, dtype=int)
            local_dispatch, local_served, local_shed = self._dispatch_island(
                buses, self.load_mw, self.generation_mw, self.max_generation_mw
            )
            dispatch[buses] = local_dispatch
            served[buses] = local_served
            shed[buses] = local_shed
            if len(island) == 1:
                continue
            local_index = {bus: index for index, bus in enumerate(island)}
            susceptance = np.zeros((len(island), len(island)))
            for branch in active_branches:
                if branch.from_bus in local_index and branch.to_bus in local_index:
                    left, right = local_index[branch.from_bus], local_index[branch.to_bus]
                    value = 1.0 / branch.reactance_pu
                    susceptance[left, left] += value
                    susceptance[right, right] += value
                    susceptance[left, right] -= value
                    susceptance[right, left] -= value
            injection = dispatch[buses] - served[buses]
            injection[0] -= float(injection.sum())
            try:
                local_angles = np.zeros(len(island))
                local_angles[1:] = np.linalg.solve(susceptance[1:, 1:], injection[1:])
            except np.linalg.LinAlgError as error:
                raise RuntimeError(f"Singular DC system inside island {island}.") from error
            angles[buses] = local_angles

        for branch in active_branches:
            flow = (angles[branch.from_bus] - angles[branch.to_bus]) / branch.reactance_pu
            branch_flow[branch.asset_id] = float(flow)
            branch_loading[branch.asset_id] = float(abs(flow) / branch.rating_mw)
        return PowerFlowState(
            bus_angles_rad=angles,
            branch_flow_mw=branch_flow,
            branch_loading_ratio=branch_loading,
            generation_mw=dispatch,
            served_load_mw=served,
            load_shed_mw=shed,
            islands=islands,
        )

    @classmethod
    def from_pandapower(cls, net) -> "DCNetwork":
        """Create a DC approximation from a pandapower network-like object.

        This adapter intentionally avoids importing pandapower, so the core
        package remains usable in environments where the existing PROPER data
        loader provides ``net`` but pandapower is not installed separately.
        Transformer and current ratings are interpreted as MW limits under a
        unity-power-factor approximation and must be reviewed for production.
        """

        bus_indices = list(net.bus.index)
        mapping = {external: internal for internal, external in enumerate(bus_indices)}
        n_buses = len(bus_indices)
        base_mva = float(getattr(net, "sn_mva", 100.0))
        load = np.zeros(n_buses)
        for _, row in net.load.iterrows():
            if bool(row.get("in_service", True)):
                load[mapping[row["bus"]]] += max(float(row.get("p_mw", 0.0)), 0.0)
        generation = np.zeros(n_buses)
        capacity = np.zeros(n_buses)
        for table_name in ("gen", "sgen"):
            table = getattr(net, table_name, None)
            if table is None:
                continue
            for _, row in table.iterrows():
                if bool(row.get("in_service", True)):
                    bus = mapping[row["bus"]]
                    power = max(float(row.get("p_mw", 0.0)), 0.0)
                    maximum = max(float(row.get("max_p_mw", power)), power)
                    generation[bus] += power
                    capacity[bus] += maximum
        external_grid = getattr(net, "ext_grid", None)
        ext_buses = []
        if external_grid is not None:
            ext_buses = [
                mapping[row["bus"]]
                for _, row in external_grid.iterrows()
                if bool(row.get("in_service", True))
            ]
        residual = max(float(load.sum() - generation.sum()), 0.0)
        if ext_buses:
            for bus in ext_buses:
                generation[bus] += residual / len(ext_buses)
                capacity[bus] += max(float(load.sum()), 1.0) * 2.0 / len(ext_buses)
        elif capacity.sum() <= 0:
            generation[0] = float(load.sum())
            capacity[0] = max(float(load.sum()) * 2.0, 1.0)
        capacity = np.maximum(capacity, generation)

        branches: list[Branch] = []
        line_table = getattr(net, "line", None)
        if line_table is not None:
            for index, row in line_table.iterrows():
                if not bool(row.get("in_service", True)):
                    continue
                from_external, to_external = row["from_bus"], row["to_bus"]
                voltage_kv = float(net.bus.loc[from_external, "vn_kv"])
                length = float(row.get("length_km", 1.0))
                parallel = max(float(row.get("parallel", 1.0)), 1.0)
                x_ohm = float(row.get("x_ohm_per_km", 0.1)) * length / parallel
                base_impedance = voltage_kv * voltage_kv / base_mva
                reactance_pu = max(x_ohm / base_impedance, 1e-8)
                max_current_ka = float(row.get("max_i_ka", np.nan))
                derating = float(row.get("df", 1.0))
                rating = (
                    np.sqrt(3.0) * voltage_kv * max_current_ka * derating * parallel
                    if np.isfinite(max_current_ka) and max_current_ka > 0
                    else max(float(load.sum()), 1.0)
                )
                branches.append(
                    Branch(
                        asset_id=f"line:{index}",
                        from_bus=mapping[from_external],
                        to_bus=mapping[to_external],
                        reactance_pu=reactance_pu,
                        rating_mw=float(rating),
                        component_kind="single_line" if parallel <= 1 else "multiple_line",
                        length_km=length,
                        metadata={"pandapower_index": int(index), "rating_approximation": "MVA treated as MW"},
                    )
                )
        trafo_table = getattr(net, "trafo", None)
        if trafo_table is not None:
            for index, row in trafo_table.iterrows():
                if not bool(row.get("in_service", True)):
                    continue
                parallel = max(float(row.get("parallel", 1.0)), 1.0)
                nominal = max(float(row.get("sn_mva", base_mva)), 1e-8)
                reactance_pu = max(float(row.get("vk_percent", 10.0)) / 100.0 * base_mva / nominal / parallel, 1e-8)
                rating = nominal * parallel * float(row.get("df", 1.0))
                branches.append(
                    Branch(
                        asset_id=f"trafo:{index}",
                        from_bus=mapping[row["hv_bus"]],
                        to_bus=mapping[row["lv_bus"]],
                        reactance_pu=reactance_pu,
                        rating_mw=rating,
                        component_kind="transformer",
                        metadata={"pandapower_index": int(index), "rating_approximation": "MVA treated as MW"},
                    )
                )
        if not branches:
            raise ValueError("The pandapower object contains no in-service lines or transformers.")
        names = tuple(str(net.bus.loc[index].get("name", index)) for index in bus_indices)
        return cls(n_buses, tuple(branches), load, generation, capacity, names, base_mva)

