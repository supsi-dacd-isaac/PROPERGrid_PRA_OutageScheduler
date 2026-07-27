"""Network data and PTDF/LODF utilities used by the clustered scheduler."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Topology:
    active_line_indices: tuple[int, ...]
    connected_components: tuple[tuple[int, ...], ...]


@dataclass
class SensitivityResult:
    active_line_indices: np.ndarray
    ptdf: np.ndarray
    lodf: np.ndarray
    islanding_contingencies: set[int]
    shift_flow: np.ndarray


class NetworkModel:
    """Dense DC network representation derived from PROPER data."""

    def __init__(self, data: Mapping):
        self.data = data
        self.names = data["names"]
        self.buses = list(self.names["buses"])
        self.lines = list(self.names["lines"])
        self.generators = list(self.names["generators"])
        self.bus_index = {name: index for index, name in enumerate(self.buses)}
        self.line_index = {name: index for index, name in enumerate(self.lines)}
        self.gen_index = {name: index for index, name in enumerate(self.generators)}

        self.cft = np.asarray(
            data.get("C_branch_bus", np.asarray(data["S"]).T), dtype=float
        )
        self.bf = np.asarray(data["B_mat"], dtype=float)
        self.base_mva = float(data.get("base_mva", 1.0))
        self.pfinj_pu = np.asarray(
            data.get("Pfinj", np.zeros(len(self.lines))), dtype=float
        ).reshape(-1)
        self.pfinj_mw = self.base_mva * self.pfinj_pu

        expected = (len(self.lines), len(self.buses))
        if self.cft.shape != expected:
            raise ValueError(f"C_branch_bus shape {self.cft.shape}; expected {expected}.")
        if self.bf.shape != expected:
            raise ValueError(f"B_mat shape {self.bf.shape}; expected {expected}.")
        if self.pfinj_pu.size != len(self.lines):
            raise ValueError(
                f"Pfinj length {self.pfinj_pu.size}; expected {len(self.lines)}."
            )

        self.limits = np.asarray(
            [float(data["f_lim"][line]) for line in self.lines], dtype=float
        )
        self.pmin = np.asarray(
            [float(data["p_min"][generator]) for generator in self.generators],
            dtype=float,
        )
        self.pmax = np.asarray(
            [float(data["p_max"][generator]) for generator in self.generators],
            dtype=float,
        )
        g2bus = list(data["g2bus"])
        if len(g2bus) != len(self.generators):
            raise ValueError("g2bus and generator names have inconsistent lengths.")
        self.generator_bus = np.asarray(
            [self.bus_index[bus] for bus in g2bus], dtype=int
        )

        references = list(data.get("ref_buses") or [self.buses[0]])
        self.preferred_reference = self.bus_index.get(references[0], 0)
        self.endpoints = tuple(
            self._endpoints(index) for index in range(len(self.lines))
        )

        self.generators_by_bus: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                generator_index
                for generator_index, bus_index in enumerate(self.generator_bus)
                if int(bus_index) == candidate_bus
            )
            for candidate_bus in range(len(self.buses))
        )

    def _endpoints(self, line_index: int) -> tuple[int, int]:
        nonzero = np.flatnonzero(np.abs(self.cft[line_index]) > 1e-12)
        if len(nonzero) != 2:
            raise ValueError(
                f"Branch {self.lines[line_index]} has {len(nonzero)} incidence "
                "entries; expected exactly two."
            )
        return int(nonzero[0]), int(nonzero[1])

    def active_line_indices(self, outaged_lines: Iterable[str]) -> tuple[int, ...]:
        unavailable = {
            self.line_index[line]
            for line in outaged_lines
            if line in self.line_index
        }
        return tuple(
            index for index in range(len(self.lines)) if index not in unavailable
        )

    def connected_components(
        self, active_indices: Sequence[int]
    ) -> tuple[tuple[int, ...], ...]:
        adjacency: list[list[int]] = [[] for _ in self.buses]
        for line_index in active_indices:
            first, second = self.endpoints[line_index]
            adjacency[first].append(second)
            adjacency[second].append(first)

        visited = np.zeros(len(self.buses), dtype=bool)
        components: list[tuple[int, ...]] = []
        for root in range(len(self.buses)):
            if visited[root]:
                continue
            stack = [root]
            visited[root] = True
            nodes: list[int] = []
            while stack:
                node = stack.pop()
                nodes.append(node)
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(tuple(sorted(nodes)))
        return tuple(components)

    @lru_cache(maxsize=1024)
    def topology(self, active_indices: tuple[int, ...]) -> Topology:
        return Topology(
            active_line_indices=active_indices,
            connected_components=self.connected_components(active_indices),
        )

    @lru_cache(maxsize=512)
    def sensitivities(self, active_indices: tuple[int, ...]) -> SensitivityResult:
        active = np.asarray(active_indices, dtype=int)
        n_active = len(active)
        n_buses = len(self.buses)
        ptdf = np.zeros((n_active, n_buses), dtype=float)
        topology = self.topology(active_indices)

        cft_active = self.cft[active, :]
        bf_active = self.bf[active, :]
        bbus = cft_active.T @ bf_active
        position = {line: local for local, line in enumerate(active_indices)}

        for component in topology.connected_components:
            nodes = list(component)
            component_lines = [
                line
                for line in active_indices
                if self.endpoints[line][0] in component
                and self.endpoints[line][1] in component
            ]
            if len(nodes) <= 1 or not component_lines:
                continue
            reference = (
                self.preferred_reference
                if self.preferred_reference in component
                else nodes[0]
            )
            nonreference = [node for node in nodes if node != reference]
            reduced = bbus[np.ix_(nonreference, nonreference)]
            try:
                inverse = np.linalg.inv(reduced)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(reduced, rcond=1e-10)
            rows = [position[line] for line in component_lines]
            local_bf = self.bf[np.ix_(component_lines, nonreference)]
            ptdf[np.ix_(rows, nonreference)] = local_bf @ inverse

        pfinj_active = self.pfinj_pu[active]
        pbusinj_active = cft_active.T @ pfinj_active
        shift_flow = self.base_mva * (pfinj_active - ptdf @ pbusinj_active)

        transfer = ptdf @ cft_active.T
        diagonal = np.diag(transfer)
        lodf = np.full_like(transfer, np.nan, dtype=float)
        islanding: set[int] = set()
        for local_index in range(n_active):
            denominator = 1.0 - diagonal[local_index]
            if abs(denominator) <= 1e-8:
                islanding.add(int(active[local_index]))
                continue
            lodf[:, local_index] = transfer[:, local_index] / denominator
            lodf[local_index, local_index] = -1.0

        return SensitivityResult(
            active_line_indices=active,
            ptdf=ptdf,
            lodf=lodf,
            islanding_contingencies=islanding,
            shift_flow=shift_flow,
        )

    def branch_screening_scores(
        self,
        active_indices: tuple[int, ...],
        injections: np.ndarray,
    ) -> dict[str, float]:
        """Estimate the maximum post-contingency loading ratio per line outage."""
        sensitivity = self.sensitivities(active_indices)
        base_flow = (
            sensitivity.ptdf @ np.asarray(injections, dtype=float)
            + sensitivity.shift_flow
        )
        active = sensitivity.active_line_indices
        active_limits = self.limits[active]
        scores: dict[str, float] = {}

        for local_index, global_index in enumerate(active):
            line = self.lines[int(global_index)]
            if int(global_index) in sensitivity.islanding_contingencies:
                scores[line] = float("inf")
                continue
            post_contingency = (
                base_flow
                + sensitivity.lodf[:, local_index] * base_flow[local_index]
            )
            ratios = np.divide(
                np.abs(post_contingency),
                active_limits,
                out=np.full_like(post_contingency, np.inf),
                where=active_limits > 0,
            )
            scores[line] = float(np.max(ratios))
        return scores
