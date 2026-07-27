"""Construction of constant-topology outage clusters.

An outage cluster is a maximal consecutive interval over which the set of
planned unavailable assets is unchanged. Empty intervals are omitted by
default because they do not depend on the maintenance schedule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OutageCluster:
    """Maximal consecutive interval with one constant planned-outage set."""

    cluster_id: str
    start_index: int
    end_index: int  # exclusive
    times: tuple[str, ...]
    active_outages: tuple[str, ...]

    @property
    def duration_steps(self) -> int:
        return self.end_index - self.start_index

    @property
    def start_time(self) -> str:
        return self.times[0]

    @property
    def end_time(self) -> str:
        return self.times[-1]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["duration_steps"] = self.duration_steps
        return result


def build_outage_clusters(
    times: Sequence[str],
    active_outages_by_time: Mapping[str, Sequence[str]],
    *,
    include_empty: bool = False,
) -> list[OutageCluster]:
    """Return maximal consecutive intervals with identical outage sets.

    Parameters
    ----------
    times:
        Ordered planning periods.
    active_outages_by_time:
        Mapping from time label to the planned outages active in that period.
    include_empty:
        If true, include intervals with no planned outages. The deterministic
        decomposition normally excludes them because they are schedule
        independent and can be handled through baseline diagnostics.
    """
    ordered_times = list(times)
    if not ordered_times:
        return []

    normalized: list[tuple[str, ...]] = []
    for time in ordered_times:
        if time not in active_outages_by_time:
            raise KeyError(f"Missing active-outage set for time {time!r}.")
        normalized.append(tuple(sorted(set(active_outages_by_time[time]))))

    clusters: list[OutageCluster] = []
    start = 0
    current = normalized[0]

    def append_cluster(first: int, last_exclusive: int, outages: tuple[str, ...]) -> None:
        if not outages and not include_empty:
            return
        index = len(clusters) + 1
        clusters.append(
            OutageCluster(
                cluster_id=f"cluster_{index:03d}",
                start_index=first,
                end_index=last_exclusive,
                times=tuple(ordered_times[first:last_exclusive]),
                active_outages=outages,
            )
        )

    for index in range(1, len(ordered_times)):
        if normalized[index] != current:
            append_cluster(start, index, current)
            start = index
            current = normalized[index]
    append_cluster(start, len(ordered_times), current)
    return clusters
