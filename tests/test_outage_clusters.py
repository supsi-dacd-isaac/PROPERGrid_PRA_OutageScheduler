from __future__ import annotations

import pytest

from scheduler_clustered.outage_clusters import build_outage_clusters


@pytest.mark.unit
def test_builds_maximal_constant_topology_intervals() -> None:
    times = [f"t{i}" for i in range(6)]
    active = {
        "t0": [],
        "t1": ["line_2", "line_1", "line_1"],
        "t2": ["line_1", "line_2"],
        "t3": ["line_3"],
        "t4": ["line_3"],
        "t5": [],
    }
    clusters = build_outage_clusters(times, active)
    assert len(clusters) == 2
    assert clusters[0].cluster_id == "cluster_001"
    assert clusters[0].times == ("t1", "t2")
    assert clusters[0].active_outages == ("line_1", "line_2")
    assert clusters[0].duration_steps == 2
    assert clusters[1].times == ("t3", "t4")


@pytest.mark.unit
def test_empty_intervals_can_be_retained() -> None:
    times = ["t0", "t1", "t2"]
    active = {"t0": [], "t1": ["A"], "t2": []}
    clusters = build_outage_clusters(times, active, include_empty=True)
    assert [cluster.active_outages for cluster in clusters] == [(), ("A",), ()]
    assert [cluster.cluster_id for cluster in clusters] == [
        "cluster_001",
        "cluster_002",
        "cluster_003",
    ]


@pytest.mark.unit
def test_empty_horizon_returns_no_clusters() -> None:
    assert build_outage_clusters([], {}) == []


@pytest.mark.unit
def test_missing_time_mapping_is_reported() -> None:
    with pytest.raises(KeyError, match="t1"):
        build_outage_clusters(["t0", "t1"], {"t0": []})
