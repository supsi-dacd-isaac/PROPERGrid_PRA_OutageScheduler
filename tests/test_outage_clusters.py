from scheduler_clustered.outage_clusters import build_outage_clusters


def test_static_topology_clusters_match_overlap_example():
    times = [f"m{i}" for i in range(12)]
    active = {time: () for time in times}
    # A: May--October => indices 4..9
    for i in range(4, 10):
        active[times[i]] = ("A",)
    # B: August--November => indices 7..10
    for i in range(7, 11):
        active[times[i]] = tuple(sorted((*active[times[i]], "B")))

    clusters = build_outage_clusters(times, active, include_empty=True)
    assert [cluster.active_outages for cluster in clusters] == [
        (),
        ("A",),
        ("A", "B"),
        ("B",),
        (),
    ]
    assert [(c.start_index, c.end_index) for c in clusters] == [
        (0, 4),
        (4, 7),
        (7, 10),
        (10, 11),
        (11, 12),
    ]
