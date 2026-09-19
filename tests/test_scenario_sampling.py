from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scheduler_clustered.outage_clusters import OutageCluster
from scheduler_clustered.scenario_sampling import (
    EmpiricalClusterScenarioSampler,
    demand_fingerprint,
)


@pytest.fixture
def sampling_data() -> dict:
    times = [f"t{i}" for i in range(8)]
    demand = pd.DataFrame(
        [
            [10.0 + i, 20.0 + 2 * i, 5.0 + 0.5 * i]
            for i in range(len(times))
        ],
        index=times,
        columns=["b0", "b1", "b2"],
    )
    return {"T": times, "nodal_demand": demand}


@pytest.fixture
def full_cluster() -> OutageCluster:
    times = tuple(f"t{i}" for i in range(8))
    return OutageCluster("cluster_001", 0, 8, times, ("line_0",))


@pytest.mark.unit
def test_stratified_scenarios_are_reproducible_and_probability_weighted(
    sampling_data: dict,
    full_cluster: OutageCluster,
) -> None:
    first = EmpiricalClusterScenarioSampler(
        sampling_data,
        scenario_count=8,
        seed=42,
        global_lognormal_sigma=0.0,
        sampling_scheme="stratified_tail",
    )
    second = EmpiricalClusterScenarioSampler(
        sampling_data,
        scenario_count=8,
        seed=999,
        global_lognormal_sigma=0.0,
        sampling_scheme="stratified_tail",
    )
    assert first.scenario_ids == second.scenario_ids
    assert sum(first.probability_mapping().values()) == pytest.approx(1.0)
    assert sorted(first.probabilities.tolist()) == pytest.approx(
        sorted([0.20] * 4 + [0.075] * 2 + [0.025] * 2)
    )

    first_scenarios = first.scenarios_for_cluster(full_cluster)
    second_scenarios = second.scenarios_for_cluster(full_cluster)
    assert [s.source_time for s in first_scenarios] == [s.source_time for s in second_scenarios]
    for scenario in first_scenarios:
        expected = sampling_data["nodal_demand"].loc[scenario.source_time].to_numpy()
        np.testing.assert_allclose(scenario.demand, expected)


@pytest.mark.unit
def test_random_sampler_uses_common_seed(sampling_data: dict, full_cluster: OutageCluster) -> None:
    a = EmpiricalClusterScenarioSampler(
        sampling_data,
        scenario_count=6,
        seed=11,
        global_lognormal_sigma=0.05,
        sampling_scheme="random",
    ).scenarios_for_cluster(full_cluster)
    b = EmpiricalClusterScenarioSampler(
        sampling_data,
        scenario_count=6,
        seed=11,
        global_lognormal_sigma=0.05,
        sampling_scheme="random",
    ).scenarios_for_cluster(full_cluster)
    assert [s.source_time for s in a] == [s.source_time for s in b]
    np.testing.assert_allclose([s.global_multiplier for s in a], [s.global_multiplier for s in b])
    for left, right in zip(a, b):
        np.testing.assert_allclose(left.demand, right.demand)


@pytest.mark.unit
def test_fingerprint_is_stable_and_sensitive() -> None:
    demand = np.asarray([1.0, 2.0, 3.0])
    assert demand_fingerprint(demand) == demand_fingerprint(demand.copy())
    assert demand_fingerprint(demand) != demand_fingerprint(demand + np.asarray([0.0, 0.0, 1e-6]))


@pytest.mark.unit
def test_sampler_validation(sampling_data: dict, full_cluster: OutageCluster) -> None:
    with pytest.raises(ValueError, match="at least two"):
        EmpiricalClusterScenarioSampler(sampling_data, scenario_count=1)
    with pytest.raises(ValueError, match="negative"):
        EmpiricalClusterScenarioSampler(sampling_data, global_lognormal_sigma=-0.1)
    with pytest.raises(ValueError, match="sampling_scheme"):
        EmpiricalClusterScenarioSampler(sampling_data, sampling_scheme="latin")
    with pytest.raises(ValueError, match="probability vector"):
        EmpiricalClusterScenarioSampler(sampling_data, scenario_count=3, probabilities=[1.0, 1.0])

    empty = OutageCluster("empty", 0, 0, (), ())
    sampler = EmpiricalClusterScenarioSampler(sampling_data, scenario_count=2)
    with pytest.raises(ValueError, match="contains no periods"):
        sampler.scenarios_for_cluster(empty)
