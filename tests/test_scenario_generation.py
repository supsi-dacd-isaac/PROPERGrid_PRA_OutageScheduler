import numpy as np
import pandas as pd

from optimizers.config import ScenarioConfig
from optimizers.scenario_generation import ScenarioBank, generate_demand_scenarios


def make_demand() -> pd.DataFrame:
    return pd.DataFrame(
        [[100.0, 50.0], [110.0, 55.0], [90.0, 45.0]],
        columns=["bus_0", "bus_1"],
    )


def test_nominal_bank() -> None:
    demand = make_demand()
    bank = ScenarioBank.nominal(demand)
    bank.validate(reference=demand)
    assert bank.probabilities == {"scenario_000": 1.0}


def test_generation_is_reproducible_and_positive() -> None:
    demand = make_demand()
    config = ScenarioConfig(n_scenarios=8, seed=123, sampling_scheme="stratified_tail")
    first = generate_demand_scenarios(demand, config)
    second = generate_demand_scenarios(demand, config)
    first.validate(reference=demand)
    assert np.isclose(sum(first.probabilities.values()), 1.0)
    for label in first.labels:
        assert np.allclose(first.demands[label], second.demands[label])
        assert (first.demands[label].to_numpy() >= 0).all()


def test_sampler_returns_exact_requested_count() -> None:
    bank = generate_demand_scenarios(make_demand(), ScenarioConfig(n_scenarios=7, seed=9))
    assert len(bank.demands) == 7
