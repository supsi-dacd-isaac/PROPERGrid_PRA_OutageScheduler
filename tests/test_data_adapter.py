from types import SimpleNamespace

import numpy as np
import pandas as pd

from optimizers.data_adapter import prepare_scos_data
from optimizers.scenario_generation import ScenarioBank


def test_prepare_scos_data_preserves_multiple_generators_per_bus() -> None:
    net = SimpleNamespace()
    net.gen = pd.DataFrame(
        {
            "bus": [0, 0],
            "min_p_mw": [0.0, 0.0],
            "max_p_mw": [100.0, 50.0],
        },
        index=[0, 1],
    )
    net._ppc = {
        "internal": {
            "Cft": np.array([[1.0, -1.0]]),
            "Bf": np.array([[10.0, -10.0]]),
        }
    }
    demand = pd.DataFrame([[60.0, 20.0], [70.0, 25.0]])
    data = {
        "network": net,
        "nodal_demand": demand,
        "outages": {
            "names": ["line_0", "gen_0"],
            "expected_duration_steps": [1, 1],
            "priorities": [1.0, 2.0],
            "cost_per_step": [1.0, 1.0],
        },
        "num_buses": 2,
        "num_branches": 1,
        "branch_capacity": [100.0],
        "max_number_of_maintenance_tasks": 1,
        "n_minus1_names": ["n1_line_0", "n1_gen_1"],
        "ref_buses": ["bus_0"],
    }
    bank = ScenarioBank.nominal(demand)
    prepared = prepare_scos_data(data, bank)
    assert prepared.bus_to_generators["bus_0"] == ["gen_0", "gen_1"]
    assert prepared.incidence.shape == (2, 1)
    assert prepared.b_flow.shape == (1, 2)
    assert list(prepared.scenario_demands["scenario_000"].columns) == ["bus_0", "bus_1"]
