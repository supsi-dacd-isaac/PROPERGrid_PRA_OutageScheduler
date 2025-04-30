# tests/test_subset_simulation.py
import pytest
from pandapower.networks import case14
from pra_psa.core.contingency import generate_n1_contingencies
from pra_psa.simulation.subset_simulation import subset_simulation

def test_subset_simulation_runs():
    net = case14()
    contingencies = generate_n1_contingencies(net)

    def score_fn(net):
        return max(abs(net.res_line.loading_percent.values))

    events, thresholds = subset_simulation(net, contingencies, score_fn, n_samples=10, max_levels=2)

    assert isinstance(events, list)
    assert isinstance(thresholds, list)
    assert all(isinstance(t, (int, float)) for t in thresholds)