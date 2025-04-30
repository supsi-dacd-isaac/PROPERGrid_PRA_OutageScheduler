# tests/test_monte_carlo.py
from pra_psa.simulation.monte_carlo import run_monte_carlo
from pra_psa.core.contingency import generate_n1_contingencies
from pandapower.networks import case14

def test_monte_carlo_runs():
    net = case14()
    contingencies = generate_n1_contingencies(net)

    def dummy_eval(net):
        return net.res_bus.vm_pu.values.mean()

    results = run_monte_carlo(net, contingencies, dummy_eval, n_runs=5)
    assert len(results) == 5
