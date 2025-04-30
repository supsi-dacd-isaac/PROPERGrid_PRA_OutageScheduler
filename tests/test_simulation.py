import os
import sys
import unittest
import numpy as np
import pandapower as pp

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pra_psa.simulation.monte_carlo import monte_carlo_simulation
from pra_psa.simulation.importance_sampling import importance_sampling
from pra_psa.simulation.mcmc_sampling import mcmc_sampling
from pra_psa.simulation.subset_simulation import subset_simulation
from pra_psa.core.contingency import generate_n1_contingencies


class TestSimulation(unittest.TestCase):
    """Test simulation functions."""

    def setUp(self):
        """Set up a test network for each test."""
        self.net = pp.create_empty_network()
        
        # Create buses
        bus1 = pp.create_bus(self.net, vn_kv=110)
        bus2 = pp.create_bus(self.net, vn_kv=110)
        bus3 = pp.create_bus(self.net, vn_kv=110)
        
        # Create lines
        pp.create_line(self.net, bus1, bus2, length_km=10, std_type="149-AL1/24-ST1A 110.0")
        pp.create_line(self.net, bus2, bus3, length_km=10, std_type="149-AL1/24-ST1A 110.0")
        pp.create_line(self.net, bus3, bus1, length_km=10, std_type="149-AL1/24-ST1A 110.0")
        
        # Create generators (first one is slack)
        pp.create_gen(self.net, bus1, p_mw=100, vm_pu=1.0, slack=True)
        pp.create_gen(self.net, bus2, p_mw=100, vm_pu=1.0)
        
        # Create loads
        pp.create_load(self.net, bus1, p_mw=50)
        pp.create_load(self.net, bus2, p_mw=50)
        pp.create_load(self.net, bus3, p_mw=50)
        
        # Run power flow to initialize
        pp.runpp(self.net)
        
        # Create contingency set
        self.contingency_set = generate_n1_contingencies(3, element_type="line")  # 3 lines
        
        # Define a simple score function
        def score_fn(net):
            try:
                pp.runpp(net)
                return max(net.res_line.loading_percent)
            except:
                return float('inf')
        
        self.score_fn = score_fn

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        # Run Monte Carlo simulation with a small number of samples
        results = monte_carlo_simulation(self.net, self.contingency_set, n_runs=2)
        
        # Check that results were returned
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

    def test_importance_sampling(self):
        """Test importance sampling."""
        # Create probabilities for each contingency
        probabilities = np.array([0.1, 0.2, 0.7])  # Sum to 1
        
        # Run importance sampling with a small number of samples
        results = importance_sampling(self.net, self.contingency_set, probabilities, n_samples=2)
        
        # Check that results were returned
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

    def test_mcmc_sampling(self):
        """Test MCMC sampling."""
        # Run MCMC sampling with a small number of samples
        rare_events, thresholds = mcmc_sampling(self.net, self.contingency_set, self.score_fn, n_samples=2)
        
        # Check that results were returned
        self.assertIsNotNone(rare_events)
        self.assertIsNotNone(thresholds)
        self.assertGreater(len(thresholds), 0)

    def test_subset_simulation(self):
        """Test subset simulation."""
        # Run subset simulation with a small number of samples
        rare_events, thresholds = subset_simulation(self.net, self.contingency_set, self.score_fn, n_samples=2)
        
        # Check that results were returned
        self.assertIsNotNone(rare_events)
        self.assertIsNotNone(thresholds)
        self.assertGreater(len(thresholds), 0)


if __name__ == '__main__':
    unittest.main() 