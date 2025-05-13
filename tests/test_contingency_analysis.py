import os
import sys
import unittest
import numpy as np
import pandapower as pp

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pra_psa.core.contingency_analysis import (
    analyze_contingency,
    apply_reference_dispatch,
    apply_load,
    calculate_lodf_and_shift
)


class TestContingencyAnalysis(unittest.TestCase):
    """Test contingency analysis functions."""

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

    def test_apply_reference_dispatch(self):
        """Test applying reference dispatch to the network."""
        # Create reference dispatch
        p_mw = [80, 120]  # Different from current values
        q_mvar = [0, 0]
        
        # Apply reference dispatch
        apply_reference_dispatch(self.net, p_mw, q_mvar)
        
        # Check that the values were applied
        self.assertEqual(self.net.gen.p_mw.iloc[0], 80)
        self.assertEqual(self.net.gen.p_mw.iloc[1], 120)
        self.assertEqual(self.net.gen.min_p_mw.iloc[0], 80)
        self.assertEqual(self.net.gen.max_p_mw.iloc[0], 80)

    def test_apply_load(self):
        """Test applying load to the network."""
        # Create new load values
        p_load = [60, 70, 80]  # Different from current values
        
        # Apply load
        apply_load(self.net, p_load)
        
        # Check that the values were applied
        self.assertEqual(self.net.load.p_mw.iloc[0], 60)
        self.assertEqual(self.net.load.p_mw.iloc[1], 70)
        self.assertEqual(self.net.load.p_mw.iloc[2], 80)

    def test_analyze_contingency(self):
        """Test analyzing a contingency."""
        # Create a contingency
        contingency = {
            'element_type': 'line',
            'element_index': 0
        }
        
        # Create a copy of the network for testing
        test_net = self.net.deepcopy()
        
        # Verify line is in service before contingency
        self.assertTrue(test_net.line.in_service.iloc[0])
        
        # Apply contingency directly
        from pra_psa.core.contingency import apply_contingency
        apply_contingency(test_net, contingency)
        
        # Verify line is out of service
        self.assertFalse(test_net.line.in_service.iloc[0])
        
        # Analyze the contingency
        result = analyze_contingency(test_net, contingency)
        
        # Check that the result has the expected keys
        self.assertIn('success', result)
        self.assertIn('loading', result)
        self.assertIn('violations', result)
        self.assertIn('severity', result)


if __name__ == '__main__':
    unittest.main() 