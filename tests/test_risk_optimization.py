import os
import sys
import pytest
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from unittest.mock import patch, MagicMock

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.gurobi_SCOS import run_SCOS_cvar
from optimizers.utils_and_constraints import (
    add_planned_outages_constraints,
    add_generators_constraints,
    add_line_power_limit_constraints,
    add_nodal_power_balance_constraints,
    add_line_dc_power_flow_constraints,
    calculate_nodal_balance
)

@pytest.fixture
def sample_data():
    """Create sample data for testing risk-aware optimization"""
    T = pd.date_range('2024-01-01', periods=24, freq='H')
    return {
        'T': T,
        'generators': ['gen_1', 'gen_2'],
        'lines': ['line_1', 'line_2'],
        'buses': ['bus_1', 'bus_2'],
        'outages': ['outage_1', 'outage_2'],
        'gen_capacity': pd.Series({'gen_1': 100.0, 'gen_2': 80.0}),
        'gen_min_output': pd.Series({'gen_1': 20.0, 'gen_2': 10.0}),
        'line_capacity': pd.Series({'line_1': 50.0, 'line_2': 40.0}),
        'demand': pd.DataFrame({
            'bus_1': np.random.uniform(30, 70, size=24),
            'bus_2': np.random.uniform(20, 50, size=24)
        }, index=T),
        'gen_bus': pd.Series({'gen_1': 'bus_1', 'gen_2': 'bus_2'}),
        'line_from': pd.Series({'line_1': 'bus_1', 'line_2': 'bus_2'}),
        'line_to': pd.Series({'line_1': 'bus_2', 'line_2': 'bus_1'}),
        'line_reactance': pd.Series({'line_1': 0.1, 'line_2': 0.1}),
        'outage_duration': pd.Series({'outage_1': 48, 'outage_2': 72}),
        'outage_window': {
            'outage_1': (0, 168),  # One week window
            'outage_2': (24, 192)  # One week window starting day 2
        },
        'risk_params': {
            'alpha': 0.95,  # Confidence level
            'beta': 0.5,    # Risk aversion parameter
            'n_scenarios': 10
        }
    }

def test_scenario_generation(sample_data):
    """Test scenario generation for risk-aware optimization"""
    n_scenarios = 10
    scenarios = generate_scenarios(sample_data, n_scenarios)

    # Check scenario structure
    assert isinstance(scenarios, dict)
    assert 'demand' in scenarios
    assert 'probabilities' in scenarios

    # Check number of scenarios
    assert len(scenarios['probabilities']) == n_scenarios
    assert scenarios['demand'].shape[1] == n_scenarios

    # Check probabilities sum to 1
    assert abs(sum(scenarios['probabilities']) - 1.0) < 1e-6

    # Check demand values are reasonable
    assert (scenarios['demand'] >= 0).all()
    assert (scenarios['demand'] <= sample_data['gen_capacity'].sum()).all()

def test_cvar_constraints():
    """Test adding CVaR constraints to the model"""
    # Create a simple test case
    model = gp.Model("cvar_test")
    model.Params.OutputFlag = 0

    # Test data
    n_scenarios = 3
    n_time = 2

    # Create variables
    vars = {
        'eta': model.addVar(name='eta'),  # VaR variable
        'z': [model.addVar(name=f'z_{s}') for s in range(n_scenarios)],  # CVaR auxiliary variables
        'loss': pd.DataFrame(
            [[model.addVar(name=f'loss_{t}_{s}')
              for s in range(n_scenarios)]
             for t in range(n_time)]
        )
    }

    # Test data
    alpha = 0.95
    beta = 0.5
    probabilities = [1/n_scenarios] * n_scenarios

    # Add CVaR constraints
    add_cvar_constraints(model, vars['eta'], vars['z'], vars['loss'],
                        alpha, probabilities)

    # Set some test values for loss variables
    for t in range(n_time):
        for s in range(n_scenarios):
            vars['loss'].iloc[t, s].lb = float(t + s)  # Set lower bounds for testing

    # Set objective (minimize CVaR)
    obj = vars['eta'] + (1/(1-alpha)) * sum(p * z
                                           for p, z in zip(probabilities, vars['z']))
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Check if model is feasible
    assert model.Status == GRB.OPTIMAL

    # Check CVaR properties
    eta_val = vars['eta'].X
    z_vals = [z.X for z in vars['z']]

    # VaR should be between min and max losses
    min_loss = min(vars['loss'].iloc[t, s].lb
                  for t in range(n_time)
                  for s in range(n_scenarios))
    max_loss = max(vars['loss'].iloc[t, s].lb
                  for t in range(n_time)
                  for s in range(n_scenarios))
    assert min_loss <= eta_val <= max_loss

    # All z values should be non-negative
    assert all(z >= 0 for z in z_vals)

def test_risk_aware_optimization(sample_data):
    """Test the complete risk-aware optimization"""
    # Run optimization
    result = run_SCOS_cvar(sample_data)

    # Check result structure
    assert isinstance(result, dict)
    assert 'outage_schedule' in result
    assert 'generation' in result
    assert 'line_power' in result
    assert 'angles' in result
    assert 'cvar' in result
    assert 'var' in result
    assert 'scenarios' in result

    # Check dimensions
    n_time = len(sample_data['T'])
    n_outages = len(sample_data['outages'])
    n_gens = len(sample_data['generators'])
    n_lines = len(sample_data['lines'])
    n_buses = len(sample_data['buses'])

    assert result['outage_schedule'].shape == (n_time, n_outages)
    assert result['generation'].shape == (n_time, n_gens)
    assert result['line_power'].shape == (n_time, n_lines)
    assert result['angles'].shape == (n_time, n_buses)

    # Check variable bounds
    assert (result['outage_schedule'] >= 0).all()
    assert (result['outage_schedule'] <= 1).all()
    assert (result['generation'] >= 0).all()
    assert (result['generation'] <= sample_data['gen_capacity'].max()).all()
    assert (abs(result['line_power']) <= sample_data['line_capacity'].max()).all()

    # Check risk measures
    assert result['var'] <= result['cvar']  # VaR should be less than or equal to CVaR
    assert result['cvar'] >= 0  # CVaR should be non-negative for our loss function

def test_risk_parameter_sensitivity():
    """Test sensitivity of the optimization to risk parameters"""
    # Create a simple test case
    data = {
        'T': pd.date_range('2024-01-01', periods=2, freq='H'),
        'generators': ['gen_1'],
        'lines': ['line_1'],
        'buses': ['bus_1', 'bus_2'],
        'gen_capacity': pd.Series({'gen_1': 100.0}),
        'gen_min_output': pd.Series({'gen_1': 0.0}),
        'line_capacity': pd.Series({'line_1': 50.0}),
        'demand': pd.DataFrame({
            'bus_1': [0.0, 0.0],
            'bus_2': [30.0, 40.0]
        }),
        'gen_bus': pd.Series({'gen_1': 'bus_1'}),
        'line_from': pd.Series({'line_1': 'bus_1'}),
        'line_to': pd.Series({'line_1': 'bus_2'}),
        'line_reactance': pd.Series({'line_1': 0.1})
    }

    # Test different risk parameters
    alphas = [0.9, 0.95]
    betas = [0.1, 0.9]
    results = {}

    for alpha in alphas:
        for beta in betas:
            key = (alpha, beta)
            results[key] = run_SCOS_cvar(data)

    # Compare results
    # More risk-averse parameters should lead to more conservative solutions
    high_risk = results[(alphas[0], betas[0])]
    low_risk = results[(alphas[1], betas[1])]

    assert low_risk['cvar'] <= high_risk['cvar']  # Lower CVaR for more conservative case

@pytest.mark.unit
class TestRiskOptimization:
    """Test suite for risk-aware optimization functions"""

    def test_cvar_scos_gurobi_initialization(self, sample_network_data, sample_demand_data, 
                                           sample_outage_data, mock_gurobi_model):
        """Test initialization of CVaR SCOS model"""
        with patch('optimizers.gurobi_SCOS.prepare_SCOS_data') as mock_prepare:
            mock_prepare.return_value = (2, [100, 100], [0, 0], [100, 100],
                                        ['gen_1', 'gen_2'], {'gen_1': 'bus_1', 'gen_2': 'bus_2'},
                                        np.eye(2), np.eye(2), np.eye(2), [1, 1], [1, 1], [100, 100])

            result = run_SCOS_cvar(sample_network_data)

            # Check if model was created with correct name
            mock_gurobi_model.assert_called_with("Transmission_Outage_Scheduling_CVaR")

            # Check if parameters were set
            assert mock_gurobi_model.setParam.call_count > 0

    @pytest.mark.gurobi
    def test_cvar_scos_gurobi_error_handling(self, sample_network_data):
        """Test error handling in CVaR SCOS model"""
        with patch('gurobipy.Model', side_effect=gp.GurobiError(10001, "Test error")):
            with pytest.raises(gp.GurobiError) as exc_info:
                run_SCOS_cvar(sample_network_data)
            assert "Test error" in str(exc_info.value)

    def test_add_planned_outages_constraints(self, mock_gurobi_model, sample_network_data, 
                                           sample_outage_data):
        """Test adding planned outages constraints"""
        add_planned_outages_constraints(
            mock_gurobi_model,
            sample_network_data['outages'],
            sample_outage_data['outage_window'],
            sample_outage_data['outage_duration']
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_generators_constraints(self, mock_gurobi_model, sample_network_data):
        """Test adding generator constraints"""
        add_generators_constraints(
            mock_gurobi_model,
            sample_network_data['generators'],
            sample_network_data['gen_capacity'],
            sample_network_data['gen_min_output']
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_line_power_limit_constraints(self, mock_gurobi_model, sample_network_data):
        """Test adding line power limit constraints"""
        add_line_power_limit_constraints(
            mock_gurobi_model,
            sample_network_data['lines'],
            sample_network_data['line_capacity']
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_nodal_power_balance_constraints(self, mock_gurobi_model, sample_network_data):
        """Test adding nodal power balance constraints"""
        add_nodal_power_balance_constraints(
            mock_gurobi_model,
            sample_network_data['buses'],
            sample_network_data['generators'],
            sample_network_data['gen_bus']
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_line_dc_power_flow_constraints(self, mock_gurobi_model, sample_network_data):
        """Test adding DC power flow constraints"""
        add_line_dc_power_flow_constraints(
            mock_gurobi_model,
            sample_network_data['lines'],
            sample_network_data['line_from'],
            sample_network_data['line_to'],
            sample_network_data['line_reactance']
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_calculate_nodal_balance(self, sample_network_data):
        """Test calculating nodal balance"""
        balance = calculate_nodal_balance(
            sample_network_data['buses'],
            sample_network_data['generators'],
            sample_network_data['gen_bus']
        )
        assert isinstance(balance, dict)
        assert all(bus in balance for bus in sample_network_data['buses'])

if __name__ == '__main__':
    pytest.main([__file__])
