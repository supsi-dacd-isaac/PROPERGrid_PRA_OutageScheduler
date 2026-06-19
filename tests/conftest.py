import os
import sys
import pytest
import pandapower as pp
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory"""
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def sample_time_periods():
    """Return sample time periods for testing"""
    return pd.date_range('2024-01-01', periods=24, freq='h')

@pytest.fixture(scope="session")
def sample_network_data():
    """Return sample network data for testing"""
    T = ['t1', 't2']
    return {
        'names': {
            'outages': ['outage_1', 'outage_2'],
            'lines': ['line_1', 'line_2'],
            'contingencies': ['cont_1', 'cont_2'],
            'buses': ['bus_1', 'bus_2'],
            'generators': ['gen_1', 'gen_2'],
            'demand_scenarios': ['scenario_1', 'scenario_2']
        },
        'T': T,
        'VoLL': 1000,
        'tail_prob_constraint': 0.95,
        'n_samples': 2,
        'use_DC_PF': True,
        'nodal_demand': pd.DataFrame({
            'bus_1': [100, 120],
            'bus_2': [80, 90]
        }, index=T),
        'network': MagicMock(),
        'max_number_of_maintenance_tasks': 2,
        'outages': {
            'indices': [1, 2],
            'names': ['outage_1', 'outage_2'],
            'type': ['line', 'line'],
            'expected_duration_steps': [1, 1],
            'cost_per_step': [100, 100],
            'priorities': [1, 1]
        },
        'generators': ['gen_1', 'gen_2'],
        'lines': ['line_1', 'line_2'],
        'buses': ['bus_1', 'bus_2'],
        'gen_capacity': pd.Series({'gen_1': 100.0, 'gen_2': 80.0}),
        'gen_min_output': pd.Series({'gen_1': 20.0, 'gen_2': 10.0}),
        'line_capacity': pd.Series({'line_1': 50.0, 'line_2': 40.0}),
        'gen_bus': pd.Series({'gen_1': 'bus_1', 'gen_2': 'bus_2'}),
        'line_from': pd.Series({'line_1': 'bus_1', 'line_2': 'bus_2'}),
        'line_to': pd.Series({'line_1': 'bus_2', 'line_2': 'bus_1'}),
        'line_reactance': pd.Series({'line_1': 0.1, 'line_2': 0.1})
    }

@pytest.fixture
def simple_network():
    """Create a simple test network"""
    net = pp.create_empty_network()

    # Create buses
    bus1 = pp.create_bus(net, vn_kv=110., name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=110., name="Bus 2")

    # Create external grid
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, va_degree=50)

    # Create generator
    pp.create_gen(net, bus=bus2, p_mw=2., vm_pu=1.01, min_p_mw=0., max_p_mw=4.)

    # Create load
    pp.create_load(net, bus=bus2, p_mw=2., q_mvar=4., name="load1")

    # Create line
    pp.create_line_from_parameters(net, from_bus=bus1, to_bus=bus2,
                                 length_km=20., r_ohm_per_km=0.876,
                                 x_ohm_per_km=0.497, c_nf_per_km=9.0,
                                 max_i_ka=0.963)
    return net
@pytest.fixture(scope="session")
def sample_variables():
    """Return sample optimization variables for testing"""
    T = ['t1', 't2']
    variables = {
        'xt': MagicMock(),
        'sxt': MagicMock(),
        'ext': MagicMock(),
        'pgen': MagicMock(),
        'pgen_c': MagicMock(),
        'd_wc': MagicMock(),
        'd_wc_c': MagicMock(),
        'f': MagicMock(),
        'f_c': MagicMock(),
        'theta': MagicMock(),
        'theta_c': MagicMock()
    }
    return variables

@pytest.fixture(scope="session")
def sample_solution():
    """Return a sample solution for testing"""
    return {
        'planned_outage_indicator[t1,outage_1]': 1.0,
        'planned_outage_indicator[t1,outage_2]': 0.0,
        'planned_outage_indicator[t2,outage_1]': 0.0,
        'planned_outage_indicator[t2,outage_2]': 1.0,
        'd_curt_tb[t1,bus_1]': 0.0,
        'd_curt_tb[t1,bus_2]': 0.0,
        'd_curt_tb[t2,bus_1]': 0.0,
        'd_curt_tb[t2,bus_2]': 0.0,
        'power_gen_tg[t1,gen_1]': 80.0,
        'power_gen_tg[t1,gen_2]': 70.0,
        'power_gen_tg[t2,gen_1]': 90.0,
        'power_gen_tg[t2,gen_2]': 80.0
    }

@pytest.fixture(scope="session")
def sample_results():
    """Return sample results for testing"""
    T = ['t1', 't2']
    return {
        'outage_schedule': pd.DataFrame({
            'outage_1': [1.0, 0.0],
            'outage_2': [0.0, 1.0]
        }, index=T),
        'loss_of_load': pd.DataFrame({
            'bus_1': [0.0, 0.0],
            'bus_2': [0.0, 0.0]
        }, index=T),
        'power_generation': pd.DataFrame({
            'gen_1': [80.0, 90.0],
            'gen_2': [70.0, 80.0]
        }, index=T),
        'CVARisk': pd.DataFrame({
            'bus_1': [0.0, 0.0],
            'bus_2': [0.0, 0.0]
        }, index=T)
    }

@pytest.fixture(scope="session")
def mock_gurobi_model():
    """Return a mock Gurobi model for testing"""
    model = MagicMock()
    model.addConstr = MagicMock()
    model.addConstrs = MagicMock()
    model.addVar = MagicMock()
    model.addVars = MagicMock()
    model.setObjective = MagicMock()
    model.optimize = MagicMock()
    model.Status = 2  # GRB.OPTIMAL
    return model

@pytest.fixture(scope="session")
def sample_T():
    """Return sample time periods"""
    return ['t1', 't2']

@pytest.fixture(scope="session")
def sample_names():
    """Return sample names dictionary"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'lines': ['line_1', 'line_2'],
        'contingencies': ['cont_1', 'cont_2'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1', 'gen_2'],
        'demand_scenarios': ['scenario_1', 'scenario_2']
    }

@pytest.fixture(scope="session")
def sample_outage_data():
    """Return sample outage data"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'outage_duration': pd.Series({'outage_1': 48, 'outage_2': 72}),
        'outage_window': {
            'outage_1': (0, 168),  # One week window
            'outage_2': (24, 192)  # One week window starting day 2
        }
    }

@pytest.fixture(scope="session")
def sample_risk_params():
    """Return sample risk parameters"""
    return {
        'alpha': 0.95,  # Confidence level
        'beta': 0.5,    # Risk aversion parameter
        'n_scenarios': 10
    }