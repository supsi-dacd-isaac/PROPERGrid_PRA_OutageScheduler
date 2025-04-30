import os
import sys
import pytest
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from unittest.mock import patch, MagicMock
from tqdm import tqdm

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.utils_and_constraints import (
    add_planned_outages_constraints,
    add_generators_constraints,
    add_line_power_limit_constraints,
    add_nodal_power_balance_constraints,
    add_line_dc_power_flow_constraints,
    calculate_nodal_balance
)

from optimizers.gurobi_SCOS import (
    add_planned_outages_constraints,
    add_generators_constraints,
    add_line_power_limit_constraints,
    add_nodal_power_balance_constraints
)

# Sample test data
def create_sample_model():
    """Create a mock Gurobi model for testing"""
    model = MagicMock()
    model.addConstr = MagicMock()
    model.addConstrs = MagicMock()
    return model

def create_sample_names():
    """Create sample names dictionary for testing"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'lines': ['line_1', 'line_2'],
        'contingencies': ['cont_1', 'cont_2'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1', 'gen_2']
    }

def create_sample_variables():
    """Create sample variables for testing"""
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

# Test fixtures
@pytest.fixture
def sample_model():
    """Create a sample Gurobi model for testing constraints"""
    model = gp.Model("test_model")
    model.Params.OutputFlag = 0  # Suppress Gurobi output
    return model

@pytest.fixture
def sample_names():
    """Create sample names dictionary for testing"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'lines': ['line_1', 'line_2'],
        'contingencies': ['cont_1', 'cont_2'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1', 'gen_2']
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing constraints."""
    T = pd.date_range('2024-01-01', periods=24, freq='h')
    data = {
        'T': T,
        'outages': ['outage_1', 'outage_2'],
        'lines': ['line_1', 'line_2'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1', 'gen_2'],
        'contingencies': ['cont_1', 'cont_2'],
        'branch_capacity': [100.0, 100.0],
        'gen_capacity': {'gen_1': 100.0, 'gen_2': 100.0},
        'gen_min_output': {'gen_1': 0.0, 'gen_2': 0.0},
        'outage_duration': {'outage_1': 2, 'outage_2': 3},
        'gen_bus': {'gen_1': 'bus_1', 'gen_2': 'bus_2'},
        'demand': pd.DataFrame(
            np.random.rand(len(T), 2) * 50,
            index=T,
            columns=['bus_1', 'bus_2']
        ),
        'line_capacity': {'line_1': 100.0, 'line_2': 100.0},
        'PTDF': np.array([[0.5, -0.5], [-0.5, 0.5]])
    }
    return data

@pytest.fixture
def sample_variables(sample_data):
    """Create sample Gurobi variables for testing."""
    M = gp.Model()
    T = sample_data['T']
    vars = {
        'xt': M.addVars(T, sample_data['outages'], vtype=gp.GRB.BINARY, name='planned_outage_indicator'),
        'sxt': M.addVars(T, sample_data['outages'], vtype=gp.GRB.BINARY, name='start_outage'),
        'ext': M.addVars(T, sample_data['outages'], vtype=gp.GRB.BINARY, name='end_outage'),
        'pgen': M.addVars(T, sample_data['generators'], name='power_generation'),
        'pgen_c': M.addVars(T, sample_data['generators'], sample_data['contingencies'], name='power_generation_contingency'),
        'f': M.addVars(T, sample_data['lines'], name='power_flow'),
        'f_c': M.addVars(T, sample_data['lines'], sample_data['contingencies'], name='power_flow_contingency'),
        'd_wc': M.addVars(T, sample_data['buses'], name='load_shedding'),
        'd_wc_c': M.addVars(T, sample_data['buses'], sample_data['contingencies'], name='load_shedding_contingency'),
        'theta': M.addVars(T, sample_data['buses'], name='voltage_angle')
    }
    return vars

@pytest.fixture
def sample_T():
    """Create sample time periods"""
    return ['t1', 't2']

# Tests for add_planned_outages_constraints
def test_add_planned_outages_constraints(sample_model, sample_names, sample_variables, sample_T):
    """Test adding planned outages constraints"""
    max_tasks = 2
    durations = {'outage_1': 1, 'outage_2': 1}
    
    result = add_planned_outages_constraints(
        sample_model, 
        sample_names['outages'], 
        sample_variables['xt'], 
        sample_variables['sxt'], 
        sample_variables['ext'], 
        max_tasks, 
        durations, 
        sample_T
    )
    
    # Verify constraints were added
    assert sample_model.addConstr.call_count > 0

# Tests for add_generators_constraints
def test_add_generators_constraints(sample_model, sample_names, sample_variables, sample_T):
    """Test adding generator constraints"""
    p_max = {'gen_1': 100.0, 'gen_2': 100.0}
    p_min = {'gen_1': 0.0, 'gen_2': 0.0}
    
    result = add_generators_constraints(
        sample_model,
        sample_variables['xt'],
        sample_names,
        pgen=sample_variables['pgen'],
        pgen_c=sample_variables['pgen_c'],
        p_max=p_max,
        p_min=p_min,
        T=sample_T
    )
    
    # Verify constraints were added
    assert sample_model.addConstr.call_count > 0

# Tests for add_line_power_limit_constraints
def test_add_line_power_limit_constraints(sample_model, sample_names, sample_variables, sample_T):
    """Test adding line power limit constraints"""
    f_lim = {'line_1': 100.0, 'line_2': 100.0}
    
    result = add_line_power_limit_constraints(
        sample_model,
        sample_variables['xt'],
        sample_names,
        f_lim,
        sample_variables['f'],
        sample_variables['f_c'],
        sample_T
    )
    
    # Verify constraints were added
    assert sample_model.addConstr.call_count > 0

# Tests for add_nodal_power_balance_constraints
def test_add_nodal_power_balance_constraints(sample_model, sample_names, sample_variables, sample_T):
    """Test adding nodal power balance constraints"""
    S = np.eye(2)
    demand = pd.DataFrame({
        'bus_1': [100, 120],
        'bus_2': [80, 90]
    }, index=sample_T)
    g2bus = ['gen_1', 'gen_2']
    
    result = add_nodal_power_balance_constraints(
        sample_model,
        sample_names,
        S,
        demand=demand,
        pgen=sample_variables['pgen'],
        pgen_c=sample_variables['pgen_c'],
        f=sample_variables['f'],
        f_c=sample_variables['f_c'],
        g2bus=g2bus,
        d_wc=sample_variables['d_wc'],
        d_wc_c=sample_variables['d_wc_c'],
        T=sample_T
    )
    
    # Check if constraints were added
    assert sample_model.addConstr.call_count > 0
    assert result == sample_model

# Tests for add_line_dc_power_flow_constraints
def test_add_line_dc_power_flow_constraints(sample_model, sample_names, sample_variables, sample_T):
    """Test adding DC power flow constraints"""
    B_mat = np.eye(2)
    f_lim = {'line_1': 100.0, 'line_2': 100.0}
    
    result = add_line_dc_power_flow_constraints(
        sample_model,
        sample_names,
        sample_variables['xt'],
        f_lim,
        sample_variables['f'],
        sample_variables['f_c'],
        sample_variables['theta'],
        sample_variables['theta_c'],
        B_mat,
        sample_T
    )
    
    # Verify constraints were added
    assert sample_model.addConstr.call_count > 0

# Tests for calculate_nodal_balance
def test_calculate_nodal_balance(sample_model, sample_names, sample_variables, sample_T):
    """Test calculating nodal balance"""
    g2bus = {'gen_1': 'bus_1', 'gen_2': 'bus_2'}
    B_mat = np.eye(2)
    demand = np.zeros((2, 2))  # 2 buses, 2 time periods

    result = calculate_nodal_balance(
        sample_model,
        sample_names,
        g2bus,
        sample_variables['pgen'],
        sample_variables['f'],
        sample_variables['d_wc'],
        B_mat,
        demand,
        sample_T
    )

    # Verify constraints were added
    assert sample_model.addConstr.call_count > 0

def test_planned_outage_constraints(sample_data, sample_variables):
    """Test the addition of planned outage constraints."""
    M = sample_variables['xt'].model
    add_planned_outages_constraints(M, sample_data, sample_variables)
    M.update()
    
    # Check number of constraints
    expected_constraints = (
        len(sample_data['T']) * len(sample_data['outages']) * 3  # Start/end/duration constraints
        + len(sample_data['outages'])  # One constraint per outage for completion
    )
    assert M.NumConstrs == expected_constraints, f"Expected {expected_constraints} constraints, got {M.NumConstrs}"

def test_generator_constraints(sample_data, sample_variables):
    """Test the addition of generator constraints."""
    M = sample_variables['pgen'].model
    add_generators_constraints(M, sample_data, sample_variables)
    M.update()
    
    # Check number of constraints
    expected_constraints = (
        len(sample_data['T']) * len(sample_data['generators']) * 2  # Min/max output
        + len(sample_data['T']) * len(sample_data['generators']) * len(sample_data['contingencies'])  # Contingency constraints
    )
    assert M.NumConstrs == expected_constraints

def test_line_power_constraints(sample_data, sample_variables):
    """Test the addition of line power constraints."""
    M = sample_variables['f'].model
    add_line_power_limit_constraints(M, sample_data, sample_variables)
    M.update()
    
    # Check number of constraints
    expected_constraints = (
        len(sample_data['T']) * len(sample_data['lines']) * 2  # Normal operation limits
        + len(sample_data['T']) * len(sample_data['lines']) * len(sample_data['contingencies']) * 2  # Contingency limits
    )
    assert M.NumConstrs == expected_constraints

def test_nodal_power_balance(sample_data, sample_variables):
    """Test the addition of nodal power balance constraints."""
    M = sample_variables['pgen'].model
    add_nodal_power_balance_constraints(M, sample_data,
                                      np.eye(len(sample_data['buses'])),  # S matrix
                                      demand=sample_data['demand'],
                                      pgen=sample_variables['pgen'],
                                      pgen_c=sample_variables['pgen_c'],
                                      f=sample_variables['f'],
                                      f_c=sample_variables['f_c'],
                                      g2bus=sample_data['gen_bus'],
                                      d_wc=sample_variables['d_wc'],
                                      d_wc_c=sample_variables['d_wc_c'],
                                      T=sample_data['T'])
    
    # Check number of constraints
    expected_constraints = (
        len(sample_data['T']) * len(sample_data['buses'])  # Base case balance
        + len(sample_data['T']) * len(sample_data['buses']) * len(sample_data['contingencies'])  # Contingency balance
    )
    assert M.NumConstrs == expected_constraints

def test_constraint_feasibility():
    """Test that the model with all constraints is feasible."""
    # Create a small test case
    T = pd.date_range('2024-01-01', periods=4, freq='h')
    data = {
        'T': T,
        'outages': ['outage_1'],
        'lines': ['line_1'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1'],
        'contingencies': ['cont_1'],
        'branch_capacity': [100.0],
        'gen_capacity': {'gen_1': 100.0},
        'gen_min_output': {'gen_1': 0.0},
        'outage_duration': {'outage_1': 2},
        'gen_bus': {'gen_1': 'bus_1'},
        'demand': pd.DataFrame(
            np.random.rand(len(T), 2) * 20,  # Lower demand for feasibility
            index=T,
            columns=['bus_1', 'bus_2']
        ),
        'line_capacity': {'line_1': 100.0},
        'PTDF': np.array([[0.5, -0.5]])
    }
    
    # Create model and variables
    M = gp.Model()
    vars = {
        'xt': M.addVars(T, data['outages'], vtype=gp.GRB.BINARY, name='planned_outage_indicator'),
        'sxt': M.addVars(T, data['outages'], vtype=gp.GRB.BINARY, name='start_outage'),
        'ext': M.addVars(T, data['outages'], vtype=gp.GRB.BINARY, name='end_outage'),
        'pgen': M.addVars(T, data['generators'], name='power_generation'),
        'pgen_c': M.addVars(T, data['generators'], data['contingencies'], name='power_generation_contingency'),
        'f': M.addVars(T, data['lines'], name='power_flow'),
        'f_c': M.addVars(T, data['lines'], data['contingencies'], name='power_flow_contingency'),
        'd_wc': M.addVars(T, data['buses'], name='load_shedding'),
        'd_wc_c': M.addVars(T, data['buses'], data['contingencies'], name='load_shedding_contingency'),
        'theta': M.addVars(T, data['buses'], name='voltage_angle')
    }
    
    # Add all constraints
    add_planned_outages_constraints(M, data, vars)
    add_generators_constraints(M, data, vars)
    add_line_power_limit_constraints(M, data, vars)
    add_nodal_power_balance_constraints(M, data,
                                      np.eye(len(data['buses'])),
                                      demand=data['demand'],
                                      pgen=vars['pgen'],
                                      pgen_c=vars['pgen_c'],
                                      f=vars['f'],
                                      f_c=vars['f_c'],
                                      g2bus=data['gen_bus'],
                                      d_wc=vars['d_wc'],
                                      d_wc_c=vars['d_wc_c'],
                                      T=data['T'])
    
    # Set objective (minimize generation)
    obj = gp.quicksum(vars['pgen'][t, g] for t in T for g in data['generators'])
    M.setObjective(obj, gp.GRB.MINIMIZE)
    
    # Optimize
    M.optimize()
    
    assert M.status == gp.GRB.OPTIMAL, "Model should be feasible"

if __name__ == '__main__':
    pytest.main([__file__]) 