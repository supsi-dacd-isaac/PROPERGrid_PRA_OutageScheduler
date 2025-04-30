import os
import sys
import pytest
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.gurobi_SCOS import run_SCOS_deterministic, run_SCOS_cvar

from optimizers.utils_and_constraints import (
    add_planned_outages_constraints,
    add_generators_constraints,
    add_line_power_limit_constraints,
    add_nodal_power_balance_constraints,
    add_line_dc_power_flow_constraints,
    calculate_nodal_balance
)
from optimizers.pre_post_processing import (
    post_process_results,
    encode_simulation_name,
    hash_simulation_params,
    simulation_exists,
    load_metadata,
    save_metadata
)
from optimizers.vis_scos_res import (
    visualize_results,
    plot_risk_pdf_cdf,
    plot_comparison_CVAR_DET_SCOS
)
from optimizers.gurobi_params import get_params


# Sample test data
def create_sample_data():
    """Create a small sample dataset for testing"""
    data = {
        'names': {
            'outages': ['outage_1', 'outage_2'],
            'lines': ['line_1', 'line_2'],
            'contingencies': ['cont_1', 'cont_2'],
            'buses': ['bus_1', 'bus_2'],
            'generators': ['gen_1', 'gen_2'],
            'demand_scenarios': ['scenario_1', 'scenario_2']
        },
        'T': ['t1', 't2'],
        'VoLL': 1000,
        'tail_prob_constraint': 0.95,
        'n_samples': 2,
        'use_DC_PF': True,
        'nodal_demand': pd.DataFrame({
            'bus_1': [100, 120],
            'bus_2': [80, 90]
        }, index=['t1', 't2']),
        'network': MagicMock(),
        'max_number_of_maintenance_tasks': 2,
        'outages': {
            'indices': [1, 2],
            'names': ['outage_1', 'outage_2'],
            'type': ['line', 'line'],
            'expected_duration_steps': [1, 1],
            'cost_per_step': [100, 100],
            'priorities': [1, 1]
        }
    }
    return data


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

    # Configure mocks to return sensible values and handle indexing
    for var_name in variables:
        mock_var = variables[var_name]
        mock_var.__getitem__.return_value = 1.0  # Default value for any index
        mock_var.return_value = 1.0  # Default value for direct calls

        # Configure the mock to handle string indices
        def getitem_mock(self, key):
            return 1.0
        mock_var.__getitem__ = getitem_mock

    return variables

def create_sample_solution():
    """Create a sample solution for testing"""
    solution = {}
    T = ['t1', 't2']
    outages = ['outage_1', 'outage_2']
    buses = ['bus_1', 'bus_2']
    generators = ['gen_1', 'gen_2']
    lines = ['line_1', 'line_2']
    contingencies = ['cont_1', 'cont_2']

    # Add planned outage indicators
    for t in T:
        for o in outages:
            solution[f'planned_outage_indicator[{t},{o}]'] = 1.0 if (t == 't1' and o == 'outage_1') or (t == 't2' and o == 'outage_2') else 0.0

    # Add power generation
    for t in T:
        for g in generators:
            solution[f'power_generation[{t},{g}]'] = 80.0 if g == 'gen_1' else 70.0

    # Add flows
    for t in T:
        for l in lines:
            solution[f'flow_tl[{t},{l}]'] = 30.0 if l == 'line_1' else 20.0

    # Add loss of load
    for t in T:
        for b in buses:
            solution[f'loss_of_load[{t},{b}]'] = 0.0

    # Add contingency loss of load
    for t in T:
        for b in buses:
            for c in contingencies:
                solution[f'loss_of_load_contingency[{t},{b},{c}]'] = 0.0

    # Add CVaR (optional)
    for t in T:
        for b in buses:
            solution[f'CVaR[{t},{b}]'] = 0.0

    return solution

def create_sample_results():
    """Create sample results for testing"""
    T = ['t1', 't2']
    return {
        'X_OutageSchedule': pd.DataFrame({
            'outage_1': [1.0, 0.0],
            'outage_2': [0.0, 1.0]
        }, index=T),
        'PowerGenerated': pd.DataFrame({
            'gen_1': [80.0, 90.0],
            'gen_2': [70.0, 80.0]
        }, index=T),
        'Line_Flows': pd.DataFrame({
            'line_1': [30.0, 30.0],
            'line_2': [20.0, 20.0]
        }, index=T),
        'WC_CURTAIL': pd.DataFrame({
            'total': [0.0, 0.0]
        }, index=T),
        'WC_CURTAIL_CON': pd.DataFrame({
            'cont_1': [0.0, 0.0],
            'cont_2': [0.0, 0.0]
        }, index=T),
        'GEN_PLUS_CURTAILED': np.array([150.0, 150.0]),
        'CVARisk': pd.DataFrame({
            'bus_1': [0.0, 0.0],
            'bus_2': [0.0, 0.0]
        }, index=T)
    }

# Test fixtures
@pytest.fixture
def sample_data():
    return create_sample_data()

@pytest.fixture
def sample_model():
    return create_sample_model()

@pytest.fixture
def sample_names():
    return create_sample_names()

@pytest.fixture
def sample_variables():
    return create_sample_variables()

@pytest.fixture
def sample_T():
    return ['t1', 't2']

@pytest.fixture
def sample_solution():
    return create_sample_solution()

@pytest.fixture
def sample_results():
    return create_sample_results()

@pytest.fixture
def mock_gurobi_model():
    with patch('gurobipy.Model') as mock_model:
        model_instance = MagicMock()
        mock_model.return_value = model_instance
        yield model_instance

@pytest.fixture
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
        'line_reactance': pd.Series({'line_1': 0.1, 'line_2': 0.1}),
        'num_branches': 2,
        'num_buses': 2  # Added this field
    }

@pytest.mark.unit
class TestOptimizers:
    """Test suite for optimization functions"""

    def test_deterministic_SCOS_gurobi_initialization(self, sample_network_data, mock_gurobi_model):
        """Test initialization of deterministic SCOS model"""
        with patch('optimizers.gurobi_SCOS.prepare_SCOS_data') as mock_prepare:
            mock_prepare.return_value = (2, [100, 100], [0, 0], [100, 100],
                                        ['gen_1', 'gen_2'], {'gen_1': 'bus_1', 'gen_2': 'bus_2'},
                                        np.eye(2), np.eye(2), np.eye(2), [1, 1], [1, 1], [100, 100])

            result = run_SCOS_deterministic(sample_network_data)

            # Check if model was created with correct name
            mock_gurobi_model.assert_called_with("deterministic_SCOS")

            # Check if parameters were set
            assert mock_gurobi_model.setParam.call_count > 0

    @pytest.mark.gurobi
    def test_deterministic_SCOS_gurobi_error_handling(self, sample_network_data):
        """Test error handling in deterministic SCOS model"""
        with patch('gurobipy.Model', side_effect=gp.GurobiError(10001, "Test error")):
            with pytest.raises(gp.GurobiError) as exc_info:
                run_SCOS_deterministic(sample_network_data)
            assert "Test error" in str(exc_info.value)

    def test_add_planned_outages_constraints(self, mock_gurobi_model, sample_network_data,
                                           sample_variables, sample_T):
        """Test adding planned outages constraints"""
        add_planned_outages_constraints(
            mock_gurobi_model,
            sample_network_data['names']['outages'],
            sample_variables['xt'],
            sample_variables['sxt'],
            sample_variables['ext'],
            sample_network_data['max_number_of_maintenance_tasks'],
            sample_network_data['outages']['expected_duration_steps'],
            sample_T
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_generators_constraints(self, mock_gurobi_model, sample_network_data,
                                     sample_variables, sample_T):
        """Test adding generator constraints"""
        add_generators_constraints(
            mock_gurobi_model,
            sample_variables['xt'],
            sample_network_data['names'],
            pgen=sample_variables['pgen'],
            pgen_c=sample_variables['pgen_c'],
            p_max=sample_network_data['gen_capacity'].values,
            p_min=sample_network_data['gen_min_output'].values,
            T=sample_T
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_line_power_limit_constraints(self, mock_gurobi_model, sample_network_data,
                                            sample_variables, sample_T):
        """Test adding line power limit constraints"""
        add_line_power_limit_constraints(
            mock_gurobi_model,
            sample_variables['xt'],
            sample_network_data['names'],
            sample_network_data['line_capacity'].values,
            sample_variables['f'],
            sample_variables['f_c'],
            sample_T
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_nodal_power_balance_constraints(self, mock_gurobi_model, sample_network_data,
                                               sample_variables, sample_T):
        """Test adding nodal power balance constraints"""
        add_nodal_power_balance_constraints(
            mock_gurobi_model,
            sample_network_data['names'],
            np.eye(2),  # S_T matrix
            demand=sample_network_data['nodal_demand'],
            pgen=sample_variables['pgen'],
            pgen_c=sample_variables['pgen_c'],
            f=sample_variables['f'],
            f_c=sample_variables['f_c'],
            g2bus=sample_network_data['gen_bus'],
            d_wc=sample_variables['d_wc'],
            d_wc_c=sample_variables['d_wc_c'],
            T=sample_T
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_add_line_dc_power_flow_constraints(self, mock_gurobi_model, sample_network_data,
                                              sample_variables, sample_T):
        """Test adding DC power flow constraints"""
        add_line_dc_power_flow_constraints(
            mock_gurobi_model,
            sample_network_data['names'],
            sample_variables['xt'],
            sample_network_data['line_capacity'].values,
            sample_variables['f'],
            sample_variables['f_c'],
            sample_variables['theta'],
            sample_variables['theta_c'],
            np.eye(2),  # B_mat matrix
            sample_T
        )
        assert mock_gurobi_model.addConstr.call_count > 0

    def test_calculate_nodal_balance(self, sample_network_data, sample_variables, sample_T):
        """Test calculating nodal balance"""
        balance = calculate_nodal_balance(
            sample_T,
            sample_variables['f'],
            sample_network_data['names'],
            np.eye(2),  # S_T matrix
            sample_variables['pgen'],
            sample_network_data['gen_bus'],
            sample_network_data['nodal_demand']
        )
        assert isinstance(balance, np.ndarray)
        assert balance.shape == (len(sample_T), len(sample_network_data['names']['buses']))

    def test_post_process_results(self, sample_solution, sample_network_data, sample_T):
        """Test post-processing of results"""
        results = post_process_results(sample_solution, sample_network_data['names'], sample_T)
        assert isinstance(results, dict)
        assert 'X_OutageSchedule' in results
        assert 'PowerGenerated' in results
        assert 'Line_Flows' in results
        assert 'WC_CURTAIL' in results
        assert 'WC_CURTAIL_CON' in results
        assert 'GEN_PLUS_CURTAILED' in results
        assert 'CVARisk' in results

    def test_encode_simulation_name(self):
        """Test encoding simulation name"""
        name = encode_simulation_name({'param1': 'value1', 'param2': 'value2'})
        assert isinstance(name, str)
        assert len(name) > 0

    def test_hash_simulation_params(self):
        """Test hashing simulation parameters"""
        hash_value = hash_simulation_params({'param1': 'value1', 'param2': 'value2'})
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    @patch('os.path.exists')
    def test_simulation_exists(self, mock_exists):
        """Test checking if simulation exists"""
        results_dir = Path("test_results")
        filename = "test_simulation"
        mock_exists.return_value = True
        assert simulation_exists(results_dir, filename)
        mock_exists.assert_called_once()

    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_load_metadata(self, mock_json_load, mock_open):
        """Test loading metadata"""
        results_dir = Path("test_results")
        filename = "test_simulation"
        mock_json_load.return_value = {'param1': 'value1'}
        metadata = load_metadata(results_dir, filename)
        assert isinstance(metadata, dict)
        assert 'param1' in metadata

    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_save_metadata(self, mock_json_dump, mock_open):
        """Test saving metadata"""
        results_dir = Path("test_results")
        filename = "test_simulation"
        metadata = {'param1': 'value1'}
        save_metadata(metadata, results_dir, filename)
        mock_json_dump.assert_called_once()

    @pytest.mark.visualization
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_results(self, mock_close, mock_savefig, mock_subplot,
                             mock_figure, sample_results, sample_network_data):
        """Test visualization of results"""
        visualize_results(sample_results, sample_network_data['names'])
        mock_figure.assert_called()
        mock_savefig.assert_called()

    @pytest.mark.visualization
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_risk_pdf_cdf(self, mock_close, mock_savefig, mock_subplot,
                              mock_figure, sample_results):
        """Test plotting risk PDF and CDF"""
        plot_risk_pdf_cdf(sample_results['CVARisk'])
        mock_figure.assert_called()
        mock_savefig.assert_called()

    @pytest.mark.visualization
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_comparison_CVAR_DET_SCOS(self, mock_close, mock_savefig, mock_subplot,
                                          mock_figure, sample_results, sample_network_data):
        """Test plotting comparison of CVaR and deterministic SCOS"""
        plot_comparison_CVAR_DET_SCOS(sample_results, sample_results, sample_network_data)
        mock_figure.assert_called()
        mock_savefig.assert_called()

if __name__ == '__main__':
    pytest.main([__file__])
