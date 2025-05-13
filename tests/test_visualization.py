import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.vis_scos_res import (
    visualize_results,
    plot_risk_pdf_cdf,
    plot_comparison_CVAR_DET_SCOS
)

# Sample test data
def create_sample_results():
    """Create sample results for testing"""
    T = ['t1', 't2']
    results = {
        'X_OutageSchedule': pd.DataFrame({
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
    return results

def create_sample_names():
    """Create sample names dictionary for testing"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'lines': ['line_1', 'line_2'],
        'contingencies': ['cont_1', 'cont_2'],
        'buses': ['bus_1', 'bus_2'],
        'generators': ['gen_1', 'gen_2']
    }

def create_sample_data():
    """Create sample data for comparison plots"""
    return {
        'outages': {
            'names': ['outage_1', 'outage_2']
        },
        'num_branches': 2,
        'num_buses': 2,
        'network': MagicMock(gen=pd.DataFrame(index=[0, 1])),
        'n_minus1_names': ['cont_1', 'cont_2']
    }

# Test fixtures
@pytest.fixture
def sample_results():
    """Create sample results for visualization testing"""
    T = pd.date_range('2024-01-01', periods=24, freq='H')
    return {
        'X_OutageSchedule': pd.DataFrame({
            'outage_1': np.random.randint(0, 2, size=24),
            'outage_2': np.random.randint(0, 2, size=24)
        }, index=T),
        'D_LossOfLoad': pd.DataFrame({
            'bus_1': np.random.uniform(0, 10, size=24),
            'bus_2': np.random.uniform(0, 10, size=24)
        }, index=T),
        'P_Generation': pd.DataFrame({
            'gen_1': np.random.uniform(50, 100, size=24),
            'gen_2': np.random.uniform(30, 80, size=24)
        }, index=T),
        'F_LinePower': pd.DataFrame({
            'line_1': np.random.uniform(-50, 50, size=24),
            'line_2': np.random.uniform(-30, 30, size=24)
        }, index=T),
        'CVARisk': pd.Series(np.random.uniform(0, 100, size=100))
    }

@pytest.fixture
def sample_names():
    """Create sample names for visualization testing"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'generators': ['gen_1', 'gen_2'],
        'lines': ['line_1', 'line_2'],
        'buses': ['bus_1', 'bus_2']
    }

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for matplotlib tests"""
    plt.switch_backend('Agg')  # Use non-interactive backend
    yield
    plt.close('all')

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
def test_visualize_results(mock_show, mock_savefig, sample_results, sample_names):
    """Test visualization of optimization results"""
    save_path = 'test_figures'
    
    visualize_results(sample_results, sample_names, save_path=save_path)
    
    # Check that figures were created and saved
    assert mock_savefig.call_count >= 4  # At least 4 figures should be saved
    mock_show.assert_not_called()  # show should not be called when saving

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
def test_plot_risk_pdf_cdf(mock_show, mock_savefig, sample_results):
    """Test plotting of risk PDF and CDF"""
    risk_data = sample_results['CVARisk']
    save_path = 'test_figures/risk.png'
    
    plot_risk_pdf_cdf(risk_data, save_path=save_path)
    
    mock_savefig.assert_called_once_with(save_path, bbox_inches='tight', dpi=300)
    mock_show.assert_not_called()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
def test_plot_comparison_CVAR_DET_SCOS(mock_show, mock_savefig):
    """Test plotting comparison between CVaR and deterministic SCOS"""
    # Create sample comparison data
    cvar_results = {
        'P_Generation': pd.DataFrame({
            'gen_1': np.random.uniform(50, 100, size=24),
            'gen_2': np.random.uniform(30, 80, size=24)
        }),
        'X_OutageSchedule': pd.DataFrame({
            'outage_1': np.random.randint(0, 2, size=24),
            'outage_2': np.random.randint(0, 2, size=24)
        })
    }
    
    det_results = {
        'P_Generation': pd.DataFrame({
            'gen_1': np.random.uniform(50, 100, size=24),
            'gen_2': np.random.uniform(30, 80, size=24)
        }),
        'X_OutageSchedule': pd.DataFrame({
            'outage_1': np.random.randint(0, 2, size=24),
            'outage_2': np.random.randint(0, 2, size=24)
        })
    }
    
    save_path = 'test_figures/comparison.png'
    
    plot_comparison_CVAR_DET_SCOS(cvar_results, det_results, save_path=save_path)
    
    mock_savefig.assert_called_once_with(save_path, bbox_inches='tight', dpi=300)
    mock_show.assert_not_called()

def test_empty_results():
    """Test handling of empty results"""
    empty_results = {
        'X_OutageSchedule': pd.DataFrame(),
        'D_LossOfLoad': pd.DataFrame(),
        'P_Generation': pd.DataFrame(),
        'F_LinePower': pd.DataFrame(),
        'CVARisk': pd.Series()
    }
    empty_names = {
        'outages': [],
        'generators': [],
        'lines': [],
        'buses': []
    }
    
    # Should not raise any exceptions
    visualize_results(empty_results, empty_names, save_path='test_figures')
    plot_risk_pdf_cdf(empty_results['CVARisk'], save_path='test_figures/risk.png')

if __name__ == '__main__':
    pytest.main([__file__]) 