import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.run_optimizer import run_optimizer

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return {
        'VoLL': 1000,
        'tail_prob_constraint': 0.05,
        'n_samples': 100,
        'use_DC_PF': True,
        'config': {
            'aggregation_time': 1,
            'load_scaling_factor': 1.0
        },
        'max_number_of_maintenance_tasks': 2
    }

@pytest.fixture
def mock_results():
    """Create mock optimization results"""
    return {
        'results_dict': {'key': 'value'},
        'solution': {'x': 1},
        'objective': 100.0
    }

class TestRunOptimizer:
    """Test suite for run_optimizer function"""

    def test_deterministic_optimization(self, sample_data, mock_results):
        """Test deterministic optimization"""
        with patch('optimizers.run_optimizer.run_SCOS_deterministic') as mock_optimizer:
            mock_optimizer.return_value = (
                mock_results['results_dict'],
                mock_results['solution'],
                mock_results['objective']
            )
            
            result = run_optimizer(sample_data, optimizer_type='deterministic')
            
            assert result[0] == mock_results['results_dict']
            assert result[1] == mock_results['solution']
            assert result[2] == mock_results['objective']
            assert result[3]['optimizer_type'] == 'deterministic'

    def test_risk_aware_optimization(self, sample_data, mock_results):
        """Test risk-aware optimization"""
        with patch('optimizers.run_optimizer.run_SCOS_cvar') as mock_optimizer:
            mock_optimizer.return_value = (
                mock_results['results_dict'],
                mock_results['solution'],
                mock_results['objective']
            )
            
            result = run_optimizer(sample_data, optimizer_type='risk_aware')
            
            assert result[0] == mock_results['results_dict']
            assert result[1] == mock_results['solution']
            assert result[2] == mock_results['objective']
            assert result[3]['optimizer_type'] == 'risk_aware'

    def test_risk_constrained_optimization(self, sample_data, mock_results):
        """Test risk-constrained optimization"""
        with patch('optimizers.run_optimizer.run_SCOS_cvar') as mock_optimizer:
            mock_optimizer.return_value = (
                mock_results['results_dict'],
                mock_results['solution'],
                mock_results['objective']
            )
            
            result = run_optimizer(sample_data, optimizer_type='risk_constrained', cvar_limit=50)
            
            assert result[0] == mock_results['results_dict']
            assert result[1] == mock_results['solution']
            assert result[2] == mock_results['objective']
            assert result[3]['optimizer_type'] == 'risk_constrained'

    def test_invalid_optimizer_type(self, sample_data):
        """Test handling of invalid optimizer type"""
        with pytest.raises(ValueError):
            run_optimizer(sample_data, optimizer_type='invalid_type')

    def test_optimization_failure(self, sample_data):
        """Test handling of optimization failure"""
        with patch('optimizers.run_optimizer.run_SCOS_deterministic') as mock_optimizer:
            mock_optimizer.return_value = (None, None, None)
            
            result = run_optimizer(sample_data, optimizer_type='deterministic')
            
            assert result == (None, None, None, None)

    def test_existing_simulation(self, sample_data, mock_results):
        """Test loading of existing simulation results"""
        with patch('optimizers.run_optimizer.simulation_exists') as mock_exists, \
             patch('optimizers.run_optimizer.load_metadata') as mock_load:
            
            mock_exists.return_value = True
            mock_load.return_value = (
                mock_results['results_dict'],
                mock_results['solution'],
                mock_results['objective'],
                {'optimizer_type': 'deterministic'}
            )
            
            result = run_optimizer(sample_data, optimizer_type='deterministic')
            
            assert result[0] == mock_results['results_dict']
            assert result[1] == mock_results['solution']
            assert result[2] == mock_results['objective']
            assert result[3]['optimizer_type'] == 'deterministic'

    def test_error_handling(self, sample_data):
        """Test error handling in run_optimizer"""
        with patch('optimizers.run_optimizer.run_SCOS_deterministic') as mock_optimizer:
            mock_optimizer.side_effect = Exception("Test error")
            
            result = run_optimizer(sample_data, optimizer_type='deterministic')
            
            assert result == (None, None, None, None)

if __name__ == '__main__':
    pytest.main([__file__]) 