import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from optimizers.pre_post_processing import (
    post_process_results,
    encode_simulation_name,
    hash_simulation_params,
    simulation_exists,
    load_metadata,
    save_metadata
)

@pytest.fixture
def sample_solution():
    """Create sample solution dictionary"""
    return {
        'planned_outage_indicator': pd.DataFrame({
            ('t1', 'outage_1'): [1],
            ('t1', 'outage_2'): [0],
            ('t2', 'outage_1'): [0],
            ('t2', 'outage_2'): [1]
        }),
        'power_generation': pd.DataFrame({
            ('t1', 'gen_1'): [100],
            ('t1', 'gen_2'): [50],
            ('t2', 'gen_1'): [80],
            ('t2', 'gen_2'): [70]
        }),
        'line_power_flow': pd.DataFrame({
            ('t1', 'line_1'): [30],
            ('t1', 'line_2'): [20],
            ('t2', 'line_1'): [25],
            ('t2', 'line_2'): [15]
        }),
        'loss_of_load': pd.DataFrame({
            ('t1', 'bus_1'): [0],
            ('t1', 'bus_2'): [0],
            ('t2', 'bus_1'): [10],
            ('t2', 'bus_2'): [5]
        })
    }

@pytest.fixture
def sample_names():
    """Create sample names dictionary"""
    return {
        'outages': ['outage_1', 'outage_2'],
        'generators': ['gen_1', 'gen_2'],
        'lines': ['line_1', 'line_2'],
        'buses': ['bus_1', 'bus_2'],
        'contingencies': ['cont_1', 'cont_2']
    }

@pytest.fixture
def sample_T():
    """Create sample time periods"""
    return ['t1', 't2']

@pytest.mark.unit
class TestPrePostProcessing:
    """Test suite for pre and post processing functions"""

    def test_post_process_results(self, sample_solution, sample_names, sample_T):
        """Test post-processing of results"""
        results = post_process_results(sample_solution, sample_names, sample_T)
        assert isinstance(results, dict)
        assert 'X_OutageSchedule' in results
        assert 'P_Generation' in results
        assert 'F_LinePower' in results
        assert 'D_LossOfLoad' in results

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
        mock_exists.return_value = True
        filename = 'test_simulation.json'
        
        assert simulation_exists(filename)
        mock_exists.assert_called_once_with(filename)

    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_load_metadata(self, mock_json_load, mock_open):
        """Test loading metadata"""
        mock_json_load.return_value = {"test": "data"}
        filename = 'test_metadata.json'
        
        metadata = load_metadata(filename)
        assert isinstance(metadata, dict)
        assert metadata == {"test": "data"}
        mock_open.assert_called_once_with(filename, 'r')

    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_save_metadata(self, mock_json_dump, mock_open):
        """Test saving metadata"""
        filename = 'test_metadata.json'
        metadata = {"test": "data"}
        
        save_metadata(metadata, filename)
        mock_open.assert_called_once_with(filename, 'w')
        mock_json_dump.assert_called_once_with(metadata, mock_open())

if __name__ == '__main__':
    pytest.main([__file__]) 