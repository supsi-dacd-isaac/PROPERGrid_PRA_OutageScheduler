import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pra_psa.utils.utils import get_project_root


class TestPathHandling(unittest.TestCase):
    """Test path handling functions to ensure they correctly resolve paths."""

    def test_get_project_root(self):
        """Test that get_project_root correctly identifies the PROPER directory."""
        project_root = get_project_root()
        self.assertEqual(project_root.name, "PROPER")
        self.assertTrue(project_root.exists())
        self.assertTrue(project_root.is_dir())

    def test_config_file_exists(self):
        """Test that the config file exists at the expected location."""
        config_path = os.path.join(project_root, 'config', 'conf_IEEE24.json')
        self.assertTrue(os.path.exists(config_path), f"Config file not found at {config_path}")

    def test_data_directory_exists(self):
        """Test that the data directory exists at the expected location."""
        data_path = os.path.join(project_root, 'data', 'powersystems', 'IEEE24')
        self.assertTrue(os.path.exists(data_path), f"Data directory not found at {data_path}")


if __name__ == '__main__':
    unittest.main() 