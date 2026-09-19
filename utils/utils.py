import os
from pathlib import Path
from typing import Sequence
import logging
import numpy as np
import json
import pandas as pd
from scipy.io import loadmat
import pandapower.networks as networks
import pandapower as pp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sbn
from statsmodels.distributions.empirical_distribution import ECDF
import hashlib

# ANSI escape code for colored text
blue_c, green_c = "\033[94m", "\033[92m"
purple_c, cyan_c, red_c, gray_c = "\033[95m", "\033[96m", "\033[91m", " "
bold_c, underline_c, reset_c = "\033[1m", "\033[4m", "\033[0m"
logger = logging.getLogger() # logging details
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)


def get_project_root(marker_files: Sequence[str] = (".git", "pyproject.toml", "setup.py")) -> Path:
    """ Return the root of the project by looking for one of the given marker  """
    # First try to find a directory named "PROPER"
    current = Path(__file__).resolve()
    logging.info(f"Current file path: {current}")
    
    for directory in (current, *current.parents):
        logging.info(f"Checking directory: {directory}")
        if directory.name == "PROPER":
            logging.info(f"Found PROPER directory: {directory}")
            return directory
    
    # If not found, fall back to marker files
    for directory in (current, *current.parents):
        if any((directory / m).exists() for m in marker_files):
            logging.info(f"Found marker file in: {directory}")
            return directory
    
    raise FileNotFoundError(f"Could not find project root (searched for {marker_files!r})")

def load_json(file_path):
    """load json file"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_matlab_data(conf: dict, file_name: str = 'hourlyDemandBus.mat', feature_name: str = 'hourlyDemandBus'):
    """ Load matlab data """
    data_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], file_name)
    try:
        logging.info(f'Loading {file_name}')
        mat_data = loadmat(data_path)
        hourly_loads = pd.DataFrame(mat_data[feature_name].T)
        try:
            bus_name_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], 'bus_names.json')
            if os.path.exists(bus_name_path):
                bus_names = load_json(bus_name_path)
                bus_names = [val for _, val in bus_names.items()]
            else:
                logging.info(f"Bus names file not found at {bus_name_path}, using default names")
                bus_names = ['bus_' + str(i) for i in range(hourly_loads.shape[1])]
        except Exception as e:
            logging.warning(f"Error loading bus names: {e}, using default names")
            bus_names = ['bus_' + str(i) for i in range(hourly_loads.shape[1])]
        if len(hourly_loads.columns) == len(bus_names):
            hourly_loads.columns = bus_names
        return hourly_loads

    except FileNotFoundError:
        logging.error(f"File {data_path} not found")
        return None


from scipy import sparse
def as_dense_array(matrix) -> np.ndarray:
    """Convert a SciPy sparse matrix/array or dense object to ndarray."""
    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)

def load_network(conf: dict):
    """ Dynamically load the network using getattr from pandapower networks"""
    try:
        logging.info('Loading network model')
        return getattr(networks, conf["panda_power_case_name"])()
    except AttributeError:
        logging.error(f"Network model {conf['panda_power_case_name']} not found")
        return None


def data_loader(conf_path: str):
    """load data from the configuration file"""
    # Get the project root directory
    project_root = get_project_root()
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(conf_path):
        # Use project root as the base for relative paths
        conf_path = os.path.join(project_root, conf_path)
    
    # Normalize the path to remove any '..' or '.' components
    conf_path = os.path.normpath(conf_path)
    
    logging.info(f"Loading configuration from: {conf_path}")
    conf = load_json(conf_path)
    panda_power_network = load_network(conf)
    try:
        logging.info("running DC power flow calculation....for _ppc initialization.")
        pp.rundcpp(panda_power_network)
    except:
        logging.warning("DC power flow calculation failed for this system. Proceeding without _ppc initialization.")
    df_hourly_nodal_demand = load_matlab_data(conf)
    return panda_power_network, df_hourly_nodal_demand, conf



def plot_show(val, xlb: str = 'x', ylb: str = 'y', ax=None, **kwargs):
    """  Plot and show """
    if ax is None:
        plt.plot(val, **kwargs)
        plt.ylabel(ylb)
        plt.xlabel(xlb)
        plt.grid()
        plt.show()
    else:
        ax.plot(val, **kwargs)
        ax.set_ylabel(ylb)
        ax.set_xlabel(xlb)
        ax.grid()
        return ax


def plot_ecdf_show(val, xlb: str = 'x', ylb: str = 'ecdf', **kwargs):
    """  Plot the empirical cumulative distribution function (ECDF) for given values.   """
    # Prepare ECDF plotting
    ecdf = ECDF(val)
    x = np.sort(val)  # Sort the values for plotting
    y = ecdf(x)  # Get ECDF values for x

    plt.step(x, y, where='post', **kwargs)  # Use step plot for ECDF
    plt.ylabel(ylb)
    plt.xlabel(xlb)
    plt.grid(True)
    plt.show()




