import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pandapower.networks as networks


# ANSI escape code for colored text
blue_c, green_c, purple_c, cyan_c, red_c, gray_c = "\033[94m", "\033[92m", "\033[95m", "\033[96m", "\033[91m", " "
bold_c, underline_c, reset_c = "\033[1m", "\033[4m", "\033[0m"

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)


def get_project_root() -> Path:
    """get project root path"""
    return Path(__file__).parent.parent


def load_json(file_path):
    """load json file"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_matlab_data(conf: dict, feature_name: str = 'hourlyDemandBus'):
    """ Load matlab data """
    data_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], 'hourlyDemandBus.mat')
    try:
        logging.info('Loading hourly demand profiles')
        mat_data = loadmat(data_path)
        hourly_loads = pd.DataFrame(mat_data[feature_name].T)
        try:
            bus_name_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], 'bus_names.json')
            bus_names = load_json(bus_name_path)
            bus_names = [val for _, val in bus_names.items()]
        except:
            bus_names = ['bus_' + str(i) for i in range(hourly_loads.shape[1])]
        if len(hourly_loads.columns) == len(bus_names):
            hourly_loads.columns = bus_names
        return hourly_loads

    except FileNotFoundError:
        logging.error(f"File {data_path} not found")
        return None


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
    conf = load_json(conf_path)
    panda_power_network = load_network(conf)
    df_hourly_nodal_demand = load_matlab_data(conf)
    return panda_power_network, df_hourly_nodal_demand
