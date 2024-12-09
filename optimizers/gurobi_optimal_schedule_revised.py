from IPython.core.pylabtools import figsize
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm
from determinisitc_outage_schedule import *

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()

# Example Usage
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """
    # Load data
    network, hourly_demand, config = data_loader('../config/conf_IEEE24.json')
    aggregation_step = config['aggregation_time']  # 'W', 'D', 'H
    cost_per_days = [1000, 2000, 1000, 1000, 2000, 2000, 5000, 5000]
    expected_duration_days = [25, 7, 55, 7, 30, 30, 30, 60]
    daily_demand = aggregate_hourly_demand(hourly_demand, aggregation_step=aggregation_step)
    if aggregation_step == 'H':
        daily_demand = daily_demand.iloc[:4000, :]  # limit the number of steps
        cost_per_step = [c_day / 24 for c_day in cost_per_days]
        expected_duration_steps = [d * 24 for d in expected_duration_days]
    elif aggregation_step == 'W':
        cost_per_step = [c_day * 7 for c_day in cost_per_days]
        expected_duration_steps = [int(d / 7) for d in expected_duration_days]
    else:
        cost_per_step = cost_per_days
        expected_duration_steps = expected_duration_days

    # example scheduled outage information
    outages = {'indices': [0, 1, 2, 3, 5, 8, 1, 2],
               'names': ['line_0', 'line_1', 'line_2', 'line_3', 'line_5', 'line_8', 'gen_1', 'gen_2'],
               'type': ['line', 'line', 'line', 'line', 'line', 'line', 'generator', 'generator'],
               'expected_duration_steps': expected_duration_steps,  # step_are_in_days for now
               'cost_per_step': cost_per_step,
               'priorities': [1, 2, 1, 3, 2, 1, 1, 3]}  # todo: this could be used 'in combination' with the step number

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [200 for _ in
                                                                                                  network.trafo]

    data = {'max_number_of_maintenance_tasks': 2,
            'nodal_demand': daily_demand,
            'outages': outages,
            'config': config,
            'network': network,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(5)]]}

    dic_res = det_optimization_model_SCOP_with_v_ph(data)
    dic_res
