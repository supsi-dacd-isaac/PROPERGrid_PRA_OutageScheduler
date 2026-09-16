from gurobi_SCOS import *
from optimizers.gurobi_SCOS_cvar import *
from utils.data_preporcess import *
import matplotlib.pyplot as plt
from visualization.monolithic_scheduler_visualization import (
    plot_comparison_CVAR_DET_SCOS, plot_risk_pdf_cdf,  visualize_results, )

# Example How to run
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """

    # Load system data and historical nodal demand data
    network, hourly_demand, config = data_loader('./config/IEEE24_scheduler.json')
    aggregation_step = config['aggregation_time']  # 'W', 'D', '12H', 'H,  etc.

    # Define PM activities (planned outages)
    comp_types = ['line', 'line', 'line', 'line', 'line', 'line', 'gen', 'gen']
    comp_ids = [0, 1, 2, 3, 5, 8, 1, 2]
    cost_per_days = [1000, 2000, 1000, 1000, 2000, 2000, 5000, 5000]  # m.u/day
    expected_duration_days = [25, 7, 55, 7, 30, 30, 30, 60]  # days/PM
    priorities = [1, 2, 1, 3, 2, 1, 1, 3]  # high = 3, medium = 2, low = 1
    # planned_outage_names = ['line_0', 'line_1', 'line_2', 'line_3', 'line_5', 'line_8', 'gen_1', 'gen_2']
    planned_outage_names = [f'{type}_{idx}' for idx, type, in zip(comp_ids, comp_types)]

    # aggregate power demand data (.mean()  .max()) over aggregation_step in { 'W', 'D', '12H',..., 'H'}
    nodal_demand_aggregated = aggregate_hourly_demand(hourly_demand*0.8,  aggregation_step=aggregation_step)

    # scale PM params: [cost/step], duration [steps]
    PM_cost_per_step, PM_duration_steps = aggregate_step_costs_and_durations(cost_per_days,
                                                                             expected_duration_days,
                                                                             aggregation_step=aggregation_step)

    # example scheduled outage information
    outages = {'indices': comp_ids,  'names': planned_outage_names,
               'type': comp_types,  'expected_duration_steps': PM_duration_steps,
               'cost_per_step': PM_cost_per_step,  'priorities': priorities}

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in network.trafo]

    DATA = {'max_number_of_maintenance_tasks': 2,
            'nodal_demand': nodal_demand_aggregated,
            'outages': outages,
            'config': config,
            'network': network,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(30)]]}

    # Run the Gurobi optimizations
    VoLL, n_samples = 1e6, 10
    use_DC_PF = False
    dic_res_det = deterministic_SCOS_gurobi(DATA, VOLL=VoLL, use_DC_PF=use_DC_PF)
    dic_res_cvar  = CVAR_SCOS_gurobi(DATA, VOLL=VoLL, use_DC_PF=use_DC_PF, alpha=0.9, n_samples=100)

    plot_comparison_CVAR_DET_SCOS(dic_res_cvar, dic_res_det, DATA)