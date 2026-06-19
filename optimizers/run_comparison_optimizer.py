from optimizers.gurobi_SCOS_cvar import *
from utils.data_preporcess import aggregate_hourly_demand, aggregate_step_costs_and_durations

def plot_comparison_CVAR_DET_SCOS(dic_res_cvar, dic_res_det, DATA):

    """ show some plots for a comparison between CVAR and DETERMINISTIC SCOScheduler"""
    names = {
        'outages': DATA['outages']['names'],  # List of outage names
        'lines': [f'line_{ll}' for ll in range(DATA['num_branches'])],  # List of line names
        'contingencies': DATA.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(DATA['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in DATA['network'].gen.index.tolist()],  # List of generator names
    }
    o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']

    n_col = 2
    n_rows = int(len(o_nam) / n_col)
    fig, ax = plt.subplots(n_rows, n_col, figsize=(10, n_rows * 4))
    ax = ax.flatten()
    for results_dictionary, style in zip([dic_res_det[0], dic_res_cvar[0]], ['-', '--']):

        X_OutageSchedule = results_dictionary["X_OutageSchedule"]
        o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']

        # Loop through outages and plot on each axis
        for i, o in enumerate(o_nam):
            if i < len(ax):  # Ensure we don't index beyond available axes
                if i == 0:
                    if style == '-':
                        X_OutageSchedule.loc[o, :].plot(ax=ax[i], linestyle=style, label=f'det-SCOP')
                    else:
                        X_OutageSchedule.loc[o, :].plot(ax=ax[i], linestyle=style, label=f'CVaR-SCOP')
                else:
                    X_OutageSchedule.loc[o, :].plot(ax=ax[i], linestyle=style)
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(f'Outage {o}')
                ax[i].set_title(f'Outage Schedule for {o}')
                ax[i].grid()
                ax[i].legend()  # Add legend
                ax[i].tick_params(axis='x', rotation=45)  # Rotate x-tick labels

    # Hide unused axes if any
    for j in range(len(o_nam), len(ax)):
        fig.delaxes(ax[j])
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle('Outage Schedule', fontsize=16)
    plt.show()

    # Plot the power generation for each generator
    n_line2_plot = 15
    fig, ax = plt.subplots(int(len(l_nam[:n_line2_plot]) / 3), 3, figsize=(20, 15))
    ax = ax.flatten()

    for results_dictionary, style in zip([dic_res_det[0], dic_res_cvar[0]], ['-', '--']):

        Line_Flows = results_dictionary["Line_Flows"]

        for i, l in enumerate(l_nam[:n_line2_plot]):
            if i < len(ax):  # Ensure we don't index beyond available axes
                if i == 0:
                    if style == '-':
                        Line_Flows.loc[l, :].plot(ax=ax[i], linestyle=style, label=f'det-SCOP')
                    else:
                        Line_Flows.loc[l, :].plot(ax=ax[i], linestyle=style, label=f'CVaR-SCOP')
                else:
                    Line_Flows.loc[l, :].plot(ax=ax[i], linestyle=style)
                ax[i].set_xlabel('Time')
                ax[i].set_ylabel(l)
                ax[i].grid()
                ax[i].legend()  # Add legend

    # Adjust layout and add a title
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# Example How to run
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """

    # Load system data and historical nodal demand data
    network, hourly_demand, config = data_loader('../config/conf_IEEE24.json')
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