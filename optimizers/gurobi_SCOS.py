from IPython.core.pylabtools import figsize
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm
import seaborn as sbn
import matplotlib.pyplot as plt
import scipy.sparse as sp

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()

def add_planned_outages_constraints(M, o_nam, xt, sxt, ext, max_tasks, durations, T):
    for t in T:  # 1) Max simultaneous outages
        M.addConstr(quicksum(xt[t, o] for o in o_nam) <= max_tasks, name=f"MaxTasks_{t}")
    for o in o_nam:  # 2) Total duration constraint
        M.addConstr(quicksum(xt[t, o] for t in T) == durations[o], name=f"Duration_{o}") 
        for t_idx, t in enumerate(T[:-1]):  #  3) Continuity constraints  - - Ensure we don't go out of bounds
            M.addConstr(sxt[T[t_idx + 1], o] >= sxt[T[t_idx], o], name=f"Cont_start_{o}_{t}")
            M.addConstr(ext[T[t_idx + 1], o] >= ext[T[t_idx], o], name=f"Cont_end_{o}_{t}")
            M.addConstr(ext[T[t_idx + 1], o] <= sxt[T[t_idx], o], name=f"end_only_after_start_{o}_{t}")
    for o in o_nam:
        for t_idx, t in enumerate(T):  # Ensure xt = 1 if started but not ended
            M.addConstr(sxt[t, o] - ext[t, o] == xt[t, o], name=f"Cont_start_end_x_{o}_{t}")
    return M


def add_generators_constraints(M, xt, names, pgen, pgen_c, p_max, p_min, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for g in names['generators']:  # Constraints with/without scheduled outage
            M = add_gen_bounds(M, pgen, p_max, p_min, t, g, xt=xt[t, g]) if (g in names['outages']) else add_gen_bounds(M, pgen, p_max, p_min, t, g)
            for c in names['contingencies']:  # Constraints with/without scheduled outage under N-1 unplanned failure
                if c[3:] == g:  
                    M.addConstr(pgen_c[t, g, c] == 0, name=f"GenNull_{t}_{g}")
                else:
                    M = add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c, xt=xt[t, g]) if (g in names['outages']) else add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c)
    return M


def add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for l in names['lines']:
            M = add_flow_up_low(M, f, f_lim, t, l, xt=xt[t, l]) if (l in names['outages']) else add_flow_up_low(M, f, f_lim, t, l) 
            for c in names['contingencies']:
                if c[3:] == l:
                    M.addConstr(f_c[t, l, c] <= 0.1, name=f"Flow_{t}_{l}_{c}_UP")
                    M.addConstr(f_c[t, l, c] >= -0.1, name=f"Flow_{t}_{l}_{c}_LOW")
                else:
                    M = add_flow_up_low(M, f_c, f_lim, t, l, c=c, xt=xt[t, l]) if (l in names['outages']) else add_flow_up_low(M, f_c, f_lim, t, l, c=c)
    return M


def add_gen_bounds(M, gen_var, p_max, p_min, t, g, c=None, xt=0.0):
    if c is None:
        M.addConstr(gen_var[t, g] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_UP")
        M.addConstr(gen_var[t, g] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_LOW")
    else:
        M.addConstr(gen_var[t, g, c] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_UP")
        M.addConstr(gen_var[t, g, c] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_LOW")
    return M


def add_flow_up_low(M, flow_var, f_lim, t, l, c=None, xt=0.0):
    if c is None:
        M.addConstr(flow_var[t, l] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_UP")
        M.addConstr(flow_var[t, l] >= -f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_LOW")
    else:
        M.addConstr(flow_var[t, l, c] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_{c}_UP")
        M.addConstr(flow_var[t, l, c] >= -f_lim[l]  * (1 - xt), name=f"Flow_{c}_{t}_{c}_LOW")
    return M

 
def add_nodal_power_balance_constraints(M, names, S_T, demand, pgen, pgen_c, f, f_c,
                                        g2bus, d_wc, d_wc_c, T):
 
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}  # Precompute generator to node mapping
    
    inflows_mat = [np.dot(S_T, np.array([f[t, l] for l in names['lines']])) for t in T]
    nodal_balance = demand.values - inflows_mat - [[(pgen[t, gen_to_node[b]] if b in gen_to_node else 0) for b in names['buses']] for t in T]
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):  # forall time steps
        for b_idx, b in enumerate(names['buses']):  # Add each constraint individually
            M.addConstr(nodal_balance[t_idx, b_idx] <= d_wc[t, b], name=f'Power_Balance_{t}_{b}_up')
            M.addConstr(nodal_balance[t_idx, b_idx] >= -d_wc[t, b], name=f'Power_Balance_{t}_{b}_low')

        for c in names['contingencies']:  # Inflow vector for contingency case
            inflows_c = np.dot(S_T, np.array([f_c[t, l, c] for l in names['lines']]))  # Get flows for the contingency case
            generated = [(pgen_c[t, gen_to_node[b], c] if b in gen_to_node else 0) for b in names['buses']]
            net_demand_less_gen = demand.iloc[t_idx, :].values - inflows_c - generated
            for b_idx, b in enumerate(names['buses']):
                M.addConstr(net_demand_less_gen[b_idx] <= d_wc_c[t, b, c],  name=f"Power_Balance_{t}_{b}_{c}_up")
                M.addConstr(net_demand_less_gen[b_idx] >= -d_wc_c[t, b, c],  name=f"Power_Balance_{t}_{b}_{c}_low")
    return M


def optimization_SCOS_M_gurobi(data, save_res_name=None):
    """
    Prepare Gurobi optimization M for:
    security-constrained outage scheduling problem
    This implementation neglects voltage constraints completely..... 
     """
    """  X_sol, D_curt, P_gen = load_and_format_solution('optimal_solution.json', nodes_names, generators_names, outages_names, T)  """
    if save_res_name is None:
        save_res_name = "optimal_solution" + data['config']['case_name'] + '_' + data['config'][
            'aggregation_time'] + ".json"
        save_res_name = '../data/results/deterministic_optimizer/' + save_res_name

    # Preprocess the data
    net = data['network']
    max_tasks = data['max_number_of_maintenance_tasks']

    names = {
        'outages': data['outages']['names'],  # List of outage names
        'lines': [f'line_{l}' for l in range(data['num_branches'])],  # List of line names
        'contingencies': data.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in net.gen.index.tolist()]  # List of generator names
    }

    T = [f'step_{t}' for t in range(len(data['nodal_demand']))]
    p_max = {gn: (v+100 if v > 0 else 200) for gn, v in zip(names['generators'], net.gen['max_p_mw'])}
    p_min = {gn: v*0 for gn, v in zip(names['generators'], net.gen['min_p_mw'])}  # todo fixme
    f_lim = {ln: v for ln, v in zip(names['lines'], data['branch_capacity'])}
    g2bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()] 
    S = net._ppc["internal"]['Cft'].A.T  # Precompute the transposed S matrix to avoid repeated computations  S[l,b]=1 if line l 'enter' bus b, -1 if it 'exit' bus b  
    if names['contingencies'] is None: # Generator indices
        names['contingencies'] = [f'n1_{l}' for l in names['lines']] + [f'n1_{g}' for g in names['generators']]

    # Pre-fetch some values for efficiency
    priority = {o_nam: outages['priorities'][o] for o, o_nam in enumerate(names['outages'])}
    durations = {o_nam: outages['expected_duration_steps'][o] for o, o_nam in enumerate(names['outages'])}
    step_cost_outage = {o_nam: outages['cost_per_step'][o] for o, o_nam in enumerate(names['outages'])}

    # ---- START ---- PROBLEM: Security-Constrained Outage Scheduling problem
    try:
        M = Model("Transmission_Outage_Scheduling")
        # ---- VARIABLES
        xt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
        sxt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="start_outage_indicator")
        ext = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="end_outage_indicator")
        pgen = M.addVars(T, names['generators'], name="power_generation")
        pgen_c = M.addVars(T, names['generators'], names['contingencies'], name="power_generation_contingency")
        d_wc = M.addVars(T, names['buses'], lb=0, name="worst_case_curtailment")
        d_wc_c = M.addVars(T, names['buses'], names['contingencies'], lb=0, name="worst_case_curtailment_contingency")
        f = M.addVars(T, names['lines'], name="flow_tl")
        f_c = M.addVars(T, names['lines'], names['contingencies'], name="flow_tl_contingency")

        # ----  OBJECTIVE FUNCTION: Maximize scheduled outages * priority & Minimize Load Curtailment
        Objective_fun = quicksum((len(T) - t_idx) / len(T) * xt[t, o] * priority[o] * step_cost_outage[o] for t_idx, t in enumerate(T) for o in names['outages'])

        Objective_fun -= quicksum(1e4 * d_wc[t, n] for n in names['buses'] for t in T) # Add the curtailment term
        Objective_fun -= quicksum(1e4 * d_wc_c[t, n, c] for n in names['buses'] for c in names['contingencies'] for t in T)  # Add the curtailment term for N-1 contingencies

        M.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' - xt: Scheduled outage decisions\n'
            f' - sxt, ext: start-end indicators for the outage task\n'
            f' - pgen, pgen_c: Generated power normal and N-1 states\n'
            f' - d_cut, d_cut_c: Worst-cases demand cut normal and N-1 states\n'
            f' - f, f_c: Power flow in normal and N-1 states\n'
            f' - theta, theta_c: Phase angles in normal operation and N-1 states\n'
            f'{blue_c} Objective Function Defined: ∑ Outage_NumberCost_Priority - DemandCurtailed(N-1 and Normal) - CostOperations(Pgen) {reset_c}\n'
        )

        # ----  CONSTRAINTS:
        logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c}')
        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        M = add_planned_outages_constraints(M, names['outages'], xt, sxt, ext, max_tasks, durations, T)

        logger.info('adding power production constraints, normal + N-1 failures')
        M = add_generators_constraints(M, xt, names, pgen=pgen, pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

        logger.info('adding line flow constraints, normal + N-1 failures')
        M = add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, T)

        logger.info('adding node balance constraints, normal + N-1 failures')
        M = add_nodal_power_balance_constraints(M, names, S, demand=data['nodal_demand'], 
                                                pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                                g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c, T=T)

        # ----  SOLVE the M
        M.update()
        M.optimize()

        """M.setParam('DualReductions', 0)
        M.computeIIS()
        M.write('iis.ilp')
        """

        # Check optimization status
        if M.status == GRB.OPTIMAL:
            logger.info(f"{blue_c} Optimal solution found! :-) :-):-){reset_c}")
            solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
            with open(save_res_name, "w") as f: # Save the solution to a file or database
                json.dump(solution, f)

            (LOADed_SOLUTION, X_OutageSchedule, WC_CURTAIL,
             WC_CURTAIL_CON, PowerGenerated, Line_Flows, PROD_and_CURT) = post_process_results(save_res_name, names, T=T)

            visualize_results(LOADed_SOLUTION, X_OutageSchedule, PowerGenerated, Line_Flows, WC_CURTAIL, names)  # plot

            return LOADed_SOLUTION, X_OutageSchedule, WC_CURTAIL, WC_CURTAIL_CON, PowerGenerated

        elif M.status == GRB.INFEASIBLE:
            logger.warning(f"{red_c} Model is infeasible! :-(:-(:-({reset_c}")
            # Run IIS to find conflicting constraints
            M.computeIIS()
            M.write("M.ilp")  # Write IIS to a file for inspection
            print("Conflicting constraints are:")
            for c in M.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
            return None
        elif M.status == GRB.UNBOUNDED:
            logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
            with open("M_status.txt", "a") as f:
                f.write("Model is unbounded.\n")
            return None
        else:
            logger.info(f"Optimization was stopped with status: {M.status}")
            return None

    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except Exception as e:
        print(e)


def post_process_results(res_path_name, names, T):
    LOADed_SOLUTION = load_json(res_path_name)
    x_temp = []
    for o in names['outages']:
        x_temp.append([LOADed_SOLUTION[f'planned_outage_indicator[{t},{o}]'] for t in T])
    X_OutageSchedule = pd.DataFrame(x_temp, index=names['outages'], columns=T)

    x_temp = []
    for l in names['lines']:
        x_temp.append([LOADed_SOLUTION[f'flow_tl[{t},{l}]'] for t in T])
    FLOWS = pd.DataFrame(x_temp, index=names['lines'], columns=T)

    x_temp = []
    for c in names['contingencies']:
        x_temp.append([sum([LOADed_SOLUTION[f'worst_case_curtailment_contingency[{t},{b},{c}]'] for b in names['buses']]) for t in T])
    WC_CURTAIL_CON = pd.DataFrame(x_temp, index=names['contingencies'], columns=T)
    WC_CURTAILED = pd.DataFrame([sum([LOADed_SOLUTION[f'worst_case_curtailment[{t},{b}]'] for b in names['buses']]) for t in T], index=T).T

    x_temp = []
    for g in names['generators']:
        x_temp.append([LOADed_SOLUTION[f'power_generation[{t},{g}]'] for t in T])
    GENERATION = pd.DataFrame(x_temp, index=names['generators'], columns=T)

    GEN_PLUS_CURTAILED = (GENERATION.sum().values + WC_CURTAILED.values)[0]

    return LOADed_SOLUTION, X_OutageSchedule, WC_CURTAILED, WC_CURTAIL_CON, GENERATION, FLOWS, GEN_PLUS_CURTAILED


def visualize_results(LOADed_SOLUTION, X_OutageSchedule, PowerGenerated, Line_Flows, WC_CURTAIL, names):
    """visualize result of the SCOS problem"""
    o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']
    fig, ax = plt.subplots(int(len(o_nam) / 4), 4)
    ax = ax.flatten()
    for i, o in enumerate(o_nam):
        X_OutageSchedule.loc[o, :].plot(ax=ax[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('Outage  ' + o)
        ax[i].grid()
    plt.title('Outage Schedule')
    plt.show()

    X_OutageSchedule.sum().plot()
    plt.xlabel('time step')
    plt.title('Total Outages')
    plt.grid()
    plt.show()

    sbn.barplot(X_OutageSchedule.T.sum())
    plt.xlabel('Outages duration')
    plt.grid()
    plt.show()

    WC_CURTAIL.T.plot()
    plt.title('curtailed demand')
    plt.xlabel('Time step')
    plt.grid()
    plt.show()

    plt.plot(PowerGenerated, ':x')
    plt.title('Generation')
    plt.xlabel('Time step')
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(int(len(l_nam) / 3) + 1, 3, figsize=(20, 10))
    ax = ax.flatten()
    for i, l in enumerate(l_nam):
        Line_Flows.loc[l, :].plot(ax=ax[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('Flow  ' + l)
        ax[i].grid()
    plt.title('LINE FLOWS')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(int(len(gen_nam) / 4) + 1, 4, figsize=(20, 10))
    ax = ax.flatten()
    for i, g in enumerate(gen_nam):
        PowerGenerated.loc[g, :].plot(ax=ax[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('Pgen ' + g)
        ax[i].grid()
    plt.title('Outage Schedule')
    plt.tight_layout()
    plt.show()


def load_and_format_solution(solution_json_path, nodes_names, generators_names, outages, T):
    solution = load_json(solution_json_path)
    x_outage = []
    for o in outages:
        x_outage.append([solution[f'xt_lines_and_gens[{t},{o}]'] for t in T])
    X_sol = pd.DataFrame(x_outage, index=outages, columns=T)

    d_curtailed = []
    for b in nodes_names:
        d_curtailed.append([solution[f'd_curt_tb[{t},{b}]'] for t in T])
    D_curt = pd.DataFrame(d_curtailed, index=nodes_names, columns=T)

    power_gen_tg = []
    for g in generators_names:
        power_gen_tg.append([solution[f'power_gen_tg[{t},{g}]'] for t in T])
    P_gen = pd.DataFrame(power_gen_tg, index=generators_names, columns=T)

    return X_sol, D_curt, P_gen


# Example Usage
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """
    # Load data
    network, hourly_demand, config = data_loader('../config/conf_IEEE24.json')
    aggregation_step = config['aggregation_time']  # 'W', 'D', 'H
    cost_per_days = [1000, 2000, 1000, 1000, 2000, 2000, 5000, 5000]
    expected_duration_days = [25, 7, 55, 7, 30, 30, 30, 60]
    daily_demand = aggregate_hourly_demand(hourly_demand * 0.5, aggregation_step=aggregation_step)
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
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in
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
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(25)]]}

    dic_res = optimization_SCOS_M_gurobi(data)
    print(dic_res)
