import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum, GurobiError
from utils.dataloader import *
from tqdm import tqdm
import seaborn as sbn


def my_quick_sum_dic(var, N1, N2=None, N3=None):
    if N2 is None and N3 is None:
        return quicksum(var[n1] for n1 in N1)
    elif N3 is None:
        return quicksum(var[n1, n2] for n2 in N2 for n1 in N1)
    else:
        return quicksum(var[n1, n2, n3] for n3 in N3 for n2 in N2 for n1 in N1)


def calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand, c = None):
    """
    Calculate inflows, generation, and nodal balance for a given set of parameters.

    Parameters:
        T (list): Time steps.
        f (ndarray): Flow data.
        names (dict): Dictionary containing 'lines' and 'buses'.
        S_T (ndarray): Connectivity matrix for buses and lines.
        pgen (ndarray): Generation data.
        gen_to_node (dict): Mapping of generators to bus nodes.
        demand (DataFrame): Demand values.

    Returns:
        nodal_balance (ndarray): The calculated nodal balance.
    """
    inflows_mat = []
    generation_all = []

    for t in T:
        if c is None:
            flows_t = np.array([f[t, l] for l in names['lines']])  # Flow values for current time step
        else:
            flows_t = np.array([f[t, l, c] for l in names['lines']])  # Flow values for current time step

        # Calculate inflows at each bus
        flows_con_t = [
            np.sum(S_T[b_idx, :][np.argwhere(S_T[b_idx, :]).flatten()] *
                   flows_t[np.argwhere(S_T[b_idx, :]).flatten()])
            for b_idx in range(len(names['buses']))
        ]

        # Calculate generation at each bus
        if c is None:
            generators_t = [pgen[t, gen_to_node[b]] if b in gen_to_node else 0 for b in names['buses']]
        else:
            generators_t = [pgen[t, gen_to_node[b], c] if b in gen_to_node else 0 for b in names['buses']]

        inflows_mat.append(flows_con_t)
        generation_all.append(generators_t)

    # Convert lists to NumPy arrays and remove any extra dimensions
    inflows_mat = np.squeeze(np.array(inflows_mat))
    generation_all = np.squeeze(np.array(generation_all))

    # Calculate nodal balance
    nodal_balance = demand.values - inflows_mat - generation_all

    return nodal_balance


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
            M = add_gen_bounds(M, pgen, p_max, p_min, t, g, xt=xt[t, g]) if (g in names['outages']) else add_gen_bounds(
                M, pgen, p_max, p_min, t, g)
            for c in names['contingencies']:  # Constraints with/without scheduled outage under N-1 unplanned failure
                if c[3:] == g:
                    M.addConstr(pgen_c[t, g, c] == 0, name=f"GenNull_{t}_{g}")
                else:
                    if g in names['outages']:
                        M = add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c, xt=xt[t, g])
                    else:
                        M = add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c)
    return M


def add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for l in names['lines']:
            if l in names['outages']:
                M = add_flow_con_up_low(M, f, f_lim, t, l, xt=xt[t, l])
            else:
                M = add_flow_con_up_low(M, f, f_lim, t, l)

            for c in names['contingencies']:
                if c[3:] == l:
                    M.addConstr(f_c[t, l, c] ==0, name=f"Flow_{t}_{l}_{c}_MIN")
                else:
                    if l in names['outages']:
                        M = add_flow_con_up_low(M, f_c, f_lim, t, l, c=c, xt=xt[t, l])
                    else:
                        M = add_flow_con_up_low(M, f_c, f_lim, t, l, c=c)
    return M


def add_gen_bounds(M, gen_var, p_max, p_min, t, g, c=None, xt=0.0):
    if c is None:
        M.addConstr(gen_var[t, g] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_UP")
        M.addConstr(gen_var[t, g] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_LOW")
    else:
        M.addConstr(gen_var[t, g, c] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_UP")
        M.addConstr(gen_var[t, g, c] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_LOW")
    return M


def add_flow_con_up_low(M, flow_var, f_lim, t, l, c=None, xt=0.0):
    if c is None:
        M.addConstr(flow_var[t, l] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_UP")
        M.addConstr(flow_var[t, l] >= -f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_LOW")
    else:
        M.addConstr(flow_var[t, l, c] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_{c}_UP")
        M.addConstr(flow_var[t, l, c] >= -f_lim[l] * (1 - xt), name=f"Flow_{c}_{t}_{c}_LOW")
    return M


def add_nodal_power_balance_constraints(M, names, S_T, demand, pgen, pgen_c, f, f_c, g2bus, d_wc, d_wc_c, T):
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}  # Precompute generator to node mapping
    nodal_balance = calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand)
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100,
                         colour="green"):  # forall time steps
        for b_idx, b in enumerate(names['buses']):  # Add each constraint individually
            M.addConstr(nodal_balance[t_idx, b_idx] <= d_wc[t, b], name=f'Power_Balance_{t}_{b}_up')
            M.addConstr(nodal_balance[t_idx, b_idx] >= -d_wc[t, b], name=f'Power_Balance_{t}_{b}_low')

    for c in tqdm(names['contingencies'],  desc="Adding Constraints", total=len(names['contingencies']), ncols=100, colour="green"):  # Inflow vector for contingency case
        nodal_balance = calculate_nodal_balance(T, f_c, names, S_T, pgen_c, gen_to_node, demand, c=c)
        for t_idx, t in enumerate(T):  # forall time steps
            for b_idx, b in enumerate(names['buses']):
                M.addConstr(nodal_balance[t_idx, b_idx] <= d_wc_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_up")
                M.addConstr(nodal_balance[t_idx, b_idx] >= -d_wc_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_low")

    return M


def add_line_dc_power_flow_constraints(M, names, xt, f_lim, f, f_c, theta, theta_c, B_mat, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        v_phases_t = np.array([theta[t, b] for b in names['buses']])
        for l_idx, l in enumerate(names['lines']):
            flow_dc = np.dot(B_mat[l_idx, :], v_phases_t)  # flow_dc = B_l theta_i - B_l theta_j
            if l in names['outages']:
                M = add_flow_con_up_low(M, f, f_lim, t, l, xt=xt[t, l])
                M.addConstr(f[t, l] == flow_dc * (1-xt[t, l]), name=f'Power_Flow_{t}_line_{l}')
            else:
                M = add_flow_con_up_low(M, f, f_lim, t, l)
                M.addConstr(f[t, l] == flow_dc, name=f'Power_Flow_{t}_line_{l}')

    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for c in names['contingencies']:
            v_phases_tc = np.array([theta_c[t, b, c] for b in names['buses']])
            for l_idx, l in enumerate(names['lines']):
                flow_dc_con = np.dot(B_mat[l_idx, :], v_phases_tc)  # flow_dc_con = B_l theta_i_con - B_l theta_j_con
                if c[3:] == l:
                    M.addConstr(f_c[t, l, c] == 0, name=f"Flow_{t}_{l}_{c}_Null")
                else:
                    if l in names['outages']:
                        M = add_flow_con_up_low(M, f_c, f_lim, t, l, c=c, xt=xt[t, l])
                        M.addConstr(f_c[t, l, c] == flow_dc_con * (1 - xt[t, l]), name=f'Power_Flow_{t}_line_{l}_{c}')
                    else:
                        M = add_flow_con_up_low(M, f_c, f_lim, t, l, c=c)
                        M.addConstr(f_c[t, l, c] == flow_dc_con, name=f'Power_Flow_{t}_line_{l}_{c}')
    return M


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
        x_temp.append(
            [sum([LOADed_SOLUTION[f'worst_case_curtailment_contingency[{t},{b},{c}]'] for b in names['buses']]) for t in
             T])
    WC_CURTAIL_CON = pd.DataFrame(x_temp, index=names['contingencies'], columns=T)
    WC_CURTAILED = pd.DataFrame(
        [sum([LOADed_SOLUTION[f'worst_case_curtailment[{t},{b}]'] for b in names['buses']]) for t in T], index=T).T

    x_temp = []
    for g in names['generators']:
        x_temp.append([LOADed_SOLUTION[f'power_generation[{t},{g}]'] for t in T])
    GENERATION = pd.DataFrame(x_temp, index=names['generators'], columns=T)

    GEN_PLUS_CURTAILED = (GENERATION.sum().values + WC_CURTAILED.values)[0]

    return LOADed_SOLUTION, X_OutageSchedule, WC_CURTAILED, WC_CURTAIL_CON, GENERATION, FLOWS, GEN_PLUS_CURTAILED


def visualize_results(results_dictionary, names):
    """visualize result of the SCOS problem"""

    X_OutageSchedule = results_dictionary["X_OutageSchedule"]
    PowerGenerated = results_dictionary["PowerGenerated"]
    Line_Flows = results_dictionary["Line_Flows"]
    WC_CURTAIL = results_dictionary["WC_CURTAIL"]
    WC_CURTAIL_CON = results_dictionary["WC_CURTAIL_CON"]
    o_nam, gen_nam, l_nam = names['outages'], names['generators'], names['lines']

    ncol = 2
    n_rows = int(len(o_nam) / ncol)
    fig, ax = plt.subplots(n_rows + 1, ncol, figsize=(12, n_rows * 5))
    ax = ax.flatten()
    for i, o in enumerate(o_nam):
        X_OutageSchedule.loc[o, :].plot(ax=ax[i])
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Outage  ' + o)
        ax[i].grid()
        # Rotate x-tick labels
        ax[i].tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit better
    fig.suptitle('Outage Schedule', fontsize=16)  # Add a main title
    plt.show()

    X_OutageSchedule.sum().plot()
    plt.xlabel('time step')
    plt.ylabel('Number of PM tasks')
    plt.title('Total Outages')
    plt.tight_layout()
    plt.grid()
    plt.show()

    sbn.barplot(X_OutageSchedule.T.sum())
    plt.xlabel('Outages duration')
    plt.ylabel('PM duration [steps]')
    plt.tight_layout()
    plt.grid()
    plt.show()


    plt.plot(WC_CURTAIL_CON, ':d', color='r', alpha=0.3, markerfacecolor='k', markeredgewidth=0.1, markeredgecolor='r')
    plt.title('Curtailed demand under N-1 failures')
    plt.xlabel('N-1 contingency')
    plt.grid()
    plt.xticks(rotation=75)  # Rotate x-ticks by 45 degrees
    plt.tight_layout()
    plt.show()

    #
    # Plot the filtered data
    WC_CURTAIL_CON.loc[(WC_CURTAIL_CON != 0).any(axis=1)].T.plot()  # Filter to show only non-zero rows
    plt.title('Curtailed demand under N-1 failures')
    plt.xlabel('Time step')
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.plot(PowerGenerated, ':x', alpha=0.5, markerfacecolor='k', markeredgewidth=0.1, markeredgecolor='b')
    plt.title('Generation')
    plt.xlabel('Time step')
    plt.grid()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(int(len(l_nam) / 3) + 1, 3, figsize=(20, 15))
    ax = ax.flatten()
    for i, l in enumerate(l_nam):
        Line_Flows.loc[l, :].plot(ax=ax[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel(l)
        ax[i].grid()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(int(len(gen_nam) / 4) + 1, 4, figsize=(20, 15))
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


def define_objective_fun(T, xt, priority, step_cost_outage, names, VOLL, d_wc, d_wc_c, pgen, pgen_c):
    """----  OBJECTIVE FUNCTION ---- :

      1) Maximize number of scheduled outages weighted by priority
      2) Value of loss load (VOLL). Minimize
      3) Minimize "unitary" generation cost
     """
    # 1) PM
    nt = len(T)
    Objective_fun = quicksum(
        (nt - t_idx) / nt * xt[t, o] * priority[o] * step_cost_outage[o] for t_idx, t in enumerate(T) for o in
        names['outages'])

    # 2) VOLL
    scale_weight = (nt + len(names['buses']))
    Objective_fun -= VOLL * quicksum(
        d_wc[t, n] for n in names['buses'] for t in T) / scale_weight  # Add the curtailment term
    Objective_fun -= VOLL * quicksum(
        d_wc_c[t, n, c] for n in names['buses'] for c in names['contingencies'] for t in T) / (
                                 scale_weight + len(names['contingencies']))

    # 3) Operational costs
    Objective_fun -= quicksum(pgen[t, g] for g in names['generators'] for t in T) / scale_weight
    Objective_fun -= quicksum(pgen_c[t, g, c] for g in names['generators'] for c in names['contingencies'] for t in T) / (
                                 scale_weight + len(names['contingencies']))
    return Objective_fun


def get_and_save_solution(M, save_res_dir=None, case_name=None, aggregation_time=None):

    if M.status == GRB.OPTIMAL:
        logger.info(f"{green_c} Optimal solution found! :-) 🎉 :-) {reset_c}")
    elif M.status == GRB.TIME_LIMIT:
        logger.warning(f"{blue_c} Solution found with time limit! {reset_c}")
    else :
        logger.error(f"{red_c} Optimization ended with status: {M.status}{reset_c}")

    if save_res_dir is None:
        save_res_dir = "../data/results/deterministic_optimizer/optimal_solution" + case_name + '_' + aggregation_time + ".json"

    # Check optimization status
    solution = None
    if M.status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT}:
        solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
        with open(save_res_dir, "w") as f:  # Save the solution to a file or database
            logger.info(f"{green_c} saving in: {reset_c} {save_res_dir}")
            json.dump(solution, f)

    elif M.status == GRB.INFEASIBLE:
        logger.warning(f"{red_c} M is Infeasible! :-(:-(:-({reset_c}")
        logger.warning(f"{red_c} Run IIS to find conflicting constraints{reset_c}")
        M.computeIIS()
        logger.warning(f"{red_c} Writing IIS to a file for inspection {reset_c}")
        M.write("M.ilp")
        print("Conflicting constraints are:")
        for c in M.getConstrs():
            if c.IISConstr:
                print(c.ConstrName)

    elif M.status == GRB.UNBOUNDED:
        logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
        with open("M_status.txt", "a") as f:
            f.write("Model is unbounded.\n")

    return solution