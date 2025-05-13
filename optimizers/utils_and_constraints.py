from gurobipy import GRB, quicksum
from pra_psa.utils.utils import *


def my_quick_sum_dic(var, N1, N2=None, N3=None):
    if N2 is None and N3 is None:
        return quicksum(var[n1] for n1 in N1)
    elif N3 is None:
        return quicksum(var[n1, n2] for n2 in N2 for n1 in N1)
    else:
        return quicksum(var[n1, n2, n3] for n3 in N3 for n2 in N2 for n1 in N1)


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


def calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand, c=None):
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

    inflows_mat, generation_all = [], []
    for t in T:
        # Flow values for current time step
        flows_t = np.array([f[t, l] for l in names['lines']]) if c is None else np.array([f[t, l, c] for l in names['lines']])

        # Calculate inflows at each bus
        flows_con_t = [np.sum(S_T[b_idx, :][np.argwhere(S_T[b_idx, :]).flatten()] * flows_t[np.argwhere(S_T[b_idx, :]).flatten()])
                       for b_idx in range(len(names['buses']))]
        # Calculate generation at each bus
        if c is None:
            generators_t = [pgen[t, gen_to_node[b]] if b in gen_to_node else 0 for b in names['buses']]
        else:
            generators_t = [pgen[t, gen_to_node[b], c] if b in gen_to_node else 0 for b in names['buses']]

        inflows_mat.append(flows_con_t)
        generation_all.append(generators_t)

    inflows_mat = np.squeeze(np.array(inflows_mat))  # Convert lists to NumPy arrays
    generation_all = np.squeeze(np.array(generation_all))  #  and remove any extra dimensions

    nodal_balance = demand.values - inflows_mat - generation_all  # Calculate nodal balance
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




import numpy as np
from scipy.sparse import coo_matrix, hstack


def build_nodal_incidence_matrix(names, g2bus, branches, T):
    """
    names['buses']: list of bus labels
    g2bus:  list of length G mapping each generator index → its bus label
    branches: list of (from_bus, to_bus) for each line
    T:       list or range of time‐steps
    """
    buses = names['buses']
    Bn, Tn = len(buses), len(T)
    Gn, Ln = len(g2bus), len(branches)

    bus2idx = {b:i for i,b in enumerate(buses)}

    rows, cols, data = [], [], []

    # --- 1) Flow incidence (size TB × (T·L)) ---
    # for each time t and branch ℓ=(i→j),
    #   injection at j += f_{t,ℓ}, at i -= f_{t,ℓ}
    for t_idx, t in enumerate(T):
        base_row = t_idx * Bn
        base_col = t_idx * Ln
        for ℓ, (i,j) in enumerate(branches):
            # out of i
            rows.append(base_row + bus2idx[i])
            cols.append(base_col + ℓ)
            data.append(-1)
            # into j
            rows.append(base_row + bus2idx[j])
            cols.append(base_col + ℓ)
            data.append(+1)

    M_flow = coo_matrix(
        (data, (rows, cols)),
        shape=(Tn*Bn, Tn*Ln)
    )

    # --- 2) Generation incidence (size TB × (T·G)) ---
    rows, cols, data = [], [], []
    for t_idx, t in enumerate(T):
        base_row = t_idx * Bn
        base_col = t_idx * Gn
        for g_idx, bus in enumerate(g2bus):
            rows.append(base_row + bus2idx[bus])
            cols.append(base_col + g_idx)
            data.append(+1)         # generation adds injection

    M_gen = coo_matrix(
        (data, (rows, cols)),
        shape=(Tn*Bn, Tn*Gn)
    )

    # --- 3) Horizontal stack to get full M ---
    # Decision vector x_f should be ordered [f..., pgen...]
    M = hstack([M_flow, M_gen], format='csr')
    return M
