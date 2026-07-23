from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def initialize_variables(model, data):
    """Define VARIABLES for the SCOS problem"""

    T = data['T']
    names = data['names']

    variables = {
        'xt': model.addVars(T, names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator"),
        'sxt': model.addVars(T, names['outages'], vtype=GRB.BINARY, name="start_outage_indicator"),
        'ext': model.addVars(T, names['outages'], vtype=GRB.BINARY, name="end_outage_indicator"),
        'pgen': model.addVars(T, names['generators'], lb=0, name="power_generation"),
        'pgen_c': model.addVars(T, names['generators'], names['contingencies'], lb=0, name="power_generation_contingency"),
        'd_wc': model.addVars(T, names['buses'], lb=0, name="loss_of_load"),
        'd_wc_c': model.addVars(T, names['buses'], names['contingencies'], lb=0, name="loss_of_load_contingency"),
        'f': model.addVars(T, names['lines'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl"),
        'f_c': model.addVars(T, names['lines'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl_contingency")
    }
    return model, variables


def define_objective_fun(model, data, variables):
    """  Build and return:
      - Objective_fun: maximize weighted outages minus VOLL and generation costs
      - terms: dict of individual Gurobi Lin Expr's for reporting
    """

    #  T, xt, priority, step_cost_outage,  names, VOLL, d_wc, d_wc_c, pgen, pgen_c
    names, T, VOLL = data['names'], data['T'], data['VoLL']

    (max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_SCOS_data(data, names)

    xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = (variables[k] for k in (
        'xt', 'sxt', 'ext', 'pgen', 'pgen_c', 'd_wc', 'd_wc_c', 'f', 'f_c'))

    # Initialize the objective function

    nt     = len(T)
    buses  = names['buses']
    outages = names['outages']
    gens   = names['generators']
    conts  = names['contingencies']

    # pre‑compute denominators
    denom_base = nt * len(buses)
    denom_cont = denom_base * len(conts)

    # helper to sum a var over any cartesian product of index sets
    def sum_over(var, *sets):
        return quicksum(var[idx] for idx in product(*sets))

    # 1) weighted scheduled outages
    pm = quicksum((nt - i) / nt * priority[o] * xt[t,o]
                  for i, t in enumerate(T) for o in outages)

    # 2) VOLL terms (normal + N‑1)
    voll    = VOLL * sum_over(d_wc,    T, buses) / denom_base
    voll_c  = VOLL * sum_over(d_wc_c,  T, buses, conts) / denom_cont

    # 3) generation cost terms (normal + N‑1)
    gen     = sum_over(pgen,   T, gens) / denom_base
    gen_c   = sum_over(pgen_c, T, gens, conts) / denom_cont

    # assemble objective
    obj = pm - (voll + voll_c + gen + gen_c)

    terms = {
        'PM': pm,
        'VoLL': voll,
        'VoLL Contingency':  voll_c,
        'Generation Cost': gen,
        'Generation Cost Contingency': gen_c,
    }
    model.setObjective(obj, GRB.MAXIMIZE)

    return model, obj, terms


def get_flows_constraints_t(S_T, b_idx, flows_t):
    F1 = S_T[b_idx, :][np.argwhere(S_T[b_idx, :]).flatten()]
    F2 = flows_t[np.argwhere(S_T[b_idx, :]).flatten()]
    return np.sum(F1*F2)


def get_gen_constraints_t(pgen, t, g2n, c, bus_names):
    if c is None:  # Calculate generation at each bus
        generators_t = [pgen[t, g2n[b]] if b in g2n else 0 for b in bus_names]
    else:
        generators_t = [pgen[t, g2n[b], c] if b in g2n else 0 for b in bus_names]
    return generators_t


def calculate_nodal_balance(T, f, names, S_T, pgen, g2n, demand, c=None):
    """  Calculate inflows, generation, and nodal balance for a given set of parameters.
    Parameters:
        T (list): Time steps.
        f (ndarray): Flow data.
        names (dict): Dictionary containing 'lines' and 'buses'.
        S_T (ndarray): Connectivity matrix for buses and lines.
        pgen (ndarray): Generation data.
        g2n (dict): Mapping of generators to bus nodes.
        demand (DataFrame): Demand values.
    Returns: nodal_balance (ndarray): The calculated nodal balance.
    """
    inflows_mat, generation_all = [], []
    for t in T:
        # prepare flows and generation at each time step
        flows_t = np.array([f[t, li] for li in names['lines']]) if c is None else np.array([f[t, li, c] for li in names['lines']])
        flows_con_t = [get_flows_constraints_t(S_T, b_idx, flows_t) for b_idx in range(len(names['buses']))]
        generators_t = get_gen_constraints_t(pgen, t, g2n, c, names['buses'])
        inflows_mat.append(flows_con_t)
        generation_all.append(generators_t)
    inflows_mat = np.squeeze(np.array(inflows_mat))  # Convert lists to NumPy arrays
    generation_all = np.squeeze(np.array(generation_all))  # and remove any extra dimensions
    nodal_balance = demand.values - inflows_mat - generation_all   # Calculate nodal balance
    return nodal_balance


def calculate_nodal_balance_vectorized(T, f, names, S_T, pgen, g2n, demand_df, c=None):
    """
    Symbolic nodal balance calculation.
    Returns: 2D list [Tn][Bn] of Gurobi LinExpr expressions
    """
    B = names['buses']
    L = names['lines']
    Tn = len(T)
    Bn = len(B)

    balance_matrix = [[None for _ in range(Bn)] for _ in range(Tn)]

    for t_idx, t in enumerate(T):
        # Flow vector for time t
        flows_t = [f[t, l] if c is None else f[t, l, c] for l in L]

        # Generation per bus at time t
        gen_expr = get_gen_constraints_t(pgen, t, g2n, c, B)

        for b_idx in range(Bn):
            # Flow contribution for bus b
            conn_indices = np.argwhere(S_T[b_idx, :]).flatten().tolist()
            flow_expr = sum(flows_t[i] for i in conn_indices)

            demand_val = demand_df.iloc[t_idx, b_idx]
            balance_matrix[t_idx][b_idx] = demand_val - flow_expr - gen_expr[b_idx]

    return balance_matrix


def add_planned_outages_constraints(M, o_nam, xt, sxt, ext, max_tasks, durations, T):
    """ Add planned outage constraints to the problem """
    for t in T:
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


def add_planned_outages_constraints_reduced_vars_example(M, outages, xt, max_tasks, durations, T):
    """Use only xt[t,o] + two kinds of constraints – no sxt/ext needed...TO BE VERIFIED."""
    # 1) No more than max_tasks outages active in any t
    M.addConstrs((quicksum(xt[t, o] for o in outages) <= max_tasks for t in T),name="MaxTasks")
    # 2) Exactly durations[o] periods of outage o
    M.addConstrs((quicksum(xt[t, o] for t in T) == durations[o]for o in outages), name="Duration")
    # 3) Contiguity: each 1 must have a 1 neighbor
    M.addConstrs((xt[T[i], o] <= xt[T[i-1], o] + xt[T[i+1], o] for o in outages for i in range(1, len(T)-1)),  name="Adjacency")
    M.addConstrs((xt[T[0], o] <= xt[T[1], o] for o in outages),  name="AdjStart")  #    For the two ends:
    M.addConstrs((xt[T[-1], o] <= xt[T[-2], o] for o in outages), name="AdjEnd")
    return M


def add_flow_lims(M, flow_var, f_lim, t, l, c=None, xt=0.0):
    if c is None:
        M.addConstr(flow_var[t, l] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_UP")
        M.addConstr(flow_var[t, l] >= -f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_LOW")
    else:
        M.addConstr(flow_var[t, l, c] <= f_lim[l] * (1 - xt), name=f"Flow_{t}_{l}_{c}_UP")
        M.addConstr(flow_var[t, l, c] >= -f_lim[l] * (1 - xt), name=f"Flow_{c}_{t}_{c}_LOW")
    return M


def add_gen_lims(M, gen_var, p_max, p_min, t, g, c=None, xt=0.0):
    if c is None:
        M.addConstr(gen_var[t, g] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_UP")
        M.addConstr(gen_var[t, g] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_LOW")
    else:
        M.addConstr(gen_var[t, g, c] <= p_max[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_UP")
        M.addConstr(gen_var[t, g, c] >= p_min[g] * (1 - xt), name=f"Gen_{t}_{g}_{c}_LOW")
    return M


def add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for l in names['lines']:
            M = add_flow_lims(M, f, f_lim, t, l, xt=xt[t, l]) if l in names['outages'] else add_flow_lims(M, f, f_lim,
                                                                                                          t, l)
            for c in names['contingencies']:
                if c[3:] == l:
                    M.addConstr(f_c[t, l, c] == 0, name=f"Flow_{t}_{l}_{c}_MIN")
                else:
                    M = add_flow_lims(M, f_c, f_lim, t, l, c=c, xt=xt[t, l]) \
                        if l in names['outages'] else add_flow_lims(M, f_c, f_lim, t, l, c=c)
    return M


def add_generators_constraints(M, xt, names, pgen, pgen_c, p_max, p_min, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):

        for g in names['generators']:  # Constraints with/without scheduled outage
            if (g in names['outages']):
                M = add_gen_lims(M, pgen, p_max, p_min, t, g, c=None, xt=xt[t, g])
            else:
                M = add_gen_lims(M, pgen, p_max, p_min, t, g, c=None, xt=0)

            for c in names['contingencies']:  # Constraints with/without scheduled outage under N-1 unplanned failure
                if c[3:] == g:
                    M.addConstr(pgen_c[t, g, c] == 0, name=f"GenNull_{t}_{g}")
                else:
                    if g in names['outages']:
                        M = add_gen_lims(M, pgen_c, p_max, p_min, t, g, c=c, xt=xt[t, g])
                    else:
                        M = add_gen_lims(M, pgen_c, p_max, p_min, t, g, c=c, xt=0)
    return M


def add_nodal_power_balance_constraints(M, names, S_T, demand, pgen, pgen_c, f, f_c, g2bus, d_wc, d_wc_c, T):
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}  # Precompute generator to node mapping
    nodal_balance = calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand)
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100,
                         colour="green"):  # forall time steps
        for b_idx, b in enumerate(names['buses']):  # Add each constraint individually
            M.addConstr(nodal_balance[t_idx, b_idx] == d_wc[t, b], name=f'Power_Balance_{t}_{b}_up')

    for c in tqdm(names['contingencies'],  desc="Adding Constraints", total=len(names['contingencies']), ncols=100, colour="green"):  # Inflow vector for contingency case
        nodal_balance = calculate_nodal_balance(T, f_c, names, S_T, pgen_c, gen_to_node, demand, c=c)
        for t_idx, t in enumerate(T):  # forall time steps
            for b_idx, b in enumerate(names['buses']):
                M.addConstr(nodal_balance[t_idx, b_idx] == d_wc_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_up")
    return M


def add_DC_power_flow_constraints(M, names, B_mat, T, xt, f, f_c):
    # add voltage phase variables
    theta = M.addVars(T, names['buses'], lb=-45, ub=+45, name="voltage_phase")
    theta_c = M.addVars(T, names['buses'], names['contingencies'], lb=-45, ub=+45, name="voltage_phase_contingency")

    # start bigM formulation
    bigM = 5e3
    z_aux_pf = M.addVars(T, names['outages'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"z_aux_pf")
    z_aux_pf_c = M.addVars(T, names['outages'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY,
                           name=f"z_aux_pf")

    # add dc power flow constraints
    # 1) Bij(theta[t,i] - theta[t,j])*(1-x[t,(ij)]) = f[t,(ij)]   --- here (ij) is a line from i to j
    # 2) Bij(theta[t,i,c] - theta[t,j,c])*(1-x[t,(ij)]) = f_c[t,(ij),c]
    idx_con_ij_list = [B_mat[l_idx, :] != 0 for l_idx in range(len(names['lines']))]
    B_ij_list = [B_mat[l_idx, idx_con_ij] for l_idx, idx_con_ij in enumerate(idx_con_ij_list)]
    outages_set = set(names['outages'])
    contingencies_set = set(names['contingencies'])
    logger.info('adding DC-PF constraints, normal + N-1 failures')
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints",
                         total=len(T), ncols=100, colour="green"):

        # Gather v_phases_t once per timestep
        v_phases_t = np.array([theta[t, b] for b in names['buses']])

        for l_idx, l in enumerate(names['lines']):
            idx_con_ij = idx_con_ij_list[l_idx]
            B_ij = B_ij_list[l_idx]

            is_outage = l in outages_set  # Outage check
            outage_factor = (1 - xt[t, l]) if is_outage else 1
            B_delta_theta = np.dot(B_ij, v_phases_t[idx_con_ij])  # todo:check if linear  worked (* outage_factor)

            # Add linear constraints for z[t, l]
            if l in outages_set:
                M.addConstr(z_aux_pf[t, l] <= B_delta_theta)
                M.addConstr(z_aux_pf[t, l] >= B_delta_theta - bigM * xt[t, l])
                M.addConstr(z_aux_pf[t, l] <= bigM * (1 - xt[t, l]))
                M.addConstr(z_aux_pf[t, l] >= -bigM * (1 - xt[t, l]))
                aux_var = z_aux_pf[t, l]
            else:
                aux_var = B_delta_theta  # No outage

            M.addConstr(f[t, l] == aux_var, name=f"Flow_{t}_{l}_DC_eq")

            # Contingency loop
            for c in contingencies_set:
                # Gather contingency phases
                if c[3:] == l:  # Contingency affecting the line
                    M.addConstr(f_c[t, l, c] == 0, name=f"Flow_{t}_{l}_{c}_DC_eq")
                else:
                    v_phases_t_cont = np.array([theta_c[t, b, c] for b in names['buses']])
                    B_delta_theta_con = np.dot(B_ij, v_phases_t_cont[idx_con_ij])  # todo: proporcoden * outage_factor
                    # Add linear constraints for z[t, l]
                    if l in outages_set:
                        # Constrain z_aux_pf to represent (1 - x[t, l]) * delta_theta
                        M.addConstr(z_aux_pf_c[t, l, c] <= B_delta_theta_con)
                        M.addConstr(z_aux_pf_c[t, l, c] >= B_delta_theta_con - bigM * xt[t, l])
                        M.addConstr(z_aux_pf_c[t, l, c] <= bigM * (1 - xt[t, l]))
                        M.addConstr(z_aux_pf_c[t, l, c] >= -bigM * (1 - xt[t, l]))
                        aux_var = z_aux_pf_c[t, l, c]
                    else:
                        aux_var = B_delta_theta_con  # No outage
                    M.addConstr(f_c[t, l, c] == aux_var, name=f"Flow_{t}_{l}_{c}_DC_eq")



def add_deterministic_constraints(Model, data, Variables):

    names, T, VoLL = data['names'], data['T'], data['VoLL']
    # NAMES
    (max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S_connectivity, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_SCOS_data(data, names)  # DATA

    xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = (Variables[k] for k in (
    'xt', 'sxt', 'ext', 'pgen', 'pgen_c', 'd_wc', 'd_wc_c', 'f', 'f_c'))

    # ----  CONSTRAINTS:
    logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c} '
                f'adding constraints for scheduled outages: '
                f'duration, number of tasks, continuity')

    Model = add_planned_outages_constraints(M=Model, o_nam=names['outages'], xt=xt,
                                            sxt=sxt, ext=ext, max_tasks=max_tasks,
                                            durations=durations, T=T)

    logger.info('adding power production constraints, normal + N-1 failures')
    Model = add_generators_constraints(M=Model, xt=xt, names=names, pgen=pgen,
                                       pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

    logger.info('adding line flow constraints, normal + N-1 failures')
    Model = add_line_power_limit_constraints(M=Model, xt=xt, names=names, f_lim=f_lim,
                                             f=f, f_c=f_c, T=T)

    logger.info('adding node balance constraints, normal + N-1 failures')
    Model = add_nodal_power_balance_constraints(Model, names, S_connectivity,
                                                demand=data['nodal_demand'],
                                                pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                                g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c,
                                                T=T)
    if data['use_DC_PF']:
        logger.info('adding DC power flow constraints')
        add_DC_power_flow_constraints(M=Model, names=names, B_mat=B_mat, T=T, xt=xt,
                                      f=f, f_c=f_c)

    return Model


def add_CVaR(model, data, Variables, d_samples, prob_scen,
             tail_prob=0.95, cvar_limit=None, objective_fun=None):
    """Add CVaR and balance constraints for each demand scenario"""

    logger = logging.getLogger(__name__)

    names, T = data['names'], data['T']

    # Prepare data and extract variables
    (_, _, _, _,
     g2bus, gen_to_node, S_connectivity,
     _, _, _, _, _) = prepare_SCOS_data(data, names)

    pgen, f = Variables['pgen'], Variables['f']

    # Precompute generator to node mapping
    gen_to_node = {b: names['generators'][i] for i, b in enumerate(g2bus)}

    # Get dimensions
    B, S = names['buses'], names['demand_scenarios']
    Tn, Bn, Sn = len(T), len(B), len(S)

    # Add variables
    d_wc_scenario = model.addVars(T, B, S, lb=0, name="Loss_of_bus_load_scenarios")
    excess = model.addVars(T, B, S, lb=0, name="Excess_over_VaR")
    VaR = model.addVars(T, B, lb=-float('inf'), name="Value_at_Risk")  # Unbounded VaR
    CVaR = model.addVars(T, B, lb=0, name="CVaR")

    logger.info('Adding demand balance constraints per scenario')
    for s_idx, s in tqdm(enumerate(S), desc="Scenario Constraints",
                         total=Sn, ncols=100, colour="blue"):
        nodal_balance = calculate_nodal_balance(T, f, names, S_connectivity, pgen,
                                                gen_to_node, d_samples[s_idx])
        # Vectorized balance constraints
        for t_idx, t in enumerate(T):
            for b_idx, b in enumerate(B):
                # Power balance constraints and excess over VaR definition
                model.addConstr(nodal_balance[t_idx, b_idx] <= d_wc_scenario[t, b, s],
                                name=f'Balance_{t}_{b}_{s}_up')
                model.addConstr(nodal_balance[t_idx, b_idx] >= -d_wc_scenario[t, b, s],
                                name=f'Balance_{t}_{b}_{s}_low')
                model.addConstr(excess[t, b, s] >= d_wc_scenario[t, b, s] - VaR[t, b],
                                name=f"Excess_{t}_{b}_{s}")

    # CVaR definition constraints (corrected formulation)
    logger.info('Adding CVaR constraints')
    for t in T:
        for b in B:
            model.addConstr(
                CVaR[t, b] == VaR[t, b] + (1 / tail_prob) *
                quicksum(prob_scen[s] * excess[t, b, s] for s in S),
                name=f"CVaR_def_{t}_{b}"
            )

    # Handle CVaR in objective or constraints

    if cvar_limit is None:
        if objective_fun is None:
            raise ValueError(
                "objective_fun must be provided if CVaR is part of the objective.")
        total_cvar = quicksum(CVaR[t, b] for t in T for b in B)
        objective_fun -= total_cvar
        logger.info("Including CVaR in the objective function")
    else:
        for t_idx, t in enumerate(T):
            for b_idx, b in enumerate(B):
                model.addConstr(CVaR[t, b] <= cvar_limit,
                                name=f"CVaR_limit_{t}_{b}")
        #total_cvar = quicksum(CVaR[t, b] for t in T for b in B)
        #model.addConstr(total_cvar <= cvar_limit, name="CVaR_limit")
        logger.info(f"Added CVaR constraint with limit: {cvar_limit}")
    return model, CVaR
