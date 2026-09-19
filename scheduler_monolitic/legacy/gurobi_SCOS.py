from gurobipy import Model, GRB, quicksum, GurobiError
from visualization.visualize_schedule import visualize_results
from utils.utils import *
from scheduler_monolitic.run_optimizer import get_and_save_solution

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def initialize_variables(M, names, T):
    """ define VARIABLES for the SCOS problem """
    xt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
    sxt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="start_outage_indicator")
    ext = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="end_outage_indicator")
    pgen = M.addVars(T, names['generators'], lb=0, name="power_generation")
    pgen_c = M.addVars(T, names['generators'], names['contingencies'], lb=0, name="power_generation_contingency")
    d_wc = M.addVars(T, names['buses'], lb=0, name="loss_of_load")
    d_wc_c = M.addVars(T, names['buses'], names['contingencies'], lb=0, name="loss_of_load_contingency")
    f = M.addVars(T, names['lines'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl")
    f_c = M.addVars(T, names['lines'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl_contingency")
    logger.info(f' \n'
                f'{blue_c}Variables Added:{reset_c}\n'
                f' - {blue_c}xt, sxt, ext{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                f' - {blue_c}pgen, pgen_c{reset_c}: Generated power planned and N-1 states\n'
                f' - {blue_c}d_cut, d_cut_c{reset_c}: loss of load planned and N-1 states\n'
                f' - {blue_c}f, f_c{reset_c}: Power flow in planned and N-1 states\n')
    return M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c


def calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand, c = None):
    """  Calculate inflows, generation, and nodal balance for a given set of parameters.
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
        if c is None:
            flows_t = np.array([f[t, l] for l in names['lines']])  # Flow values for current time step
        else:
            flows_t = np.array([f[t, l, c] for l in names['lines']])  # Flow values for current time step

        # Calculate inflows at each bus
        flows_con_t = [np.sum(S_T[b_idx, :][np.argwhere(S_T[b_idx, :]).flatten()] *
                              flows_t[np.argwhere(S_T[b_idx, :]).flatten()]) for b_idx in range(len(names['buses'])) ]

        if c is None: # Calculate generation at each bus
            generators_t = [pgen[t, gen_to_node[b]] if b in gen_to_node else 0 for b in names['buses']]
        else:
            generators_t = [pgen[t, gen_to_node[b], c] if b in gen_to_node else 0 for b in names['buses']]

        inflows_mat.append(flows_con_t)
        generation_all.append(generators_t)

    inflows_mat = np.squeeze(np.array(inflows_mat))  # Convert lists to NumPy arrays and remove any extra dimensions
    generation_all = np.squeeze(np.array(generation_all))
    return demand.values - inflows_mat - generation_all  # Calculate nodal balance


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
            if (g in names['outages']):
                M = add_gen_bounds(M, pgen, p_max, p_min, t, g, c=None, xt=xt[t, g])
            else:
                M = add_gen_bounds(M, pgen, p_max, p_min, t, g, c=None, xt=0)

            for c in names['contingencies']:  # Constraints with/without scheduled outage under N-1 unplanned failure
                if c[3:] == g:
                    M.addConstr(pgen_c[t, g, c] == 0, name=f"GenNull_{t}_{g}")
                else:
                    if g in names['outages']:
                        M = add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c, xt=xt[t, g])
                    else:
                        M = add_gen_bounds(M, pgen_c, p_max, p_min, t, g, c=c, xt=0)
    return M


def add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for l in names['lines']:
            if l in names['outages']:
                M = add_flow_up_low(M, f, f_lim, t, l, xt=xt[t, l])
            else:
                M = add_flow_up_low(M, f, f_lim, t, l)

            for c in names['contingencies']:
                if c[3:] == l:
                    M.addConstr(f_c[t, l, c] == 0, name=f"Flow_{t}_{l}_{c}_MIN")
                else:
                    if l in names['outages']:
                        M = add_flow_up_low(M, f_c, f_lim, t, l, c=c, xt=xt[t, l])
                    else:
                        M = add_flow_up_low(M, f_c, f_lim, t, l, c=c)
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


def define_objective_fun(T, xt, priority, step_cost_outage, names, VOLL, d_wc, d_wc_c, pgen, pgen_c):
    """ OBJECTIVE FUNCTION:
      1) Maximize number of scheduled outages weighted by priority
      2) Value of loss load (VOLL). Minimize
      3) Minimize "unitary" generation cost
     """
    # todo: check weights and scaling factors
    # todo: move cost of PM to budget constraint

    # 1) PM
    nt = len(T)
    # * step_cost_outage[o]
    Objective_fun = quicksum(
        (nt - t_idx) / nt * xt[t, o] * priority[o] for t_idx, t in enumerate(T) for o in names['outages'])

    # 2) VoLL
    scale_weight1 = (nt * len(names['buses']))
    scale_weight2 = (scale_weight1 * len(names['contingencies']))
    Objective_fun -= VOLL * quicksum(d_wc[t, n] for n in names['buses'] for t in T) / scale_weight1  # Add the curtailment term
    Objective_fun -= VOLL * quicksum(d_wc_c[t, n, c] for n in names['buses'] for c in names['contingencies'] for t in T)/scale_weight2

    # 3) Operational costs
    Objective_fun -= quicksum(pgen[t, g] for g in names['generators'] for t in T) / scale_weight1
    Objective_fun -= quicksum(
        pgen_c[t, g, c] for g in names['generators'] for c in names['contingencies'] for t in T) / scale_weight2
    return Objective_fun


def prepare_guroby_SCOS_data(data, names):
    # Preprocess the data
    net = data['network']
    _ppc_internal = net._ppc["internal"]
    max_tasks = data['max_number_of_maintenance_tasks']
    T = [f'step_{t}' for t in range(len(data['nodal_demand']))]
    p_max = {gn: (v + 100 if v > 0 else 200) for gn, v in zip(names['generators'], net.gen['max_p_mw'])}
    p_min = {gn: v * 0 for gn, v in zip(names['generators'], net.gen['min_p_mw'])}  # todo fixme
    f_lim = {ln: v for ln, v in zip(names['lines'], data['branch_capacity'])}
    g2bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]

    # Precompute generator to node mapping
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}

    #S = net._ppc["internal"]['Cft'].A.T  # S[l,b]=1 if line l 'enter' bus b, -1 if it 'exit' bus b
    #B_mat = np.real(_ppc_internal['Bf'].A)

    # B_mat = np.real(_ppc_internal['Bf'].A)
    S = as_dense_array(_ppc_internal["Cft"]).T
    Bf = as_dense_array(_ppc_internal["Bf"])
    B_mat = np.real(Bf)

    B_lines = np.max(B_mat, axis=1)
    if names['contingencies'] is None:  # Generator indices
        names['contingencies'] = [f'n1_{l}' for l in names['lines']] + [f'n1_{g}' for g in names['generators']]

    # Pre-fetch some values for efficiency
    priority = {o_nam: data['outages']['priorities'][o] for o, o_nam in enumerate(names['outages'])}
    durations = {o_nam: data['outages']['expected_duration_steps'][o] for o, o_nam in enumerate(names['outages'])}
    step_cost_outage = {o_nam: data['outages']['cost_per_step'][o] for o, o_nam in enumerate(names['outages'])}

    return T, max_tasks, p_max, p_min, f_lim, g2bus, gen_to_node, S, B_mat, B_lines, priority, durations, step_cost_outage


def deterministic_SCOS_gurobi(data, VOLL=1e7, use_DC_PF=False, save_res_name=None):
    """ Deterministic security-constrained outage planner"""
    # todo:
    #  1. check if adding DC power flow equations and voltage angles makes it more interesting.
    #  2. added big-M linear constraints to replace quadratic y(1-x) <= b terms on the dc power flow equations

    names = {
            'outages': data['outages']['names'],  # List of outage names
            'lines': [f'line_{ll}' for ll in range(data['num_branches'])],  # List of line names
            'contingencies': data.get('n_minus1_names', None),  # List of contingency names
            'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus names
            'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()],  # List of generator names
            }

    # get data ready for the optimization
    (T, max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_guroby_SCOS_data(data, names)

    # ---- START ----
    # PROBLEM: Deterministic Security-Constrained Outage Scheduling problem
    try:
        # ---- define the model
        M = Model("Transmission_Outage_Scheduling_Deterministic")
        # ---- VARIABLES
        M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = initialize_variables(M, names, T)

        # ----  OBJECTIVE FUNCTION ---- :
        Objective_fun = define_objective_fun(T, xt, priority, step_cost_outage, names, VOLL, d_wc, d_wc_c, pgen, pgen_c)
        M.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c} Objective Function Defined: ∑ Outage_NumberCost_Priority'
            f' - ∑ LossOfLoad(N-1 and Normal) - '
            f' - ∑ CostOperations(Pgen) {reset_c}\n'
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

        if use_DC_PF:
            # add voltage phase variables
            theta = M.addVars(T, names['buses'], lb=-45, ub=+45, name="voltage_phase")
            theta_c = M.addVars(T, names['buses'], names['contingencies'], lb=-45, ub=+45, name="voltage_phase_contingency")

            # start bigM formulation
            bigM = 5e3
            z_aux_pf = M.addVars(T, names['outages'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"z_aux_pf")
            z_aux_pf_c = M.addVars(T, names['outages'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"z_aux_pf")

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

                    is_outage = l in outages_set # Outage check
                    outage_factor = (1 - xt[t, l]) if is_outage else 1
                    B_delta_theta = np.dot(B_ij, v_phases_t[idx_con_ij])  # todo:check if linearization worked (* outage_factor)

                    # Add linear constraints for z[t, l]
                    if l in outages_set:
                        # Constrain z_aux_pf to represent (1 - x[t, l]) * delta_theta
                        M.addConstr(z_aux_pf[t, l] <= B_delta_theta)
                        M.addConstr(z_aux_pf[t, l] >= B_delta_theta - bigM * xt[t, l])
                        M.addConstr(z_aux_pf[t, l] <= bigM * (1 - xt[t, l]))
                        M.addConstr(z_aux_pf[t, l] >= -bigM * (1 - xt[t, l]))
                        aux_var = z_aux_pf[t, l]
                    else:
                        aux_var = B_delta_theta  # No outage

                    M.addConstr(f[t, l] == aux_var, name=f"Flow_{t}_{l}_DC_eq")
                    # todo: check if Add constraints for main flow has been linearized
                    # M.addConstr(f[t, l] == B_delta_theta, name=f"Flow_{t}_{l}_DC_eq")

                    # Contingency loop
                    for c in contingencies_set:
                        # Gather contingency phases
                        if c[3:] == l:  # Contingency affecting the line
                            M.addConstr(f_c[t, l, c] == 0, name=f"Flow_{t}_{l}_{c}_DC_eq")
                        else:
                            v_phases_t_cont = np.array([theta_c[t, b, c] for b in names['buses']])
                            B_delta_theta_con = np.dot(B_ij, v_phases_t_cont[idx_con_ij])  # todo: check all is good zio can * outage_factor

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


        # ----  SOLVE the M
        M.setParam('MIPGap', 0.05)  #  Acceptable optimality gap
        M.setParam('Heuristics', 0.5)  # Emphasize heuristics
        M.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
        M.setParam('Presolve', 2)  # Enable aggressive pre-solve
        M.setParam('Threads', 8)  # Use 8 threads for parallel computation
        M.setParam('TimeLimit', 3600)  # Set a one-hour time limit

        # Check optimization status
        save_res_dir = ("../outputs/schedule_results/monolitic_scheduler/optim_sol_det_" +
                        data['config']['case_name'] + '_' + data['config']['aggregation_time'] + ".json")

        M.update()

        M.optimize()


        solution = get_and_save_solution(M, save_res_dir=save_res_dir,
                                         case_name=data['config']['case_name'],
                                         aggregation_time=data['config']['aggregation_time'])

        if solution is not None:
            (LOADed_SOLUTION, results_dictionary) = post_process_results(save_res_dir, names, T=T)
            visualize_results(results_dictionary, names)
            return results_dictionary, solution
        else:
            return None, solution

    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except Exception as e:
        print(e)


def post_process_results(res_path_name, names, T):
    SOLUTION = load_json(res_path_name)
    x_temp = []
    for o in names['outages']:
        x_temp.append([SOLUTION[f'planned_outage_indicator[{t},{o}]'] for t in T])
    X_OutageSchedule = pd.DataFrame(x_temp, index=names['outages'], columns=T)

    x_temp = []
    for l in names['lines']:
        x_temp.append([SOLUTION[f'flow_tl[{t},{l}]'] for t in T])
    FLOWS = pd.DataFrame(x_temp, index=names['lines'], columns=T)

    x_temp = []
    for c in names['contingencies']:
        x_temp.append(
            [sum([SOLUTION[f'loss_of_load_contingency[{t},{b},{c}]'] for b in names['buses']]) for t in
             T])
    WC_CURTAIL_CON = pd.DataFrame(x_temp, index=names['contingencies'], columns=T)
    WC_CURTAILED = pd.DataFrame(
        [sum([SOLUTION[f'loss_of_load[{t},{b}]'] for b in names['buses']]) for t in T], index=T).T

    x_temp = []
    for g in names['generators']:
        x_temp.append([SOLUTION[f'power_generation[{t},{g}]'] for t in T])
    GENERATION = pd.DataFrame(x_temp, index=names['generators'], columns=T)

    GEN_PLUS_CURTAILED = (GENERATION.sum().values + WC_CURTAILED.values)[0]

    results_dictionary = {
        "X_OutageSchedule": X_OutageSchedule,
        "PowerGenerated": GENERATION,
        "Line_Flows": FLOWS,
        "WC_CURTAIL": WC_CURTAILED,
        "WC_CURTAIL_CON": WC_CURTAIL_CON,
        "GEN_PLUS_CURTAILED": GEN_PLUS_CURTAILED
    }

    return SOLUTION, results_dictionary


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

