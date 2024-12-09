from gurobipy import Model, GRB, GurobiError
from optimizers.utils_and_constraints import *
from utils.data_preporcess import prepare_data_4_gurobi_security_constrained_outage_planning as prepare_data

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def initialize_variables(M, nams, T):
    # ---- VARIABLES
    xt = M.addVars(T, nams['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
    sxt = M.addVars(T, nams['outages'], vtype=GRB.BINARY, name="start_outage_indicator")
    ext = M.addVars(T, nams['outages'], vtype=GRB.BINARY, name="end_outage_indicator")
    pgen = M.addVars(T, nams['generators'], lb=0, name="power_generation")
    pgen_c = M.addVars(T, nams['generators'], nams['contingencies'], lb=0, name="power_generation_contingency")
    d_wc = M.addVars(T, nams['buses'], lb=0, name="worst_case_curtailment")
    d_wc_c = M.addVars(T, nams['buses'], nams['contingencies'], lb=0, name="worst_case_curtailment_contingency")
    f = M.addVars(T, nams['lines'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl")
    f_c = M.addVars(T, nams['lines'], nams['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY,
                    name="flow_tl_contingency")

    logger.info(f' \n'
                f'{blue_c}Variables Added:{reset_c}\n'
                f' - {blue_c}xt, sxt, ext{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                f' - {blue_c}pgen, pgen_c{reset_c}: Generated power normal and N-1 states\n'
                f' - {blue_c}d_cut, d_cut_c{reset_c}: Worst-cases demand cut normal and N-1 states\n'
                f' - {blue_c}f, f_c{reset_c}: Power flow in normal and N-1 states\n'
    )

    return M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c


def objective_function(M, nams, T, priority, step_cost_outage, xt, d_wc, d_wc_c,
                       pgen, pgen_c, cost_delta=1e6):

    # ----  OBJECTIVE FUNCTION ---- :
    # 1) outages costs
    #nT = len(T)
    # Objective_fun = quicksum((nT - t_idx) / nT * xt[t, o] * priority[o] for t_idx, t in enumerate(T) for o in nams['outages'])

    # 2) Minimize Load-Generation Missmatch
    # scaler = (nT + len(nams['buses']))
    # Objective_fun -= cost_delta * quicksum(d_wc[t, n] for n in nams['buses'] for t in T) #/ scaler  # Add the mismatch term
    # Objective_fun -= cost_delta * quicksum(d_wc_c[t, n, c] for n in nams['buses'] for c in nams['contingencies'] for t in T)  #/ (scaler + len(nams['contingencies']))

    # 3) Minimize Operational Generation Costs
    # Objective_fun -= quicksum(pgen[t, g] for g in nams['generators'] for t in T)  # / scaler
    # Objective_fun -= quicksum(pgen_c[t, g, c] for g in nams['generators'] for c in nams['contingencies'] for t in T)  #/ (scaler + len(nams['contingencies']))

    Objective_fun = my_quick_sum_dic(xt, T, N2=nams['outages'], N3=None)
    Objective_fun -= cost_delta * my_quick_sum_dic(d_wc, T, N2=nams['buses'], N3=None)
    Objective_fun -= cost_delta * my_quick_sum_dic(d_wc_c, T, N2=nams['buses'], N3=nams['contingencies'])
    Objective_fun -= my_quick_sum_dic(pgen, T, N2=nams['generators'], N3=None)
    Objective_fun -= my_quick_sum_dic(pgen_c, T, N2=nams['generators'], N3=nams['contingencies'])

    M.setObjective(Objective_fun, GRB.MAXIMIZE)

    logger.info(
        f'{blue_c} Objective Function Defined:{reset_c} \n'
        f'Maximize (total number of planned outages + priorities weight) - \n'
        f'(total Cost of Missmatch demand generation (Normal and N-1)) -\n'
        f'(total Cost of Operations(Pgen)) {reset_c}\n'
    )

    return M

def add_all_constraints(M, nams,
                        xt, sxt, ext, max_tasks, durations,
                        pgen, pgen_c, p_max, p_min,
                        S, data, g2bus, d_wc, d_wc_c, f, f_c, f_lim, T):
    logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
    M = add_planned_outages_constraints(M, nams['outages'], xt, sxt, ext, max_tasks, durations, T)

    logger.info('adding power production constraints, normal + N-1 failures')
    M = add_generators_constraints(M, xt, nams, pgen=pgen, pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

    logger.info('adding line flow constraints, normal + N-1 failures')
    M = add_line_power_limit_constraints(M, xt, nams, f_lim, f, f_c, T)

    logger.info('adding node balance constraints, normal + N-1 failures')
    M = add_nodal_power_balance_constraints(M, nams, S, demand=data['nodal_demand'],
                                            pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                            g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c, T=T)
    return M


def optimization_SCOS_M_gurobi(data, save_res_name=None):
    """Prepare Gurobi optimization M for:  security-constrained outage scheduling problem .This implementation neglects voltage constraints completely"""
    if save_res_name is None:
        save_res_name = "optimal_solution" + data['config']['case_name'] + '_' + data['config'][
            'aggregation_time'] + ".json"
        save_res_name = '../data/results/deterministic_optimizer/' + save_res_name

    nams = {
        'outages': data['outages']['names'],  # List of outage nams
        'lines': [f'line_{l}' for l in range(data['num_branches'])],  # List of line nams
        'contingencies': data.get('n_minus1_nams', None),  # List of contingency nams
        'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus nams
        'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()]  # List of generator nams
    }

    (T, max_tasks, p_max, p_min,
     f_lim, g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_data(data, nams)

    # ---- START ---- PROBLEM: Security-Constrained Outage Scheduling problem
    try:
        M = Model("Transmission_Outage_Scheduling")
        M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = initialize_variables(M, nams, T)

        # ----  OBJECTIVE FUNCTION ---- :
        M = objective_function(M, nams, T, priority, step_cost_outage, xt, d_wc, d_wc_c, pgen, pgen_c, cost_delta=1e4)
        # ----  CONSTRAINTS:
        logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c}')
        M = add_all_constraints(M, nams, xt, sxt, ext, max_tasks, durations, pgen, pgen_c, p_max, p_min, S, data, g2bus, d_wc, d_wc_c, f, f_c, f_lim, T)

        # ----  SOLVE the M
        M.setParam('MIPGap', 0.1)  #  Acceptable optimality gap
        M.setParam('Heuristics', 0.5)  # Emphasize heuristics
        M.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
        M.setParam('Presolve', 2)  # Enable aggressive pre-solve
        M.setParam('Threads', 8)  # Use 8 threads for parallel computation
        M.setParam('TimeLimit', 1200)  # Set a time limit

        M.update()
        M.optimize()

        """M.setParam('DualReductions', 0)
        M.computeIIS()
        M.write('iis.ilp') """

        # Check optimization status
        if M.status == GRB.OPTIMAL:
            logger.info(f"{blue_c} Optimal solution found! :-) :-):-){reset_c}")
            solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
            with open(save_res_name, "w") as f:  # Save the solution to a file or database
                json.dump(solution, f)

            (SOLUTION, X_OP, WC_cut_norm, WC_cut_con, Pgen, Flows, Prod) \
                = post_process_results(save_res_name, nams, T=T)

            visualize_results(X_OP, Pgen, Flows, WC_cut_norm, WC_cut_con, nams)  # plot

            return SOLUTION, X_OP, WC_cut_norm, WC_cut_con, Pgen

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


def deterministic_optimization_model_SCOP_with_voltage_phases(data, save_res_name=None):
    """  Prepare Gurobi optimization M for:  SCOS_M security-constrained outage scheduling problem """

    if save_res_name is None:
        save_res_name = "optimal_solution" + data['config']['case_name'] + '_' + data['config'][
            'aggregation_time'] + ".json"
        save_res_name = '../data/results/deterministic_optimizer/' + save_res_name

    nams = {
        'outages': data['outages']['names'],  # List of outage nams
        'lines': [f'line_{l}' for l in range(data['num_branches'])],  # List of line nams
        'contingencies': data.get('n_minus1_nams', None),  # List of contingency nams
        'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus nams
        'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()]  # List of generator nams
    }

    (T, max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_data(data, nams)

    # ---- START ---- PROBLEM: Security-Constrained Outage Scheduling problem
    try:
        M = Model("Transmission_Outage_Scheduling_with_phases")

        M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = initialize_variables(M, nams, T)  # ---- VARIABLES
        theta = M.addVars(T, nams['buses'], lb=-10, ub=10, name="nodal_phase_tb")
        theta_c = M.addVars(T, nams['buses'], nams['contingencies'], lb=-10, ub=10, name="nodal_phase_tb_contingency")

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' {blue_c}- theta, theta_c {reset_c}: Phase angles in normal operation and N-1 states \n'
        )

        # ----  OBJECTIVE FUNCTION:
        M = objective_function(M, nams, T, priority, step_cost_outage, xt, d_wc, d_wc_c, pgen, pgen_c, cost_delta=1e4)

        # ----  CONSTRAINTS:
        logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c}')
        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        M = add_planned_outages_constraints(M, nams['outages'], xt, sxt, ext, max_tasks, durations, T)

        logger.info('adding power production constraints, normal + N-1 failures')
        M = add_generators_constraints(M, xt, nams, pgen=pgen, pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

        logger.info('adding line flow constraints, normal + N-1 failures')
        M = add_line_dc_power_flow_constraints(M, nams, xt, f_lim, f, f_c, theta, theta_c, B_mat, T)

        logger.info('adding node balance constraints, normal + N-1 failures')
        M = add_nodal_power_balance_constraints(M, nams, S, demand=data['nodal_demand'],
                                                pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                                g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c, T=T)

        # ---- SOLVE the M
        M.update()  # Ensure the model is up-to-date




        # Set default parameters before attempting to load tuning file
        M.setParam('MIPGap', 0.1)  # Increase acceptable optimality gap
        M.setParam('Heuristics', 0.8)  # Emphasize heuristics
        M.setParam('Cuts', 1)  # Allow Gurobi to generate more cuts
        M.setParam('NodeLimit', 1e6)  # Limit nodes explored
        M.setParam('Presolve', 1)  # Moderate presolve
        M.setParam('Threads', 6)  # Use 6 threads for parallel computation
        M.setParam('TimeLimit', 600)  # Set a 10-minute time limit

        try:
            M.read('tune.prm')  # Load tuned parameters if available
        except Exception:
            logger.info("No tune.prm file found, running tuning...")
            M.tune()  # Run tuning to optimize parameter settings
            M.write('tune.prm')  # Save the tuned parameters for future runs

        # Optimize the model
        M.optimize()

        """
        M.setParam('DualReductions', 0)
        M.computeIIS()
        M.write('iis.ilp')
        """

        # Check optimization status
        if M.status == GRB.OPTIMAL:
            logger.info(f"{blue_c} Optimal solution found! :-) :-):-){reset_c}")
            solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
            with open(save_res_name, "w") as f:  # Save the solution to a file or database
                json.dump(solution, f)

            (LOADed_SOLUTION, X_OutageSchedule, WC_CURTAIL,
             WC_CURTAIL_CON, PowerGenerated, Line_Flows, PROD_and_CURT) = post_process_results(save_res_name, nams,
                                                                                               T=T)

            visualize_results(LOADed_SOLUTION, X_OutageSchedule, PowerGenerated, Line_Flows, WC_CURTAIL, nams)  # plot

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
