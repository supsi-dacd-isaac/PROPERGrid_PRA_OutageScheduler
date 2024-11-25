from IPython.core.pylabtools import figsize
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
from tqdm import tqdm
from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from utils_and_constraints import visualize_results
from gurobi_SCOS import (calculate_nodal_balance, prepare_guroby_SCOS_data, add_planned_outages_constraints, add_nodal_power_balance_constraints,
                         add_generators_constraints, add_line_power_limit_constraints, add_gen_bounds)


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def transform_to_fat_tailed(noise, power):
    # Apply a power transformation (e.g., squaring the normal samples)
    fat_tailed_samples = np.sign(noise) * np.abs(noise)**power
    return fat_tailed_samples



def demand_sampler(nodal_demand, n_samples: int, random_seed: int = 42):
    """
    Generate new demand samples for each bus based on the existing `nodal_demand`, by sampling around
    each existing demand value with added noise.  """

    np.random.seed(random_seed)  # Reset the random seed for reproducibility
    nodal_demand_samples = []
    for s_idx in range(n_samples):
        for scenario in range(n_samples):  # For the current bus, sample around each existing load value
            # For each scenario, perturb the value by adding Gaussian noise
            # 5% noise as an example and transform it to a fat-tailed distribution
            noise = np.random.normal(loc=0, scale=0.05 * nodal_demand)
            nodal_demand_samples.append(nodal_demand  + transform_to_fat_tailed(noise, power=1.3))
    return nodal_demand_samples

def CVAR_SCOS_gurobi(data, n_samples=20, save_res_name=None):
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

    names = {
        'outages': data['outages']['names'],  # List of outage names
        'lines': [f'line_{l}' for l in range(data['num_branches'])],  # List of line names
        'contingencies': data.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()],  # List of generator names
        'demand_scenarios': [f'demand_{s}' for s in range(n_samples)]  # List of demand scenarios
    }

    (T, max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_guroby_SCOS_data(data, names)

    # ---- START ---- PROBLEM: SECURITY-Constrained Outage Scheduling with CVaR missmatch minimization
    try:
        # ---- define the model
        M = Model("Transmission_Outage_Scheduling_CVaR")
        # ---- VARIABLES
        xt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
        sxt = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="start_outage_indicator")
        ext = M.addVars(T, names['outages'], vtype=GRB.BINARY, name="end_outage_indicator")
        pgen = M.addVars(T, names['generators'], lb=0, name="power_generation")
        pgen_c = M.addVars(T, names['generators'], names['contingencies'],lb=0, name="power_generation_contingency")
        d_wc = M.addVars(T, names['buses'], lb=0, name="worst_case_curtailment")
        d_wc_c = M.addVars(T, names['buses'], names['contingencies'], lb=0, name="worst_case_curtailment_contingency")
        f = M.addVars(T, names['lines'], lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="flow_tl")
        f_c = M.addVars(T, names['lines'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl_contingency")

        # ----  OBJECTIVE FUNCTION ---- :
        # 1) Maximize scheduled outages * priority & Minimize Load Curtailment
        nt = len(T)
        Objective_fun = quicksum((nt - t_idx) / nt * xt[t, o] * priority[o] * step_cost_outage[o] for t_idx, t in enumerate(T) for o in names['outages'])

        # 2) Minimize Load Curtailment
        scale_weight = (nt+len(names['buses']))
        Objective_fun -= 1e6 * quicksum(d_wc[t, n] for n in names['buses'] for t in T)/scale_weight  # Add the curtailment term
        Objective_fun -= 1e6 * quicksum(d_wc_c[t, n, c] for n in names['buses'] for c in names['contingencies'] for t in T)/(scale_weight + len(names['contingencies']))

        # 2) Minimize "unitary" generation cost
        Objective_fun -= quicksum(pgen[t, g] for g in names['generators'] for t in T)/scale_weight
        Objective_fun -= quicksum(pgen_c[t, g, c] for g in names['generators'] for c in names['contingencies'] for t in T)/(scale_weight + len(names['contingencies']))

        # --------------  CVaR minimization

        # Define the loss of load variables for each bus, time step, and scenario
        d_wc_scenario = M.addVars(T, names['buses'], names['demand_scenarios'], lb=0, name="Loss_of_bus_load_scenarios")
        # Add auxiliary variables for CVaR calculation
        VarDev = M.addVars(T, names['buses'], names['demand_scenarios'], vtype=GRB.CONTINUOUS, lb=0,
                           name="Deviation_from_VaR")
        # the VAR is the d_wc
        CVaR = M.addVars(T, names['buses'], vtype=GRB.CONTINUOUS, lb=0, name="CVaR")

        alpha = 0.05  # Define alpha for CVaR (tail percentage)
        # Define deviation variables: how much the loss deviates from VaR for each scenario
        M.addConstrs(d_wc_scenario[t, b, s] - d_wc[t, b] <= VarDev[t, b, s] for t in T for b in names['buses'] for s in
                     names['demand_scenarios'])

        M.addConstrs(d_wc[t, b] + (1 / (len(names['demand_scenarios']) * alpha)) * quicksum(
                VarDev[t, b, s] for s in names['demand_scenarios']) == CVaR[t, b]for t in T for b in names['buses']
        )

        # Objective: Minimize CVaR (overall risk)
        Objective_fun -= quicksum(CVaR[t, b] for t in T for b in names['buses'])

        M.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' - xt, sxt, ext: s Scheduled outage decisions, start-end indicators for the outage task\n' 
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
        M = add_line_power_limit_constraints(M, xt, names, f_lim, f, f_c, B_lines, T)

        logger.info('adding node balance constraints, normal + N-1 failures')
        M = add_nodal_power_balance_constraints(M, names, S, demand=data['nodal_demand'],
                                                pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                                g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c, T=T)


        logger.info('adding node balance for each demand scenario')

        nodal_demand_samples = demand_sampler(data['nodal_demand'], n_samples=n_samples, random_seed=42)
        gen_to_node = {b: names['generators'][idx] for idx, b in
                       enumerate(g2bus)}  # Precompute generator to node mapping

        for s_idx, s in tqdm(enumerate(names['demand_scenarios']), desc="Adding Constraints", total=n_samples, ncols=100,colour="green"):  # forall time steps

            nodal_balance = calculate_nodal_balance(T, f, names, S, pgen, gen_to_node, nodal_demand_samples[s_idx])

            for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100,
                                 colour="green"):  # forall time steps
                for b_idx, b in enumerate(names['buses']):  # Add each constraint individually
                    M.addConstr(nodal_balance[t_idx, b_idx] <= d_wc_scenario[t, b, s], name=f'Power_Balance_{t}_{b}_{s}_up')
                    M.addConstr(nodal_balance[t_idx, b_idx] >= -d_wc_scenario[t, b, s], name=f'Power_Balance_{t}_{b}_{s}_low')


        # ----  SOLVE the M
        M.setParam('MIPGap', 0.005)  #  Acceptable optimality gap
        M.setParam('Heuristics', 0.5)  # Emphasize heuristics
        M.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
        M.setParam('Presolve', 2)  # Enable aggressive pre-solve
        M.setParam('Threads', 8)  # Use 8 threads for parallel computation
        M.setParam('TimeLimit', 3600)  # Set a one-hour time limit

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
            with open(save_res_name, "w") as f:  # Save the solution to a file or database
                json.dump(solution, f)

            (LOADed_SOLUTION, X_OutageSchedule, WC_CURTAIL,
             WC_CURTAIL_CON, PowerGenerated, Line_Flows, PROD_and_CURT) = post_process_results(save_res_name, names,
                                                                                               T=T)

            visualize_results(X_OutageSchedule, PowerGenerated, Line_Flows, WC_CURTAIL, WC_CURTAIL_CON, names)

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

    dic_res = CVAR_SCOS_gurobi(data)
    print(dic_res)
