from gurobipy import Model, GRB, quicksum, GurobiError
from utils.utils import *
from optimizers.run_optimizer import  get_and_save_solution
from visualization.visualize_schedule import visualize_results
import seaborn as sbn
from gurobi_SCOS import (define_objective_fun, load_json, initialize_variables,
                         calculate_nodal_balance, prepare_guroby_SCOS_data, add_planned_outages_constraints, add_nodal_power_balance_constraints,
                         add_generators_constraints, add_line_power_limit_constraints)

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


def plot_risk_pdf_cdf(CVARisk):

    # Aggregate CVaR across all buses for each time step
    cvar_over_time = CVARisk.sum(axis=0)  # Sum of CVaR for all buses at each step

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(cvar_over_time, marker='o', linestyle='-', color='red')
    plt.title("CVaR Evolution Over Time (Global Aggregation)", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Total CVaR", fontsize=12)

    # Set custom ticks for every 30 steps
    step_interval = 30
    ticks = range(0, len(cvar_over_time), step_interval)
    plt.xticks(ticks, labels=[f"step_{i}" for i in ticks], rotation=45)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Flatten the CVARisk dataframe into a single array
    all_bus_data = CVARisk.values.flatten()
    # Create the combined distribution plot
    plt.figure(figsize=(10, 8))
    sbn.histplot(all_bus_data, kde=True, bins=30, color='green', alpha=0.7)
    plt.title("Combined CVaR Distribution for All Buses", fontsize=14)
    plt.xlabel("CVaR", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    plt.figure(figsize=(10, 8))
    for bus in CVARisk.index:
        sbn.ecdfplot(CVARisk.loc[bus], label=bus, linewidth=1.5)
    plt.title("CVaR Distribution Across All Buses", fontsize=14)
    plt.xlabel("CVaR", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Buses", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Select a specific bus for visualization

    for b_idx, bus_to_plot in enumerate(CVARisk.index):
        bus_data = CVARisk.loc[bus_to_plot]
        plt.figure(figsize=(8, 6))
        sbn.histplot(bus_data, kde=True, bins=10, color='blue', alpha=0.7)
        plt.title(f"CVaR Distribution for {bus_to_plot}", fontsize=14)
        plt.xlabel("CVaR", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


def post_process_results_cvar(res_path_name, names, T):
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

    cvar_temp = []
    for g in names['buses']:
        cvar_temp.append([SOLUTION[f'CVaR[{t},{g}]'] for t in T])
    CVARisk = pd.DataFrame(cvar_temp, index=names['buses'], columns=T)

    results_dictionary = {
        "X_OutageSchedule": X_OutageSchedule,
        "PowerGenerated": GENERATION,
        "Line_Flows": FLOWS,
        "WC_CURTAIL": WC_CURTAILED,
        "WC_CURTAIL_CON": WC_CURTAIL_CON,
        "GEN_PLUS_CURTAILED": GEN_PLUS_CURTAILED,
        "CVARisk": CVARisk,
    }

    return SOLUTION, results_dictionary


def CVAR_SCOS_gurobi(data,
                     n_samples:int=20,
                     alpha=0.1,
                     VOLL=1e7, use_DC_PF=True, save_res_name=None):
    """   Probabilistic Security-Constrained Outage Scheduling problem with CVaR missmatch minimization"""
    #  nodal demand samples are included

    names = {
            'outages': data['outages']['names'],  # List of outage names
            'lines': [f'line_{ll}' for ll in range(data['num_branches'])],  # List of line names
            'contingencies': data.get('n_minus1_names', None),  # List of contingency names
            'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus names
            'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()],  # List of generator names
            'demand_scenarios': [f'demand_{s}' for s in range(n_samples)]  # List of demand scenarios
            }

    (T, max_tasks, p_max, p_min, f_lim,
     g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_guroby_SCOS_data(data, names)

    # ---- START ----
    # PROBLEM: Probabilistic Security-Constrained Outage Scheduling problem with CVaR missmatch minimization
    try:
        # ---- define the model
        M = Model("Transmission_Outage_Scheduling_CVaR")

        M, xt, sxt, ext, pgen, pgen_c, d_wc, d_wc_c, f, f_c = initialize_variables(M, names, T) # ---- VARIABLES

        # ----  OBJECTIVE FUNCTION ---- :
        Objective_fun = define_objective_fun(T, xt, priority, step_cost_outage, names, VOLL, d_wc, d_wc_c, pgen, pgen_c)

        # --------------  CVaR minimization
        # Define the loss of load variables for each bus, time step, and scenario
        d_wc_scenario = M.addVars(T, names['buses'], names['demand_scenarios'], lb=0, name="Loss_of_bus_load_scenarios")
        VarDev = M.addVars(T, names['buses'], names['demand_scenarios'], vtype=GRB.CONTINUOUS, lb=0, name="Deviation_from_VaR")
        CVaR = M.addVars(T, names['buses'], vtype=GRB.CONTINUOUS, lb=0, name="CVaR")
        # Add aux vars,  VaR = loss of load planned case (d_wc)
        M.addConstrs(d_wc_scenario[t, b, s] - d_wc[t, b] <= VarDev[t, b, s] for t in T for b in names['buses'] for s in names['demand_scenarios'])
        # Define deviation variables and alpha percentile for CVaR
        M.addConstrs(d_wc[t, b] + (1 / (len(names['demand_scenarios']) * alpha)) *
                     quicksum(VarDev[t, b, s] for s in names['demand_scenarios']) == CVaR[t, b] for t in T for b in names['buses'])
        # add to objective function --> Minimize CVaR (overall risk)
        Objective_fun -= quicksum(CVaR[t, b] for t in T for b in names['buses'])

        M.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' - VarDev, CVaR \n'
            f'{blue_c} Objective Function Defined: ∑ Outage_NumberCost_Priority'
            f' - ∑ LossOfLoad(N-1 and Normal) - '
            f' - ∑ CostOperations(Pgen) '
            f' - ∑ CVaR(LossOfLoad){reset_c}\n'
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
            theta = M.addVars(T, names['buses'], lb=-5, ub=+5, name="voltage_phase")
            theta_c = M.addVars(T, names['buses'], names['contingencies'], lb=-5, ub=+5, name="voltage_phase_contingency")
            # add dc power flow constraints
            idx_con_ij_list = [B_mat[l_idx, :] != 0 for l_idx in range(len(names['lines']))]
            B_ij_list = [B_mat[l_idx, idx_con_ij] for l_idx, idx_con_ij in enumerate(idx_con_ij_list)]
            outages_set = set(names['outages'])
            contingencies_set = set(names['contingencies'])
            logger.info('adding DC-PF constraints, normal + N-1 failures')
            for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
                # Gather v_phases_t once per timestep
                v_phases_t = np.array([theta[t, b] for b in names['buses']])
                for l_idx, l in enumerate(names['lines']):
                    idx_con_ij = idx_con_ij_list[l_idx]
                    B_ij = B_ij_list[l_idx]

                    # Outage check
                    is_outage = l in outages_set
                    outage_factor = 1 - xt[t, l] if is_outage else 1
                    sum_B_times_Delta_Theta = np.dot(B_ij, v_phases_t[idx_con_ij]) * outage_factor
                    # Add constraints for main flow
                    M.addConstr(f[t, l] == sum_B_times_Delta_Theta, name=f"Flow_{t}_{l}_DC_eq")

                    # Contingency loop
                    for c in contingencies_set:
                        # Gather contingency phases
                        v_phases_t_cont = np.array([theta_c[t, b, c] for b in names['buses']])
                        sum_B_times_Delta_Theta_con = np.dot(B_ij, v_phases_t_cont[idx_con_ij]) * outage_factor

                        if c[3:] == l:  # Contingency affecting the line
                            M.addConstr(f_c[t, l, c] == sum_B_times_Delta_Theta_con * 0, name=f"Flow_{t}_{l}_{c}_DC_eq")
                        else:
                            M.addConstr(f_c[t, l, c] == sum_B_times_Delta_Theta_con, name=f"Flow_{t}_{l}_{c}_DC_eq")

        logger.info('adding node balance constraints for all demand scenarios (normal planned not contingencies)')

        nodal_demand_samples = demand_sampler(data['nodal_demand'], n_samples=n_samples, random_seed=42)
        gen_to_node = {b: names['generators'][idx] for idx, b in
                       enumerate(g2bus)}  # Precompute generator to node mapping

        for s_idx, s in tqdm(enumerate(names['demand_scenarios']), desc="Adding Constraints", total=n_samples, ncols=100,colour="green"):  # forall time steps
            nodal_balance = calculate_nodal_balance(T, f, names, S, pgen, gen_to_node, nodal_demand_samples[s_idx])
            for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):  # forall time steps
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

        # Check optimization status
        save_res_dir = ("../data/schedule_results/cvar_optimizer/optimal_solution" +
                        data['config']['case_name'] + '_' + data['config']['aggregation_time'] + ".json")

        solution = get_and_save_solution(M, save_res_dir=save_res_dir,
                                         case_name=data['config']['case_name'],
                                         aggregation_time=data['config']['aggregation_time'])

        if solution is not None:
            (LOADed_SOLUTION, results_dictionary) = post_process_results_cvar(save_res_dir, names, T=T)
            visualize_results(results_dictionary, names)
            plot_risk_pdf_cdf(results_dictionary['CVARisk'])
            return results_dictionary, solution
        else:
            return None, solution

    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except Exception as e:
        print(e)


