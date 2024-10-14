from IPython.core.pylabtools import figsize
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm
import seaborn as sbn
import matplotlib.pyplot as plt
import scipy.sparse as sp

from gurobi_SCOS import *

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()



def det_optimization_model_SCOP(data, save_res_name=None):
    """  Prepare Gurobi optimization M for:  SCOS_M security-constrained outage scheduling problem """

    if save_res_name is None:
        save_res_name = "optimal_solution" + data['config']['case_name'] + '_' + data['config'][
            'aggregation_time'] + ".json"
        save_res_name = '../data/results/deterministic_optimizer/' + save_res_name
 
    names = {
        'outages': data['outages']['names'],  # List of outage names
        'lines': [f'line_{l}' for l in range(data['num_branches'])],  # List of line names
        'contingencies': data.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(data['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in data['network'].gen.index.tolist()]  # List of generator names
    }

    (T, max_tasks, p_max, p_min, f_lim, g2bus, gen_to_node, S, B_mat, B_lines,
     priority, durations, step_cost_outage) = prepare_guroby_SCOS_data(data, names)

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
        f = M.addVars(T, names['lines'], lb=-GRB.INFINITY, name="flow_tl")
        f_c = M.addVars(T, names['lines'], names['contingencies'], lb=-GRB.INFINITY, name="flow_tl_contingency")
        theta = M.addVars(T, names['buses'], lb=-10, ub=10, name="nodal_phase_tb")
        theta_c = M.addVars(T, names['buses'], names['contingencies'], lb=-20, ub=20, name="nodal_phase_tb_contingency")

        # ----  OBJECTIVE FUNCTION: Maximize scheduled outages * priority & Minimize Load Curtailment
        Objective_fun = quicksum(
            (len(T) - t_idx) / len(T) * xt[t, o] * priority[o] * step_cost_outage[o] for t_idx, t in enumerate(T) for o
            in names['outages'])

        # minimize load shedding
        scaler = (len(T)+len(names['buses']))
        Objective_fun -= 1e3 * quicksum(d_wc[t, n] for n in names['buses'] for t in T)/scaler  # Add the curtailment term
        Objective_fun -= 1e3 * quicksum(d_wc_c[t, n, c] for n in names['buses'] for c in names['contingencies'] for t in T)/(scaler + len(names['contingencies']))

        # minimize unitary generation cost
        Objective_fun -= quicksum(pgen[t, g] for g in names['generators'] for t in T)/scaler
        Objective_fun -= quicksum(pgen_c[t, g, c] for g in names['generators'] for c in names['contingencies'] for t in T)/(scaler + len(names['contingencies']))

        M.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:\n'
            f' - xt, sxt, ext : Scheduled outage decisions, start-end indicators for the outage task\n' 
            f' - pgen, pgen_c: Generated power normal and N-1 states\n'
            f' - d_wc, d_wc_c: Curtailments normal and N-1 states\n'
            f' - f, f_c: Power flow in normal and N-1 states\n'
            f' - theta, theta_c: Phase angles in normal operation and N-1 states\n'
            f'Objective Function Defined: ∑ Outage_NumberCost_Priority - DemandCurtailed(N-1 and Normal) - CostOperations(Pgen) {reset_c}\n'
        )

        # ----  CONSTRAINTS:
        logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c}')
        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        M = add_planned_outages_constraints(M, names['outages'], xt, sxt, ext, max_tasks, durations, T)

        logger.info('adding power production constraints, normal + N-1 failures')
        M = add_generators_constraints(M, xt, names, pgen=pgen, pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

        logger.info('adding line flow constraints, normal + N-1 failures')
        M = add_line_constraints(M, xt, names, f_lim, f, f_c, theta, theta_c, B_mat, T)

        logger.info('adding dc power flow constraints, normal + N-1 failures')
        M = add_DCPF_constraints(M, data['nodal_demand'], B_mat, pgen, d_wc, theta, pgen_c, d_wc_c, theta_c, names, g2bus, T)

        # ----  SOLVE the M
        M.update()

        M.setParam('MIPGap', 0.001)  # Increase acceptable optimality gap
        M.setParam('Heuristics', 0.5)  # Emphasize heuristics
        M.setParam('Cuts', 2)  # Allow Gurobi to generate more cuts
        M.setParam('Presolve', 2)  # Enable aggressive presolve
        M.setParam('Threads', 8)  # Use 8 threads for parallel computation
        M.setParam('TimeLimit', 3600)  # Set a one-hour time limit

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


def add_DCPF_constraints(M, demand, B_mat, pgen, d_wc, theta, pgen_c, d_wc_c, theta_c, names, g2bus, T):
    # Precompute generator to node mapping
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}
    S_T = B_mat.T  # Transpose to get line-to-node mapping
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        generation = np.array([(pgen[t, gen_to_node[b]] if b in gen_to_node else 0) for b_idx, b in enumerate(names['buses'])])
        in_flows = [quicksum(S_T[b_idx] * theta[t, b]) for b_idx, b in enumerate(names['buses'])]  # Nodal voltage phases at time t
        nodal_balance = demand.iloc[t_idx, :].values - in_flows - generation

        # Add power balance constraints for normal case
        for b_idx, b in enumerate(names['buses']):
            M.addConstr(nodal_balance[b_idx] <= d_wc[t, b], name=f'Power_Balance_{t}_{b}_up')
            M.addConstr(nodal_balance[b_idx] >= -d_wc[t, b], name=f'Power_Balance_{t}_{b}_low')

        for c in names['contingencies']:  # Contingency case
            generation_c = np.array([(pgen_c[t, gen_to_node[b], c] if b in gen_to_node else 0) for b_idx, b in enumerate(names['buses'])])
            in_flows_c = [quicksum(S_T[b_idx] * theta_c[t, b, c]) for b_idx, b in enumerate(names['buses'])]
            nodal_balance_c = demand.iloc[t_idx, :].values - in_flows_c - generation_c
            for b_idx, b in enumerate(names['buses']):
                M.addConstr(nodal_balance_c[b_idx] <= d_wc_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_up")
                M.addConstr(nodal_balance_c[b_idx] >= -d_wc_c[t, b, c], name=f"Power_Balance_{t}_{b}_{c}_low")
    return M


def add_line_constraints(M, xt, names, f_lim, f, f_c, theta, theta_c, B_mat, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        v_phases_t = np.array([theta[t, b] for b in names['buses']])
        for l_idx, l in enumerate(names['lines']):
            flow_dc = np.dot(B_mat[l_idx, :], v_phases_t)  # flow_dc = B_l theta_i - B_l theta_j
            if l in names['outages']:
                M = add_flow_up_low(M, f, f_lim, t, l, xt=xt[t, l])
                M.addConstr(f[t, l] == flow_dc * (1-xt[t, l]), name=f'Power_Flow_{t}_line_{l}')
            else:
                M = add_flow_up_low(M, f, f_lim, t, l)
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
                        M = add_flow_up_low(M, f_c, f_lim, t, l, c=c, xt=xt[t, l])
                        M.addConstr(f_c[t, l, c] == flow_dc_con * (1-xt[t, l]), name=f'Power_Flow_{t}_line_{l}_{c}')
                    else:
                        M = add_flow_up_low(M, f_c, f_lim, t, l, c=c)
                        M.addConstr(f_c[t, l, c] == flow_dc_con, name=f'Power_Flow_{t}_line_{l}_{c}')
    return M


# Example Usage
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """
    # Load data
    network, hourly_demand, config = data_loader('../config/conf_IEEE24.json')
    aggregation_step = config['aggregation_time']  # 'W', 'D', 'H
    cost_per_days = [1000, 2000, 1000, 1000, 2000, 2000, 5000, 5000]
    expected_duration_days = [25, 7, 55, 7, 30, 30, 30, 60]
    daily_demand = aggregate_hourly_demand(hourly_demand, aggregation_step=aggregation_step)
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

    dic_res = det_optimization_model_SCOP(data)
    dic_res
