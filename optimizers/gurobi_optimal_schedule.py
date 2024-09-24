import pyomo.environ as pe
from pyomo.opt import SolverFactory
from utils.utils import *
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
import pandapower.networks

from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm


def optimization_SCOS_model_pyomo(data):
    """
        Prepare Pyomo optimization model for:
        SCOS_model
        security-constrained outage scheduling problem
    """

    # Preprocess the data
    net = data['network']
    outages = data['outages']
    nodal_demand = data['nodal_demand']
    n_minus1_names = data.get('n_minus1_names', None)

    reference_bus = data['ref_buses']
    num_buses = data['num_buses']
    num_branches = data['num_branches']
    nodes_names = [f'bus_{b}' for b in range(num_buses)]  # Bus indices
    branch_names = [f'line_{l}' for l in range(num_branches)]
    generators_names = [f'gen_{g}' for g in net.gen.index.tolist()]

    Pg_max = {gn: v for gn, v in zip(generators_names, net.gen['max_p_mw'])}
    Pg_min = {gn: v for gn, v in zip(generators_names, net.gen['min_p_mw'])}
    from_to_bus = net.line[['from_bus', 'to_bus']].values.tolist() + net.trafo[['hv_bus', 'lv_bus']].values.tolist()
    branch_capacity = {bn: v for bn, v in zip(branch_names, data['branch_capacity'])}

    gen_is_in_bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]

    T = [f'step_{t}' for t in range(len(nodal_demand))]
    n_planning_steps = len(T)

    PPC_internal = net._ppc["internal"]
    Incidence = PPC_internal['Cft'].A
    branch_susceptance = np.real(PPC_internal['Bf'].A).max(axis=1)
    branch_susceptance = {bn: v for bn, v in zip(branch_names, branch_susceptance)}

    if n_minus1_names is None:
        n_minus1_names = [f'n1_{l}' for l in branch_names] + [f'n1_{g}' for g in generators_names]

    # Pre-fetch some values
    dic_outage_priority = {o_nam: outages['priorities'][o] for o, o_nam in enumerate(outages['names'])}
    dic_outage_duration = {o_nam: outages['expected_duration_steps'][o] for o, o_nam in enumerate(outages['names'])}
    max_number_of_maintenance_tasks_t = 3

    # ---- START ---- Pyomo Model ----
    model = pe.ConcreteModel()

    # Define sets
    model.T = pe.Set(initialize=T)
    model.O = pe.Set(initialize=outages['names'])
    model.G = pe.Set(initialize=generators_names)
    model.B = pe.Set(initialize=nodes_names)
    model.Branch = pe.Set(initialize=branch_names)
    model.N1 = pe.Set(initialize=n_minus1_names)

    # Define variables
    model.xt = pe.Var(model.T, model.O, within=pe.Binary)  # Outage decisions
    model.p_t_gen = pe.Var(model.T, model.G, within=pe.NonNegativeReals)  # Power generation
    model.p_tn1c_gen = pe.Var(model.T, model.G, model.N1, within=pe.NonNegativeReals)
    model.d_worst_case = pe.Var(within=pe.NonNegativeReals)  # Worst-case curtailment
    model.d_curt_tbc = pe.Var(model.T, model.B, model.N1, within=pe.NonNegativeReals)
    model.flow_tl = pe.Var(model.T, model.Branch, within=pe.Reals)
    model.flow_tlc = pe.Var(model.T, model.Branch, model.N1, within=pe.Reals)
    model.theta_tb = pe.Var(model.T, model.B, bounds=(-0.3, 0.3))  # Phase angles
    model.theta_tbc = pe.Var(model.T, model.B, model.N1, bounds=(-0.3, 0.3))

    # Objective function: Maximize outages and minimize demand curtailment
    def objective_function(model):
        return sum(
            model.xt[t, o] * dic_outage_priority[o] for t in model.T for o in model.O) - 0.01 * model.d_worst_case

    model.objective = pe.Objective(rule=objective_function, sense=pe.maximize)

    # Constraints
    def max_maintenance_constraint(model, t):
        return sum(model.xt[t, o] for o in model.O) <= max_number_of_maintenance_tasks_t

    model.Max_Maintenance_Tasks = pe.Constraint(model.T, rule=max_maintenance_constraint)

    def outage_duration_constraint(model, o):
        return sum(model.xt[t, o] for t in model.T) == dic_outage_duration[o]

    model.Duration_Constraint = pe.Constraint(model.O, rule=outage_duration_constraint)

    def continuity_constraint(model, o, t):
        if t != T[-1]:  # Skip last time step
            return model.xt[T[T.index(t) + 1], o] >= model.xt[t, o]
        else:
            return pe.Constraint.Skip

    model.Continuity_Constraint = pe.Constraint(model.O, model.T, rule=continuity_constraint)

    # Add more constraints for power generation, flow, nodal power balance, etc.

    # Solve the model using a solver (CBC, Gurobi, etc.)
    opt = SolverFactory('cbc')  # Use 'gurobi' if Gurobi is installed
    results = opt.solve(model, tee=True)

    # Extract results
    solution = {}
    for v in model.component_objects(pe.Var, active=True):
        varobject = getattr(model, str(v))
        solution[str(v)] = varobject.value

    return solution


def add_variables(SCOS_model, T, outages, generators_names, n_minus1_names, nodes_names, branch_names):
    # 2. Define Variables
    xt = SCOS_model.addVars(T, outages['names'], vtype=GRB.BINARY, name="xt_lines_and_gens")
    # Generation variables
    p_t_gen = SCOS_model.addVars(T, generators_names, lb=0, name="power_gen_tg")
    p_tn1c_gen = SCOS_model.addVars(T, generators_names, n_minus1_names, lb=0, name="power_gen_tn1c")

    # Demand curtailment variables
    d_worst_case = SCOS_model.addVar(lb=0, name="worst-case-curtailment")
    d_curt_tbc = SCOS_model.addVars(T, nodes_names, n_minus1_names, lb=0, name="d_curt_tbc")

    # Power flow variables
    flow_tl = SCOS_model.addVars(T, branch_names, name="flow_tl")
    flow_tlc = SCOS_model.addVars(T, branch_names, n_minus1_names, name="flow_tlc")

    # Phase angle variables (bounded between -0.3 and 0.3 radians)
    theta_tb = SCOS_model.addVars(T, nodes_names, lb=-0.3, ub=0.3, name="theta_tb")
    theta_tbc = SCOS_model.addVars(T, nodes_names, n_minus1_names, lb=-0.3, ub=0.3, name="theta_tbc")
    return SCOS_model, xt, p_t_gen, p_tn1c_gen, d_worst_case, d_curt_tbc, flow_tl, flow_tlc, theta_tb, theta_tbc


def add_planned_outages_constraints(SCOS_model, T, outages, xt, max_tasks_t, dic_outage_duration, n_planning_steps):
    # Constraints for the duration, number of tasks at time t, and continuity of outages when they are scheduled
    for o in outages['names']:
        SCOS_model.addConstr(quicksum(xt[t, o] for t in T) == dic_outage_duration[o], f"Duration_con_{o}")
        for t in range(n_planning_steps - 1):
            SCOS_model.addConstr(xt[T[t + 1], o] >= xt[T[t], o], f"Continuity_out_con_{o}_{t}")
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        SCOS_model.addConstr(quicksum(xt[t, o] for o in outages['names']) <= max_tasks_t,
                             f"Max_tasks_con_{t}")  # for all time steps max task <= 3 
    return SCOS_model


def add_generators_constraints(SCOS_model, T, outages_names, n_minus1_names, generators_names, reference_bus, theta_tb,
                               p_t_gen, p_tn1c_gen, Pg_max, Pg_min, xt):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for rb in reference_bus:  # (Reference phase angle = 0constraint)
            SCOS_model.addConstr(theta_tb[t, rb] == 0, f"Reference_Bus_cnstr_{t}_phase_equal_0")

        for g in generators_names:
            if g in outages_names:
                SCOS_model.addConstr(p_t_gen[t, g] <= Pg_max[g] * (1 - xt[t, g]),
                                     f"Generation_Max_cnstr_{t}_{g}_with_scheduled_outage")
                SCOS_model.addConstr(p_t_gen[t, g] >= Pg_min[g] * (1 - xt[t, g]),
                                     f"Generation_Min_cnstr_{t}_{g}_with_scheduled_outage")
            else:
                SCOS_model.addConstr(p_t_gen[t, g] <= Pg_max[g], f"Generation_Max_cnstr_{t}_{g}")
                SCOS_model.addConstr(p_t_gen[t, g] >= Pg_min[g], f"Generation_Min_cnstr_{t}_{g}")

            for c in n_minus1_names:
                if c[3:] == g:
                    SCOS_model.addConstr(p_tn1c_gen[t, g, c] == 0,
                                         f"Generation_Max_cnstr_{t}_{g}_N_minus_1_{c}_must_be_null")
                elif g in outages_names:
                    SCOS_model.addConstr(p_tn1c_gen[t, g, c] <= Pg_max[g] * (1 - xt[t, g]),
                                         f"Generation_Max_cnstr_{t}_{g}_N_minus_1_{c}")
                    SCOS_model.addConstr(p_tn1c_gen[t, g, c] >= Pg_min[g] * (1 - xt[t, g]),
                                         f"Generation_Min_cnstr_{t}_{g}_N_minus_1_{c}")
                else:
                    SCOS_model.addConstr(p_tn1c_gen[t, g, c] <= Pg_max[g],
                                         f"Generation_Max_cnstr_{t}_{g}_N_minus_1_{c}")
                    SCOS_model.addConstr(p_tn1c_gen[t, g, c] >= Pg_min[g],
                                         f"Generation_Min_cnstr_{t}_{g}_N_minus_1_{c}")
    return SCOS_model


def add_line_flow_constraints(branch_names, SCOS_model, T, outages, branch_capacity, flow_tl, flow_tlc, xt,
                              n_minus1_names):

    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        for l in branch_names:
            if l in outages['names']:
                SCOS_model.addConstr(flow_tl[t, l] <= branch_capacity[l] * (1 - xt[t, l]),
                                     f"f_cnstr_{t}_{l}_with_so_up")
                SCOS_model.addConstr(flow_tl[t, l] >= -branch_capacity[l] * (1 - xt[t, l]),
                                     f"f_cnstr_{t}_{l}_with_so_low")
            else:
                SCOS_model.addConstr(flow_tl[t, l] <= branch_capacity[l], f"f_cnstr_{t}_{l}_up")
                SCOS_model.addConstr(flow_tl[t, l] >= -branch_capacity[l], f"f_cnstr_{t}_{l}_low")

            for c in n_minus1_names:
                if c[3:] == l:
                    SCOS_model.addConstr(flow_tlc[t, l, c] == 0, f"f_cnstr_{t}_{l}_{c}_must_be_0")
                elif not (c[3:] == l) and l in outages['names']:
                    SCOS_model.addConstr(flow_tlc[t, l, c] <= branch_capacity[l] * (1 - xt[t, l]),
                                         f"f_cnstr_{t}_{l}_N_minus_1_{c}")
                    SCOS_model.addConstr(flow_tlc[t, l, c] >= -branch_capacity[l] * (1 - xt[t, l]),
                                         f"f_cnstr_{t}_{l}_N_minus_1_{c}")
                else:
                    SCOS_model.addConstr(flow_tlc[t, l, c] <= branch_capacity[l],
                                         f"f_cnstr_{t}_{l}_N_minus_1_{c}_up")
                    SCOS_model.addConstr(flow_tlc[t, l, c] >= -branch_capacity[l],
                                         f"f_cnstr_{t}_{l}_N_minus_1_{c}_low")


    return SCOS_model


def add_nodal_power_balance_constraints(SCOS_model, T, nodes_names, branch_names, generators_names, S_lk, nodal_demand, d_curt_tbc, n_minus1_names,
                                        p_t_gen, p_tn1c_gen, flow_tl, flow_tlc, gen_is_in_bus, d_worst_case):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        # Power balance constraints for each node
        for b_idx, b in enumerate(nodes_names):  # Add Nodal Power Balance total_Flow = Production - Demand  constraint using DC Power Flow approximation
            demand_tb = nodal_demand.iloc[t_idx, b_idx]
            power_injection_tb = quicksum([S_lk[l, b] * flow_tl[t, l] for l in branch_names])
            if f'gen_in_bus{b_idx}' in gen_is_in_bus:
                g = [generators_names[idx] for idx, g2b in enumerate(gen_is_in_bus) if g2b == b][0]
                SCOS_model.addConstr(power_injection_tb == p_t_gen[t, g] - demand_tb, f"Power_Balance_{t}_{b}")
            else:
                SCOS_model.addConstr(power_injection_tb == demand_tb, f"Power_Balance_{t}_{b}")
            for c in n_minus1_names:  # Add Nodal Power Balance for unplanned failures ..... N-1 contingency cases
                power_injection_tbc = quicksum([S_lk[l, b] * flow_tlc[t, l, c] for l in branch_names])
                if f'gen_in_bus{b_idx}' in gen_is_in_bus:
                    g = [generators_names[b_idx] for idx, g2b in enumerate(gen_is_in_bus) if g2b == b][0]
                    tmp_c = power_injection_tbc == p_tn1c_gen[t, g, c] - demand_tb + d_curt_tbc[t, b, c]
                    SCOS_model.addConstr(tmp_c, f"Power_Balance_{t}_{b}_{c}")
                else:
                    tmp_c = power_injection_tbc == demand_tb - d_curt_tbc[t, b, c]
                    SCOS_model.addConstr(tmp_c, f"Power_Balance_{t}_{b}_{c}_nogen")
    # add mean curtailment constraint
    mean_demand_curtailed = quicksum([d_curt_tbc[t, b, c] for t in T for b in nodes_names for c in n_minus1_names])
    SCOS_model.addConstr(mean_demand_curtailed <= d_worst_case, f"Max_Demand_Curtailment_{t}_{b}_{c}")
    return SCOS_model


def add_dc_line_flow_constraints(SCOS_model, T, from_to_bus, branch_susceptance, flow_tl,
                                 theta_tb, flow_tlc, theta_tbc, branch_names, outage_names, n_minus1_names):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints", total=len(T), ncols=100, colour="green"):
        """" DC-PF flow approximation -
                flow_l = B_l (Theta_i - Theta_j) where l --> (i,j)
                flow_l = 0 if line == c """
        for l_idx, l in enumerate(branch_names):

            from_b, to_b = f'bus_{from_to_bus[l_idx][0]}', f'bus_{from_to_bus[l_idx][1]}'  # todo make {line: fb, tb} dic
            dc_pf_l = flow_tl[t, l] == branch_susceptance[l] * (theta_tb[t, from_b] - theta_tb[t, to_b])
            SCOS_model.addConstr(dc_pf_l, f"DCPF_{t}_{l}")

            for c in n_minus1_names:
                if not(c[3:] == l) and l in outage_names:
                    dc_pf_contingency = (flow_tlc[t, l, c] ==
                                         branch_susceptance[l] * (theta_tbc[t, from_b, c] - theta_tbc[t, to_b, c]))
                    SCOS_model.addConstr(dc_pf_contingency, f"DCPF_{t}_{l}_{c}")

    return SCOS_model



def optimization_SCOS_model_gurobi(data):
    """
        Prepare Gurobi optimization model for:
        SCOS_model
        security-constrained outage scheduling problem
    """

    # Preprocess the data
    net = data['network']
    outages = data['outages']
    nodal_demand = data['nodal_demand']
    n_minus1_names = data.get('n_minus1_names', None)

    reference_bus = data['ref_buses']
    num_buses = data['num_buses']
    num_branches = data['num_branches']
    max_tasks_t = data['max_number_of_maintenance_tasks']

    nodes_names = [f'bus_{b}' for b in range(num_buses)]  # Bus indices
    branch_names = [f'line_{l}' for l in range(num_branches)]
    generators_names = [f'gen_{g}' for g in net.gen.index.tolist()]
    outages_names = outages['names']

    Pg_max = {gn: v for gn, v in zip(generators_names, net.gen['max_p_mw'])}
    Pg_min = {gn: v for gn, v in zip(generators_names, net.gen['min_p_mw'])}

    from_to_bus = net.line[['from_bus', 'to_bus']].values.tolist() + net.trafo[['hv_bus', 'lv_bus']].values.tolist()
    branch_capacity = {bn: v for bn, v in zip(branch_names, data['branch_capacity'])}

    gen_is_in_bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]

    T = [f'step_{t}' for t in range(len(nodal_demand))]
    n_planning_steps = len(T)

    PPC_internal = net._ppc["internal"]
    Incidence = PPC_internal['Cft'].A  # Incidence-matrix   
    branch_susceptance = {bn: v for bn, v in zip(branch_names, np.real(PPC_internal['Bf'].A).max(axis=1))}

    S_lk = {(l, k): 0 for k in nodes_names for l in branch_names}
    for k in net.bus.index.tolist():
        for l in net.line.index.tolist():
            S_lk[branch_names[l], nodes_names[k]] = Incidence[l, k]
            S_lk[branch_names[l], nodes_names[k]] = Incidence[l, k]

    # Generator indices
    if n_minus1_names is None:
        n_minus1_names = [f'n1_{l}' for l in branch_names] + [f'n1_{g}' for g in generators_names]

    # Pre-fetch some values for efficiency
    dic_outage_priority = {o_nam: outages['priorities'][o] for o, o_nam in enumerate(outages['names'])}
    dic_outage_duration = {o_nam: outages['expected_duration_steps'][o] for o, o_nam in enumerate(outages['names'])}

    # ---- START ---- Prepare the SCOS_model
    try:
        # 1. Define the Security-Constrained Outage Scheduling Model
        SCOS_model = Model("Transmission_Outage_Scheduling")

        SCOS_model, xt, p_t_gen, p_tn1c_gen, d_worst_case, d_curt_tbc, flow_tl, flow_tlc, theta_tb, theta_tbc = add_variables(
            SCOS_model, T, outages, generators_names, n_minus1_names, nodes_names, branch_names)

        # 3. Objective Function: Maximize scheduled outages and minimize demand curtailment
        Objective_fun = quicksum(xt[t, o] * dic_outage_priority[o] for t in T for o in outages['names'])
        Objective_fun -= 0.01 * d_worst_case  # Add the curtailment term
        SCOS_model.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' - xt: Scheduled outage decisions\n'
            f' - p_t_gen: Generated power in normal operation\n'
            f' - p_tn1c_gen: Generated power in N-1 contingency\n'
            f' - d_worst_case: Worst-case demand curtailment\n'
            f' - d_curt_tbc: Curtailment in N-1 contingency for all nodes\n'
            f' - flow_tl: Power flow in normal operation\n'
            f' - flow_tlc: Power flow in N-1 contingency\n'
            f' - theta_tb: Phase angles in normal operation\n'
            f' - theta_tbc: Phase angles in N-1 contingency\n'
            f'{blue_c}Objective Function Defined: {Objective_fun} {reset_c}\n'
        )

        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        SCOS_model = add_planned_outages_constraints(SCOS_model, T, outages, xt, max_tasks_t, dic_outage_duration,
                                                     n_planning_steps)

        logger.info('adding power production constraints, normal + N-1 failures')
        SCOS_model = add_generators_constraints(SCOS_model, T, outages_names, n_minus1_names, generators_names,
                                                reference_bus, theta_tb,
                                                p_t_gen, p_tn1c_gen, Pg_max, Pg_min, xt)

        logger.info('adding line flow constraints, normal + N-1 failures')
        SCOS_model = add_line_flow_constraints(branch_names, SCOS_model, T, outages, branch_capacity, flow_tl, flow_tlc,
                                               xt, n_minus1_names)

        logger.info('adding node balance constraints, normal + N-1 failures')
        SCOS_model = add_nodal_power_balance_constraints(SCOS_model, T, nodes_names, branch_names, generators_names,
                                                         S_lk, nodal_demand, d_curt_tbc, n_minus1_names,
                                                         p_t_gen, p_tn1c_gen, flow_tl, flow_tlc, gen_is_in_bus,
                                                         d_worst_case)

        logger.info('adding dc power flow constraints, normal + N-1 failures')
        SCOS_model = add_dc_line_flow_constraints(SCOS_model, T, from_to_bus, branch_susceptance, flow_tl,
                                                  theta_tb, flow_tlc, theta_tbc, branch_names, outages_names,
                                                  n_minus1_names)

        # Solve the model
        SCOS_model.optimize()

        # Results
        return {
            'objective_value': SCOS_model.objVal,
            'xt': {t: {o: xt[t, o].X for o in outages['names']} for t in T},
            'p_t_gen': {t: {g: p_t_gen[t, g].X for g in generators_names} for t in T},
            'd_curt_tbc': {(t, b): {c: d_curt_tbc[t, b, c].X for c in n_minus1_names} for t in T for b in nodes_names} ,
            'flow_tl': {t: {l: flow_tl[t, l].X for l in branch_names} for t in T},
        }

    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except Exception as e:
        print(e)

    # Example usage
    # data = load_data()  # Implement your own data loading function
    # results = optimization_SCOS_model(data)


# Example Usage
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """

    # Load data
    network, hourly_demand, config = data_loader('../config/conf_IEEE24.json')
    daily_demand = aggregate_hourly_demand(hourly_demand)

    # example scheduled outage information
    outages = {'indices': [0, 2, 3, 5, 8, 1, 2],
               'names': ['line_0', 'line_2', 'line_3', 'line_5', 'line_8', 'gen_1', 'gen_2'],
               'type': ['line', 'line', 'line', 'line', 'line', 'generator', 'generator'],
               'expected_duration_steps': [24, 24, 24, 48, 48, 120, 120],
               'cost_per_step': [1000, 1000, 1000, 2000, 2000, 5000, 5000],
               'priorities': [1, 1, 3, 2, 1, 1, 3]}

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in network.trafo]

    data = {'max_number_of_maintenance_tasks': 3,
            'nodal_demand': daily_demand,
            'outages': outages,
            'network': network,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names':  [f'n1_{l}' for l in [f'line_{k}' for k in range(5)]]}


    dic_reults = optimization_SCOS_model_gurobi(data)