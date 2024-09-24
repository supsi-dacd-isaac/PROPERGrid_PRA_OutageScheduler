"""
examplaination goes here


also, refer to:
M. Rocha, M. F. Anjos, M. Gendreau, (2023)
Scheduling maintenance with uncertain duration on power transmission systems
https://www.gerad.ca/fr/papers/G-2023-08.pdf

and
Jenny Liu, Mostafa Kazemi et al. Security-Constrained Optimal Scheduling of Transmission Outages With Load Curtailment
IEEE TRANSACTIONS ON POWER SYSTEMS, VOL. 33, NO. 1, JANUARY 2018

"""
import pandapower.networks
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from utils.dataloader import *
import numpy as np
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm

def add_security_constraints(SCOS_model, data, hourly_demand, config):
    # todo: Add security constraints.....N-1 line failures, flow must be secure.....
    pass


def optimization_SCOS_model(data):
    """
        Prepare PuLP optimization for:
        SCOS_model
        security-constrained outage scheduling problem
    """

    # 0. Preprocess the data
    net = data['network']
    outages = data['outages']
    nodal_demand = data['nodal_demand']
    n_minus1_names = data.get('n_minus1_cases', None)

    reference_bus = data['ref_buses']  # Assuming bus 0 is the reference bus
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

    T = [f'step_{t}' for t in range(len(nodal_demand))]  # Time steps for the planning horizon, e.g., hourly, daily, weekly
    n_planning_steps = len(T)

    PPC_internal = net._ppc["internal"]
    Incidence = PPC_internal['Cft'].A   # Incidence-matrix [(n_lines + n_transformers) x n_buses] +1 if line l 'leaves/from' bus k, -1 if line l 'enters/to' bus k, 0 otherwise
    branch_susceptance = np.real(PPC_internal['Bf'].A).max(axis=1)
    branch_susceptance = {bn: v for bn, v in zip(branch_names, branch_susceptance)}


    S_lk = {(l, k): 0 for k in nodes_names for l in branch_names}
    for k in net.bus.index.tolist():
        for l in net.line.index.tolist():
                S_lk[branch_names[l], nodes_names[k]] = Incidence[l, k]
                S_lk[branch_names[l], nodes_names[k]] = Incidence[l, k]

     # Generator indices
    if n_minus1_names is None:
        n_minus1_names = [f'n1_{l}' for l in branch_names] + [f'n1_{g}' for g in generators_names]

    # Pre-fetch some values for efficiency
    dic_outage_priority = {o_nam: outages['priorities'][o] for o, o_nam in enumerate(outages['names'])}  # Outage priority
    dic_outage_duration = {o_nam: outages['expected_duration_steps'][o] for o, o_nam in enumerate(outages['names'])}  # Outage expected durations
    maximum_number_of_maintenance_tasks_t = 3  # arbitrary value for the maximum number of maintenance tasks at time t

    # ---- START ---- Prepare the SCOS_model
    # 1. Define the Security-Constrained Outage Scheduling (Mixed Integer Linear Program MILP)
    logger.info(f'{blue_c} Initializing Transmission_Outage_Scheduling problem in PulP {reset_c}')
    SCOS_model = LpProblem(name="Transmission_Outage_Scheduling", sense=LpMaximize)

    # 2. Define Variables
    xt = LpVariable.dicts(name="xt_lines_and_gens",
                          indices=((t, o) for t in T for o in outages['names']),
                          cat='Binary')
    # Generation variables
    p_t_gen = LpVariable.dicts(name="power_gen_tg", indices=((t, g) for t in T for g in generators_names),
                               lowBound=0)  # Power generation variables for normal operation
    p_tn1c_gen = LpVariable.dicts(name="power_gen_tn1c",
                                  indices=((t, g, c) for t in T for g in generators_names for c in n_minus1_names),
                                  lowBound=0)  # Power generation variables, unplanned failures

    # Demand curtailment variables
    d_worst_case = LpVariable(name="worst-case-curtailment", lowBound=0)
    d_curt_tbc = LpVariable.dicts(name="d_curt_tbc",
                                  indices=((t, b, c) for t in T for b in nodes_names for c in n_minus1_names),
                                  lowBound=0)

    # Power flow variables
    flow_tl = LpVariable.dicts(name="flow_tl", indices=((t, l) for t in T for l in branch_names))
    flow_tlc = LpVariable.dicts(name="flow_tlc", indices=((t, l, c) for t in T for l in branch_names for c in n_minus1_names))

    # Phase angle variables (bounded between -0.3 and 0.3 radians)
    theta_tb = LpVariable.dicts(name="theta_tb", indices=((t, b) for t in T for b in nodes_names),
                                lowBound=-0.3, upBound=0.3)
    theta_tbc = LpVariable.dicts(name="theta_tbc",
                                 indices=((t, b, c) for t in T for b in nodes_names for c in n_minus1_names),
                                 lowBound=-0.3, upBound=0.3)
    logger.info(
        f'{blue_c}Variables Added:\n'
        f' - xt: Scheduled outage decisions\n'
        f' - p_t_gen: Generated power in normal operation\n'
        f' - p_tn1c_gen: Generated power in N-1 contingency\n'
        f' - d_worst_case: Worst-case demand curtailment\n'
        f' - d_curt_tbc: Curtailment in N-1 contingency for all nodes\n'
        f' - flow_tl: Power flow in normal operation\n'
        f' - flow_tlc: Power flow in N-1 contingency\n'
        f' - theta_tb: Phase angles in normal operation\n'
        f' - theta_tbc: Phase angles in N-1 contingency{reset_c}'
    )

    # 3. Objective Function: Maximize scheduled outages and minimize demand curtailment
    # objective 1) Maximize number of scheduled outages in the planning period T (weighted by priority)
    # objective 2) Minimize the worst-case demand curtailed d_worst_case
    Objective_fun = [xt[t, o] * dic_outage_priority[o] - 0.01*d_worst_case for t in T for o in outages['names']]
    SCOS_model += lpSum(Objective_fun), "Maximize_Scheduled_Outages"

    # 4. Constraints in https://www.gerad.ca/fr/papers/G-2023-08.pdf, Appendix A
    # (25 - 34) constraints maximum number of maintenance tasks, outage duration and continuity constraints
    for t in T:
        SCOS_model += lpSum([xt[t, o] for o in outages['names']]) <= data['max_number_of_maintenance_tasks'], f"Max_Maintenance_Tasks_{t}"

    for o in outages['names']:
        # 1. Outage duration must match exactly the expected duration
        SCOS_model += lpSum([xt[t, o] for t in T]) == dic_outage_duration[o], f"Duration_Constraint_Outage_{o}"

        # 2. Continuity constraint: If an outage starts, it must continue for consecutive steps
        for t in range(n_planning_steps - 1):
            SCOS_model += xt[T[t + 1], o] >= xt[T[t], o], f"Continuity_Constraint_Outage_{o}_{t}"


    n_constraints_approx = len(T)*(len(generators_names)+ len(branch_names)+ len(nodes_names))*(1+len(n_minus1_names))
    logger.info(f'{blue_c} ADDING CONSTRAINTS FOR EACH TIME STEP: Aproximativelly ~~ {n_constraints_approx} constraints {reset_c}')

    for t_idx, t in tqdm(enumerate(T), desc="Building SCOS model ---- > Adding Constraints", total=len(T), ncols=100, colour="green"):

        # (Reference phase angle constraint)
        for rb in reference_bus:
            SCOS_model += theta_tb[t, rb] == 0, f"Reference_Bus_Constraint_{t}_phase equal 0"

        for g in generators_names:  # Power generation
            if g in outages['names']:  # Undamaged grid case, besides the scheduled outages
                SCOS_model += p_t_gen[t, g] <= Pg_max[g] * (1 - xt[t, g]), f"Generation_Max_Constraint_{t}_{g}_with_scheduled_outage"
                SCOS_model += p_t_gen[t, g] >= Pg_min[g] * (1 - xt[t, g]), f"Generation_Min_Constraint_{t}_{g}_with_scheduled_outage"
            else:
                SCOS_model += p_t_gen[t, g] <= Pg_max[g], f"Generation_Max_Constraint_{t}_{g}"
                SCOS_model += p_t_gen[t, g] >= Pg_min[g], f"Generation_Min_Constraint_{t}_{g}"

            for c in n_minus1_names:  # N-1 contingency cases...generation constraints
                if c[3:] == g:
                    SCOS_model += p_tn1c_gen[t, g, c] == 0, f"Generation_Max_Constraint_{t}_{g}_N_minus_1_{c}_must be null"
                elif not (c[3:] == g) and g in outages['names']:
                    SCOS_model += p_tn1c_gen[t, g, c] <= Pg_max[g] * (1 - xt[t, g]), f"Generation_Max_Constraint_{t}_{g}_N_minus_1_{c}"
                    SCOS_model += p_tn1c_gen[t, g, c] >= Pg_min[g] * (1 - xt[t, g]), f"Generation_Min_Constraint_{t}_{g}_N_minus_1_{c}"
                else:
                    SCOS_model += p_tn1c_gen[t, g, c] <= Pg_max[g], f"Generation_Max_Constraint_{t}_{g}_N_minus_1_{c}"
                    SCOS_model += p_tn1c_gen[t, g, c] >= Pg_min[g], f"Generation_Min_Constraint_{t}_{g}_N_minus_1_{c}"

        for l in branch_names:  # Line maximum loading constraint
            if l in outages['names']:
                SCOS_model += flow_tl[t, l] <= branch_capacity[l] * (1 - xt[t, l]), f"Flow_Constraint_{t}_{l}_with_so_up"
                SCOS_model += flow_tl[t, l] >= -branch_capacity[l] * (1 - xt[t, l]), f"Flow_Constraint_{t}_{l}_with_so_low"
            else:
                SCOS_model += flow_tl[t, l] <= branch_capacity[l], f"Flow_Constraint_{t}_{l}_up"
                SCOS_model += flow_tl[t, l] >= -branch_capacity[l], f"Flow_Constraint_{t}_{l}_low"
            for c in n_minus1_names:  # N-1 contingency cases....add flow limit constraints
                if c[3:] == l:
                    SCOS_model += flow_tlc[t, l, c] == 0, f"Flow_Constraint_{t}_{l}_{c} must be == 0"
                elif not(c[3:] == l) and l in outages['names']:
                    SCOS_model += flow_tlc[t, l, c] <= branch_capacity[l] * (1 - xt[t, l]), f"Flow_Constraint_{t}_{l}_{c}_with_so_up"
                    SCOS_model += flow_tlc[t, l, c] >= -branch_capacity[l] * (1 - xt[t, l]), f"Flow_Constraint_{t}_{l}_{c}_with_so_low"
                else:
                    SCOS_model += flow_tlc[t, l, c] <= branch_capacity[l], f"Flow_Constraint_{t}_{l}_{c}_up"
                    SCOS_model += flow_tlc[t, l, c] >= -branch_capacity[l], f"Flow_Constraint_{t}_{l}_{c}_low"


        for b_idx, b in enumerate(nodes_names):  # Add Nodal Power Balance total_Flow = Production - Demand  constraint using DC Power Flow approximation
            demand_tb = nodal_demand.iloc[t_idx, b_idx]
            power_injection_tb = lpSum([S_lk[l, b] * flow_tl[t, l] for l in branch_names])
            if f'gen_in_bus{b_idx}' in gen_is_in_bus:
                g = [generators_names[idx] for idx, g2b in enumerate(gen_is_in_bus) if g2b == b][0]
                SCOS_model += (power_injection_tb + p_t_gen[t, g] == demand_tb), f"Power_Balance_{t}_{b}"
            else:
                SCOS_model += (power_injection_tb == demand_tb), f"Power_Balance_{t}_{b}"

            for c in n_minus1_names:  # Add Nodal Power Balance for unplanned failures ..... N-1 contingency cases
                power_injection_tbc = lpSum([S_lk[l, b] * flow_tlc[t, l, c] for l in branch_names])

                if f'gen_in_bus{b_idx}' in gen_is_in_bus:
                    g = [generators_names[b_idx] for idx, g2b in enumerate(gen_is_in_bus) if g2b == b][0]
                    SCOS_model += (power_injection_tbc + p_tn1c_gen[t, g, c] == demand_tb - d_curt_tbc[t, b, c]), f"Power_Balance_{t}_{b}_{c}"
                else:
                    SCOS_model += (power_injection_tbc  == demand_tb - d_curt_tbc[t, b, c]), f"Power_Balance_{t}_{b}_{c}_nogen"

                # add worst-case constraint on the curtailment, e.g. d_curt_tbc[t, b, c] <= d_worst_case for all t, b, c
                SCOS_model += d_curt_tbc[t, b, c] <= d_worst_case, f"Max_Demand_Curtailment_{t}_{b}_{c}"

        for l_idx, l in enumerate(branch_names):  # DC Power Flow approximation
            """" DC-PF flow approximation --> P_l = B_l (Theta_i - Theta_j) ...l --> (i,j)"""
            from_b, to_b = f'bus_{from_to_bus[l_idx][0]}', f'bus_{from_to_bus[l_idx][1]}' #todo make {line: fb, tb} dic
            SCOS_model += (flow_tl[t, l] == branch_susceptance[l] * (theta_tb[t, from_b] - theta_tb[t, to_b])), f"DCPF_{t}_{l}"
            if c in generators_names:
                SCOS_model += (flow_tlc[t, l, c] == branch_susceptance[l] * (theta_tbc[t, from_b, c] - theta_tbc[t, to_b, c])), f"DCPF_{t}_{l}_{c}"

    # Solve the SCOS model
    # SCOS_model.solve()
    SCOS_model.solve(PULP_CBC_CMD())
    # Output results
    results = {v.name: v.varValue for v in SCOS_model.variables()}
    return results


# Main Execution
def main(conf_path):
    # Load data
    network, hourly_demand, config = data_loader(conf_path)

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
            'branch_capacity': branch_capacity}  # todo: how to use net.line['max_i_ka'] --> convert to --> max flow in MW?

    # Run optimization SCOS_model
    results = optimization_SCOS_model(data)

    # Print results
    for var_name, value in results.items():
        print(f"{var_name}: {value}")


# Example Usage
if __name__ == "__main__":
    """ Prepare data for the outage scheduling problem """
    conf_path = '../config/conf_IEEE24.json'
    main(conf_path)
