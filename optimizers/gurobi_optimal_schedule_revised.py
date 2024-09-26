
from gurobipy import Model, GRB, quicksum, GurobiError
import numpy as np
from utils.dataloader import *
from utils.data_preporcess import aggregate_hourly_demand
from tqdm import tqdm
import seaborn as sbn
import matplotlib.pyplot as plt


def add_planned_outages_constraints(model, o_nam, xt, sxt, ext, max_tasks, durations, T):
    for t in T:# 1) Max simultaneous outages
        model.addConstr(quicksum(xt[t, o] for o in o_nam) <= max_tasks, name=f"MaxTasks_{t}")
    for o in o_nam:# 2) Total duration constraint
        model.addConstr(quicksum(xt[t, o] for t in T) == durations[o], name=f"Duration_{o}")
        # 3) Continuity constraints
        for t_idx, t in enumerate(T[:-1]):  # Ensure we don't go out of bounds
            model.addConstr(sxt[T[t_idx + 1], o] >= sxt[T[t_idx], o], name=f"Cont_start_{o}_{t}")
            model.addConstr(ext[T[t_idx + 1], o] >= ext[T[t_idx], o], name=f"Cont_end_{o}_{t}")
            model.addConstr(ext[T[t_idx + 1], o] <= sxt[T[t_idx], o], name=f"end_only_after_start_{o}_{t}")
    for o in o_nam:
        for t_idx, t in enumerate(T):  # Ensure xt = 1 if started but not ended
            model.addConstr(sxt[t, o] - ext[t, o] == xt[t, o], name=f"Cont_start_end_x_{o}_{t}")
    return model


def add_generators_constraints(model, xt, o_nam, con_nam, gen_nam, ref_b, theta, pgen, pgen_c, p_max, p_min, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints",
                         total=len(T), ncols=100, colour="green"):
        for rb in ref_b:  # Reference phase angle constraint
            model.addConstr(theta[t, rb] == 0, f"RefBus_{t}")
        for g in gen_nam:
            if g in o_nam:  # Constraints with scheduled outage
                model.addConstr(pgen[t, g] <= p_max[g] * (1 - xt[t, g]), name=f"GenMaxOutage_{t}_{g}")
                model.addConstr(pgen[t, g] >= p_min[g] * (1 - xt[t, g]), name=f"GenMinOutage_{t}_{g}")
            else:  # Constraints without scheduled outage
                model.addConstr(pgen[t, g] <= p_max[g], name=f"GenMax_{t}_{g}")
                model.addConstr(pgen[t, g] >= p_min[g], name=f"GenMin_{t}_{g}")
            for c in con_nam:
                if c[3:] == g:
                    model.addConstr(pgen_c[t, g, c] == 0,  name=f"GenNull_{t}_{g}")
                elif g in o_nam:
                    model.addConstr(pgen_c[t, g, c] <= p_max[g] * (1 - xt[t, g]), name=f"GenMaxOutageN1_{t}_{g}")
                    model.addConstr(pgen_c[t, g, c] >= p_min[g] * (1 - xt[t, g]), name=f"GenMinOutageN1_{t}_{g}")
                else:
                    model.addConstr(pgen_c[t, g, c] <= p_max[g], name=f"GenMaxN1_{t}_{g}")
    return model


def add_line_flow_constraints(model, xt, linod_nam, o_nam, con_nam, f_lim, f, f_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints",
                         total=len(T), ncols=100, colour="green"):
        for l in linod_nam:
            name = f"Flow_{t}_{l}"
            if l in o_nam:
                model.addConstr(f[t, l] <= f_lim[l] * (1 - xt[t, l]), name=name + "_Outage")
                model.addConstr(f[t, l] >= -f_lim[l] * (1 - xt[t, l]),   name=name + "_OutageLow")
            else:
                model.addConstr(f[t, l] <= f_lim[l], name=name + "_Max")
                model.addConstr(f[t, l] >= -f_lim[l], name=name + "_Min")
            for c in con_nam:
                if c[3:] == l:
                    model.addConstr(f_c[t, l, c] == 0,    name=f"FlowNull_{t}_{l}_{c}")
                elif l in o_nam:
                    model.addConstr(f_c[t, l, c] <= f_lim[l] * (1 - xt[t, l]), name=name + "_OutageN1")
                    model.addConstr(f_c[t, l, c] >= -f_lim[l] * (1 - xt[t, l]),  name=name + "_OutageN1Low")
                else:
                    model.addConstr(f_c[t, l, c] <= f_lim[l],  name=name + "_N1")
                    model.addConstr(f_c[t, l, c] >= -f_lim[l], name=name + "_N1Low")
    return model


def add_nodal_power_balance_constraints(model, nod_nam, linod_nam, gen_nam, con_nam, S, demand,  pgen, pgen_c, f, f_c, g2bus, d_wc, d_wc_c, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints",
                         total=len(T), ncols=100, colour="green"):
        for b_idx, b in enumerate(nod_nam):
            demand_tb = demand.iat[t_idx, b_idx]
            pow_b = quicksum([S.at[l, b] * f[t, l] for l in linod_nam])
            if b in g2bus:  # do we have generators in node b?
                g = [gen_nam[idx] for idx, g2b in enumerate(g2bus) if g2b == b][0]
                model.addConstr(demand_tb - pow_b - pgen[t, g] <= d_wc[t], f"Power_Balance_{t}_{b}")
            else:
                model.addConstr(demand_tb - pow_b <= d_wc[t], f"Power_Balance_{t}_{b}")
            for c in con_nam:  # Add N-1 contingency cases
                power_injection_tbc = quicksum([S.at[l, b] * f_c[t, l, c] for l in linod_nam])
                if b in g2bus:
                    g = [gen_nam[idx] for idx, g2b in enumerate(g2bus) if g2b == b][0]
                    tmp_c = demand_tb - power_injection_tbc - pgen_c[t, g, c] <= d_wc_c[t, c]
                    model.addConstr(tmp_c, f"Power_Balance_{t}_{b}_{c}")
                else:
                    tmp_c = demand_tb - power_injection_tbc <= d_wc_c[t, c]
                    model.addConstr(tmp_c, f"Power_Balance_{t}_{b}_{c}_nogen")
    return model


def add_dc_line_flow_constraints(model, f2b, B, f, theta, f_c, theta_c, linod_nam, o_nam, con_nam, T):
    for t_idx, t in tqdm(enumerate(T), desc="Adding Constraints",
                         total=len(T), ncols=100, colour="green"):
        for l_idx, l in enumerate(linod_nam):
            from_b, to_b = f'bus_{f2b[l_idx][0]}', f'bus_{f2b[l_idx][1]}'
            dc_pf_l = f[t, l] == B[l] * (theta[t, from_b] - theta[t, to_b])
            name = f"DC_power_flow_{t}_{l}"
            model.addConstr(dc_pf_l, name=name)
            for c in con_nam:
                if not(c[3:] == l) and l in o_nam:
                    dc_pf_contingency = (f_c[t, l, c] == B[l] * (theta_c[t, from_b, c] - theta_c[t, to_b, c]))
                    model.addConstr(dc_pf_contingency, name= name + f"_{c}")
    return model


def optimization_SCOS_model_gurobi(data):
    """  Prepare Gurobi optimization model for:  SCOS_model security-constrained outage scheduling problem """
    """  X_sol, D_curt, P_gen = load_and_format_solution('optimal_solution.json', nodes_names, generators_names, outages_names, T)  """
    # Initialize logger
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger()

    # Preprocess the data
    net = data['network']
    max_tasks = data['max_number_of_maintenance_tasks']
    nod_nam = [f'bus_{b}' for b in range(data['num_buses'])]  # Bus indices
    linod_nam = [f'line_{l}' for l in range(data['num_branches'])]
    gen_nam = [f'gen_{g}' for g in net.gen.index.tolist()]
    o_nam = data['outages']['names']
    con_nam = data.get('n_minus1_names', None)
    T = [f'step_{t}' for t in range(len(data['nodal_demand']))]
    p_max = {gn: v for gn, v in zip(gen_nam, net.gen['max_p_mw'])}
    p_min = {gn: v for gn, v in zip(gen_nam, net.gen['min_p_mw'])}

    f2b = net.line[['from_bus', 'to_bus']].values.tolist() + net.trafo[['hv_bus', 'lv_bus']].values.tolist()
    f_lim = {bn: v for bn, v in zip(linod_nam, data['branch_capacity'])}
    g2bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]

    B = {bn: v for bn, v in zip(linod_nam, np.real(net._ppc["internal"]['Bf'].A).max(axis=1))}
    S = pd.DataFrame(net._ppc["internal"]['Cft'].A, index=linod_nam, columns=nod_nam)

    # Generator indices
    if con_nam is None:
        con_nam = [f'n1_{l}' for l in linod_nam] + [f'n1_{g}' for g in gen_nam]

    # Pre-fetch some values for efficiency
    priority = {o_nam: outages['priorities'][o] for o, o_nam in enumerate(outages['names'])}
    durations = {o_nam: outages['expected_duration_steps'][o] for o, o_nam in enumerate(outages['names'])}
    cost_per_step = {o_nam: outages['cost_per_step'][o] for o, o_nam in enumerate(outages['names'])}
    # ---- START ----
    try:
        # PROBLEM: Security-Constrained Outage Scheduling problem
        model = Model("Transmission_Outage_Scheduling")

        # VARIABLES
        xt = model.addVars(T, o_nam, vtype=GRB.BINARY, name="planned_outage_indicator")
        sxt = model.addVars(T, o_nam, vtype=GRB.BINARY, name="start_outage_indicator")
        ext = model.addVars(T, o_nam, vtype=GRB.BINARY, name="end_outage_indicator")
        pgen = model.addVars(T, gen_nam, lb=0, name="power_generation")
        pgen_c = model.addVars(T, gen_nam, con_nam, lb=0, name="power_generation_contingency")
        d_wc = model.addVars(T, lb=0, name="worst_case_curtailment")
        d_wc_c = model.addVars(T, con_nam, lb=0, name="worst_case_curtailment_contingency")
        f = model.addVars(T, linod_nam, name="flow_tl")
        f_c = model.addVars(T, linod_nam, con_nam, name="flow_tl_contingency")
        theta = model.addVars(T, nod_nam, lb=-25, ub=25, name="nodal_phase_tb")
        theta_c = model.addVars(T, nod_nam, con_nam, lb=-25, ub=25, name="nodal_phase_tb_contingency")

        # OBJECTIVE FUNCTION: Maximize scheduled outages * priority & Minimize Load Curtailment
        Objective_fun = quicksum((len(T)-t_idx) / len(T) * xt[t, o] * priority[o] * cost_per_step[o] for t_idx, t in enumerate(T) for o in o_nam)
        Objective_fun -= quicksum(1e4 * d_wc[t] + 1e4 * d_wc_c[t, c] for t in T for c in con_nam)  # Add the curtailment term

        model.setObjective(Objective_fun, GRB.MAXIMIZE)

        logger.info(
            f'{blue_c}Variables Added:{reset_c}\n'
            f' - xt: Scheduled outage decisions\n'
            f' - sxt, ext: start-end indicators for the outage task\n'
            f' - pgen, pgen_c: Generated power normal and N-1 states\n'
            f' - d_cut, d_cut_c: Worst-cases demand cut normal and N-1 states\n' 
            f' - f, f_c: Power flow in normal and N-1 states\n' 
            f' - theta, theta_c: Phase angles in normal operation and N-1 states\n' 
            f'{blue_c} Objective Function Defined: ∑_t (x_t_o * priority_o * (T-t)/T) - d_curtailed_worst_case {reset_c}\n'
        )

        # CONSTRAINTS:
        logger.info(f'{blue_c} ADDING CONSTRAINTS: {reset_c}')
        logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
        model = add_planned_outages_constraints(model, o_nam, xt, sxt, ext, max_tasks, durations, T)

        logger.info('adding power production constraints, normal + N-1 failures')
        model = add_generators_constraints(model, xt, o_nam, con_nam, gen_nam, ref_b=data['ref_buses'],
                                           theta=theta, pgen=pgen, pgen_c=pgen_c, p_max=p_max, p_min=p_min, T=T)

        logger.info('adding line flow constraints, normal + N-1 failures')
        model = add_line_flow_constraints(model, xt, linod_nam, o_nam, con_nam, f_lim, f, f_c, T)

        logger.info('adding node balance constraints, normal + N-1 failures')
        model = add_nodal_power_balance_constraints(model, nod_nam, linod_nam, gen_nam, con_nam, S, demand=data['nodal_demand'],
                                                    pgen=pgen, pgen_c=pgen_c, f=f, f_c=f_c,
                                                    g2bus=g2bus, d_wc=d_wc, d_wc_c=d_wc_c, T=T)

        logger.info('adding dc power flow constraints, normal + N-1 failures')
        model = add_dc_line_flow_constraints(model, f2b, B, f, theta, f_c, theta_c, linod_nam, o_nam, con_nam, T)

        # Solve the model
        model.update()
        model.optimize()

        # Check optimization status
        if model.status == GRB.OPTIMAL:
            logger.info(f"{blue_c} Optimal solution found! :-) :-):-){reset_c}")
            # Retrieve and save variable values
            solution = {v.VarName: v.X for v in model.getVars()}
            # Save the solution to a file or database
            with open("optimal_solution" + data['config']['case_name'] + ".json", "w") as f:
                json.dump(solution, f)

        elif model.status == GRB.INFEASIBLE:
            logger.warning(f"{red_c} Model is infeasible! :-(:-(:-({reset_c}")

            # Run IIS to find conflicting constraints
            model.computeIIS()
            model.write("model.ilp")  # Write IIS to a file for inspection
            print("Conflicting constraints are:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)

        elif model.status == GRB.UNBOUNDED:
            logger.warning("Model is unbounded.")
            # Save unbounded model status to log or file
            with open("model_status.txt", "a") as f:
                f.write("Model is unbounded.\n")

        else:
            logger.info(f"Optimization was stopped with status: {model.status}")

        LOADed_SOLUTION = load_json("optimal_solution.json")
        x_temp = []
        for o in o_nam:
            x_temp.append([LOADed_SOLUTION[f'planned_outage_indicator[{t},{o}]'] for t in T])
        X_OutageSchedule = pd.DataFrame(x_temp, index=o_nam, columns=T)

        x_temp = []
        for c in con_nam:
            x_temp.append([LOADed_SOLUTION[f'worst_case_curtailment_contingency[{t},{c}]'] for t in T])
        WC_CURTAIL_CON = pd.DataFrame(x_temp, index=con_nam, columns=T)
        WC_CURTAIL = pd.DataFrame([LOADed_SOLUTION[f'worst_case_curtailment[{t}]'] for t in T])

        x_temp = []
        for g in gen_nam:
            x_temp.append([LOADed_SOLUTION[f'power_generation[{t},{g}]'] for t in T])
        PowerGenerated = pd.DataFrame(x_temp, index=gen_nam, columns=T)

        # Results
        return LOADed_SOLUTION, X_OutageSchedule, WC_CURTAIL, WC_CURTAIL_CON, PowerGenerated

    except GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

    except Exception as e:
        print(e)


def visualize_results(solution, X_OutageSchedule, PowerGenerated, WC_CURTAIL, o_nam, gen_nam):
    fig, ax = plt.subplots(int(len(o_nam)/4), 4)
    ax = ax.flatten()
    [X_OutageSchedule.loc[o,:].plot(ax=ax[i]) for i, o in enumerate(o_nam)]
    for a in ax:
        a.set_xlabel('time')
    plt.title('Outage Schedule')
    plt.grid()
    plt.show()

    X_OutageSchedule.sum().plot()
    plt.xlabel('time step')
    plt.title('Total Outages')
    plt.grid()
    plt.show()

    sbn.barplot(X_OutageSchedule.T.sum())
    plt.xlabel('Outages duration')
    plt.grid()
    plt.show()

    WC_CURTAIL.plot()
    plt.title('curtailed demand')
    plt.show()

    [plt.plot([solution[f'nodal_phase_tb[step_{t},bus_{b}]'] for t in range(364)]) for b in range(12)]
    plt.title('node phases')
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(int(len(gen_nam) / 4)+1, 4)
    ax = ax.flatten()
    for i, g in enumerate(gen_nam):
        PowerGenerated.loc[g, :].plot(ax=ax[i])
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('Pgen ' + g)
    plt.title('Outage Schedule')
    plt.grid()
    plt.show()


def load_and_format_solution(solution_json_path, nodes_names, generators_names, outage_names, T):
    solution = load_json(solution_json_path)
    x_outage = []
    for o in outage_names:
        x_outage.append([solution[f'xt_lines_and_gens[{t},{o}]'] for t in T])
    X_sol = pd.DataFrame(x_outage, index=outage_names, columns=T)

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
    daily_demand = aggregate_hourly_demand(hourly_demand, aggregation_step='H')

    # example scheduled outage information
    outages = {'indices': [0, 1, 2, 3, 5, 8, 1, 2],
               'names': ['line_0', 'line_1',  'line_2', 'line_3', 'line_5', 'line_8', 'gen_1', 'gen_2'],
               'type': ['line', 'line', 'line', 'line', 'line', 'line', 'generator', 'generator'],
               'expected_duration_steps': [12, 7, 11, 7, 10, 20, 20, 60],  # step_are_in_days for now
               'cost_per_step': [1000, 2000, 1000, 1000, 2000, 2000, 5000, 5000],
               'priorities': [1, 2, 1, 3, 2, 1, 1, 3]}  # todo: this could be used 'in combination' with the step number

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in network.trafo]

    data = {'max_number_of_maintenance_tasks': 3,
            'nodal_demand': daily_demand,
            'outages': outages,
            'config': config,
            'network': network,
            'num_buses': num_buses,
            'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(25)]]}

    dic_reults = optimization_SCOS_model_gurobi(data)