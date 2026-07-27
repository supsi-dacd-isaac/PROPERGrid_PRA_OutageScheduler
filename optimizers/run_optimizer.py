import os
import sys
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gurobipy import GRB, Model
import pickle
from utils.data_preporcess import *



### PREPROCESSING
def load_data_from_conf_grid_case(conf_path=None):
    """load data from grid configuration file"""
    if conf_path is None:
        # Try multiple possible locations for the config file
        possible_paths = [
            os.path.join(project_root, 'config', 'conf_IEEE24_scheduler.json'),
            os.path.join(os.path.dirname(project_root), 'config', 'conf_IEEE24_scheduler.json'),
            os.path.join(os.path.dirname(os.path.dirname(project_root)), 'config', 'conf_IEEE24_scheduler.json'),
            'C:\\Users\\roberto.rocchetta.in\\OneDrive - SUPSI\\Documenti\\GitHub\\PROPER\\config\\conf_IEEE24_scheduler.json']
        for path in possible_paths:
            logging.info(f"Trying config path: {path}")
            if os.path.exists(path):
                conf_path = path
                logging.info(f"Found config file at: {conf_path}")
                break

        if conf_path is None:
            raise FileNotFoundError(f"Could not find config file in any of the expected locations: {possible_paths}")
    else:
        conf_path = conf_path

    # Load system data and historical nodal demand data
    network, hourly_demand, config = data_loader(conf_path)
    aggregation_step = config['aggregation_time']  # 'W', 'D', '12H', 'H,  etc.
    scaling_factor = config['load_scaling_factor']  # scaling factor for the demand

    # -----  Prepare branch bus info
    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in
                                                                                                  network.trafo]

    # ----- Define PM activities (planned outages)
    comp_types, comp_ids, cost_per_days = config['comp_types'], config['comp_ids'], config['cost_per_days']
    expected_duration_days, priorities = config['expected_duration_days'], config['priorities']
    planned_outage_names = [f'{type}_{idx}' for idx, type, in zip(comp_ids, comp_types)]

    # -----  Others
    # aggregate power demand data (.mean()  .max()) over aggregation_step in { 'W', 'D', '12H',..., 'H'}
    nodal_demand_aggregated = aggregate_hourly_demand(hourly_demand * scaling_factor, aggregation_step=aggregation_step)

    # scale PM params: [cost/step], duration [steps]
    PM_cost_per_step, PM_duration_steps = aggregate_step_costs_and_durations(cost_per_days,
                                                                             expected_duration_days,
                                                                             aggregation_step=aggregation_step)

    # -----  Summary outage information
    outages = {'indices': comp_ids, 'names': planned_outage_names,
               'type': comp_types, 'expected_duration_steps': PM_duration_steps,
               'cost_per_step': PM_cost_per_step, 'priorities': priorities}

    # -----  Data Dictionary
    DATA = {'max_number_of_maintenance_tasks': config['max_number_of_maintenance_tasks'],
            'nodal_demand': nodal_demand_aggregated,
            'outages': outages, 'config': config,
            'network': network, 'num_buses': num_buses, 'num_branches': num_branches,
            'ref_buses': ['bus_12'],
            'branch_capacity': branch_capacity,
            'n_minus1_names': [f'n1_{l}' for l in [f'line_{k}' for k in range(10)]],
            'VoLL': config['VoLL'],
            'n_samples': config['n_samples'],
            'tail_prob_constraint': config['tail_prob_constraint'],
            'use_DC_PF': config['use_DC_PF']}

    # ----  Prepare names of variables for the optimization problem
    names = {
            'outages': DATA['outages']['names'],  # List of outage names
            'lines': [f'line_{lines}' for lines in range(DATA['num_branches'])],  # List of line names
            'contingencies': DATA.get('n_minus1_names', None),  # List of contingency names
            'buses': [f'bus_{busses}' for busses in range(DATA['num_buses'])],  # List of bus names
            'generators': [f'gen_{gens}' for gens in DATA['network'].gen.index.tolist()],  # List of generator names
            'demand_scenarios': [f'demand_{s}' for s in range(config['n_samples'])]  # List of demand scenarios
            }

    DATA['names'] = names
    DATA['T'] = [f'step_{t}' for t in range(len(nodal_demand_aggregated))]
    return DATA


def prepare_SCOS_data(data, names):
    # Preprocess the data
    net = data['network']
    max_tasks = data['max_number_of_maintenance_tasks']
    p_max = {gn: (v + 100 if v > 0 else 200) for gn, v in zip(names['generators'], net.gen['max_p_mw'])}
    p_min = {gn: v * 0 for gn, v in zip(names['generators'], net.gen['min_p_mw'])}  # todo fixme
    f_lim = {ln: v for ln, v in zip(names['lines'], data['branch_capacity'])}
    g2bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]
    # Precompute generator to node mapping
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}
    S = net._ppc["internal"]['Cft'].A.T  # S[l,b]=1 if line l 'enter' bus b, -1 if it 'exit' bus b
    B_mat = np.real(net._ppc["internal"]['Bf'].A)
    B_lines = np.max(B_mat, axis=1)
    if names['contingencies'] is None:  # Generator indices
        names['contingencies'] = [f'n1_{l}' for l in names['lines']] + [f'n1_{g}' for g in names['generators']]
    # Pre-fetch some values for efficiency
    priority = {o_nam: data['outages']['priorities'][o] for o, o_nam in enumerate(names['outages'])}
    durations = {o_nam: data['outages']['expected_duration_steps'][o] for o, o_nam in enumerate(names['outages'])}
    step_cost_outage = {o_nam: data['outages']['cost_per_step'][o] for o, o_nam in enumerate(names['outages'])}
    return max_tasks, p_max, p_min, f_lim, g2bus, gen_to_node, S, B_mat, B_lines, priority, durations, step_cost_outage


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


def load_metadata(results_dir: Path, filename: str):
    file_path = results_dir / f"{filename}.pkl"  # ← fix extension
    if file_path.exists():
        with open(file_path, 'rb') as f:         # ← open in binary mode
            data = pickle.load(f)
            params = data['params']
            dic_res = data['schedule_results']
            solution = data['solution']
            objective = data['objective']
            return dic_res, solution, objective, params

    return None, None, None, None


### POSTPROCESSING and SAVE UTILS
def analyze_model_solution(model: Model):
    summary = {
        'objective_value': model.ObjVal,
        'runtime_sec': model.Runtime,
        'mip_gap': model.MIPGap,
        'node_count': model.NodeCount,
        'iteration_count': model.IterCount,
    }

    vars_by_type = {'x': [], 'P': [], 'f': [], 'zeta': [], 'other': []}
    for var in model.getVars():
        name = var.VarName
        if 'x_' in name:
            vars_by_type['x'].append((name, var.X))
        elif 'P_' in name:
            vars_by_type['P'].append((name, var.X))
        elif 'f_' in name:
            vars_by_type['f'].append((name, var.X))
        elif 'zeta' in name:
            vars_by_type['zeta'].append((name, var.X))
        else:
            vars_by_type['other'].append((name, var.X))

    dfs = {k: pd.DataFrame(v, columns=['VarName', 'Value']) for k, v in vars_by_type.items()}

    tight_constraints = []
    for c in model.getConstrs():
        if abs(c.Slack) < 1e-6:
            tight_constraints.append((c.ConstrName, c.Slack))

    tight_df = pd.DataFrame(tight_constraints, columns=['Constraint', 'Dual'])

    objective_terms = {
        'PM_cost': dfs['x']['Value'].sum(),
        'shed_cost': dfs['zeta']['Value'].sum(),
        'gen_cost': dfs['P']['Value'].sum(),
    }

    return {
        'summary': summary,
        'variables': dfs,
        'tight_constraints': tight_df,
        'objective_terms': objective_terms
    }


def post_process_results_older_version(res_path_name, names, T):
    """ loading gurobi solution and post-processing the schedule_results"""
    try:
        SOLUTION = load_json(res_path_name)
    except FileNotFoundError:
        print(f"File {res_path_name} not found")
        return None, None

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
    try:
        cvar_temp = []
        for g in names['buses']:
            cvar_temp.append([SOLUTION[f'CVaR[{t},{g}]'] for t in T])
        CVARisk = pd.DataFrame(cvar_temp, index=names['buses'], columns=T)
    except:
        CVARisk = pd.DataFrame([])

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


def post_process_results(solution, names, T):
    """Load Gurobi solution and post‑process schedule_results into DataFrames."""

    # helper: build a DataFrame for any (items, value_fn)
    def make_df(items, value_fn):
        return pd.DataFrame([[value_fn(t, item) for t in T] for item in items], index=items, columns=T)

    # outage schedule
    lambda_fun = lambda t, o: solution[f'planned_outage_indicator[{t},{o}]']
    X_OutageSchedule = make_df(names['outages'], lambda_fun)

    # line flows
    lambda_fun = lambda t, l: solution[f'flow_tl[{t},{l}]']
    Line_Flows = make_df(names['lines'], lambda_fun)

    # contingency curtailments
    lambda_fun = lambda t, c: sum(solution[f'loss_of_load_contingency[{t},{b},{c}]']for b in names['buses'])
    WC_CURTAIL_CON = make_df(names['contingencies'],lambda_fun)

    # total curtailed (no index, just one row)
    WC_CURTAILED = pd.DataFrame([[sum(solution[f'loss_of_load[{t},{b}]'] for b in names['buses']) for t in T]],
        index=['total'], columns=T)

    # generation
    lambda_fun = lambda t, g: solution[f'power_generation[{t},{g}]']
    PowerGenerated = make_df(names['generators'], lambda_fun)

    # gen + curtailed
    GEN_PLUS_CURTAILED = (PowerGenerated.sum(axis=0).values + WC_CURTAILED.values[0])

    # optional CVaR
    try:
        CVARisk = make_df(names['buses'], lambda t, b: solution[f'CVaR[{t},{b}]'])
    except KeyError:
        CVARisk = pd.DataFrame()

    results = {
        "X_OutageSchedule": X_OutageSchedule,
        "PowerGenerated":   PowerGenerated,
        "Line_Flows":       Line_Flows,
        "WC_CURTAIL":       WC_CURTAILED,
        "WC_CURTAIL_CON":   WC_CURTAIL_CON,
        "GEN_PLUS_CURTAILED": GEN_PLUS_CURTAILED,
        "CVARisk":          CVARisk,
    }
    return results


def hash_simulation_params(params: dict) -> str:
    """Generate a short hash for a given dict of parameters."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]


def get_simulation_filename(params: dict, prefix: str = "sim") -> str:
    hash_id = hash_simulation_params(params)
    return f"{prefix}_{hash_id}"


def encode_simulation_name(params: dict) -> str:
    """Encode simulation parameters into a compact filename."""
    parts = [f"{key}={str(val).replace(' ', '')}" for key, val in sorted(params.items())]
    return "__".join(parts)


def simulation_exists(results_dir: Path, filename: str, ext: str = ".pkl") -> bool:
    """Check if a simulation file already exists."""
    file_path = results_dir / f"{filename}{ext}"
    return file_path.exists()


def save_metadata(dic_res, results_dir: Path, filename: str, solution=None, objective=None, params=None):
    data = {
        'params': params,
        'schedule_results': dic_res,
        'solution': solution,
        'objective': objective
    }
    with open(results_dir / f"{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def get_gurobi_solution(M, save_res_dir=None):
    """ preparing solution and saving it """
    if M.status == GRB.OPTIMAL:
        logger.info(f"{green_c} Optimal solution found! :-) 🎉 :-) {reset_c}")
    elif M.status == GRB.TIME_LIMIT:
        logger.warning(f"{blue_c} Solution found with time limit! {reset_c}")
    else:
        logger.error(f"{red_c} Optimization ended with status: {M.status}{reset_c}")

    # Check optimization status
    solution = None
    if M.status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT}:
        try:
            solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
            if save_res_dir is not None:
                with open(save_res_dir, "w") as f:  # Save the solution to a file or database
                    logger.info(f"{green_c} saving in: {reset_c} {save_res_dir}")
                    json.dump(solution, f)
        except:
            logger.error(f"{red_c} Error saving solution {reset_c}")
            raise

    elif M.status == GRB.INFEASIBLE:
        logger.warning(f"{red_c} M is Infeasible! :-(:-(:-({reset_c}")
        logger.warning(f"{red_c} Run IIS to find conflicting constraints{reset_c}")
        M.computeIIS()
        logger.warning(f"{red_c} Writing IIS to a file for inspection {reset_c}")
        M.write("M.ilp")
        print("Conflicting constraints are:")
        for c in M.getConstrs():
            if c.IISConstr:
                print(c.ConstrName)

    elif M.status == GRB.UNBOUNDED:
        logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
        with open("M_status.txt", "a") as f:
            f.write("Model is unbounded.\n")

    return solution




def get_and_save_solution(M, save_res_dir=None, case_name=None, aggregation_time=None):
    if M.status == GRB.OPTIMAL:
        logger.info(f"{green_c} Optimal solution found! :-) 🎉 :-) {reset_c}")
    elif M.status == GRB.TIME_LIMIT:
        logger.warning(f"{blue_c} Solution found with time limit! {reset_c}")
    else :
        logger.error(f"{red_c} Optimization ended with status: {M.status}{reset_c}")

    if save_res_dir is None:
        save_res_dir = "../outputs/schedule_results/monolitic_optim_" + case_name + '_' + aggregation_time + ".json"

    # Check optimization status
    solution = None
    if M.status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT}:
        solution = {v.VarName: v.X for v in M.getVars()}  # Retrieve and save variable values
        with open(save_res_dir, "w") as f:  # Save the solution to a file or database
            logger.info(f"{green_c} saving in: {reset_c} {save_res_dir}")
            json.dump(solution, f)

    elif M.status == GRB.INFEASIBLE:
        logger.warning(f"{red_c} M is Infeasible! :-(:-(:-({reset_c}")
        logger.warning(f"{red_c} Run IIS to find conflicting constraints{reset_c}")
        M.computeIIS()
        logger.warning(f"{red_c} Writing IIS to a file for inspection {reset_c}")
        M.write("M.ilp")
        print("Conflicting constraints are:")
        for c in M.getConstrs():
            if c.IISConstr:
                print(c.ConstrName)

    elif M.status == GRB.UNBOUNDED:
        logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
        with open("M_status.txt", "a") as f:
            f.write("Model is unbounded.\n")

    return solution