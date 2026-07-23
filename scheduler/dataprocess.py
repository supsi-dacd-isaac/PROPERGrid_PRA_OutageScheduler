
from optimizers.gurobi_SCOS import *
from scheduler.utils_pre_process import aggregate_hourly_demand, aggregate_step_costs_and_durations

# ANSI escape code for colored text
blue_c, green_c, purple_c, cyan_c, red_c, gray_c = "\033[94m", "\033[92m", "\033[95m", "\033[96m", "\033[91m", " "
bold_c, underline_c, reset_c = "\033[1m", "\033[4m", "\033[0m"

# logging details
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


"""def get_project_root() -> Path: 
    return Path(__file__).parent.parent
"""

from scipy import sparse


def as_dense_array(matrix) -> np.ndarray:
    """Convert a SciPy sparse matrix/array or dense object to ndarray."""
    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def get_project_root(marker_files: Sequence[str] = (".git", "pyproject.toml", "setup.py")) -> Path:
    """
    Return the root of the project by looking for one of the given marker files/directories.
    Starts from this file and walks up until it finds a parent containing any marker.
    Raises FileNotFoundError if no marker is found.
    """
    # First try to find a directory named "PROPER"
    current = Path(__file__).resolve()
    logging.info(f"Current file path: {current}")

    for directory in (current, *current.parents):
        logging.info(f"Checking directory: {directory}")
        if directory.name == "PROPER":
            logging.info(f"Found PROPER directory: {directory}")
            return directory

    # If not found, fall back to marker files
    for directory in (current, *current.parents):
        if any((directory / m).exists() for m in marker_files):
            logging.info(f"Found marker file in: {directory}")
            return directory

    raise FileNotFoundError(f"Could not find project root (searched for {marker_files!r})")


def load_json(file_path):
    """load json file"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_matlab_data(conf: dict, file_name: str = 'hourlyDemandBus.mat', feature_name: str = 'hourlyDemandBus'):
    """ Load matlab data """
    data_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], file_name)
    try:
        logging.info(f'Loading {file_name}')
        mat_data = loadmat(data_path)
        hourly_loads = pd.DataFrame(mat_data[feature_name].T)
        try:
            bus_name_path = os.path.join(get_project_root(), conf['data_path'], conf['case_name'], 'bus_names.json')
            if os.path.exists(bus_name_path):
                bus_names = load_json(bus_name_path)
                bus_names = [val for _, val in bus_names.items()]
            else:
                logging.info(f"Bus names file not found at {bus_name_path}, using default names")
                bus_names = ['bus_' + str(i) for i in range(hourly_loads.shape[1])]
        except Exception as e:
            logging.warning(f"Error loading bus names: {e}, using default names")
            bus_names = ['bus_' + str(i) for i in range(hourly_loads.shape[1])]
        if len(hourly_loads.columns) == len(bus_names):
            hourly_loads.columns = bus_names
        return hourly_loads

    except FileNotFoundError:
        logging.error(f"File {data_path} not found")
        return None


def load_network(conf: dict):
    """ Dynamically load the network using getattr from pandapower networks"""
    try:
        logging.info('Loading network model')
        return getattr(networks, conf["panda_power_case_name"])()
    except AttributeError:
        logging.error(f"Network model {conf['panda_power_case_name']} not found")
        return None


def data_loader(conf_path: str):
    """load data from the configuration file"""
    # Get the project root directory
    project_root = get_project_root()

    # Convert to absolute path if it's relative
    if not os.path.isabs(conf_path):
        # Use project root as the base for relative paths
        conf_path = os.path.join(project_root, conf_path)

    # Normalize the path to remove any '..' or '.' components
    conf_path = os.path.normpath(conf_path)

    logging.info(f"Loading configuration from: {conf_path}")
    conf = load_json(conf_path)
    panda_power_network = load_network(conf)
    try:
        logging.info("running DC power flow calculation....for _ppc initialization.")
        pp.rundcpp(panda_power_network)
    except:
        logging.warning("DC power flow calculation failed for this system. Proceeding without _ppc initialization.")
    df_hourly_nodal_demand = load_matlab_data(conf)
    return panda_power_network, df_hourly_nodal_demand, conf


def plot_show(val, xlb: str = 'x', ylb: str = 'y', ax=None, **kwargs):
    """  Plot and show """
    if ax is None:
        plt.plot(val, **kwargs)
        plt.ylabel(ylb)
        plt.xlabel(xlb)
        plt.grid()
        plt.show()
    else:
        ax.plot(val, **kwargs)
        ax.set_ylabel(ylb)
        ax.set_xlabel(xlb)
        ax.grid()
        return ax


def plot_ecdf_show(val, xlb: str = 'x', ylb: str = 'ecdf', **kwargs):
    """  Plot the empirical cumulative distribution function (ECDF) for given values.   """
    # Prepare ECDF plotting
    ecdf = ECDF(val)
    x = np.sort(val)  # Sort the values for plotting
    y = ecdf(x)  # Get ECDF values for x

    plt.step(x, y, where='post', **kwargs)  # Use step plot for ECDF
    plt.ylabel(ylb)
    plt.xlabel(xlb)
    plt.grid(True)
    plt.show()


def prepare_data(conf_path = None):
    # Load system data and historical nodal demand data
    if conf_path is None:
        conf_path = '../config/conf_IEEE24_scheduler.json'

    network, hourly_demand, config = data_loader(conf_path)
    aggregation_step = config['aggregation_time']  # 'W', 'D', '12H', 'H,  etc.
    # Define PM activities (planned outages)
    comp_types = config['comp_types']
    comp_ids = config['comp_ids']
    cost_per_days = config['cost_per_days']  # m.u/day
    expected_duration_days = config['expected_duration_days']  # days/PM
    priorities = config['priorities']  # high = 3, medium = 2, low = 1
    planned_outage_names = [f'{type}_{idx}' for idx, type, in zip(comp_ids, comp_types)]

    # PREPARE DATA INPUT DICTIONARY
    nodal_demand_aggregated = aggregate_hourly_demand(hourly_demand, aggregation_step=aggregation_step)

    PM_cost_per_step, PM_duration_steps = aggregate_step_costs_and_durations(cost_per_days, expected_duration_days,
                                                                             aggregation_step=aggregation_step)

    # example scheduled outage information
    outages = {'indices': comp_ids,
               'names': planned_outage_names,
               'type': comp_types,
               'expected_duration_steps': PM_duration_steps,
               'cost_per_step': PM_cost_per_step,
               'priorities': priorities}

    num_branches = len(network.trafo) + len(network.line)
    num_buses = len(network.bus)
    branch_capacity = [175 if max_i_ka <= 1 else 500 for max_i_ka in network.line['max_i_ka']] + [400 for _ in
                                                                                                  network.trafo]
    n_minus1_names = [f'n1_{l}' for l in [f'line_{k}' for k in range(30)]]

    DATA = {'max_tasks': 2,
            'nodal_demand': nodal_demand_aggregated,
            'outages': outages,
            'config': config,
            'network': network,
            'num_buses': num_buses, 'num_branches': num_branches,
            'ref_buses': ['bus_12'], 'branch_capacity': branch_capacity,
            'n_minus1_names': n_minus1_names}

    names = {
        'outages': DATA['outages']['names'],  # List of outage names
        'lines': [f'line_{ll}' for ll in range(DATA['num_branches'])],  # List of line names
        'contingencies': DATA.get('n_minus1_names', None),  # List of contingency names
        'buses': [f'bus_{b}' for b in range(DATA['num_buses'])],  # List of bus names
        'generators': [f'gen_{g}' for g in DATA['network'].gen.index.tolist()],  # List of generator names
    }

    if names['contingencies'] is None:  # Generator indices
        names['contingencies'] = [f'n1_{l}' for l in names['lines']] + [f'n1_{g}' for g in names['generators']]

    DATA['names'] = names

    # Preprocess the data
    T = [f'step_{t}' for t in range(len(DATA['nodal_demand']))]
    DATA['p_max'] = {gn: (v + 100 if v > 0 else 200) for gn, v in
                     zip(names['generators'], DATA['network'].gen['max_p_mw'])}
    DATA['p_min'] = {gn: v * 0 for gn, v in zip(names['generators'], DATA['network'].gen['min_p_mw'])}  # todo fixme
    DATA['f_lim'] = {ln: v for ln, v in zip(names['lines'], DATA['branch_capacity'])}
    DATA['g2bus'] = [f'bus_{g}' for g in DATA['network'].gen['bus'].values.tolist()]

    # Precompute generator to node mapping
    DATA['gen_to_node'] = {b: names['generators'][idx] for idx, b in enumerate(DATA['g2bus'])}
    #DATA['S'] = DATA['network']._ppc["internal"]['Cft'].A.T  # S[l,b]=1 if line l 'enter' bus b, -1 if it 'exit' bus b
    #DATA['B_mat'] = np.real(DATA['network']._ppc["internal"]['Bf'].A)


    # Pandapower/PYPOWER internal matrices
    internal_ppc = DATA["network"]._ppc["internal"]
    Cft = as_dense_array(internal_ppc["Cft"])
    Bf = as_dense_array(internal_ppc["Bf"])

    #print("Cft:", Cft.shape)
    #print("Expected branches:", DATA["num_branches"])
    #print("Expected buses:", DATA["num_buses"])

    # Preserve the orientation used by the existing implementation
    DATA["S"] = Cft.T
    DATA["B_mat"] = np.real(Bf)
    DATA["B_lines"] = np.max(DATA["B_mat"], axis=1)

    # Pre-fetch some values for efficiency
    DATA['priority'] = {o_nam: DATA['outages']['priorities'][o] for o, o_nam in enumerate(names['outages'])}
    DATA['durations'] = {o_nam: DATA['outages']['expected_duration_steps'][o] for o, o_nam in
                         enumerate(names['outages'])}
    DATA['step_cost_outage'] = {o_nam: DATA['outages']['cost_per_step'][o] for o, o_nam in enumerate(names['outages'])}

    DATA['T'] = T  # value of loss load

    DATA['VoLL'] = 1e6  # value of loss load
    DATA['n_samples'] = 50
    DATA['use_DC_PF'] = False
    return DATA



def get_solution_dic(model, data):

    names = data['names']
    T = data['T']

    """ get solution dictionary Check optimization status"""
    if model.status == GRB.OPTIMAL:
        logger.info(f"{green_c} Optimal solution found! :-) 🎉 :-) {reset_c}")
    elif model.status == GRB.TIME_LIMIT:
        logger.warning(f"{blue_c} Solution found with time limit! {reset_c}")
    else:
        logger.error(f"{red_c} Optimization ended with status: {model.status}{reset_c}")

    # Check optimization status
    SOLUTION = None
    if model.status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT}:
        SOLUTION = {v.VarName: v.X for v in model.getVars()}  # Retrieve and save variable values

    elif model.status == GRB.INFEASIBLE:
        logger.warning(f"{red_c} M is Infeasible! :-(:-(:-({reset_c}")
        logger.warning(f"{red_c} Run IIS to find conflicting constraints{reset_c}")
        model.computeIIS()
        logger.warning(f"{red_c} Writing IIS to a file for inspection {reset_c}")
        model.write("M.ilp")
        print("Conflicting constraints are:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(c.ConstrName)
        return None

    elif model.status == GRB.UNBOUNDED:
        logger.warning("Model is unbounded.")  # Save unbounded M status to log or file
        with open("M_status.txt", "a") as f:
            f.write("Model is unbounded.\n")
        return None

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

    results_variables_dictionary = {
        "X_OutageSchedule": X_OutageSchedule,
        "PowerGenerated": GENERATION,
        "Line_Flows": FLOWS,
        "WC_CURTAIL": WC_CURTAILED,
        "WC_CURTAIL_CON": WC_CURTAIL_CON,
        "GEN_PLUS_CURTAILED": GEN_PLUS_CURTAILED,
    }

    result_objective_function = model.getObjective().getValue()
    return results_variables_dictionary, result_objective_function