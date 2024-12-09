from utils.utils import *
from utils.dataloader import *


def build_incidence_matrix(from_to_bus, num_buses, num_lines):
    S_k_l = {(k, l): 0 for k in range(num_buses) for l in range(num_lines)}
    for l, (from_bus, to_bus) in enumerate(from_to_bus):
        S_k_l[from_bus, l] = -1
        S_k_l[to_bus, l] = 1
    return S_k_l


def build_susceptance_matrix(from_to_bus, B, num_buses, num_lines):
    B_k_l = {(k, l): 0 for k in range(num_buses) for l in range(num_lines)}
    for l, (from_bus, to_bus) in enumerate(from_to_bus):
        B_k_l[from_bus, l] = -B[l]
        B_k_l[to_bus, l] = B[l]
    return B_k_l


def aggregate_hourly_demand(data, start_date='2023-01-01 00:00', aggregation_step='D'):
    """
    Aggregates hourly nodal demand by a specified step (e.g., daily, weekly).
    Returns: pd.DataFrame: A DataFrame with the aggregated demand using the maximum value over the specified step.
    """
    df_loads_24 = pd.DataFrame(data.values, columns=data.columns, index=pd.date_range(start_date, periods=len(data), freq='h'))
    if not pd.api.types.is_datetime64_any_dtype(df_loads_24.index):  # Ensure that the DataFrame index is a datetime index for proper resampling
        df_loads_24.index = pd.to_datetime(df_loads_24.index)
    return df_loads_24.resample(aggregation_step).mean()


def aggregate_step_costs_and_durations(cost_per_days, expected_duration_days, aggregation_step):
    """
    Aggregates step costs and durations based on the given aggregation step.

    Parameters:
    cost_per_days (list): List of cost per day for each task
    expected_duration_days (list): List of expected durations in days for each task
    aggregation_step (str): The time step for aggregation ('H' for hours, 'W' for weeks, 'D' for days, etc.)

    Returns:
    tuple: Aggregated costs and durations for the chosen aggregation step
    """

    # Define the conversion factors for each aggregation step
    conversion_factors = {
        'H': {'cost': 1 / 24, 'duration': 24},  # cost per hour, duration in hours
        '3H': {'cost': 1 / 8, 'duration': 8},  # cost per 6 hours, duration in 6-hour periods
        '6H': {'cost': 1 / 4, 'duration': 4},  # cost per 6 hours, duration in 6-hour periods
        '12H': {'cost': 1 / 2, 'duration': 2},  # cost per 12 hours, duration in 12-hour periods
        'W': {'cost': 7, 'duration': 1 / 7},  # cost per week, duration in weeks
        'M': {'cost': 30, 'duration': 1 / 30},  # cost per month, duration in months
        'D': {'cost': 1, 'duration': 1},  # cost per day, duration in days
        'Q': {'cost': 90, 'duration': 1 / 90}  # cost per quarter, duration in quarters
    }

    # Check if the aggregation_step is valid
    if aggregation_step not in conversion_factors:
        raise ValueError(
            f"Invalid aggregation step '{aggregation_step}'. Choose from {list(conversion_factors.keys())}.")

    # Get the conversion factors for the selected aggregation step
    cost_factor = conversion_factors[aggregation_step]['cost']
    duration_factor = conversion_factors[aggregation_step]['duration']

    # Aggregate costs and durations
    cost_per_step = [c_day * cost_factor for c_day in cost_per_days]
    expected_duration_steps = [d * duration_factor for d in expected_duration_days]

    return cost_per_step, expected_duration_steps


def prepare_data_for_outage_scheduling_problem(conf_path: str):
    """ Prepare data for the outage scheduling problem """
    panda_power_network, df_hourly_nodal_demand, conf = data_loader(conf_path)
    if panda_power_network is None or df_hourly_nodal_demand is None:
        logging.error("Data loading failed. Cannot prepare data for the outage scheduling problem.")
        return None

    # incidence matrix with elements S_{l,k} = -1 if line l 'enters' bus k, 1 if line l 'leaves' bus k, 0 otherwise
    S_lk = panda_power_network._ppc["internal"]["Cft"].A

    # Susceptance line-bus matrix with elements B_{l,k} = -Bl if line l 'enters' bus k, Bl if line l 'leaves' bus k, 0 otherwise
    B_lk = panda_power_network._ppc["internal"]["Bf"].A

    df_daily_nodal_demand = aggregate_hourly_demand(df_hourly_nodal_demand)


    # Extract bus and line information from the pandapower network
    buses = panda_power_network.bus
    lines = panda_power_network.line
    generators = panda_power_network.gen
    transformers = panda_power_network.trafo

    # Extract necessary information from buses, lines, and generators
    bus_info = buses[['name', 'vn_kv', 'type']]  # Bus voltage levels and types
    line_info = lines[['from_bus', 'to_bus', 'length_km', 'max_i_ka']]  # Line information like capacity and connections
    gen_info = generators[['bus', 'p_mw', 'max_p_mw']]  # Generator bus connections and capacities

    # Prepare the final data structure to return
    outage_scheduling_data = {
        "buses": bus_info,
        "lines": line_info,
        "generators": gen_info,
        "transformers": transformers,
        "hourly_nodal_demand": df_hourly_nodal_demand,  # Hourly demand for each bus
        "network": panda_power_network  # Complete pandapower network object for further reference
    }

    return outage_scheduling_data


def prepare_data_4_gurobi_security_constrained_outage_planning(data, names):
    # Preprocess the data
    net = data['network']
    max_tasks = data['max_number_of_maintenance_tasks']
    T = [f'step_{t}' for t in range(len(data['nodal_demand']))]
    p_max = {gn: (v if v > 0 else 200) for gn, v in zip(names['generators'], net.gen['max_p_mw'])}
    p_min = {gn: v * 0 for gn, v in zip(names['generators'], net.gen['min_p_mw'])}  # todo fixme
    f_lim = {ln: v for ln, v in zip(names['lines'], data['branch_capacity'])}
    g2bus = [f'bus_{g}' for g in net.gen['bus'].values.tolist()]

    # Precompute generator to node mapping
    gen_to_node = {b: names['generators'][idx] for idx, b in enumerate(g2bus)}
    # Precompute the transposed S matrix
    # to avoid repeated computations  S[l,b]=1 if line l 'enter' bus b, -1 if it 'exit' bus b
    S = net._ppc["internal"]['Cft'].A.T
    B_mat = np.real(net._ppc["internal"]['Bf'].A)
    B_lines = np.max(B_mat, axis=1)

    if names['contingencies'] is None:  # Generator indices
        names['contingencies'] = [f'n1_{l}' for l in names['lines']] + [f'n1_{g}' for g in names['generators']]

    # Pre-fetch some values for efficiency
    priority = {o_nam: data['outages']['priorities'][o] for o, o_nam in enumerate(names['outages'])}
    durations = {o_nam: data['outages']['expected_duration_steps'][o] for o, o_nam in enumerate(names['outages'])}
    step_cost_outage = {o_nam: data['outages']['cost_per_step'][o] for o, o_nam in enumerate(names['outages'])}

    return T, max_tasks, p_max, p_min, f_lim, g2bus, gen_to_node, S, B_mat, B_lines, priority, durations, step_cost_outage
