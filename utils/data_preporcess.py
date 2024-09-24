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


