import pandapower as pp
import pandas as pd
import numpy as np
from pra_psa.reliability_performance import *


def get_OPF_gen(net, opf_solver=pp.rundcopp, distributed_slack=True):
    opf_solver(net, distributed_slack)
    return net.res_gen.p_mw, net.res_gen.q_mvar, net


def get_PF_loading(net, pf_solver=pp.runpp, distributed_slack=True):
    pf_solver(net, distributed_slack)  # Unpack the kwargs dictionary and pass it as keyword arguments
    return net.res_line.loading_percent, net


def apply_reference_dispatch(network, reference_p_mw=None, reference_q_mvar=None):
    if reference_p_mw is not None:
        for gen, p_mw in zip(network.gen.index, reference_p_mw):
            network.gen.at[gen, "p_mw"] = p_mw
            network.gen.at[gen, "max_p_mw"] = p_mw
            network.gen.at[gen, "min_p_mw"] = p_mw
    if reference_q_mvar is not None:
        for gen, q_mvar in zip(network.gen.index, reference_q_mvar):
            network.gen.at[gen, "q_mvar"] = q_mvar
            network.gen.at[gen, "max_q_mvar"] = q_mvar
            network.gen.at[gen, "min_q_mvar"] = q_mvar


def apply_load(network, p_load):
    for load, p in zip(network.load.index, p_load):
        network.load.at[load, "p_mw"] = p


def calculate_lodf_and_shift(network, pf_solver=pp.runpp):
    """Calculate Line Outage Distribution Factors (LODF) and shift factors."""
    n_lines = len(network.line)
    lodf_matrix = np.zeros((n_lines, n_lines))
    shift_factors = np.zeros((n_lines, n_lines))
    
    # Get base case flows
    pf_solver(network)
    base_flows = network.res_line.p_from_mw.values
    
    for i in range(n_lines):
        # Create a copy of the network for contingency
        net_cont = network.copy()
        # Outage line i
        net_cont.line.at[i, "in_service"] = False
        
        # Run power flow
        try:
            pf_solver(net_cont)
            cont_flows = net_cont.res_line.p_from_mw.values
            
            # Calculate LODF and shift factors
            for j in range(n_lines):
                if i != j:
                    lodf_matrix[j, i] = (cont_flows[j] - base_flows[j]) / base_flows[i]
                    shift_factors[j, i] = cont_flows[j] - base_flows[j]
        except:
            # If power flow fails, set factors to infinity
            lodf_matrix[:, i] = np.inf
            shift_factors[:, i] = np.inf
    
    return lodf_matrix, shift_factors


def analyze_contingency(network, contingency, reference_dispatch=None):
    """
    Analyze a contingency scenario and return detailed schedule_results.
    
    Args:
        network: pandapower network
        contingency: dictionary with 'element_type' and 'element_index'
        reference_dispatch: tuple of (p_mw, q_mvar) for reference dispatch
        
    Returns:
        dict: Analysis schedule_results including:
            - success: bool indicating if power flow converged
            - loading: line loading percentages
            - violations: list of constraint violations
            - severity: severity score of the contingency
    """
    # Create a deep copy of the network for analysis
    net = network.deepcopy()
    
    # Apply reference dispatch if provided
    if reference_dispatch is not None:
        p_mw, q_mvar = reference_dispatch
        apply_reference_dispatch(net, p_mw, q_mvar)
    
    # Apply contingency
    element_type = contingency['element_type']
    element_index = contingency['element_index']
    
    if element_type == 'line':
        net.line.loc[element_index, 'in_service'] = False
    elif element_type == 'gen':
        net.gen.loc[element_index, 'in_service'] = False
    elif element_type == 'trafo':
        net.trafo.loc[element_index, 'in_service'] = False
    else:
        raise ValueError(f"Unknown element type: {element_type}")
    
    # Run power flow
    try:
        pp.runpp(net)
        success = True
        loading = net.res_line.loading_percent
        violations = []
        
        # Check for violations
        if any(loading > 100):
            violations.append('line_overload')
        if any(net.res_bus.vm_pu < 0.9) or any(net.res_bus.vm_pu > 1.1):
            violations.append('voltage_violation')
        if any(net.res_gen.p_mw > net.gen.max_p_mw) or any(net.res_gen.p_mw < net.gen.min_p_mw):
            violations.append('generator_limit')
            
        # Calculate severity score
        severity = max(loading) if loading is not None else float('inf')
        
    except:
        success = False
        loading = None
        violations = ['power_flow_divergence']
        severity = float('inf')
    
    return {
        'success': success,
        'loading': loading,
        'violations': violations,
        'severity': severity
    }


def compute_ptdf_matrix(network):
    #todo: complete the function
    # Extract the square susceptance matrix (Bbus) from the network
    Bbus = network._ppc["internal"]['Bbus']
    # Compute the inverse of the susceptance matrix Bf
    Bf_inv = np.linalg.inv(Bbus)
    # Initialize the PTDF matrix
    n_lines = Bbus .shape[0]  # Number of transmission lines (branches)
    n_buses = Bbus.shape[1]  # Number of buses
    ptdf_matrix = np.zeros((n_lines, n_buses))  # PTDF matrix: lines x buses
    # Compute the PTDF for each line and each bus
    for i in range(n_lines):
        for k in range(n_buses):
            # PTDF calculation for line i and bus k
            ptdf_matrix[i, k] = Bf[i, k] * (Bf_inv[i, k] - Bf_inv[i, k])
    return ptdf_matrix


def compute_line_flows_from_sensitivity(net, sensitivity_matrix, base_line_flows, new_loads):
    """
    Compute new line flows based on the sensitivity matrix and changes in load.

    Parameters:
    - net: The network object.
    - sensitivity_matrix (np.ndarray): Sensitivity matrix (lines x buses), mapping load changes to line flow changes.
    - base_line_flows (np.ndarray): Base line flows in the network under reference loading.
    - new_loads (np.ndarray): New load values at each bus.

    Returns:
    - line_flows (np.ndarray): Updated line flows.
    - net: Updated network object.
    """
    # Calculate the difference in loads from the base case (assuming `new_loads` is an absolute load vector)
    base_loads = net.load.p_mw.values  # Assuming base load values are stored here
    delta_loads = new_loads - base_loads

    # Calculate the line flow changes using the sensitivity matrix
    delta_line_flows = sensitivity_matrix @ delta_loads

    # Compute the new line flows by adding the change to the base flows
    new_line_flows = base_line_flows + delta_line_flows

    # Update the network with the new line loading percentages
    net.res_line['loading_percent'] = new_line_flows / net.line[
        'max_p_mw'].values * 100  # Assuming max line capacity in 'max_p_mw'

    return net.res_line['loading_percent'], net


def compute_sensitivity_matrix_and_base_flows(network,
                                              opf_solver=pp.rundcopp,
                                              pf_solver=pp.rundcpp,
                                              perturbation=1e-2):
    """
    Compute the sensitivity matrix and base line flows for a given network.

    Parameters:
    - network: The electrical network object.
    - opf_solver: The OPF solver to use for the base case calculation.
    - perturbation (float): Small value to perturb power injections to compute sensitivities.

    Returns:
    - sensitivity_matrix (np.ndarray): Sensitivity matrix mapping active power injection changes to line flow changes.
    - base_line_flows (np.ndarray): Base line flows under the reference operating point.
    - network: Updated network with base OPF schedule_results.
    """
    # Step 1: Run base case OPF to get reference injections and flows
    reference_p_mw, reference_q_mvar, network = get_OPF_gen(network, opf_solver=opf_solver)
    base_line_flows = network.res_line.loading_percent.values

    # Get number of buses and lines
    num_buses = len(network.bus)
    num_lines = len(network.line)

    # Initialize sensitivity matrix (lines x buses)
    sensitivity_matrix = np.zeros((num_lines, num_buses))

    # Step 2: Compute sensitivity by perturbing each bus load
    for bus_id in range(num_buses):
        # Perturb active power injection at the current bus
        network.load.p_mw.iloc[bus_id] += perturbation  # Slight increase in load

        # Run power flow with perturbed load
        pf_solver(network)  # DC power flow is usually sufficient for sensitivity analysis

        # Calculate change in line flows due to perturbation
        perturbed_line_flows = network.res_line.loading_percent.values
        delta_flows = (perturbed_line_flows - base_line_flows) / perturbation

        # Populate sensitivity matrix column for current bus
        sensitivity_matrix[:, bus_id] = delta_flows

        # Reset the perturbation
        network.load.p_mw.iloc[bus_id] -= perturbation

    return sensitivity_matrix, base_line_flows, network, reference_p_mw, reference_q_mvar