import pandapower as pp
import pandas as pd
import pandapower.networks as pn
import numpy as np
from pra_psa.reliability_performance import *
from utils.dataloader import data_loader
from utils.utils import *


def get_OPF_gen(net, opf_solver=pp.rundcopp):
    opf_solver(net)
    return net.res_gen.p_mw, net.res_gen.q_mvar, net


def get_PF_loading(net, pf_solver=pp.rundcpp):
    pf_solver(net)
    return net.res_line.loading_percent, net


def apply_reference_dispatch(network, reference_p_mw=None, reference_q_mvar=None):
    if reference_p_mw is not None:
        for gen, p_mw in zip(network.gen.index, reference_p_mw):
            network.gen.at[gen, "p_mw"] = p_mw
            network.gen.at[gen, "max_p_mw"] = p_mw
            network.gen.at[gen, "min_p_mw"] = p_mw

    if reference_q_mvar is not None:
        for gen, q_mvar in zip(network.gen.index, reference_p_mw):
            network.gen.at[gen, "sn_mva"] = q_mvar
            network.gen.at[gen, "max_q_mvar"] = q_mvar
            network.gen.at[gen, "min_q_mvar"] = q_mvar

    """
    # how to apply reference dispatch to the network 
    """
    network.ext_grid['min_p_mw'] = network.res_ext_grid['p_mw']
    network.ext_grid['max_p_mw'] = network.res_ext_grid['p_mw']
    return network


def apply_load(network, p_load):
    for load, pl_j in zip(network.load.index, p_load):
        network.load.at[load, "p_mw"] = pl_j
    return network


def apply_nk_contingency(network, failure_event):
    """
    Apply an N-k contingency to the network by setting specified lines or generators out of service.

    Parameters:
    - network: The original pandapower network object.
    - Nk_failure_event: A dictionary or list of dictionaries specifying the elements to take out of service.
                       Each dictionary must have two elements:
                        'element_type' in {line, gen, trafo}
                       'element_index' integer.

    Returns:
    - contingency_network: A deep copy of the network with the specified elements out of service.
    """

    contingency_network = network.deepcopy()
    # Ensure Nk_failure_event is always treated as a list for uniformity
    if type(failure_event) is not list:
        failure_event = [failure_event]

        for failure_k in failure_event:
            try:
                element_type = failure_k.get('element_type')
                element_index = failure_k.get('element_index')

                if element_type == "line":
                    # Remove the line by setting it out of service
                    contingency_network.line.at[element_index, "in_service"] = False
                elif element_type == "generator":
                    # Remove the generator by setting it out of service
                    contingency_network.gen.at[element_index, "in_service"] = False
                elif element_type == "trafo":
                    # Remove the generator by setting it out of service
                    contingency_network.trafo.at[element_index, "in_service"] = False
                else:
                    raise ValueError("Invalid element type for N-k....use {line, gen, trafo}, not ", element_type)

            except KeyError as e:
                logger.error(f"Invalid element index or type in contingency: {failure_k}. Error: {e}")
                continue  # Skip to the next failure event if an error occurs

    return contingency_network


def calculate_LODF_and_shift(network, pf_solver=pp.runpp):
    """
    Calculate the shift in power flows for each line in the Pandapower network
    when that line is taken out of service.

    Parameters:
    - network: The Pandapower network to analyze.

    Returns:
    - p_f_shift_df: A DataFrame containing the power flow shifts for each line.
    """
    # List to store the shifts in power flows
    p_f_shift, LODF = [], []
    # Run the power flow for the base case (all lines in service)
    pf_solver(network)
    original_flow = network.res_line['p_from_mw'].copy()  # Store the original flow
    for outaged_line_idx in network.line.index:  # Iterate over each line in the network
        # Set all lines to in_service (necessary to ensure correct initial state)
        network.line['in_service'] = True
        network.line.at[outaged_line_idx, 'in_service'] = False  # Set the line to out of service
        # Power flow on the outaged line before outage
        P_k_pre = network.res_line.loc[outaged_line_idx, 'p_from_mw']
        pf_solver(network)  # Run the power flow again after the outage
        shifted_flow = network.res_line['p_from_mw']  # Store the shifted flow after the outage
        # Calculate the shift in power flow due to the outage
        flow_shift = shifted_flow - original_flow
        # Calculate LODF for each line relative to the outaged line
        LODF.append(flow_shift / P_k_pre)
        p_f_shift.append(flow_shift)
    # Convert list of flow shifts to a DataFrame for easier analysis
    p_f_shift_df = pd.DataFrame(p_f_shift, index=network.line.index, columns=network.line.index)
    LODF = pd.DataFrame(LODF, index=network.line.index, columns=network.line.index)
    return p_f_shift_df, LODF
