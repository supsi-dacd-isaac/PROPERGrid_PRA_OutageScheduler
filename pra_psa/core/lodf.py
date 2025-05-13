import pandapower as pp
import pandapower.networks as pn
import numpy as np
from pypower.makePTDF import makePTDF
from pypower.makeLODF import makeLODF
from pandapower.converter import to_ppc

def compute_ptdf(net):
    """
    Compute the Power Transfer Distribution Factor matrix.
    """
    ppc = to_ppc(net, calculate_voltage_angles=True)
    return makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'], ppc['gen'])


def compute_lodf(net):
    """
    Compute the Line Outage Distribution Factor matrix.
    """
    ppc = to_ppc(net, calculate_voltage_angles=True)
    ptdf = makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'], ppc['gen'])
    return makeLODF(ppc['bus'], ppc['branch'], ptdf)


def estimate_contingency_flows(base_flows, lodf_matrix, outage_line_index):
    """
    Estimate new line flows under a contingency using LODF.
    """
    f_outaged = base_flows[outage_line_index]
    delta = lodf_matrix[:, outage_line_index] * f_outaged
    new_flows = base_flows + delta
    new_flows[outage_line_index] = 0.0
    return new_flows

