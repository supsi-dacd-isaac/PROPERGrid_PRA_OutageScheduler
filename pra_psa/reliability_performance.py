import numpy as np

thresholds_map = {'high_loading_percent': 98.0,   'low_vm_pu': .92, 'high_vm_pu': 1.08}


def g_fun_loading(loading_percent, upper_threshold=thresholds_map['high_loading_percent']):
    """  reliability performance functions for line loading
    # g_k: reliability performance functions (components)
    # w = max_k(g_k): worst-case reliability performance function (system)"""
    # g_k(x)>0 = failure state for component k
    # g_k(x)=0 failure boundary for component k
    # g_k(x)< 0 Save state for component k
    # w(x)>0 = system failure (at least one component fails to meet the requirement)
    # g = np.minimum(loading_percent, 150) - upper_threshold
    g = loading_percent - upper_threshold
    w = max(g)
    comp_failure = g >= 0
    sys_failure = w >= 0
    return g, w, comp_failure, sys_failure


def g_fun_low_voltage_magnitude(voltage_magnitudes_pu, lower_thresholds=thresholds_map['low_vm_pu']):
    """  reliability performance functions for under voltage

    - g_k: reliability performance fun (components); w = max_k(g_k): worst-case reliability performance fun (system)

    """
    # g_k(x)>0 = failure state for component k;
    # g_k(x)=0 failure boundary for component k;
    # g_k(x)< 0 Save state for component k
    # w(x)>0 = system failure (at least one component fails to meet the requirement)
    if lower_thresholds is not None and len(lower_thresholds) == len(voltage_magnitudes_pu):
        g = [th - v_pu for th, v_pu in zip(lower_thresholds, voltage_magnitudes_pu)]
    else:
        g = thresholds_map['low_vm_pu'] - voltage_magnitudes_pu
    w = max(g)
    comp_failure = g >= 0
    sys_failure = w >= 0
    return g, w, comp_failure, sys_failure


def g_fun_high_voltage_magnitude(voltage_magnitudes_pu, upper_thresholds=thresholds_map['high_vm_pu']):
    """  reliability performance functions for over voltage     """
    if upper_thresholds is not None and len(upper_thresholds) == len(voltage_magnitudes_pu):
        g = [v_pu - th for th, v_pu in zip(upper_thresholds, voltage_magnitudes_pu)]
    else:
        g = voltage_magnitudes_pu - thresholds_map['high_vm_pu']

    w = max(g)
    comp_failure = g >= 0
    sys_failure = w >= 0
    return g, w, comp_failure, sys_failure


