import numpy as np


def max_line_loading(flows, ratings):
    return np.max(np.abs(flows) / ratings)


def count_violations(flows, ratings):
    return np.sum(np.abs(flows) > ratings)


def voltage_violation_count(voltages, vmin=0.95, vmax=1.05):
    return np.sum((voltages < vmin) | (voltages > vmax))


def line_overload_mask(flows, ratings):
    return np.abs(flows) > ratings
