import pytest
import numpy as np
from pra_psa.reliability_performance import (
    g_fun_loading,
    g_fun_low_voltage_magnitude,
    g_fun_high_voltage_magnitude,
    thresholds_map
)

def test_g_fun_loading():
    # Test case 1: All loads below threshold
    loading_percent = np.array([80, 85, 90])
    g, w, comp_failure, sys_failure = g_fun_loading(loading_percent)
    assert np.all(g < 0)  # All components should be safe
    assert w < 0  # System should be safe
    assert not np.any(comp_failure)  # No component failures
    assert not sys_failure  # No system failure

    # Test case 2: Some loads above threshold
    loading_percent = np.array([80, 99, 90])
    g, w, comp_failure, sys_failure = g_fun_loading(loading_percent)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state
    assert np.any(comp_failure)  # Some component failures
    assert sys_failure  # System failure

    # Test case 3: Custom threshold
    loading_percent = np.array([80, 85, 90])
    custom_threshold = 85
    g, w, comp_failure, sys_failure = g_fun_loading(loading_percent, upper_threshold=custom_threshold)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state

def test_g_fun_low_voltage_magnitude():
    # Test case 1: All voltages above lower threshold
    voltage_magnitudes = np.array([0.95, 0.96, 0.97])
    g, w, comp_failure, sys_failure = g_fun_low_voltage_magnitude(voltage_magnitudes)
    assert np.all(g < 0)  # All components should be safe
    assert w < 0  # System should be safe
    assert not np.any(comp_failure)  # No component failures
    assert not sys_failure  # No system failure

    # Test case 2: Some voltages below lower threshold
    voltage_magnitudes = np.array([0.95, 0.91, 0.97])
    g, w, comp_failure, sys_failure = g_fun_low_voltage_magnitude(voltage_magnitudes)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state
    assert np.any(comp_failure)  # Some component failures
    assert sys_failure  # System failure

    # Test case 3: Custom thresholds
    voltage_magnitudes = np.array([0.95, 0.96, 0.97])
    custom_thresholds = np.array([0.96, 0.96, 0.96])
    g, w, comp_failure, sys_failure = g_fun_low_voltage_magnitude(voltage_magnitudes, lower_thresholds=custom_thresholds)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state

def test_g_fun_high_voltage_magnitude():
    # Test case 1: All voltages below upper threshold
    voltage_magnitudes = np.array([1.05, 1.06, 1.07])
    g, w, comp_failure, sys_failure = g_fun_high_voltage_magnitude(voltage_magnitudes)
    assert np.all(g < 0)  # All components should be safe
    assert w < 0  # System should be safe
    assert not np.any(comp_failure)  # No component failures
    assert not sys_failure  # No system failure

    # Test case 2: Some voltages above upper threshold
    voltage_magnitudes = np.array([1.05, 1.09, 1.07])
    g, w, comp_failure, sys_failure = g_fun_high_voltage_magnitude(voltage_magnitudes)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state
    assert np.any(comp_failure)  # Some component failures
    assert sys_failure  # System failure

    # Test case 3: Custom thresholds
    voltage_magnitudes = np.array([1.05, 1.06, 1.07])
    custom_thresholds = np.array([1.06, 1.06, 1.06])
    g, w, comp_failure, sys_failure = g_fun_high_voltage_magnitude(voltage_magnitudes, upper_thresholds=custom_thresholds)
    assert np.any(g >= 0)  # Some components should be in failure state
    assert w >= 0  # System should be in failure state 