import pytest
import pandapower as pp
import numpy as np
import pandas as pd
from pra_psa.contingency_analysis import (
    get_OPF_gen,
    get_PF_loading,
    apply_reference_dispatch,
    apply_load,
    apply_nk_contingency,
    compute_lodf_matrix,
    compute_ptdf_matrix,
    extract_branch_flows_and_loading,
    calculate_lodf_and_shift,
    compute_line_flows_from_sensitivity,
    compute_sensitivity_matrix_and_base_flows
)

@pytest.fixture
def simple_network():
    """Create a simple test network"""
    net = pp.create_empty_network()
    
    # Create buses
    bus1 = pp.create_bus(net, vn_kv=110., name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=110., name="Bus 2")
    
    # Create external grid
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, va_degree=50)
    
    # Create generator
    pp.create_gen(net, bus=bus2, p_mw=2., vm_pu=1.01, min_p_mw=0., max_p_mw=4.)
    
    # Create load
    pp.create_load(net, bus=bus2, p_mw=2., q_mvar=4., name="load1")
    
    # Create line
    pp.create_line_from_parameters(net, from_bus=bus1, to_bus=bus2,
                                 length_km=20., r_ohm_per_km=0.876,
                                 x_ohm_per_km=0.497, c_nf_per_km=9.0,
                                 max_i_ka=0.963)
    return net

def test_get_OPF_gen(simple_network):
    p_mw, q_mvar, net = get_OPF_gen(simple_network)
    assert isinstance(p_mw, np.ndarray)
    assert isinstance(q_mvar, np.ndarray)
    assert len(p_mw) == len(simple_network.gen)
    assert len(q_mvar) == len(simple_network.gen)

def test_get_PF_loading(simple_network):
    flows, loading, net = get_PF_loading(simple_network)
    assert isinstance(flows, np.ndarray)
    assert isinstance(loading, np.ndarray)
    assert len(flows) == len(simple_network.line) + len(simple_network.trafo)
    assert len(loading) == len(simple_network.line) + len(simple_network.trafo)

def test_apply_reference_dispatch(simple_network):
    # First run OPF to get reference values
    p_mw, q_mvar, _ = get_OPF_gen(simple_network)
    
    # Apply reference dispatch
    net = apply_reference_dispatch(simple_network, p_mw, q_mvar)
    
    # Check if generator values were set correctly
    assert np.allclose(net.gen.p_mw.values, p_mw)
    assert np.allclose(net.gen.sn_mva.values, q_mvar)

def test_apply_load(simple_network):
    new_load = np.array([3.0])  # New load value
    net = apply_load(simple_network, new_load)
    assert np.allclose(net.load.p_mw.values, new_load)

def test_apply_nk_contingency(simple_network):
    # Test line contingency
    failure_event = {"element_type": "line", "element_index": 0}
    cont_net = apply_nk_contingency(simple_network, failure_event)
    assert not cont_net.line.in_service.iloc[0]
    
    # Test generator contingency
    failure_event = {"element_type": "generator", "element_index": 0}
    cont_net = apply_nk_contingency(simple_network, failure_event)
    assert not cont_net.gen.in_service.iloc[0]
    
    # Test multiple contingencies
    failure_events = [
        {"element_type": "line", "element_index": 0},
        {"element_type": "generator", "element_index": 0}
    ]
    cont_net = apply_nk_contingency(simple_network, failure_events)
    assert not cont_net.line.in_service.iloc[0]
    assert not cont_net.gen.in_service.iloc[0]
    
    # Test invalid element type
    with pytest.raises(ValueError):
        apply_nk_contingency(simple_network, {"element_type": "invalid", "element_index": 0})
    
    # Test invalid element index
    with pytest.raises(KeyError):
        apply_nk_contingency(simple_network, {"element_type": "line", "element_index": 999})

def test_compute_lodf_matrix(simple_network):
    lodf = compute_lodf_matrix(simple_network)
    assert isinstance(lodf, np.ndarray)
    assert lodf.shape == (len(simple_network.line), len(simple_network.line))

def test_compute_ptdf_matrix(simple_network):
    ptdf = compute_ptdf_matrix(simple_network)
    assert isinstance(ptdf, np.ndarray)
    # PTDF shape should be (number of lines, number of buses)
    assert ptdf.shape == (len(simple_network.line), len(simple_network.bus))

def test_extract_branch_flows_and_loading(simple_network):
    # First run power flow
    pp.rundcpp(simple_network)
    
    flows, loading = extract_branch_flows_and_loading(simple_network)
    assert isinstance(flows, np.ndarray)
    assert isinstance(loading, np.ndarray)
    assert len(flows) == len(simple_network.line) + len(simple_network.trafo)
    assert len(loading) == len(simple_network.line) + len(simple_network.trafo)
    
    # Test error when power flow hasn't been run
    net = pp.create_empty_network()
    with pytest.raises(RuntimeError):
        extract_branch_flows_and_loading(net)

def test_calculate_lodf_and_shift(simple_network):
    p_f_shift_df, LODF = calculate_lodf_and_shift(simple_network)
    assert isinstance(p_f_shift_df, pd.DataFrame)
    assert isinstance(LODF, pd.DataFrame)
    assert p_f_shift_df.shape == (len(simple_network.line), len(simple_network.line))
    assert LODF.shape == (len(simple_network.line), len(simple_network.line))

def test_compute_line_flows_from_sensitivity(simple_network):
    # First compute sensitivity matrix and base flows
    sensitivity_matrix, base_line_flows, net, _, _ = compute_sensitivity_matrix_and_base_flows(simple_network)
    
    # Test with new loads
    new_loads = np.array([2.5])  # Slightly higher load
    loading, net = compute_line_flows_from_sensitivity(net, sensitivity_matrix, base_line_flows, new_loads)
    
    assert isinstance(loading, np.ndarray)
    assert len(loading) == len(simple_network.line)

def test_compute_sensitivity_matrix_and_base_flows(simple_network):
    sensitivity_matrix, base_line_flows, net, ref_p_mw, ref_q_mvar = compute_sensitivity_matrix_and_base_flows(simple_network)
    
    assert isinstance(sensitivity_matrix, np.ndarray)
    assert isinstance(base_line_flows, np.ndarray)
    assert isinstance(ref_p_mw, np.ndarray)
    assert isinstance(ref_q_mvar, np.ndarray)
    
    # Check shapes
    assert sensitivity_matrix.shape == (len(simple_network.line), len(simple_network.bus))
    assert len(base_line_flows) == len(simple_network.line)
    assert len(ref_p_mw) == len(simple_network.gen)
    assert len(ref_q_mvar) == len(simple_network.gen) 