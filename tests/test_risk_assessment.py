import pytest
import pandapower as pp
import numpy as np
from pra_psa.risk_assessment import (
    NaiveProbLoadModel,
    NaiveProbFailureModel,
    runPRA_LODF,
    runPRA
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

def test_naive_prob_load_model():
    model = NaiveProbLoadModel()
    
    # Test sampling
    mean = 100
    std = 10
    n_samples = 1000
    samples = model.sample(mean, std, n_samples)
    
    assert len(samples) == n_samples
    assert all(model.min_L <= s <= model.max_L for s in samples)
    
    # Test that samples are normally distributed around mean
    sample_mean = np.mean(samples)
    assert abs(sample_mean - mean) < std  # Mean should be within one std of target
    
    # Test edge cases
    # Very small mean and std
    samples = model.sample(mean=0.1, std=0.01, n_sam=100)
    assert all(model.min_L <= s <= model.max_L for s in samples)
    
    # Very large mean and std
    samples = model.sample(mean=1000, std=100, n_sam=100)
    assert all(model.min_L <= s <= model.max_L for s in samples)
    
    # Zero std
    samples = model.sample(mean=100, std=0, n_sam=100)
    assert all(s == 100 for s in samples)

def test_naive_prob_failure_model():
    model = NaiveProbFailureModel()
    
    # Test single failure
    failure_set = [{"element_index": 0, "element_type": "line"}]
    probs = model.get_probabilities(failure_set)
    
    assert len(probs) == 2  # Normal state + one failure
    assert abs(sum(probs) - 1.0) < 1e-10  # Probabilities should sum to 1
    
    # Test multiple failures
    failure_set = [
        {"element_index": 0, "element_type": "line"},
        {"element_index": 1, "element_type": "line"}
    ]
    probs = model.get_probabilities(failure_set)
    
    assert len(probs) == 3  # Normal state + two failures
    assert abs(sum(probs) - 1.0) < 1e-10  # Probabilities should sum to 1
    
    # Test empty failure set
    failure_set = []
    probs = model.get_probabilities(failure_set)
    assert len(probs) == 1  # Only normal state
    assert abs(probs[0] - 1.0) < 1e-10  # Normal state probability should be 1

def test_runPRA_LODF(simple_network):
    # Create load time series
    load_time_series = [np.array([2.0])]  # Single time step
    
    # Run PRA with LODF
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA_LODF(
        simple_network,
        load_time_series
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)
    
    # Check shapes
    assert len(Mean_System_Risk_t) == len(load_time_series)
    assert Mean_Risk_c_id_t.shape[0] == len(load_time_series)
    assert len(Pf_cont_t) == len(load_time_series)
    assert len(worst_case_severity) == len(load_time_series)
    
    # Test with custom probability models
    custom_load_model = NaiveProbLoadModel()
    custom_failure_model = NaiveProbFailureModel()
    
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA_LODF(
        simple_network,
        load_time_series,
        prob_load_model=custom_load_model,
        prob_cont_model=custom_failure_model
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)
    
    # Test with N-k contingency set
    n_minus_k_set = [{"element_index": 0, "element_type": "line"}]
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA_LODF(
        simple_network,
        load_time_series,
        n_minus_k_set=n_minus_k_set
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)

def test_runPRA(simple_network):
    # Create load time series
    load_time_series = [np.array([2.0])]  # Single time step
    
    # Run PRA
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA(
        simple_network,
        load_time_series=load_time_series
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)
    
    # Check shapes
    assert len(Mean_System_Risk_t) == len(load_time_series)
    assert Mean_Risk_c_id_t.shape[0] == len(load_time_series)
    assert len(Pf_cont_t) == len(load_time_series)
    assert len(worst_case_severity) == len(load_time_series)
    
    # Test with custom probability models
    custom_load_model = NaiveProbLoadModel()
    custom_failure_model = NaiveProbFailureModel()
    
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA(
        simple_network,
        load_time_series=load_time_series,
        prob_load_model=custom_load_model,
        prob_cont_model=custom_failure_model
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)
    
    # Test with N-k contingency set
    n_minus_k_set = [{"element_index": 0, "element_type": "line"}]
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA(
        simple_network,
        load_time_series=load_time_series,
        n_minus_k_set=n_minus_k_set
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray)
    
    # Test with different solvers
    Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity = runPRA(
        simple_network,
        load_time_series=load_time_series,
        pf_solver=pp.rundcpp,
        opf_solver=pp.rundcopp
    )
    
    assert isinstance(Mean_System_Risk_t, np.ndarray)
    assert isinstance(Mean_Risk_c_id_t, np.ndarray)
    assert isinstance(Pf_cont_t, np.ndarray)
    assert isinstance(worst_case_severity, np.ndarray) 