import pytest
import pandapower as pp

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