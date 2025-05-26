from optimizers.gurobi_SCOS import *
from config.config import *
from config.config import ConfigData as CONF
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
add_variables_msg = CONF.add_variables_msg

def add_variables(model: Model, data: dict, optim_type: str = 'deterministic'):
    """Add scheduling variables to the optimization model."""
    names = data['names']
    T = data['T']

    xt = model.addVars(T, names['outages'], vtype=GRB.BINARY, name="planned_outage_indicator")
    sxt = model.addVars(names['outages'], lb=0, name="start_time")

    pgen = model.addVars(T, names['generators'], lb=0, name="power_generation")
    d_wc = model.addVars(T, names['buses'], lb=0, name="loss_of_load")
    f = model.addVars(T, names['lines'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl")

    f_c = model.addVars(T, names['lines'], names['contingencies'], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl_contingency")
    d_wc_c = model.addVars(T, names['buses'], names['contingencies'], lb=0, name="loss_of_load_contingency")
    pgen_c = model.addVars(T, names['generators'], names['contingencies'], lb=0, name="power_generation_contingency")

    VARIABLES = {
        'xt': xt, 'sxt': sxt,
        'pgen': pgen, 'pgen_c': pgen_c,
        'd_wc': d_wc, 'd_wc_c': d_wc_c,
        'f': f, 'f_c': f_c
    }

    logger.info(add_variables_msg)

    if optim_type == 'deterministic':
        return model, VARIABLES

    elif optim_type == 'robust':
        VARIABLES['z'] = model.addVar(lb=-GRB.INFINITY, name="robust_obj")
        return model, VARIABLES

    elif optim_type == 'risk_constrained':
        S = names['demand_scenarios']
        VARIABLES.update({
            'd_wc_scenario': model.addVars(T, names['buses'], S, lb=0, name="Loss_of_bus_load_scenarios"),
            'excess': model.addVars(T, names['buses'], S, lb=0, name="Excess_over_VaR"),
            'VaR': model.addVars(T, names['buses'], lb=-float('inf'), name="Value_at_Risk"),
            'CVaR': model.addVars(T, names['buses'], lb=0, name="CVaR")
        })
        return model, VARIABLES

    else:
        raise ValueError(f"Unknown optimization type: {optim_type}")

