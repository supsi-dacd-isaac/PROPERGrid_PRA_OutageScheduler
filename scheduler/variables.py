
from optimizers.gurobi_SCOS import *
from config.logger_config import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def add_variables(model: Model, data:dict, optim_type: str = 'deterministic'):

    """ Add scheduling variables """
    O = data['names']['outages']
    G = data['names']['generators']
    L = data['names']['lines']
    B = data['names']['buses']
    C = data['names']['contingencies']
    T = data['T']

    xt       = model.addVars(T, O, vtype=GRB.BINARY, name="planned_outage_indicator")
    sxt      = model.addVars(T, O, vtype=GRB.BINARY, name="start_outage_indicator")
    ext      = model.addVars(T, O, vtype=GRB.BINARY, name="end_outage_indicator")

    #xt_is_on = model.addVars(T, O, vtype=GRB.BINARY, name="planned_outage_indicator_active")
    #is_o     = model.addVars(O, vtype=GRB.BINARY, name="is_pm_scheduled")  # define variables
    #VARIABLES['is_o'] = is_o
    #VARIABLES['xt_is_on'] = xt_is_on

    pgen = model.addVars(T, G, lb=0, name="power_generation")
    d_wc = model.addVars(T, B, lb=0, name="loss_of_load")
    f    = model.addVars(T, L, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl")

    f_c     = model.addVars(T, L, C, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flow_tl_contingency")
    d_wc_c  = model.addVars(T, B, C, lb=0, name="loss_of_load_contingency")
    pgen_c  = model.addVars(T, G, C, lb=0, name="power_generation_contingency")

    VARIABLES = {}
    VARIABLES['xt'] = xt

    VARIABLES['sxt'] = sxt
    VARIABLES['ext'] = ext
    VARIABLES['pgen'] = pgen
    VARIABLES['pgen_c'] = pgen_c
    VARIABLES['d_wc'] = d_wc
    VARIABLES['d_wc_c'] = d_wc_c
    VARIABLES['f'] = f
    VARIABLES['f_c'] = f_c

    logger.info(f' \n'
                f'{blue_c}Variables Added:{reset_c}\n'
                f' - {blue_c}xt, sxt, ext{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                f' - {blue_c}pgen, pgen_c{reset_c}: Generated power planned and N-1 states\n'
                f' - {blue_c}d_cut, d_cut_c{reset_c}: loss of load planned and N-1 states\n'
                f' - {blue_c}f, f_c{reset_c}: Power flow in planned and N-1 states\n')

    if optim_type == 'deterministic':
        return model, VARIABLES

    elif optim_type == 'robust':
        # Add robust variables here if needed z = worst-case cost of ephi scenario formulation
        z = model.addVar(lb=-GRB.INFINITY, name="robust_obj")
        VARIABLES['z'] = z
        return model, VARIABLES

    elif optim_type == 'risk_constrained':
        # Add variables for cvar computation
        S = data['names']['demand_scenarios']
        d_wc_scenario = model.addVars(T, B, S, lb=0, name="Loss_of_bus_load_scenarios")
        excess = model.addVars(T, B, S, lb=0, name="Excess_over_VaR")
        VaR = model.addVars(T, B, lb = -float('inf'), name="Value_at_Risk")  # Unbounded VaR
        CVaR = model.addVars(T, B, lb=0, name="CVaR")
        VARIABLES['d_wc_scenario'] = d_wc_scenario
        VARIABLES['excess'] = excess
        VARIABLES['VaR'] = VaR
        VARIABLES['CVaR'] = CVaR
        return model, VARIABLES


