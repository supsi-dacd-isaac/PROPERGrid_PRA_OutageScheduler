from optimizers.gurobi_SCOS import *
from config.config import ConfigData as CONF
from IPython.display import display, HTML
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
html_latex_expr = HTML(CONF.objective_html)


def objective_function(data, variables):
    """Define and return the objective components for the optimization problem."""
    # Unpack variables
    xt = variables['xt']
    pgen = variables['pgen']
    pgen_c = variables['pgen_c']
    d_wc = variables['d_wc']
    d_wc_c = variables['d_wc_c']

    # Unpack data
    T = data['T']
    names = data['names']
    outages = names['outages']
    buses = names['buses']
    generators = names['generators']
    contingencies = names['contingencies']
    priority = data['priority']

    # Scaling factors
    nt = len(T)
    weight1 = 1 / (nt * len(buses))
    weight2 = 1 / (nt * len(buses) * len(contingencies))

    # 1) PM priority score
    PM_priority = quicksum(
        ((nt - t_idx) / nt) * xt[t, o] * priority[o]
        for t_idx, t in enumerate(T) for o in outages
    )

    # 2) Loss of load penalties
    VOLL = data['VoLL']
    VOLL_cost = VOLL * quicksum(d_wc[t, n] for t in T for n in buses) * weight1
    VOLL_cost_contingency = VOLL * quicksum(d_wc_c[t, n, c] for t in T for n in buses for c in contingencies) * weight2

    # 3) Operational costs
    OP_cost = quicksum(pgen[t, g] for t in T for g in generators) * weight1
    OP_cost_contingency = quicksum(pgen_c[t, g, c] for t in T for g in generators for c in contingencies) * weight2

    # Display LaTeX expression
    display(html_latex_expr)

    # Return objective components
    return {
        'PM_priority': PM_priority,
        'VOLL_cost': VOLL_cost,
        'VOLL_cost_contingency': VOLL_cost_contingency,
        'OP_cost': OP_cost,
        'OP_cost_contingency': OP_cost_contingency,
        'Total': PM_priority - (VOLL_cost + VOLL_cost_contingency + OP_cost + OP_cost_contingency)
    }