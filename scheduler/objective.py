from optimizers.gurobi_SCOS import *
from config.config import *
from IPython.display import display, HTML
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

html_latex_expr = HTML(r""" <div style=\"color:orange;\"> 
                            \[ \max_X \Bigl( \sum_o C_{PM,o}(X) - \omega_1 \sum_t\sum_b C_{VoLL,t,b}(X) \\
                                - \omega_2 \sum_t\sum_b\sum_C C_{VoLL,t,b}(X|C)    
                                 - \omega_3 \sum_t\sum_b C_{OP,b,t}(X)    
                                  - \omega_4 \sum_t\sum_b\sum_C C_{OP,b,t}(X|C) \Bigr) \] </div> """)


def objective_function(data, variables):
    # Unpack variables
    xt = variables['xt']
    pgen = variables['pgen']
    pgen_c = variables['pgen_c']
    d_wc = variables['d_wc']
    d_wc_c = variables['d_wc_c']

    T = data['T']
    outages = data['names']['outages']
    buses = data['names']['buses']
    generators = data['names']['generators']
    contingencies = data['names']['contingencies']
    priority = data['priority']

    # scaling factors
    nt = len(data['T'])
    weight1, weight2 = 1/(nt * len(buses)),  1/(nt * len(buses) * len(contingencies))

    # 1) PM priority score
    PM_priority = quicksum((nt - t_idx) / nt * xt[t, o] * priority[o] for t_idx, t in enumerate(T) for o in outages)

    # 2) Loss of load penalty proportional to VoLL
    VOLL_cost = data['VoLL'] * quicksum(d_wc[t, n] for t in T for n in buses) * weight1
    VOLL_cost_contingency = data['VoLL'] * quicksum(
        d_wc_c[t, n, c] for t in T for n in buses for c in contingencies) * weight2

    # 3) Operational costs
    OP_cost = quicksum(pgen[t, g] for t in T for g in generators) * weight1
    OP_cost_contingency = quicksum(pgen_c[t, g, c] for t in T for g in generators for c in contingencies) * weight2

    #  display LaTeX
    display(html_latex_expr)

    # Collect objectives
    Objectives = {'PM_priority': PM_priority,
                  'VOLL_cost': VOLL_cost,
                  'VOLL_cost_contingency': VOLL_cost_contingency,
                  'OP_cost': OP_cost,
                  'OP_cost_contingency': OP_cost_contingency,
                  'Total': (PM_priority - (VOLL_cost + VOLL_cost_contingency) - (OP_cost + OP_cost_contingency))}

    return Objectives
