from gurobipy import Model, quicksum, tuplelist, GRB, LinExpr
from config.config import *
import time
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_all_generator_constraints(model: Model, data: dict, variables: dict) -> Model:
    """Add generation bounds and contingency constraints."""
    logger.info('Adding power production constraints (normal + N-1 failures)')
    start = time.time()

    T = data['T']
    names = data['names']
    gens = names['generators']
    contingencies = names['contingencies']
    outages_set = set(names['outages'])
    p_max = data['p_max']
    p_min = data['p_min']

    xt = variables['xt']
    pgen = variables['pgen']
    pgen_c = variables['pgen_c']

    # Precompute outage factors
    logger.info(f"Precomputing factors for {len(T) * len(gens)} generation constraints")
    factors = {(t, g): xt[t, g] if g in outages_set else 0.0 for t in T for g in gens}

    # Normal operation bounds
    model.addConstrs((pgen[t, g] <= p_max[g] * (1 - factors[t, g]) for t in T for g in gens), name="GenUp")
    model.addConstrs((pgen[t, g] >= p_min[g] * (1 - factors[t, g]) for t in T for g in gens), name="GenLow")

    # Contingency constraints
    zero_c = []
    bound_c = []

    for t in T:
        for g in gens:
            for c in contingencies:
                if c[3:] == g:  # failed generator in contingency
                    zero_c.append((t, g, c))
                else:
                    bound_c.append((t, g, c))

    model.addConstrs((pgen_c[t, g, c] == 0 for t, g, c in zero_c), name="GenNull")
    model.addConstrs((pgen_c[t, g, c] <= p_max[g] * (1 - factors[t, g]) for t, g, c in bound_c), name="GenCUp")
    model.addConstrs((pgen_c[t, g, c] >= p_min[g] * (1 - factors[t, g]) for t, g, c in bound_c), name="GenCLow")

    logger.info(f"Done - build time: {time.time() - start:.3f}s")
    return model


def add_planned_outages_constraints(model: Model, data: dict, variables: dict) -> Model:
    """Add planned outage constraints using integer start times for efficient formulation."""
    logger.info('Adding constraints for scheduled outages: max tasks, duration, and activation logic')
    start_time = time.time()

    T = data['T']
    out = data['names']['outages']
    maxT = data['max_tasks']
    dur = data['durations']
    dur = {k: round(dur[k]) for k in dur}  # integer

    # Get variables
    xt = variables['xt']
    start_vars = variables['sxt']

    # 1) No more than maxT simultaneous outages
    model.addConstrs((quicksum(xt[t, o] for o in out) <= maxT for t in T), name="MaxTasks")

    # 2) Each outage must be active for exactly its duration
    model.addConstrs((quicksum(xt[t, o] for t in T) == dur[o] for o in out), name="Duration")

    # 3) Link xt to start_time: xt[t, o] = 1 iff t in [start[o], start[o] + dur[o] - 1]
    bigM = 1e4  # big enough constant (max time index)

    # Map time labels to integer indices
    time_index = {t: i for i, t in enumerate(T)}

    for o in out:
        for t in T:
            ti = time_index[t]
            model.addConstr(ti - start_vars[o] + bigM * (1 - xt[t, o]) >= 0,
                name=f"start_lb_{t}_{o}" )
            model.addConstr(start_vars[o] + dur[o] - 1 - ti + bigM * (1 - xt[t, o]) >= 0,
                name=f"start_ub_{t}_{o}" )

    logger.info(f"Done - build time: {time.time() - start_time:.3f}s")
    return model


def add_line_power_limit_constraints(model: Model, data: dict, variables: dict) -> Model:
    logger.info('Adding line flow constraints, normal + N-1 failures')
    start = time.time()

    lines = data['names']['lines']
    contingencies = data['names']['contingencies']
    outages = set(data['names']['outages'])
    T = data['T']

    f = variables['f']
    f_c = variables['f_c']
    xt = variables['xt']  # binary outage variables: indexed by (t, outage)
    f_lim = data['f_lim']

    # Map each contingency to the outage line it affects
    cont_line = {c: next(l for l in lines if c.endswith(l)) for c in contingencies}

    # Build factor: for each (t,l), is line l out at time t?
    # Assume outages represent lines, so check if line l is in outages active at t
    factor = {}
    for t in T:
        for l in lines:
            factor[(t, l)] = quicksum(xt[t, o] for o in outages if o == l)

    # Base case flow limits
    model.addConstrs((f[t, l] <= f_lim[l] * (1 - factor[(t, l)]) for t in T for l in lines), name="FlowBase_UP")
    model.addConstrs((f[t, l] >= -f_lim[l] * (1 - factor[(t, l)]) for t in T for l in lines), name="FlowBase_LOW")

    # Contingency case: zero flow on failed line; normal bounds otherwise
    for t in T:
        for l in lines:
            for c in contingencies:
                if cont_line[c] == l:
                    model.addConstr(f_c[t, l, c] == 0, name=f"FlowC_Zero_{t}_{l}_{c}")
                else:
                    model.addConstr(f_c[t, l, c] <= f_lim[l] * (1 - factor[(t, l)]), name=f"FlowC_UP_{t}_{l}_{c}")
                    model.addConstr(f_c[t, l, c] >= -f_lim[l] * (1 - factor[(t, l)]), name=f"FlowC_LOW_{t}_{l}_{c}")

    logger.info(f"Done - build time: {time.time() - start:.3f}s")
    return model


def add_nodal_power_balance_constraints(model: Model, data: dict, variables: dict) -> Model:
    logger.info('Adding node balance constraints, normal + N-1 failures')
    start = time.time()

    names, S_T, demand, g2bus, T = data['names'], data['S'], data['nodal_demand'], data['g2bus'], data['T']
    pgen, pgen_c, f, f_c, d_wc, d_wc_c = (
        variables['pgen'], variables['pgen_c'],
        variables['f'], variables['f_c'],
        variables['d_wc'], variables['d_wc_c']
    )
    buses, contingencies = names['buses'], names['contingencies']

    gen_to_node = {b: names['generators'][i] for i, b in enumerate(g2bus)}

    base_bal = calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand)
    cont_bal = {c: calculate_nodal_balance(T, f_c, names, S_T, pgen_c, gen_to_node, demand, c=c) for c in contingencies}

    for (t, b_idx), expr in base_bal.items():
        model.addConstr(expr <= d_wc[t, buses[b_idx]], name=f"PB_up_{t}_{buses[b_idx]}")
        model.addConstr(expr >= -d_wc[t, buses[b_idx]], name=f"PB_low_{t}_{buses[b_idx]}")

    for c in contingencies:
        mb = cont_bal[c]
        for (t, b_idx), expr in mb.items():
            model.addConstr(expr <= d_wc_c[t, buses[b_idx], c], name=f"PB_{c}_up_{t}_{buses[b_idx]}")
            model.addConstr(expr >= -d_wc_c[t, buses[b_idx], c], name=f"PB_{c}_low_{t}_{buses[b_idx]}")

    logger.info(f"Done - build time: {time.time() - start:.3f}s")
    return model


def calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand, c=None):
    nodal_balance = {}

    buses = names['buses']
    lines = names['lines']

    for t_idx, t in enumerate(T):
        for b_idx, bus in enumerate(buses):
            # Sum of flows for bus b at time t
            # S_T[b_idx, :] is a vector with +1/-1 for line incidence on bus
            expr = LinExpr()
            for l_idx, line in enumerate(lines):
                coeff = S_T[b_idx, l_idx]
                if coeff != 0:
                    if c is None:
                        expr.addTerms(coeff, f[t, line])
                    else:
                        expr.addTerms(coeff, f[t, line, c])

            # Add generation at bus b at time t
            if bus in gen_to_node:
                if c is None:
                    expr -= pgen[t, gen_to_node[bus]]
                else:
                    expr -= pgen[t, gen_to_node[bus], c]


            # Nodal balance = demand - (inflows - generation)
            nodal_balance[(t, b_idx)] = demand.values[t_idx, b_idx] - expr

    return nodal_balance