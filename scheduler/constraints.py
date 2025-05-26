
from gurobipy import Model, quicksum, tuplelist
from config.config import *
import time
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def add_all_generator_constraints(model: Model, data: dict, variables: dict) -> Model:
    """
    Add generation bounds and contingency constraints in a single pass.
    Uses Gurobi's addConstrs for vectorized constraint creation.
    """

    logger.info('adding power production constraints, normal + N-1 failures')
    start = time.time()

    T               = data['T']
    gens            = data['names']['generators']
    contingencies   = data['names']['contingencies']
    outages_set     = set(data['names']['outages'])
    p_max           = data['p_max']
    p_min           = data['p_min']
    xt              = variables['xt']
    pgen            = variables['pgen']
    pgen_c          = variables['pgen_c']

    # Precompute factors for each (t,g) and normal generation bounds
    logger.info(f"precomputing factors for {len(T) * len(gens)} generation constraints")
    factors = {  (t, g): (xt[t, g] if g in outages_set else 0.0) for t in T for g in gens }
    model.addConstrs((pgen[t, g] <= p_max[g] * (1 - factors[t, g]) for t in T for g in gens) ,name="GenUp")
    model.addConstrs((pgen[t, g] >= p_min[g] * (1 - factors[t, g]) for t in T for g in gens), name="GenLow")

    # Contingency generation
    zero_c, bound_c_up, bound_c_low = [], [], []
    for t in T:
        for g in gens:
            for c in contingencies:
                if c[3:] == g:
                    zero_c.append((t, g, c))
                else:
                    bound_c_up.append((t, g, c))
                    bound_c_low.append((t, g, c))

    # Add zero-output constraints and Add bounds for non-failed contingencies
    model.addConstrs((pgen_c[t, g, c] == 0 for t, g, c in zero_c), name="GenNull")
    model.addConstrs((pgen_c[t, g, c] <= p_max[g] * (1 - factors[t, g]) for t, g, c in bound_c_up), name="GenCUp")
    model.addConstrs((pgen_c[t, g, c] >= p_min[g] * (1 - factors[t, g]) for t, g, c in bound_c_low), name="GenCLow")
    logger.info(f"Done - build time:   {time.time() - start:.3f}s")
    return model


def add_planned_outages_constraints(model: Model, data: dict, variables: dict) -> Model:
    logger.info('adding constraints for scheduled outages: duration, number of tasks, continuity')
    start = time.time()
    out, maxT, dur, T = data['names']['outages'], data['max_tasks'], data['durations'], data['T']
    xt, st, ed = variables['xt'], variables['sxt'], variables['ext']

    # 1) ≤ maxT simultaneous outages
    model.addConstrs((quicksum(xt[t, o_i]  for o_i in out) <= maxT for t in T), name="MaxTasks")
    # 2) exact total duration
    model.addConstrs((quicksum(xt[t, o_i]  for t in T) == dur[o_i]  for o_i in out), name="Duration")
    # 3) monotonicity & end-after-start
    model.addConstrs((st[T[ i +1] ,o_i] >= st[T[i] ,o_i] for o_i in out for i in range(len(T ) -1)), name="StartPM")
    model.addConstrs((ed[T[ i +1] ,o_i] >= ed[T[i] ,o_i] for i in range(len(T ) -1) for o_i in out), name="EndPM")
    model.addConstrs((ed[T[ i +1] ,o_i] <= st[T[i] ,o_i] for i in range(len(T ) -1) for o_i in out), name="EndAfterStart")
    # 4) link xt = st−ed
    model.addConstrs((st[t ,o_i ] -ed[t ,o_i] == xt[t ,o_i] for t in T for o_i in out), name="LinkPM")
    logger.info(f"Done - build time:   {time.time() - start:.3f}s")
    return model


def add_line_power_limit_constraints(model: Model, data: dict, variables: dict) -> Model:
    logger.info('adding line flow constraints, normal + N-1 failures')
    start = time.time()
    lines, conts, outages, T = data['names']['lines'], data['names']['contingencies'], set(data['names']['outages']), \
    data['T']
    f, f_c, xt, f_lim = variables['f'], variables['f_c'], variables['xt'], data['f_lim']
    # xt_is_on , is_o= variables['xt_is_on'] , variables['is_o']

    factor = {(t, l): xt[t, l] if l in outages else 0 for t in T for l in lines}
    cont_line = {c: next(l for l in lines if c.endswith(l)) for c in conts}
    model.addConstrs((f[t, l] <= f_lim[l] * (1 - factor[t, l]) for t in T for l in lines), name="FlowBase_UP")
    model.addConstrs((f[t, l] >= -f_lim[l] * (1 - factor[t, l]) for t in T for l in lines), name="FlowBase_LOW")
    FlowC_UP  = (f_c[t, l, c] == 0 if cont_line[c] == l else f_c[t, l, c] <= f_lim[l] * (1 - factor[t, l])  for t in T for l in lines for c in conts)
    FlowC_LOW = (f_c[t, l, c] == 0 if cont_line[c] == l else f_c[t, l, c] >= -f_lim[l] * (1 - factor[t, l]) for t in T for l in
         lines for c in conts)

    model.addConstrs(FlowC_UP, name="FlowC_UP")
    model.addConstrs(FlowC_LOW, name="FlowC_LOW")
    logger.info(f"Done - build time:   {time.time() - start:.3f}s")
    return model


def add_nodal_power_balance_constraints(model: Model, data: dict, variables: dict) -> Model:

    logger.info('adding node balance constraints, normal + N-1 failures')
    start = time.time()
    names, S_T, demand, g2bus, T = data['names'], data['S'], data['nodal_demand'], data['g2bus'], data['T']
    pgen, pgen_c, f, f_c , d_wc, d_wc_c = (variables['pgen'], variables['pgen_c'],
                                           variables['f'], variables['f_c'], variables['d_wc'], variables['d_wc_c'])
    buses , contraints   = names['buses'], names['contingencies']

    # 1) precompute full balance matrices
    base_bal = calculate_nodal_balance(T, f, names, S_T, pgen,   {b: names['generators'][i] for i ,b in enumerate(g2bus)}, demand)
    cont_bal = \
        {c: calculate_nodal_balance(T, f_c, names, S_T, pgen_c, {b: names['generators'][i] for i, b in enumerate(g2bus)},
                                   demand, c=c) for c in contraints}

    # 2) build index ranges once
    T_idx = range(len(T))
    B_idx = range(len(buses))

    # 3) vectorized adds for base case
    model.addConstrs((base_bal[t, b] <= d_wc[T[t], buses[b]] for t in T_idx for b in B_idx), name="PB_up")
    model.addConstrs((base_bal[t, b] >= -d_wc[T[t], buses[b]] for t in T_idx for b in B_idx), name="PB_low")

    # 4) vectorized adds for each contingency
    for c in contraints:
        mb = cont_bal[c]
        model.addConstrs((mb[t, b] <= d_wc_c[T[t], buses[b], c] for t in T_idx for b in B_idx), name=f"PB_{c}_up")
        model.addConstrs((mb[t, b] >= -d_wc_c[T[t], buses[b], c] for t in T_idx for b in B_idx), name=f"PB_{c}_low")
    logger.info(f"Done - build time:   {time.time() - start:.3f}s")
    return model


def calculate_nodal_balance(T, f, names, S_T, pgen, gen_to_node, demand, c = None):
    """
    Calculate inflows, generation, and nodal balance for a given set of parameters.

    Parameters:
        T (list): Time steps.
        f (ndarray): Flow data.
        names (dict): Dictionary containing 'lines' and 'buses'.
        S_T (ndarray): Connectivity matrix for buses and lines.
        pgen (ndarray): Generation data.
        gen_to_node (dict): Mapping of generators to bus nodes.
        demand (DataFrame): Demand values.

    Returns:
        nodal_balance (ndarray): The calculated nodal balance.
    """
    inflows_mat = []
    generation_all = []

    for t in T:
        if c is None:
            flows_t = np.array([f[t, l] for l in names['lines']])  # Flow values for current time step
        else:
            flows_t = np.array([f[t, l, c] for l in names['lines']])  # Flow values for current time step

        # Calculate inflows at each bus
        flows_con_t = [
            np.sum(S_T[b_idx, :][np.argwhere(S_T[b_idx, :]).flatten()] *
                   flows_t[np.argwhere(S_T[b_idx, :]).flatten()])
            for b_idx in range(len(names['buses']))
        ]

        # Calculate generation at each bus
        if c is None:
            generators_t = [pgen[t, gen_to_node[b]] if b in gen_to_node else 0 for b in names['buses']]
        else:
            generators_t = [pgen[t, gen_to_node[b], c] if b in gen_to_node else 0 for b in names['buses']]

        inflows_mat.append(flows_con_t)
        generation_all.append(generators_t)

    # Convert lists to NumPy arrays and remove any extra dimensions
    inflows_mat = np.squeeze(np.array(inflows_mat))
    generation_all = np.squeeze(np.array(generation_all))

    # Calculate nodal balance
    nodal_balance = demand.values - inflows_mat - generation_all

    return nodal_balance


def add_nodal_power_balance_constraints_sparse(model: Model, data: dict, variables: dict) -> Model:
    logger.info('adding node balance constraints, normal + N-1 failures')
    start = time.time()

    # unpack
    names, S_T, demand, g2bus, T = (
        data['names'], data['S'], data['nodal_demand'], data['g2bus'], data['T']
    )
    pgen, pgen_c = variables['pgen'], variables['pgen_c']
    f, f_c = variables['f'], variables['f_c']
    d_wc, d_wc_c = variables['d_wc'], variables['d_wc_c']
    buses, contingencies = names['buses'], names['contingencies']

    # precompute balance exprs
    base_bal = calculate_nodal_balance(
        T, f, names, S_T, pgen,
        {b: names['generators'][i] for i, b in enumerate(g2bus)},
        demand
    )
    cont_bal = {
        c: calculate_nodal_balance(
            T, f_c, names, S_T, pgen_c,
            {b: names['generators'][i] for i, b in enumerate(g2bus)},
            demand, c=c
        )
        for c in contingencies
    }

    # map every model var to a unique column index
    all_vars = model.getVars()
    varcol = {v: i for i, v in enumerate(all_vars)}

    # containers for the sparse‐matrix data
    rows, cols, vals = [], [], []
    senses, rhs = [], []
    row = 0

    # helper to extract a LinExpr into COO lists
    def push_expr(expr, sense, right):
        nonlocal row
        for i in range(expr.size()):
            v = expr.getVar(i)
            c = expr.getCoeff(i)
            rows.append(row)
            cols.append(varcol[v])
            vals.append(c)
        senses.append(sense)
        rhs.append(right)
        row += 1

    T_idx = range(len(T))
    B_idx = range(len(buses))

    # 1) base‐case up & low
    for t in T_idx:
        for b in B_idx:
            # ≤ case: base_bal - demand ≤ 0
            expr_up = base_bal[t, b] - d_wc[T[t], buses[b]]
            push_expr(expr_up, '<=', 0)

            # ≥ case: -base_bal - demand ≤ 0
            expr_low = -base_bal[t, b] - d_wc[T[t], buses[b]]
            push_expr(expr_low, '<=', 0)

    # 2) each contingency
    for c in contingencies:
        mb = cont_bal[c]
        for t in T_idx:
            for b in B_idx:
                expr_up = mb[t, b] - d_wc_c[T[t], buses[b], c]
                push_expr(expr_up, '<=', 0)

                expr_low = -mb[t, b] - d_wc_c[T[t], buses[b], c]
                push_expr(expr_low, '<=', 0)

    # one bulk insertion
    A = tuplelist(zip(rows, cols, vals))
    model.addMConstrs(A, sense=senses, rhs=rhs, name="PB_sparse")

    logger.info(f"Done - build time: {time.time() - start:.3f}s")
    return model
