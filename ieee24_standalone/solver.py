"""One joint DC-SCOPF oracle and three finite-candidate master searches.

The masters optimize start choices with a small MILP and re-score each proposed
schedule using the same frozen scenario bank. They are matheuristics: the
reported risk is evaluated exactly for each candidate under this model, but
global optimality over all outage schedules is not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from .model import StudyData


@dataclass(frozen=True)
class ScheduleResult:
    method: str
    starts: dict[str, int | None]
    utility: float
    selection_score: float
    deterministic_loss_mw: float
    mean_loss_mw: float
    var_loss_mw: float
    cvar_loss_mw: float
    dro_cvar_loss_mw: float
    max_loss_mw: float
    losses_mw: list[float]
    candidates_evaluated: int
    slave_calls: int
    runtime_s: float


def SLAVE_DCSCOPF(
    data: StudyData, demand_mw: np.ndarray, line_out=(), gen_out=(),
    contingencies: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, float]:
    """Joint base/N-1 DC LP; return DNS in MW for each surviving state.

    Base dispatch is shared through bounded corrective generator redispatch.
    Load shedding and local generation spill make every topology feasible.
    """
    net = data.network
    nb, ng, nl = len(net.bus_ids), len(net.gen_bus), len(net.branch_from)
    d = np.asarray(demand_mw, dtype=float)
    if d.shape != (nb,) or np.any(d < 0) or not np.all(np.isfinite(d)):
        raise ValueError("demand_mw must be a finite nonnegative vector in MATPOWER bus order")
    initial_lines, initial_gens = set(line_out), set(gen_out)
    if contingencies is None:
        contingencies = tuple([("line", int(i)) for i in data.settings["contingency_lines"]] +
                              [("gen", int(i)) for i in data.settings["contingency_generators"]])
    valid = [(typ, int(i)) for typ, i in contingencies
             if (typ == "line" and i not in initial_lines) or (typ == "gen" and i not in initial_gens)]
    if any(typ not in ("line", "gen") or not 0 <= i < (nl if typ == "line" else ng) for typ, i in valid):
        raise ValueError("Invalid contingency index")
    states = [("base", -1)] + valid
    stride = ng + nb + nl + nb + nb
    nvar = len(states) * stride + 1
    z = nvar - 1
    c = np.zeros(nvar)
    c[z] = 1.0
    bounds = []
    eq_r, eq_c, eq_v, beq = [], [], [], []
    ub_r, ub_c, ub_v, bub = [], [], [], []
    ramp_fraction = float(data.settings["redispatch_fraction"])
    if not 0 <= ramp_fraction <= 1:
        raise ValueError("redispatch_fraction must be in [0,1]")

    def add_eq(pairs, rhs):
        row = len(beq)
        for col, value in pairs:
            if value:
                eq_r.append(row); eq_c.append(col); eq_v.append(float(value))
        beq.append(float(rhs))

    def add_ub(pairs, rhs):
        row = len(bub)
        for col, value in pairs:
            if value:
                ub_r.append(row); ub_c.append(col); ub_v.append(float(value))
        bub.append(float(rhs))

    for state_idx, (kind, failed) in enumerate(states):
        off_l = initial_lines | ({failed} if kind == "line" else set())
        off_g = initial_gens | ({failed} if kind == "gen" else set())
        offset = state_idx * stride
        p = offset
        theta = p + ng
        flow = theta + nb
        shed = flow + nl
        spill = shed + nb

        # A separate reference is required for each connected island.
        parent = list(range(nb))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for line in range(nl):
            if line not in off_l:
                a, b = root(int(net.branch_from[line])), root(int(net.branch_to[line]))
                parent[a] = b
        reference = {}
        for bus in range(nb):
            reference.setdefault(root(bus), bus)
        reference_buses = set(reference.values())
        local_capacity = np.bincount(net.gen_bus, weights=net.gen_pmax, minlength=nb)
        for g in range(ng):
            cap = 0.0 if g in off_g else float(net.gen_pmax[g])
            bounds.append((0.0, cap))
        for bus in range(nb):
            bounds.append((0.0, 0.0) if bus in reference_buses else (-100.0, 100.0))
        for line in range(nl):
            cap = 0.0 if line in off_l else float(net.branch_rate[line])
            bounds.append((-cap, cap))
        bounds.extend((0.0, float(x)) for x in d)
        bounds.extend((0.0, float(x)) for x in local_capacity)

        for bus in range(nb):
            row = [(p + g, 1) for g in range(ng) if net.gen_bus[g] == bus]
            row += [(shed + bus, 1), (spill + bus, -1)]
            row += [(flow + line, -1) for line in range(nl) if net.branch_from[line] == bus]
            row += [(flow + line, 1) for line in range(nl) if net.branch_to[line] == bus]
            add_eq(row, d[bus])
        for line in range(nl):
            if line in off_l:
                continue
            b = net.base_mva / (net.branch_x[line] * net.branch_tap[line])
            add_eq([(flow + line, 1), (theta + net.branch_from[line], -b),
                    (theta + net.branch_to[line], b)], -b * net.branch_shift[line])
        add_ub([(shed + bus, 1) for bus in range(nb)] + [(z, -1)], 0.0)
        if state_idx:
            for g in range(ng):
                if g in off_g or g in initial_gens:
                    continue
                cap = ramp_fraction * float(net.gen_pmax[g])
                add_ub([(p + g, 1), (g, -1)], cap)
                add_ub([(g, 1), (p + g, -1)], cap)
    bounds.append((0.0, None))
    Aeq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(len(beq), nvar)).tocsr()
    Aub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(bub), nvar)).tocsr()
    worst = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                    bounds=bounds, method="highs")
    if not worst.success:
        raise RuntimeError(f"DC-SCOPF worst-state phase failed: {worst.message}")
    # Lexicographic second phase chooses a reproducible, low-total-DNS
    # corrective dispatch among solutions with optimal worst-state DNS.
    tie_break = np.zeros(nvar)
    for state_idx in range(len(states)):
        offset = state_idx * stride
        tie_break[offset + ng + nb + nl:offset + ng + nb + nl + nb] = 1.0
        tie_break[offset + ng + nb + nl + nb:offset + stride] = 1e-3
    tie_break[:ng] = 1e-6
    bounds[z] = (0.0, float(worst.x[z]) + 1e-6)
    res = linprog(tie_break, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"DC-SCOPF total-DNS phase failed: {res.message}")
    return {("base" if kind == "base" else f"{kind}:{failed}"):
            max(0.0, float(sum(res.x[i * stride + ng + nb + nl:
                                       i * stride + ng + nb + nl + nb])))
            for i, (kind, failed) in enumerate(states)}


def _risk(loss: np.ndarray, probability: np.ndarray, alpha: float, rho: float) -> tuple[float, float, float]:
    """Weighted upper CVaR and exact finite-support TV worst-case CVaR LP."""
    tail = 1.0 - alpha
    order = np.argsort(-loss)
    remaining, weighted_tail, quantile = tail, 0.0, 0.0
    for i in order:
        taken = min(remaining, probability[i])
        if taken > 0:
            weighted_tail += taken * loss[i]
            quantile = float(loss[i])
            remaining -= taken
        if remaining <= 1e-12:
            break
    n = len(loss)
    # q = adverse distribution, w = tail envelope, u = |q-p| epigraph.
    c = np.zeros(3 * n)
    c[n:2 * n] = -loss
    eq = np.zeros((2, 3 * n))
    eq[0, :n] = 1
    eq[1, n:2 * n] = 1
    ub = np.zeros((3 * n + 1, 3 * n))
    rhs = np.zeros(3 * n + 1)
    for i in range(n):
        ub[i, i] = -1; ub[i, n + i] = tail
        ub[n + i, i] = 1; ub[n + i, 2 * n + i] = -1; rhs[n + i] = probability[i]
        ub[2 * n + i, i] = -1; ub[2 * n + i, 2 * n + i] = -1; rhs[2 * n + i] = -probability[i]
    ub[-1, 2 * n:] = 1
    rhs[-1] = 2 * rho
    result = linprog(c, A_ub=ub, b_ub=rhs, A_eq=eq, b_eq=[1.0, 1.0],
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"TV-CVaR LP failed: {result.message}")
    return quantile, float(weighted_tail / tail), float(-result.fun)


def _event_dns(data: StudyData, scores: dict[str, float], s: int, t: int, off_lines: set[int]) -> float:
    """Expected DNS for competing first line-failure events within one step."""
    rates = []
    lines = []
    for line in data.settings["contingency_lines"]:
        if line in off_lines:
            continue
        spec = data.hazard[line]
        base = float(spec.get("base_rate_per_h", 0.0))
        if base <= 0:
            continue
        transverse = data.wind_speed_mps[s, t, line] * abs(np.sin(np.deg2rad(
            data.wind_direction_deg[s, t, line] - float(spec.get("bearing_deg", 0.0)))))
        exponent = float(spec.get("speed_slope_per_mps", 0.0)) * max(
            0.0, transverse - float(spec.get("speed_threshold_mps", 0.0)))
        rate = min(float(spec.get("max_rate_per_h", 1e6)), base * np.exp(min(exponent, 50)))
        rates.append(rate); lines.append(line)
    total = sum(rates)
    if total <= 0:
        return scores["base"]
    p0 = np.exp(-float(data.duration_h[t]) * total)
    return float(p0 * scores["base"] + sum((1 - p0) * r / total * scores[f"line:{line}"]
                                                for line, r in zip(lines, rates)))


def _evaluate(data: StudyData, starts: tuple[int | None, ...], sample_indices,
              baseline_cache: dict) -> tuple[np.ndarray, np.ndarray, int]:
    ntime = len(data.timestamps)
    losses = np.zeros(len(sample_indices))
    increments = np.zeros((len(sample_indices), ntime))
    calls = 0
    topology = []
    for t in range(ntime):
        off_l = frozenset(int(task["asset_id"]) for task, start in zip(data.tasks, starts)
                           if start is not None and task["asset_type"] == "line"
                           and start <= t < start + int(task["duration_steps"]))
        off_g = frozenset(int(task["asset_id"]) for task, start in zip(data.tasks, starts)
                           if start is not None and task["asset_type"] == "gen"
                           and start <= t < start + int(task["duration_steps"]))
        topology.append((off_l, off_g))
    clusters = []
    first = 0
    for t in range(1, ntime + 1):
        if t == ntime or topology[t] != topology[first]:
            if any(topology[first]):
                clusters.append((first, t, *topology[first]))
            first = t
    horizon_h = float(data.duration_h.sum())
    representative_count = int(data.settings["representatives_per_topology"])
    for j, s in enumerate(sample_indices):
        for first, last, off_l, off_g in clusters:
            steps = list(range(first, last))
            if representative_count and len(steps) > representative_count:
                load = data.demand_mw[s, first:last].sum(axis=1)
                wind = data.wind_speed_mps[s, first:last].max(axis=1)
                chosen = {first + int(np.argmax(load))}
                if representative_count > 1:
                    chosen.add(first + int(np.argmax(wind)))
                for t_extra in np.linspace(first, last - 1, representative_count, dtype=int):
                    if len(chosen) >= representative_count:
                        break
                    chosen.add(int(t_extra))
                for t_extra in steps:
                    if len(chosen) >= representative_count:
                        break
                    chosen.add(t_extra)
                steps = sorted(chosen)[:representative_count]
                weights = [float(data.duration_h[first:last].sum()) / len(steps) / horizon_h] * len(steps)
            else:
                weights = [float(data.duration_h[t]) / horizon_h for t in steps]
            for t, weight in zip(steps, weights):
                if (s, t) not in baseline_cache:
                    baseline_scores = SLAVE_DCSCOPF(data, data.demand_mw[s, t])
                    baseline_cache[s, t] = _event_dns(data, baseline_scores, s, t, set())
                    calls += 1
                candidate_scores = SLAVE_DCSCOPF(data, data.demand_mw[s, t], off_l, off_g)
                candidate = _event_dns(data, candidate_scores, s, t, off_l)
                calls += 1
                increments[j, t] = weight * max(0.0, candidate - baseline_cache[s, t])
        losses[j] = float(increments[j].sum())
    return losses, increments, calls


def _master(data: StudyData, method: str, deterministic: ScheduleResult | None = None) -> ScheduleResult:
    tic = perf_counter()
    settings = data.settings
    if method != "DET" and deterministic is None:
        deterministic = _master(data, "DET")
    tasks, ntime = data.tasks, len(data.timestamps)
    choices = []  # (task index, start, utility, occupancy)
    groups = []
    for j, task in enumerate(tasks):
        group = []
        for start in range(int(task["earliest_start"]), int(task["latest_start"]) + 1):
            value = float(task["utility"]) - float(task.get("delay_penalty_per_step", 0.0)) * (
                start - int(task["earliest_start"]))
            group.append(len(choices))
            choices.append((j, start, value, tuple(range(start, start + int(task["duration_steps"])))))
        if bool(task.get("deferrable", False)):
            group.append(len(choices)); choices.append((j, None, 0.0, ()))
        groups.append(group)
    n = len(choices)
    utility = np.array([x[2] for x in choices])
    A = np.zeros((len(tasks) + ntime, n))
    for j, group in enumerate(groups):
        A[j, group] = 1
    for k, (_, _, _, active) in enumerate(choices):
        A[len(tasks) + np.asarray(active, dtype=int), k] = 1
    lower = np.r_[np.ones(len(tasks)), np.zeros(ntime)]
    upper = np.r_[np.ones(len(tasks)), np.full(ntime, int(settings["max_active"]))]
    floor = None if deterministic is None else deterministic.utility - float(settings["utility_tolerance"])
    proxy = np.zeros(n)
    base_load = float(np.dot(data.probabilities, data.demand_mw.sum(axis=2).mean(axis=1)))
    for k, (j, start, _, active) in enumerate(choices):
        if start is not None:
            load = np.dot(data.probabilities, data.demand_mw[:, active, :].sum(axis=2).mean(axis=1))
            proxy[k] = float(load / max(base_load, 1.0)) * len(active)
    seen = []
    baseline_cache = {}
    evaluated = 0
    calls = 0
    best_key = None
    best_starts = None
    representative = next((i for i, name in enumerate(data.scenario_ids)
                           if name == settings.get("deterministic_scenario_id", data.scenario_ids[0])), None)
    if representative is None:
        raise ValueError("deterministic_scenario_id is absent from the fixed scenario bank")
    for _ in range(int(settings["max_candidates"])):
        rows = [A]; lo = [lower]; hi = [upper]
        if floor is not None:
            rows.append(utility[None, :]); lo.append(np.array([floor])); hi.append(np.array([np.inf]))
        for selected in seen:
            cut = np.zeros((1, n)); cut[0, list(selected)] = 1
            rows.append(cut); lo.append(np.array([-np.inf])); hi.append(np.array([len(tasks) - 1]))
        objective = -utility + float(settings["proxy_gain"]) * proxy
        candidate = milp(objective, integrality=np.ones(n), bounds=Bounds(0, 1),
                         constraints=LinearConstraint(np.vstack(rows), np.concatenate(lo), np.concatenate(hi)),
                         options={"disp": False})
        if candidate.x is None:
            if evaluated == 0:
                raise ValueError("No feasible schedule satisfies windows, concurrency and utility floor")
            break
        selected = tuple(int(np.argmax(candidate.x[group])) + group[0] for group in groups)
        seen.append(selected)
        starts = tuple(choices[k][1] for k in selected)
        u = float(sum(utility[list(selected)]))
        indices = [representative] if method == "DET" else list(range(len(data.scenario_ids)))
        loss, increments, count = _evaluate(data, starts, indices, baseline_cache)
        calls += count; evaluated += 1
        if method == "DET":
            risk_value = float(loss[0])
            key = (float(settings["loss_scale"]) * risk_value - u, risk_value, -u)
        else:
            _, cvar, dro = _risk(loss, data.probabilities, settings["alpha"], settings["tv_radius"])
            risk_value = cvar if method == "CVAR" else dro
            key = (risk_value, float(np.dot(data.probabilities, loss)), -u)
        if best_key is None or key < best_key:
            best_key, best_starts = key, starts
        # Candidate-specific response of the risk proxy. The exact slave score
        # determines incumbent selection; this update only steers later MILPs.
        if method == "DET":
            time_loss = increments[0]
        else:
            order = np.argsort(-loss)
            tail_count = max(1, int(np.ceil((1 - settings["alpha"]) * len(loss))))
            time_loss = increments[order[:tail_count]].mean(axis=0)
        for k in selected:
            active = choices[k][3]
            if active:
                proxy[k] += float(sum(time_loss[list(active)]))
    assert best_starts is not None
    all_loss, _, count = _evaluate(data, best_starts, list(range(len(data.scenario_ids))), baseline_cache)
    calls += count
    deterministic_loss, _, count = _evaluate(data, best_starts, [representative], baseline_cache)
    calls += count
    var, cvar, dro = _risk(all_loss, data.probabilities, settings["alpha"], settings["tv_radius"])
    selected_utility = sum(float(task["utility"]) - float(task.get("delay_penalty_per_step", 0.0)) *
                           (start - int(task["earliest_start"]))
                           for task, start in zip(tasks, best_starts) if start is not None)
    score = float(best_key[0]) if best_key is not None else 0.0
    return ScheduleResult(method, {str(t["task_id"]): start for t, start in zip(tasks, best_starts)},
                          float(selected_utility), score, float(deterministic_loss[0]),
                          float(np.dot(data.probabilities, all_loss)), var, cvar, dro,
                          float(np.max(all_loss)), all_loss.tolist(), evaluated, calls,
                          perf_counter() - tic)


def Master_DET(data: StudyData) -> ScheduleResult:
    """Select by utility minus representative-path security loss."""
    return _master(data, "DET")


def Master_CVAR(data: StudyData, deterministic: ScheduleResult | None = None) -> ScheduleResult:
    """Minimize evaluated empirical CVaR among utility-feasible candidates."""
    return _master(data, "CVAR", deterministic)


def Master_DRO_CVAR(data: StudyData, deterministic: ScheduleResult | None = None) -> ScheduleResult:
    """Minimize exact TV worst-case CVaR of each evaluated candidate."""
    return _master(data, "DRO_CVAR", deterministic)
