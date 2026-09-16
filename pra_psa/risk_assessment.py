from pra_psa.core.contingency_analysis import *
from pra_psa.reliability_performance import g_fun_loading
from utils.utils import *

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NaiveProbLoadModel:
    def __init__(self):
        self.min_L = 0
        self.max_L = 1_000

    def sample(self, mean, std, n_sam):
        return [np.clip(self.min_L, np.random.normal(mean, std * 0.1), self.max_L) for _ in range(n_sam)]


class NaiveProbFailureModel:
    def __init__(self):
        self.model_params = (0.05, 0.3)

    def get_probabilities(self, failure_set, state=None):
        """ example of occurrence probability for the contingency set (independent of the system state and uniform in this naive case)"""
        Prob_contingent_states = [np.random.uniform(self.model_params[0], self.model_params[1]) * 1 / len(failure_set) for _ in failure_set]
        prob_normal_and_failures = [1 - sum(Prob_contingent_states)] + Prob_contingent_states
        return prob_normal_and_failures

def runPRA(
    network,
    n_minus_k_set=None,
    load_time_series=None,
    prob_cont_model=NaiveProbFailureModel(),
    prob_load_model=NaiveProbLoadModel(),
    pf_solver=pp.rundcpp,
    opf_solver=pp.rundcopp,
    n_load_samples=25,
    loading_limit_percent=100.0,
    contingency_loading_limit_percent=100.0,
    distributed_slack=False,
):
    """
    Preventive-dispatch PRA.

    For each load state:
    1. solve OPF on the intact network;
    2. freeze generator dispatch;
    3. keep the external grid as slack/reference;
    4. evaluate base and N-k states by PF under sampled loads;
    5. aggregate failure probability and expected severity.
    """

    if load_time_series is None:
        raise ValueError("load_time_series must be provided.")

    if n_minus_k_set is None:
        logger.info("Empty n_minus_k_set; using N-1 line and transformer failures.")
        n_minus_k_set = (
            [{"element_index": l, "element_type": "line"} for l in network.line.index]
            +
            [{"element_index": t, "element_type": "trafo"} for t in network.trafo.index]
        )

    n_time = len(load_time_series)
    n_cont = len(n_minus_k_set)
    n_states = 1 + n_cont

    worst_loading_t = np.full(n_time, np.nan)
    pf_t = np.full(n_time, np.nan)
    mean_system_risk_t = np.full(n_time, np.nan)
    mean_risk_c_t = np.full((n_time, n_states), np.nan)

    w_loading_tsc = np.full((n_time, n_load_samples, n_states), np.nan)
    ext_grid_p_tsc = np.full((n_time, n_load_samples, n_states), np.nan)
    system_fail_tsc = np.full((n_time, n_load_samples, n_states), np.nan)
    severity_tsc = np.full((n_time, n_load_samples, n_states), np.nan)
    n_overload_tsc = np.full((n_time, n_load_samples, n_states), np.nan)

    for t_id, load_t in enumerate(load_time_series):

        # ------------------------------------------------------------
        # 1. Reference OPF on intact network
        # ------------------------------------------------------------
        try:
            ref_net = copy.deepcopy(network)

            ref_net.ext_grid["in_service"] = True

            if "slack" in ref_net.gen.columns:
                ref_net.gen["slack"] = False

            ref_net.line["max_loading_percent"] = loading_limit_percent

            ref_net = apply_load(ref_net, load_t)

            reference_p_mw, reference_q_mvar, ref_net = get_OPF_gen(
                ref_net,
                opf_solver=opf_solver,
            )

            ref_net = apply_reference_dispatch(
                ref_net,
                reference_p_mw=reference_p_mw,
                reference_q_mvar=reference_q_mvar,
            )

            # Preventive dispatch is now fixed.
            ref_dispatch_net = copy.deepcopy(ref_net)

            logger.info("Solved OPF reference dispatch for time %s.", t_id)

        except Exception as exc:
            logger.error("Reference OPF failed at time %s: %s", t_id, exc)
            continue

        # ------------------------------------------------------------
        # 2. Sample load uncertainty and contingency probabilities
        # ------------------------------------------------------------
        load_samples = prob_load_model.sample(
            load_t,
            load_t * 0.2,
            n_sam=n_load_samples,
        )

        p_cont = np.asarray(
            prob_cont_model.get_probabilities(failure_set=n_minus_k_set),
            dtype=float,
        )

        if len(p_cont) != n_states:
            raise ValueError(
                "Contingency probability vector must contain one normal-state "
                f"probability plus {n_cont} contingency probabilities."
            )

        p_load = np.full(n_load_samples, 1.0 / n_load_samples)

        # ------------------------------------------------------------
        # 3. Evaluate base and contingency PF states
        # ------------------------------------------------------------
        for s_id, random_load in enumerate(load_samples):

            # Base state
            try:
                base_net = copy.deepcopy(ref_dispatch_net)
                base_net = apply_load(base_net, random_load)

                loading, base_net = get_PF_loading(
                    base_net,
                    pf_solver=pf_solver,
                    distributed_slack=distributed_slack,
                )

                g_base, w_base, _, sys_fail_base = g_fun_loading(
                    loading_percent=loading,
                    upper_threshold=loading_limit_percent,
                )

                w_loading_tsc[t_id, s_id, 0] = w_base
                system_fail_tsc[t_id, s_id, 0] = sys_fail_base
                severity_tsc[t_id, s_id, 0] = g_base[g_base > 0].sum()
                n_overload_tsc[t_id, s_id, 0] = np.sum(g_base > 0)

                if len(base_net.ext_grid) > 0 and hasattr(base_net, "res_ext_grid"):
                    ext_grid_p_tsc[t_id, s_id, 0] = base_net.res_ext_grid["p_mw"].sum()

            except Exception as exc:
                logger.error(
                    "Base PF failed at time %s, sample %s: %s",
                    t_id,
                    s_id,
                    exc,
                )
                continue

            # Contingency states
            for c_id, contingency in enumerate(n_minus_k_set, start=1):
                try:
                    cont_net = copy.deepcopy(ref_dispatch_net)
                    cont_net = apply_load(cont_net, random_load)
                    cont_net = apply_nk_contingency(
                        cont_net,
                        failure_event=contingency,
                    )

                    loading, cont_net = get_PF_loading(
                        cont_net,
                        pf_solver=pf_solver,
                        distributed_slack=distributed_slack,
                    )

                    g_c, w_c, _, sys_fail_c = g_fun_loading(
                        loading_percent=loading,
                        upper_threshold=contingency_loading_limit_percent,
                    )

                    w_loading_tsc[t_id, s_id, c_id] = w_c
                    system_fail_tsc[t_id, s_id, c_id] = sys_fail_c
                    severity_tsc[t_id, s_id, c_id] = g_c[g_c > 0].sum()
                    n_overload_tsc[t_id, s_id, c_id] = np.sum(g_c > 0)

                    if len(cont_net.ext_grid) > 0 and hasattr(cont_net, "res_ext_grid"):
                        ext_grid_p_tsc[t_id, s_id, c_id] = cont_net.res_ext_grid["p_mw"].sum()

                except Exception as exc:
                    logger.error(
                        "PF failed at time %s, sample %s, contingency %s/%s: %s",
                        t_id,
                        s_id,
                        c_id,
                        n_cont,
                        exc,
                    )

        # ------------------------------------------------------------
        # 4. Risk aggregation
        # ------------------------------------------------------------
        valid_fail = np.nan_to_num(system_fail_tsc[t_id], nan=1.0)
        valid_severity = np.nan_to_num(severity_tsc[t_id], nan=np.nan)

        for c_id in range(n_states):
            mean_severity_c = np.nansum(p_load * valid_severity[:, c_id])
            mean_risk_c_t[t_id, c_id] = p_cont[c_id] * mean_severity_c

        mean_system_risk_t[t_id] = np.nansum(mean_risk_c_t[t_id, :])

        pf_t[t_id] = np.nansum(
            p_load[:, None] * p_cont[None, :] * valid_fail
        )

        worst_loading_t[t_id] = np.nanmax(w_loading_tsc[t_id, :, :])

    return {
        "mean_system_risk_t": mean_system_risk_t,
        "mean_risk_c_t": mean_risk_c_t,
        "pf_t": pf_t,
        "worst_loading_t": worst_loading_t,
        "w_loading_tsc": w_loading_tsc,
        "ext_grid_p_tsc": ext_grid_p_tsc,
        "system_fail_tsc": system_fail_tsc,
        "severity_tsc": severity_tsc,
        "n_overload_tsc": n_overload_tsc,
    }


