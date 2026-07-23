from pra_psa.core.contingency_analysis import *
from tqdm import tqdm 
import logging

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


def runPRA(network, n_minus_k_set=None,
           load_time_series=None,
           prob_cont_model=NaiveProbFailureModel(),
           prob_load_model=NaiveProbLoadModel(),
           pf_solver=pp.rundcpp,
           opf_solver=pp.rundcopp):
    """
    Compute worst-case reliability scores for a network using N-1 or N-k contingencies.

    Parameters:
    - network: pandapower network object, the electrical network to simulate.
    - n_minus_k_set: Set of contingencies to simulate (element failures). If None, defaults to N-1 line failures.
    - load_samples: Array or list of load samples. If None, random samples based on the nominal load will be generated.
    - pf_solver: Power flow solver to use. Defaults to DC power flow solver (pp.rundcpp).

    Returns:
    - w_loading_lc: Matrix of worst-case line loading for each load sample and contingency.
    - is_unsafe_lc: Binary matrix indicating whether the system is unsafe for each load sample and contingency.
    """

    # Use N-1 contingency if no N-k set is provided
    if n_minus_k_set is None:
        logger.info('Empty n_minus_k_set....using N-1 line failures')
        n_minus_k_set = [{"element_index": l, "element_type": "line"} for l in network.line.index]
        n_minus_k_set += [{"element_index": t, "element_type": "trafo"} for t in network.trafo.index]

    n_load_samples = 25

    n_load_time_steps = len(load_time_series)
    n_contingencies = len(n_minus_k_set)

    # Preallocate result arrays
    worst_case_severity_cont_time_t = np.zeros((n_load_time_steps,))
    Pf_cont_t = np.zeros((n_load_time_steps,))
    Mean_System_Risk_t = np.zeros((n_load_time_steps,))
    Mean_Risk_c_id_t = np.zeros((n_load_time_steps,1 + n_contingencies))
    w_loading_tlc = np.zeros((n_load_time_steps, n_load_samples, 1 + n_contingencies))
    ext_grid_p_mw_tlc = np.zeros((n_load_time_steps, n_load_samples, 1 + n_contingencies))
    system_fail_tlc = np.zeros((n_load_time_steps, n_load_samples, 1 + n_contingencies))
    g_tail_tlc = np.zeros((n_load_time_steps, n_load_samples, 1 + n_contingencies))
    n_lin_overload_tlc = np.zeros((n_load_time_steps, n_load_samples, 1 + n_contingencies))

    for t_id, load_t in enumerate(load_time_series):

        network.ext_grid['in_service'] = False
        network.gen['slack'] = True
        network.line['max_loading_percent'] = 100
        try:  # Apply load at time t get reference power dispatch, apply the dispatch and then analyze load and failure scenarios
            network = apply_load(network, load_t)
            reference_p_mw, reference_q_mvar, network = get_OPF_gen(network, opf_solver=opf_solver)
            # sensitivity_matrix, base_line_flows, network, reference_p_mw, reference_q_mvar \
            #     = compute_sensitivity_matrix_and_base_flows( network)
            logger.info(f" Solved OPF for base case and undamaged network")
            network = apply_reference_dispatch(network, reference_p_mw=reference_p_mw, reference_q_mvar=reference_q_mvar)  # Optimal reference dispatch
        except Exception as e:
            logger.error(f"Failed to apply reference dispatch: {e}")  # Skip to next time step
            continue


        # calculate_LODF_and_shift(network, pf_solver=pp.runpp)
        network.line['max_loading_percent'] = 180
        logger.info(f'{blue_c} Sampling {n_load_samples} random load from a load model and get failure probabilities from the contingency model {reset_c}')
        load_samples = prob_load_model.sample(load_t, load_t * 0.2, n_sam=n_load_samples)
        prob_t_normal_and_contingencies = prob_cont_model.get_probabilities(failure_set=n_minus_k_set)

        for l_id, random_load in tqdm(enumerate(load_samples), desc="Evaluate Random Load Normal and Contingengy State",
                                      total=len(load_samples), ncols=100, colour="blue"): # Apply random load

            network = apply_load(network, random_load)

            try:  # Run power flow on the undamaged network
                loading, network = get_PF_loading(network, pf_solver=pf_solver, distributed_slack=True)
                logger.info(f"{green_c} Solved: PF for load sample {l_id}/{n_load_samples} on undamaged network {reset_c}")
            except Exception as e:
                logger.error(f"Power flow failed for load sample {l_id} on undamaged network: {e}")
                continue

            # Compute base case loading and safety status
            g_base, w_base, _, sys_fail_base = g_fun_loading(loading_percent=loading)
            w_loading_tlc[t_id, l_id, 0] = w_base
            ext_grid_p_mw_tlc[t_id, l_id, 0]  = network.res_ext_grid['p_mw'].values[0]
            system_fail_tlc[t_id, l_id, 0] = sys_fail_base
            g_tail_tlc[t_id, l_id, 0] = g_base[g_base > 0].sum()  # severity score as excess loading
            n_lin_overload_tlc[t_id, l_id, 0] = sum(g_base > 0)

            # Evaluate contingencies (N-k failures)
            for c_id, nkf in enumerate(n_minus_k_set):
                try:
                    cont_network = apply_nk_contingency(network, failure_event=nkf)  # Apply contingency
                    loading, cont_network = get_PF_loading(cont_network, pf_solver=pf_solver)
                    if c_id % 100 == 0 and c_id > 0:
                        logger.info(f"{green_c} Solved: PF for load sample {l_id}/{n_load_samples} for contingency {c_id}/{n_contingencies}  {reset_c}")
                    # Get line loading and system failure status for contingency
                    g_c, w_c, comp_fail_c, sys_fail_c = g_fun_loading(loading_percent=loading)

                    ext_grid_p_mw_tlc[t_id, l_id, c_id + 1] = cont_network.res_ext_grid['p_mw'].values[0]
                    w_loading_tlc[t_id,l_id, c_id + 1] = w_c
                    system_fail_tlc[t_id,l_id, c_id + 1] = sys_fail_c
                    g_tail_tlc[t_id, l_id, c_id + 1] = g_c[g_c > 0].sum()
                    n_lin_overload_tlc[t_id,l_id, c_id + 1] = sum(g_c > 0)

                except Exception as e:
                    logger.error(f"Power flow failed for load sample {l_id}, contingency {c_id}/{n_contingencies}: {e}")
                    logger.error(f"Error for contingency {nkf}")
                    logger.error(f"Error for contingency {nkf}")
                    ext_grid_p_mw_tlc[t_id, l_id, 0] =  np.nan
                    w_loading_tlc[t_id, l_id, c_id + 1] = np.nan  # Mark as unknown
                    system_fail_tlc[t_id, l_id, c_id + 1] = np.nan  # Mark as unknown
                    g_tail_tlc[t_id, l_id, c_id + 1] = np.nan  # Mark as unknown
                    n_lin_overload_tlc[t_id,l_id, c_id + 1] = np.nan  # Mark as unknown

        worst_case_severity_cont_time_t[t_id] = np.max(w_loading_tlc[t_id, :, 1:])
        Pf_cont_t[t_id] = np.sum(prob_t_normal_and_contingencies[1:])  # dummy probability value.... 1 - \sum_c Prob(c) ~= Prob(normal)

        Mean_Risk_c_id_t[t_id, 1] = prob_t_normal_and_contingencies[0] * np.mean(g_tail_tlc[t_id, :, 0])
        Mean_Risk_c_id_t[t_id, 1:] = [prob_t_normal_and_contingencies[c_id + 1] * np.mean(g_tail_tlc[t_id, :, c_id + 1]) for c_id, con in enumerate(n_minus_k_set)]
        Mean_System_Risk_t[t_id] = sum(Mean_Risk_c_id_t[t_id,:])

    return Mean_System_Risk_t, Mean_Risk_c_id_t, Pf_cont_t, worst_case_severity_cont_time_t



