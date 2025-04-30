from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pra_psa.probabilistic_models.condcopulas.kernels.beta_kernel import SequentialKDE

# from xgboost import XGBRegressor  # model clss for predictor
# from mbtr.mbtr import MBT # example of different model class  (https://github.com/supsi-dacd-isaac/mbtr)
from pra_psa.probabilistic_models.condcopulas.copulas.gaussian_copulas import GaussianCopula

Selected_case = 'SwissGrid' # 'SwissGrid' or 'IEEE118' or 'IEEE24'

if __name__ == '__main__':

    def find_project_root(start_path, marker='README.md'):
        """
        Recursively search for the project root by looking for a specific marker file or directory.
        """
        for parent in start_path.parents:
            if (parent / marker).exists():
                return parent
        return None


    # Set the path to the script file's location
    current_file_path = Path(__file__).resolve()
    project_root = find_project_root(current_file_path, marker='.git')  # Or use 'README.md', 'setup.py'

    if Selected_case == 'SwissGrid':

        if project_root is not None:
            pkl_file_path = project_root / 'data' / 'powersystems' / 'SwissGrid' / 'swissgrid.pkl'  # read annual data of Swissgrid load
            # Load the pickle file with error handling
            try:
                df = pd.read_pickle(pkl_file_path)
                print("Pickle file loaded successfully.")
            except FileNotFoundError:
                print(f"Error: The file '{pkl_file_path}' was not found.")
            except Exception as e:
                print(f"An error occurred while loading the file: {e}")
        else:
            print("Error: Project root not found. Please ensure a marker file (e.g., .git, README.md) is available.")

        df = df[~df.index.duplicated(keep='first')]
        df.name = 'load'
        steps_per_day = 96
        n_scenarios = 10
        n_scenarios_gc = 10
        h_kernel = 0.05

    elif Selected_case == 'IEEE118':
        conf_path = '../../config/conf_IEEE118.json'
        net_data_24, df_nodal_loads, conf24 = data_loader(conf_path)
        df_nodal_loads.index = pd.date_range(start='1/1/2020', periods=len(df_nodal_loads), freq='H')
        df_nodal_loads.name = 'load'
        steps_per_day = 24
        n_scenarios = 20
        n_scenarios_gc = 20
        h_kernel = 0.001

    elif Selected_case == 'IEEE24':
        conf_path = '../../config/conf_IEEE24.json'
        net_data_24, df_nodal_loads, conf24 = data_loader(conf_path)
        df_nodal_loads.index = pd.date_range(start='1/1/2020', periods=len(df_nodal_loads), freq='H')
        df_nodal_loads.name = 'load'
        steps_per_day = 24
        n_scenarios = 30
        n_scenarios_gc = 30
        h_kernel = 0.05

    if Selected_case != 'SwissGrid':
        df = df_nodal_loads.iloc[:, 0].copy()
        df.name = 'load'

    # make a dataset with day of the week as a covariate and add lags of the target and the covariate
    cov_name = 'day'
    df = pd.concat([df, pd.Series(df.index.dayofweek, name=cov_name, index=df.index)], axis=1)


    def get_lags(df, lags, column):
        return pd.concat([df[column].shift(l).rename('{}_lag_{:03d}'.format(column, l)) for l in lags], axis=1)

    x = pd.concat([df, get_lags(df, np.arange(0, steps_per_day), 'load')], axis=1)
    x = pd.concat([x, get_lags(df, -np.arange(1, steps_per_day + 1), 'day')], axis=1)
    y = get_lags(df, -np.arange(1, steps_per_day + 1), 'load')
    remove_these = x.isna().any(axis=1) | y.isna().any(axis=1)
    x = x[~remove_these]
    y = y[~remove_these]

    print_starts = [0, 1, 2, steps_per_day + 2, steps_per_day * 2 + 2]
    print('{} Columns in x: {}\b'.format('#' * 10, '#' * 10))
    [print(*x.columns[s:print_starts[i + 1]]) for i, s in enumerate(print_starts[:-1])]
    print('{} Columns in y: {}\b'.format('#' * 10, '#' * 10))
    print_starts = [0, steps_per_day]
    [print(*y.columns[s:print_starts[i + 1]]) for i, s in enumerate(print_starts[:-1])]

    d_s = 0
    query_steps = np.arange(0, steps_per_day * 10, steps_per_day)
    tr_start = int(steps_per_day * (150 + d_s))
    tr_end = int(steps_per_day * (160 + d_s))
    te_end = int(steps_per_day * (170 + d_s))
    x_tr, y_tr = x.iloc[tr_start:tr_end, :], y.iloc[tr_start:tr_end, :]
    x_te, y_te = x.iloc[tr_end:te_end, :], y.iloc[tr_end:te_end, :]

    # ##################################################################################################################
    # ############################# Sequential KDE #####################################################################
    # ##################################################################################################################

    # ----------------------------- prepare data for the sequential KDE -----------------------------------------------
    target = y_tr['load_lag_-01'].values
    target_past = x_tr[[c for c in x_tr.columns if 'load' in c and '-' not in c]]
    target_past = target_past[np.sort(target_past.columns)[::-1]].values

    covariate_past = x_tr[[c for c in x_tr.columns if cov_name in c and '-' not in c]]
    covariate_past = covariate_past[np.sort(covariate_past.columns)[::-1]].values
    covariate_future = x_tr[[c for c in x_te.columns if cov_name in c and '-' in c]].values

    target_past_q = x_te[[c for c in x_te.columns if 'load' in c and '-' not in c]]
    target_past_q = target_past_q[np.sort(target_past_q.columns)[::-1]].values
    covariate_past_q = x_te[[c for c in x_te.columns if cov_name in c and '-' not in c]]
    covariate_past_q = covariate_past_q[np.sort(covariate_past_q.columns)[::-1]].values
    covariate_future_q = x_te[[c for c in x_te.columns if cov_name in c and '-' in c]].values
    # covariate_past, covariate_future = None, None
    categorical_vars = np.hstack([np.ones(covariate_past.shape[1]), np.zeros(target_past.shape[1] + 1)]).astype(bool)
    query_steps = np.arange(0, steps_per_day * 4, steps_per_day)

    # ----------------------------- without covariates -----------------------------
    skde = SequentialKDE().fit(target, target_past)
    with ProcessPoolExecutor() as executor:
        scenarios = list(tqdm(executor.map(partial(skde.sample_scenario,
                                                   target_past_query=target_past_q[query_steps],
                                                   n_sa=steps_per_day, h_kernel=h_kernel, n_support=20),
                                           np.arange(n_scenarios)), total=n_scenarios))

    # ----------------------------- with covariates -----------------------------
    skde_cat = SequentialKDE().fit(target, target_past, covariate_past)
    with ProcessPoolExecutor() as executor:
        scenarios_kr = list(
            tqdm(executor.map(partial(skde_cat.sample_scenario, target_past_query=target_past_q[query_steps],
                                      n_sa=steps_per_day, covariate_past_query=covariate_past_q[query_steps],
                                      covariate_future_query=covariate_future_q[query_steps],
                                      h_kernel=h_kernel, n_support=20, categorical_vars=categorical_vars,
                                      categorical_kernel='kronecker'),
                              np.arange(n_scenarios)), total=n_scenarios))

    # ##################################################################################################################
    # ############################# Conditional Gaussian Copula ########################################################
    # ##################################################################################################################

    gc = GaussianCopula(estimation_method='vanilla').fit(pd.concat([x_tr, y_tr], axis=1),
                                                         conditioning_vars=x_tr.columns)
    gc_scenarios = [gc.sample(n_scenarios_gc, x_te.values[q, :]) for q in query_steps]
    for i, q in enumerate(query_steps):
        fig, ax = plt.subplots(3, 1, layout='tight', figsize=(10, 8))
        ax[0].plot(np.arange(-steps_per_day - 1, 0), target_past_q[q, :])
        ax[0].plot(np.squeeze(np.dstack(scenarios)[i]), color='darkred', alpha=0.2)
        ax[0].plot(y_te.iloc[q, :].values)
        ax[0].set_title('No covariates')
        ax[1].plot(np.arange(-steps_per_day - 1, 0), target_past_q[q, :])
        ax[1].plot(np.squeeze(np.dstack(scenarios_kr)[i]), color='darkred', alpha=0.2)
        ax[1].plot(y_te.iloc[q, :].values)
        ax[1].set_title('Categoricals - Kronecker')
        ax[2].plot(np.arange(-steps_per_day - 1, 0), target_past_q[q, :])
        ax[2].plot(np.squeeze(np.dstack(gc_scenarios[i])), color='darkred', alpha=0.2)
        ax[2].plot(y_te.iloc[q, :].values)
        ax[2].set_title('Conditional Gaussian Copula')

        plt.show()