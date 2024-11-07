from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasters.condcopulas.kernels.beta_kernel import SequentialKDE
from forecasters.condcopulas.copulas.gaussian_copulas import GaussianCopula
from pathlib import Path

from utils.dataloader import *
from forecasters.probabilisticmodel import Probability_model_nodal_load as Model
# from xgboost import XGBRegressor  # model clss for predictor
# from mbtr.mbtr import MBT # example of different model class  (https://github.com/supsi-dacd-isaac/mbtr)
from forecasters.condcopulas.copulas.gaussian_copulas import GaussianCopula


def find_project_root(start_path, marker='README.md'):
    for parent in start_path.parents:
        if (parent / marker).exists():
            return parent
    return None


Selected_case = 'SwissGrid'  # 'SwissGrid' or 'IEEE118' or 'IEEE24'

if __name__ == '__main__':


    if Selected_case == 'SwissGrid':

        current_file_path = Path(__file__).resolve()
        project_root = find_project_root(current_file_path, marker='.git')  # Or use 'README.md', 'setup.py'
        if project_root is not None:
            pkl_file_path = project_root / 'data' / 'powersystems' / 'SwissGrid' / 'swissgrid.pkl'  # read annual data of Swissgrid load
            try:  # Load the pickle file with error handling
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
        conf_path = '../config/conf_IEEE118.json'
        net_data_24, df_nodal_loads, conf24 = data_loader(conf_path)
        df_nodal_loads.index = pd.date_range(start='1/1/2020', periods=len(df_nodal_loads), freq='H')
        df_nodal_loads.name = 'load'
        steps_per_day = 24
        n_scenarios = 20
        n_scenarios_gc = 20
        h_kernel = 0.001

    elif Selected_case == 'IEEE24':
        conf_path = '../config/conf_IEEE24.json'
        net_data_24, df_nodal_loads, conf24 = data_loader(conf_path)
        df_nodal_loads.index = pd.date_range(start='1/1/2020', periods=len(df_nodal_loads), freq='H')
        df_nodal_loads.name = 'load'
        steps_per_day = 24
        n_scenarios = 30
        n_scenarios_gc = 30
        h_kernel = 0.05


    def get_lags(df, lags, column):
        return pd.concat([df[column].shift(l).rename('{}_lag_{:03d}'.format(column, l)) for l in lags], axis=1)

    query_nodes = [0, 1, 9, 12, 19]
    n_query_steps = 3
    fig, ax = plt.subplots(n_query_steps, len(query_nodes), layout='tight', figsize=(10, 8))

    for nd_i, node in enumerate(query_nodes):
        if Selected_case != 'SwissGrid':
            df = df_nodal_loads.iloc[:, node].copy()

        # make a dataset with day of the week as a covariate and add lags of the target and the covariate
        cov_name = 'day'
        df = pd.concat([df, pd.Series(df.index.dayofweek, name=cov_name, index=df.index)], axis=1)
        x = pd.concat([df, get_lags(df, np.arange(0, steps_per_day), 'load')], axis=1)
        x = pd.concat([x, get_lags(df, -np.arange(1, steps_per_day + 1), 'day')], axis=1)
        y = get_lags(df, -np.arange(1, steps_per_day + 1), 'load')
        remove_these = x.isna().any(axis=1) | y.isna().any(axis=1)
        x = x[~remove_these]
        y = y[~remove_these]

        print_starts = [0, 1, 2, steps_per_day + 2, steps_per_day * 2 + 2]
        print('{} Columns in x: {}\b'.format('#' * 10, '#' * 10))
        #[print(*x.columns[s:print_starts[i + 1]]) for i, s in enumerate(print_starts[:-1])]
        print('{} Columns in y: {}\b'.format('#' * 10, '#' * 10))
        print_starts = [0, steps_per_day]
        #[print(*y.columns[s:print_starts[i + 1]]) for i, s in enumerate(print_starts[:-1])]

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

        categorical_vars = np.hstack([np.ones(covariate_past.shape[1]), np.zeros(target_past.shape[1] + 1)]).astype(bool)
        query_steps = [13, 43, 180]
        # ##################################################################################################################
        # ############################# Conditional Gaussian Copula ########################################################
        # ##################################################################################################################

        gc = GaussianCopula(estimation_method='vanilla').fit(pd.concat([x_tr, y_tr], axis=1),
                                                             conditioning_vars=x_tr.columns)
        gc_scenarios = [gc.sample(n_scenarios_gc, x_te.values[q, :]) for q in query_steps]

        print('plotting')
        for i, q in enumerate(query_steps):

            ax[i, nd_i].plot(np.arange(-steps_per_day - 1, 0), target_past_q[q, :])
            ax[i, nd_i].plot(np.squeeze(np.dstack(gc_scenarios[i])), color='darkred', alpha=0.1)
            ax[i, nd_i].plot(y_te.iloc[q, :].values)
            ax[i, nd_i].set_ylabel(f'Load at node {nd_i}')
            ax[i, nd_i].set_title(f'Step {q} h ')
            ax[i, nd_i].grid()

    plt.show()