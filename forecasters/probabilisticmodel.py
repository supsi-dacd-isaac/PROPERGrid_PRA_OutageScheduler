import numpy as np
from scipy.stats import norm
from statsmodels.distributions.copula.api import (CopulaDistribution, GumbelCopula, IndependenceCopula)
from scipy.stats import ecdf as ECDF
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from colorama import Fore, Back, Style
from scipy.stats import multivariate_normal
import pandas as pd


class Demand_sampler:
    """ demand sampler class
            - net: a pandapower network model
            - model: a probabilistic model for the injections
                                a class that must have a method
                                    1) self.fit(data)
                                    2) self.sample(n)
    """
    def __init__(self, net, model=None):
        self.mean_P_mw = net.load['p_mw']
        self.mean_Q_mvar = net.load['q_mvar']
        self.std_P_mw = net.load['p_mw'].abs() * 0.1
        self.std_Q_mvar = net.load['q_mvar'].abs() * 0.1
        self.is_fitted = False
        self.prob_model = model


    def fit(self, data):
        """ Fit a model to data.
        Inputs:
            - data (np.ndarray): containing nodal injection data
        Returns:
          - probabilistic model: Trained
        """
        self.is_fitted = False
        try:
            if self.prob_model is not None:
                self.prob_model.fit(data)
                self.is_fitted = True
                return self.prob_model
            else:
                print('No data or model provided')
                return None
        except ValueError as e:
            print(f'Error fitting model: {e}')


    def generate_samples(self, num_samples:int=1000):
        """
        Generate random samples of demand using a fitted model or independent normal distributions.
        Parameters:
            num_samples (int): Number of samples to generate (default: 1000).
        Returns:
            tuple[np.ndarray, np.ndarray]:
                - P (np.ndarray): Array of shape (num_samples,) containing active power demand samples.
                - Q (np.ndarray): Array of shape (num_samples,) containing reactive power demand samples.
        """
        if self.is_fitted:
            # If model is fitted, sample from non-correlated distributions
            P, Q = self.prob_model.sample(num_samples=num_samples)
        else:  # If model is not fitted, generate uncorrelated gaussian samples
            # Generate random samples for active  and reactive power demand
            P, Q = [], []
            print('the model has not been fitted yet....using uncorrelated gaussian')
            for _ in range(num_samples):
                P.append(np.random.normal(loc=self.mean_P_mw, scale=self.std_P_mw).tolist())
                Q.append(np.random.normal(loc=self.mean_Q_mvar, scale=self.std_Q_mvar).tolist())
        return P, Q


class ProbabilisticModel:
    """abstract class for the probability load demand model"""
    def __init__(self, name: str = 'abstract class for the probabilistic model'):
        self.name = name

    def sample(self, num_samples: int):
        P, Q = np.zeros(num_samples), np.zeros(num_samples)
        return P, Q

    def fit(self, data):
        pass


class ProbNodalLoad:
    """ probabilistic model for the nodal demand (vertical loads)"""
    def __init__(self,
                 net=None, window_size_x: int = 24, window_size_y: int = 6, model_class=None):
        self.name = 'Nodal load forecaster'
        if net is not None:
            self.reference_P_mw = net.load['p_mw']
            self.reference_Q_mvar = net.load['q_mvar']
        if model_class is None:
            print('model_class is None. Using XGBRegressor as nodal load forecaster')
            self.model_class = XGBRegressor
        else:
            self.model_class = model_class
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.marginal_models, self.residual_ECDFs, self.ErrorCovariances= None, None, None

    def preprocess_nodal_load_data(self, nodal_data, use_time_indices: bool = False,
                                   load_id: int = 0,  test_size: float = 0.2):
        """ preprocess data for load_id"""
        X, y = [], []
        test_size = min(max(test_size, 0.05), .95)
        hour_id = np.array([np.linspace(0, 23, 24) for _ in range(364)]).flatten()
        day_id = np.array([[np.linspace(i, i, 24) for i in range(7)] for _ in range(52)]).flatten()
        for j in range(self.window_size_x, nodal_data.shape[1] - self.window_size_y):
            if use_time_indices:
                X_t = np.hstack([nodal_data[load_id, j - self.window_size_x:j], hour_id[j], day_id[j]])
            else:
                X_t = nodal_data[load_id, j - self.window_size_x:j]
            y_t = nodal_data[load_id, j:self.window_size_y + j]
            X.append(X_t)
            y.append(y_t)
        X = np.array(X)
        y = np.array(y)
        return train_test_split(X, y, test_size=test_size)

    def fit_marginal_injection_models(self, nodal_demand_data):
        """
        Function to forecast marginal loads using historical data.
        Parameters:
        - nodal_injections_data: numpy array of shape (n_loads, n_hours)
        - window_size_x: Size of the past loads for forecasting
        - window_size_y: Size of the prediction window for forecasting
        Returns:
        - marginal_models: LIST with n_loads marginal forecasters (predict y_expected load)
        - marginal_residual_ecdfs: LIST with empirical-cdf for the prediction residuals (y_pred () + q() probability model)
        NOTE:
        - marginal_models predict the nodal loads [y_expected = f(x)]
        - marginal_residual_ecdfs give the probabilistic model [y_expected+ ecdf(x) ]
        """
        n_marginal_predictors = len(nodal_demand_data)
        marginal_models, residual_ecdfs, residuals_list = [], [], []
        for ld_i in range(n_marginal_predictors):
            print(f'prepare training data for the node: {ld_i + 1}/{n_marginal_predictors}')
            X_tr, X_te, y_tr, y_te = self.preprocess_nodal_load_data(nodal_data=nodal_demand_data, load_id=ld_i)
            """fit predictor"""
            print(f'training model....')
            model = self.model_class()
            marginal_models.append(model.fit(X_tr, y_tr))
            print(f'done.')
            """simple residual model"""
            errors = marginal_models[ld_i].predict(X_tr) - y_tr
            residuals_list.append(errors)
            residual_ecdfs.append(ECDF(errors.flatten()))
        # Store trained model
        self.marginal_models = marginal_models  # this predicts expectation
        self.residual_ECDFs = residual_ecdfs   # this samples variability around the expected value
        Covariances = [pd.DataFrame(res_ld).cov() for res_ld in residuals_list]
        self.ErrorCovariances = Covariances
        return marginal_models, residual_ecdfs, Covariances

    def predict(self, X_pred_list):
        """ predict expected values
        INPUT:
        X_pred_list = [X_pred_load_1, X_pred_load_2, ..., X_pred_load_n]
        OUTPUT:
        y_pred = [y_pred_1,..., y_pred_n]
        y_pred_i = [demand_t1, demand_t2,...,demand_tT]_i, T = prediction horizon
        """
        y_pred = []

        if self.marginal_models is not None:
            for load_id, model in enumerate(self.marginal_models):
                y_pred.append(model.predict(X_pred_list[load_id])[0])
        else:
            print('Trained nodal load models not found')
        return y_pred

    def sample(self, X_pred_list, n_samples=200):
        """ random sampler for the nodal injections        """
        random_nodal_injection_time_t = []
        expected_y_pred = self.predict(X_pred_list=X_pred_list)
        if self.ErrorCovariances is not None:
            for load_id, cov_mat in enumerate(self.ErrorCovariances):
                samples_errors = multivariate_normal.rvs(mean=expected_y_pred[load_id] * 0, cov=cov_mat, size=n_samples)
                samples_nodal_forecast = expected_y_pred[load_id] + samples_errors
                random_nodal_injection_time_t.append(samples_nodal_forecast)
            return np.array(random_nodal_injection_time_t), expected_y_pred
        else:
            print('Trained nodal load models not found')
            return None

