import numpy as np
from scipy.stats import ecdf as ECDF
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pra_psa.utils.utils import *


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


class ProbabilisticModel(ABC):
    """Abstract base class for the probabilistic load demand model"""
    def __init__(self, name: str = 'Abstract probabilistic model', **kwargs):
        self.name = name

    @abstractmethod
    def sample(self, x, num_samples: int):
        """Method to sample probabilistic predictions"""
        pass

    @abstractmethod
    def predict(self, x):
        """Method to predict"""
        pass

    @abstractmethod
    def fit_marginals(self, x):
        """Method to fit the model"""
        pass

    def save(self, filepath):
        """Save the trained model to a file"""
        with open(filepath, 'wb') as f:
            joblib.dump(self, f)
        print(f'Model saved to {filepath}')

    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            loaded_model = joblib.load(f)
        print(f'Model loaded from {filepath}')
        return loaded_model


class Probability_model_nodal_load(ProbabilisticModel):
    """Probabilistic model for nodal demand (vertical loads)"""

    def __init__(self, net=None, window_size_past_x: int = 24, horizon_prediction_steps: int = 6, model_class=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = 'Nodal load forecaster'
        if net is not None:
            self.reference_P_mw = net.load['p_mw']
            self.reference_Q_mvar = net.load['q_mvar']
            self.n_predictors = len(net.bus)
        else:
            logger.warning(f'Network model not provided')
            self.reference_P_mw = None
            self.reference_Q_mvar = None
            self.n_nodes = None

        if model_class is None:
            logger.info('model_class is None. Using XGBRegressor as nodal load forecaster')
            self.model_class = XGBRegressor
        else:
            self.model_class = model_class

        self.window_size_x = window_size_past_x
        self.window_size_y = horizon_prediction_steps
        self.marginal_models, self.residual_ECDFs, self.ErrorCovariances = None, None, None

    def fit_marginals_non_parallel(self, nodal_demand_data):
        """Fit marginal models for nodal demand forecasting"""
        assert self.n_predictors == len(nodal_demand_data)
        marginal_models, residual_ECDFs, residuals_list = [], [], []
        for ld_i in range(self.n_predictors):
            logger.info(f'Preparing training data for node: {ld_i + 1}/{self.n_predictors}')
            X_tr, X_te, y_tr, y_te = preprocess_nodal_load_data(nodal_data=nodal_demand_data, load_id=ld_i,
                                                                window_size_x=self.window_size_x,  window_size_y=self.window_size_y)
            model = self.model_class()
            marginal_models.append(model.fit(X_tr, y_tr))
            logger.info(f'{blue_c}Fitting completed for node: {ld_i + 1}/{self.n_predictors}{reset_c}')
            # Calculate residuals and empirical CDF
            errors = marginal_models[ld_i].predict(X_tr) - y_tr
            residuals_list.append(errors)
            residual_ECDFs.append(ECDF(errors.flatten()))

        self.marginal_models = marginal_models
        self.residual_ECDFs = residual_ECDFs
        self.ErrorCovariances = [pd.DataFrame(res_ld).cov() for res_ld in residuals_list]
        return marginal_models, residual_ECDFs, self.ErrorCovariances

    def fit_marginals(self, x):
        """Fit marginal models for nodal demand forecasting in parallel"""
        assert self.n_predictors == len(x)

        marginal_models, residual_ECDFs, error_covariances = [], [], []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    fit_single_marginal,
                    ld_i,
                    x,
                    self.model_class,
                    self.window_size_x,
                    self.window_size_y,
                    preprocess_nodal_load_data
                )
                for ld_i in range(self.n_predictors)
            ]

            for future in as_completed(futures):
                model, residual_ECDF, covariance_matrix = future.result()
                marginal_models.append(model)
                residual_ECDFs.append(residual_ECDF)
                error_covariances.append(covariance_matrix)

        # Store trained models and residuals
        self.marginal_models = marginal_models
        self.residual_ECDFs = residual_ECDFs
        self.ErrorCovariances = error_covariances

        return marginal_models, residual_ECDFs, error_covariances

    def predict(self, x):
        """Predict expected values for future nodal demand"""
        y = []
        if self.marginal_models is not None:
            for load_id, marginal_load_model in enumerate(self.marginal_models):
                y.append(marginal_load_model.predict(x[load_id])[0])
        else:
            print('Trained nodal load models not found')
        return y

    def sample(self, x, n_samples=200):
        """Random sampler for nodal injections"""
        random_nodal_injection_time_t = []
        expected_y_pred = self.predict(x=x)
        if self.ErrorCovariances is not None:
            for load_id, cov_mat in enumerate(self.ErrorCovariances):
                samples_errors = multivariate_normal.rvs(mean=expected_y_pred[load_id] * 0, cov=cov_mat, size=n_samples)
                samples_nodal_forecast = expected_y_pred[load_id] + samples_errors
                random_nodal_injection_time_t.append(samples_nodal_forecast)
            return np.array(random_nodal_injection_time_t), expected_y_pred
        else:
            print('Trained nodal load models not found')
            return None



def fit_single_marginal(ld_i, nodal_demand_data, model_class, window_size_x, window_size_y, preprocess_nodal_load_data):
    """Helper function to fit a model for a single load"""
    logger.info(f'Preparing training data for node: {ld_i + 1}')
    X_tr, X_te, y_tr, y_te = preprocess_nodal_load_data(nodal_data=nodal_demand_data, load_id=ld_i,
                                                        window_size_x=window_size_x, window_size_y=window_size_y)
    model = model_class()
    model.fit(X_tr, y_tr)
    # Calculate residuals and empirical CDF
    errors = model.predict(X_tr) - y_tr
    residual_ECDF = ECDF(errors.flatten())
    covariance_matrix = pd.DataFrame(errors).cov()
    return model, residual_ECDF, covariance_matrix


def preprocess_nodal_load_data(nodal_data, use_time_indices: bool = False, load_id: int = 0,
                               test_size: float = 0.2, window_size_x: int = 24, window_size_y: int = 12):
    """Preprocess data for a specific load"""
    X, y = [], []
    test_size = min(max(test_size, 0.05), .95)
    hour_id = np.array([np.linspace(0, 23, 24) for _ in range(364)]).flatten()
    day_id = np.array([[np.linspace(i, i, 24) for i in range(7)] for _ in range(52)]).flatten()
    for j in range(window_size_x, nodal_data.shape[1] - window_size_y):
        if use_time_indices:
            X_t = np.hstack([nodal_data.iloc[load_id, j - window_size_x:j], hour_id[j], day_id[j]])
        else:
            X_t = nodal_data.iloc[load_id, j - window_size_x:j]
        y_t = nodal_data.iloc[load_id, j:window_size_y + j]
        X.append(X_t)
        y.append(y_t)
    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=test_size)
