from utils.dataloader import data_loader
from forecasters.probabilisticmodel import Probability_model_nodal_load as Model
from xgboost import XGBRegressor  # model clss for predictor
# from mbtr.mbtr import MBT # example of different model class  (https://github.com/supsi-dacd-isaac/mbtr)


if __name__ == '__main__':
    # Example: load data network
    conf_path = 'config/conf_IEEE118.json'
    net_data118, df_loads118, conf118 = data_loader(conf_path)

    conf_path = 'config/conf_IEEE24.json'
    net_data_24, df_loads_24, conf24 = data_loader(conf_path)

    # Example: define and fit nodal injection model (example) 
    model = Model(net=net_data_24,
                  model_class=XGBRegressor,
                  window_size_past_x=24,
                  horizon_prediction_steps=24)  # predict next 24 hours using past 24

    marginals, res_ECDF, Cov = model.fit_marginals(x=df_loads_24.T)

    # Example: sampler for nodal injections
    t = 120  # a time step (the hour of the year)
    plot_samples = True
    X_pred_list = []
    for load_id in range(model.n_predictors):
        X_pred_list.append([df_loads_24.T.iloc[load_id, t - model.window_size_x: t]])
        load_samples = model.sample(x=X_pred_list, n_samples=50)
        expected_y_pred = model.predict(x=X_pred_list)