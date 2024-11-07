import matplotlib.pyplot as plt
from utils.dataloader import *
from forecasters.probabilisticmodel import Probability_model_nodal_load as Model
from xgboost import XGBRegressor  # model clss for predictor
# from mbtr.mbtr import MBT # example of different model class  (https://github.com/supsi-dacd-isaac/mbtr)

if __name__ == '__main__':

    conf_path = '../config/conf_IEEE24.json'
    forecast_model_path = os.path.join(get_project_root(), 'forecasters', 'models', 'prob_model_ieee24_xgboost')
    net_data_24, df_loads_24, conf24 = data_loader(conf_path)

    # Example: define and fit nodal injection model (example)
    model = Model(net=net_data_24,
                  model_class=XGBRegressor,
                  window_size_past_x=24,
                  horizon_prediction_steps=24)  # predict next 24 hours using past 24

    try:
        model = model.load(forecast_model_path)
    except:
        model.fit_marginals(x=df_loads_24.T)

    # Example: sampler for nodal injections
    t = 120  # a time step (the hour of the year)
    plot_samples = True
    X_pred_list = []
    for load_id in range(model.n_predictors):
        X_pred_list.append([df_loads_24.T.iloc[load_id, t - model.window_size_x: t]])
    load_samples = model.sample(x=X_pred_list, n_samples=50)
    expected_y_pred = model.predict(x=X_pred_list)

    # plot
    from utils.plotters import *
    plt.plot(pd.DataFrame(expected_y_pred).T)
    plt.xlabel('time index')
    plt.ylabel('load [MW]')
    plt.title('Expected load')
    plt.grid()
    plt.show()

    for bus_id in [7, 9, 11, 21]:
        plt.plot(pd.DataFrame(load_samples[0][bus_id]).T, 'r', alpha=0.1)
        plt.xlabel('time index')
        plt.ylabel(f'Demand sample for bus {bus_id} [MW]')
        plt.grid()
        plt.show()



