from utils.dataloader import data_loader
from utils.data_preporcess import aggregate_hourly_demand
from forecasters.probabilisticmodel import Probability_model_nodal_load as Model
from xgboost import XGBRegressor  # model clss for predictor
# from mbtr.mbtr import MBT # example of different model class  (https://github.com/supsi-dacd-isaac/mbtr)


if __name__ == '__main__':
    # Example: load data network
    conf_path = 'config/conf_IEEE118.json'
    net_data118, df_loads118, conf118 = data_loader(conf_path)

    conf_path = 'config/conf_IEEE24.json'
    net_data_24, df_loads_24, conf24 = data_loader(conf_path)

    df_loads_daily = aggregate_hourly_demand(df_loads_24)

    len(df_loads_daily)

