from utils.utils import *
from pra_psa.risk_assessment import runPRA

if __name__ == '__main__':

    conf_path = './config/IEEE24_scheduler.json'
    net_data24, df_loads24, conf24 = data_loader(conf_path)

    # Use only busses with non-zero loads
    load_samples = [lsam[lsam > 0] for lsam in df_loads24.iloc[:24, :].values]

    PRA_results = runPRA(net_data24, load_time_series=load_samples)
