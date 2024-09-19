from utils.dataloader import load_json, data_loader


if __name__ == '__main__':
    conf_path = 'config/conf_IEEE118.json'
    net_data118, df_loads118 = data_loader(conf_path)

    conf_path = 'config/conf_IEEE24.json'
    net_data_24, df_loads_24 = data_loader(conf_path)



