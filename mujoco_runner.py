import torch
import yaml
import numpy as np
import pandas as pd
from mujoco_dataloader import testloader
from mujoco_pidnn import pidnn_driver

with open("mujoco_config.yaml", "r") as f:
    all_configs = yaml.safe_load(f)
    common_config = all_configs['COMMON'].copy()

active_data_config_name = 'SIMULATION_STOP'
active_model_config_name = 'INTERNAL_O3'
noise = 0.01

active_data_config = all_configs[active_data_config_name].copy()
active_data_config.update(common_config)

active_model_config = all_configs[active_model_config_name].copy()
active_model_config.update(active_data_config)
config = active_model_config

config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])
config['noise'] = noise

config['SAVE_PLOT'] = False
config['SAVE_MODEL'] = False

pidnn_driver(config)


def test_all_models():
    for noise in common_config['NOISE_CONFIGS']:
        dicts_testdata = []
        dicts_ood = []

        for active_data_config_name in common_config['DATA_CONFIGS']:
            active_data_config = all_configs[active_data_config_name].copy()
            active_data_config.update(common_config)

            dict_testdata = dict({})
            dict_ood = dict({})

            for active_model_config_name in common_config['MODEL_CONFIGS']:
                active_model_config = all_configs[active_model_config_name].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                config['noise'] = noise

                config['datafile'] = 'data.csv'
                config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
                config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])

                model = torch.load(config['dirname'] + config['model_name'] + '_' + str(config['noise']) + '.pt')
                model.eval()

                dict_testdata[active_model_config_name] = testloader(config, config['dirname'] + config['datafile'], model).item()

                config['datafile'] = 'ood.csv'
                config['vx_range'] += config['ood_delta']

                dict_ood[active_model_config_name] = testloader(config, config['dirname'] + config['datafile'], model).item()
            
            dicts_testdata.append(dict_testdata)
            dicts_ood.append(dict_ood)
        
        df_testdata = pd.DataFrame(dicts_testdata, index = common_config['DATA_CONFIGS'])
        df_ood = pd.DataFrame(dicts_ood, index = common_config['DATA_CONFIGS'])
        df_testdata.to_csv('Models/Noise_' + f'{int(100*noise)}/' + 'inferences_testdata.csv')
        df_ood.to_csv('Models/Noise_' + f'{int(100*noise)}/' + 'inferences_ood.csv')

# test_all_models()


            
