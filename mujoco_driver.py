import torch
import yaml
import numpy as np
import pandas as pd
from mujoco_datagen_artificial import artificial_datagen
from mujoco_datagen_simulation import simulation_datagen
from mujoco_dataloader import testloader
from mujoco_pidnn import pidnn_driver

with open("mujoco_config.yaml", "r") as f:
    all_config = yaml.safe_load(f)
    common_config = all_config['COMMON'].copy()

data_configs = ['ARTIFICIAL_NOSTOP', 'ARTIFICIAL_STOP', 'SIMULATION_NOSTOP', 'SIMULATION_STOP']
model_configs = ['BORDER_O1', 'INTERNAL_O3', 'STRONG_FF', 'WEAK_FF']
noise_configs = [0.0, 0.01, 0.02, 0.05]

def generate_all_datasets():
    generate_data = False
    generate_ood = True

    for active_data_config in data_configs:
        active_data_config = all_config[active_data_config].copy()
        active_data_config.update(common_config)
        config = active_data_config

        # For normal data generation

        config['datafile'] = 'data.csv'
        config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
        config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])

        if generate_data:
            if config['artificial_data']:
                artificial_datagen(config)
            else:
                simulation_datagen(config)

        # For ood generation

        config['datafile'] = 'ood.csv'
        config['vx_range'] += config['ood_delta']

        if generate_ood:
            if config['artificial_data']:
                artificial_datagen(config)
            else:
                simulation_datagen(config)

# generate_all_datasets()
        
def train_all_models():
    for noise in noise_configs:
        for active_data_config in data_configs:
            active_data_config = all_config[active_data_config].copy()
            active_data_config.update(common_config)

            for active_model_config in model_configs:
                active_model_config = all_config[active_model_config].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
                config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])
                config['noise'] = noise

                pidnn_driver(config)

train_all_models()

def test_all_models():
    for noise in noise_configs:
        dicts_testdata = []
        dicts_ood = []

        for active_data_config_name in data_configs:
            active_data_config = all_config[active_data_config_name].copy()
            active_data_config.update(common_config)

            dict_testdata = dict({})
            dict_ood = dict({})

            for active_model_config_name in model_configs:
                active_model_config = all_config[active_model_config_name].copy()
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
        
        df_testdata = pd.DataFrame(dicts_testdata, index = data_configs)
        df_ood = pd.DataFrame(dicts_ood, index = data_configs)
        df_testdata.to_csv(f'inferences_testdata_{noise}.csv')
        df_ood.to_csv(f'inferences_ood_{noise}.csv')

test_all_models()


            
