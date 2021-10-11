import torch
import yaml
import numpy as np
import pandas as pd
from mujoco_datagen_toy import toy_datagen
from mujoco_datagen_simulation import simulation_datagen
from mujoco_dataloader import testloader
from mujoco_pidnn import pidnn_driver

with open("mujoco_config.yaml", "r") as f:
    all_configs = yaml.safe_load(f)
    common_config = all_configs['COMMON'].copy()

def generate_all_datasets():
    generate_data = common_config['GENERATE_DATA']
    generate_ood = common_config['GENERATE_OOD']

    for active_data_config_name in common_config['DATA_CONFIGS']:
        active_data_config = all_configs[active_data_config_name].copy()
        active_data_config.update(common_config)
        config = active_data_config

        # For normal data generation

        config['datafile'] = 'data.csv'
        config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
        config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])

        if generate_data:
            if config['toy_data']:
                toy_datagen(config)
            else:
                simulation_datagen(config)

        # For ood generation

        config['datafile'] = 'ood.csv'
        config['vx_range'] += config['ood_delta']

        if generate_ood:
            if config['toy_data']:
                toy_datagen(config)
            else:
                simulation_datagen(config)

# generate_all_datasets()
        
def train_all_models():
    for noise in common_config['NOISE_CONFIGS']:
        for active_data_config_name in common_config['DATA_CONFIGS']:
            active_data_config = all_configs[active_data_config_name].copy()
            active_data_config.update(common_config)

            for active_model_config_name in common_config['MODEL_CONFIGS']:
                active_model_config = all_configs[active_model_config_name].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
                config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])
                config['noise'] = noise
                config['modeldir'] = 'Models/Noise_' + f'{int(100*noise)}/' + active_data_config_name + '/'

                print(f'======================={active_data_config_name}, {active_model_config_name}=======================')
                pidnn_driver(config)

# train_all_models()

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

                model = torch.load('Models/Noise_' + f'{int(100*noise)}/{active_data_config_name.lower()}/' + config['model_name'] + '.pt')
                model.eval()

                dict_testdata[active_model_config_name] = testloader(config, config['datadir'] + config['datafile'], model).item()

                config['datafile'] = 'ood.csv'
                config['vx_range'] += config['ood_delta']

                dict_ood[active_model_config_name] = testloader(config, config['datadir'] + config['datafile'], model).item()
            
            dicts_testdata.append(dict_testdata)
            dicts_ood.append(dict_ood)
        
        df_testdata = pd.DataFrame(dicts_testdata, index = common_config['DATA_CONFIGS'])
        df_ood = pd.DataFrame(dicts_ood, index = common_config['DATA_CONFIGS'])
        df_testdata.to_csv('Models/Noise_' + f'{int(100*noise)}/' + 'inferences_testdata.csv')
        df_ood.to_csv('Models/Noise_' + f'{int(100*noise)}/' + 'inferences_ood.csv')

test_all_models()


            
