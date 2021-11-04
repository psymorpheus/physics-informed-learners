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
active_model_config_name = 'WEAK_FF'
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
# pidnn_driver_advanced(config)