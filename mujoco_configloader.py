import yaml
import numpy as np

ACTIVE_DATA_CONFIG = 'ARTIFICIAL_NOSTOP'

try:
    with open("mujoco_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        common_config = config['COMMON'].copy()
        active_config = config[ACTIVE_DATA_CONFIG].copy()
        common_config.update(active_config)
        config = common_config
except:
    print("Some problem with the YAML file, exiting.")
    import sys
    sys.exit(1)

config['vx_range'] = np.linspace(config['VX_START'], config['VX_END'], num = config['VX_VALUES'], dtype=np.float32)
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TOTAL_ITERATIONS'], step = config['TIMESTEP'])

if __name__=="__main__":
    if config['artificial_data']:
        from mujoco_datagen_artificial import artificial_datagen
        artificial_datagen(config)
    else:
        from mujoco_datagen_simulation import simulation_datagen
        simulation_datagen(config)
